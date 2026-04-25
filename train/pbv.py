from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from bmpt.core.distributed import (
    cleanup_distributed,
    init_distributed,
    is_main_process,
    move_to_device,
    reduce_metrics,
)
from bmpt.core.logging import MetricsEmitter, StepMetricsLogger
from bmpt.data.dataloader import build_dataloader
from bmpt.data.processor import process_all_sources
from bmpt.manager import Manager
from bmpt.prompt.composer_manager import Composer
from bmpt.tokenizer import load_tokenizer
from bmpt.toolbox import ToolBox


def _to_2d_ids(ids: torch.Tensor) -> torch.Tensor:
    if ids.dim() == 1:
        return ids.unsqueeze(0)
    return ids


def _trim_padding_1d(ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    ids = ids.to(dtype=torch.long)
    non_pad = (ids != pad_token_id).nonzero(as_tuple=False)
    if non_pad.numel() == 0:
        return ids.new_empty((0,), dtype=torch.long)
    end = int(non_pad[-1].item()) + 1
    return ids[:end]


def _split_plan_steps(plan_text: str, max_steps: int) -> list[str]:
    rows = [line.strip() for line in plan_text.replace("\r\n", "\n").split("\n")]
    steps: list[str] = []
    for row in rows:
        if not row:
            continue
        chunks = [chunk.strip() for chunk in row.split(";")]
        for chunk in chunks:
            if chunk:
                steps.append(chunk)
                if len(steps) >= max_steps:
                    return steps
    return steps[:max_steps]


def _split_plan_divs(plan_steps: list[str], div_num: int) -> list[list[str]]:
    if div_num <= 0:
        raise ValueError(f"algorithm.div_num must be >= 1, got {div_num}")

    total_steps = len(plan_steps)
    base_size = total_steps // div_num
    extra = total_steps % div_num

    divs: list[list[str]] = []
    cursor = 0
    for idx in range(div_num):
        size = base_size + (1 if idx < extra else 0)
        divs.append(plan_steps[cursor : cursor + size])
        cursor += size
    return divs


def _clone_past_key_values(past_key_values: Any, repeats: int) -> Any:
    if repeats <= 1:
        return past_key_values
    if isinstance(past_key_values, tuple):
        expanded_layers = []
        for layer in past_key_values:
            if isinstance(layer, tuple):
                expanded_layers.append(
                    tuple(t.repeat_interleave(repeats, dim=0) for t in layer)
                )
            else:
                expanded_layers.append(layer.repeat_interleave(repeats, dim=0))
        return tuple(expanded_layers)
    return past_key_values


@torch.no_grad()
def _sample_with_kv_cache(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    temperature: float,
    top_p: float,
    num_samples: int = 1,
) -> list[torch.Tensor]:
    prompt_ids = _to_2d_ids(prompt_ids).to(dtype=torch.long)
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device=device)
    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long)

    was_training = model.training
    model.eval()

    prefill = model(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        use_cache=True,
        return_dict=True,
    )
    next_logits = prefill.logits[:, -1, :]
    past = _clone_past_key_values(prefill.past_key_values, repeats=num_samples)

    if num_samples > 1:
        next_logits = next_logits.repeat_interleave(num_samples, dim=0)

    samples: list[list[int]] = [[] for _ in range(next_logits.size(0))]
    finished = torch.zeros(next_logits.size(0), dtype=torch.bool, device=device)
    input_token: torch.Tensor | None = None

    for _ in range(max_new_tokens):
        if input_token is None:
            step_logits = next_logits
        else:
            step_mask = torch.ones_like(input_token, dtype=torch.long)
            outputs = model(
                input_ids=input_token,
                attention_mask=step_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            step_logits = outputs.logits[:, -1, :]
            past = outputs.past_key_values

        if temperature > 0.0:
            probs = torch.softmax(step_logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumsum > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                next_token_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_idx.gather(-1, next_token_in_sorted)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(step_logits, dim=-1, keepdim=True)

        next_token = next_token.to(dtype=torch.long)
        for i in range(next_token.size(0)):
            if not finished[i]:
                token_id = int(next_token[i, 0].item())
                samples[i].append(token_id)
                if token_id == eos_token_id:
                    finished[i] = True

        input_token = next_token
        if bool(finished.all().item()):
            break

    if was_training:
        model.train()

    return [torch.tensor(s, dtype=torch.long, device=device) for s in samples]


def _completion_logprob_batch(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    completions: list[torch.Tensor],
    require_grad: bool,
) -> torch.Tensor:
    if not completions:
        return torch.empty(0, device=prompt_ids.device)

    model_device = next(model.parameters()).device
    prompt_ids = _to_2d_ids(prompt_ids).to(device=model_device, dtype=torch.long)
    prompt_len = int(prompt_ids.shape[1])

    full_rows: list[torch.Tensor] = []
    completion_lens: list[int] = []
    for comp in completions:
        comp = comp.to(device=model_device, dtype=torch.long)
        full = torch.cat([prompt_ids[0], comp], dim=0)
        full_rows.append(full)
        completion_lens.append(int(comp.numel()))

    max_len = max(int(row.numel()) for row in full_rows)
    pad_id = int(prompt_ids[0, -1].item())
    batch_ids = torch.full(
        (len(full_rows), max_len),
        fill_value=pad_id,
        dtype=torch.long,
        device=model_device,
    )
    batch_mask = torch.zeros_like(batch_ids, dtype=torch.long)

    for i, row in enumerate(full_rows):
        length = int(row.numel())
        batch_ids[i, :length] = row
        batch_mask[i, :length] = 1

    run_context = torch.enable_grad() if require_grad else torch.no_grad()
    with run_context:
        outputs = model(
            input_ids=batch_ids,
            attention_mask=batch_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        sums: list[torch.Tensor] = []
        for i, comp_len in enumerate(completion_lens):
            if comp_len <= 0:
                sums.append(log_probs.new_zeros(()))
                continue
            start = prompt_len - 1
            end = start + comp_len
            target = batch_ids[i, prompt_len : prompt_len + comp_len]
            token_lp = log_probs[i, start:end, :].gather(-1, target.unsqueeze(-1)).squeeze(-1)
            sums.append(token_lp.sum())
        return torch.stack(sums, dim=0)


def _decode_ids(tokenizer: Any, ids: torch.Tensor) -> str:
    if ids.numel() == 0:
        return ""
    return tokenizer.decode(ids.detach().cpu().tolist(), skip_special_tokens=True)


def _encode_text_1d(tokenizer: Any, text: str, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    return encoded["input_ids"][0].to(device=device, dtype=torch.long)


def _compose_single_input_ids(composer: Composer, outputs_1d: list[torch.Tensor]) -> torch.Tensor:
    outputs_2d = [out.unsqueeze(0) for out in outputs_1d]
    composed = composer.compose(outputs=outputs_2d)
    length = int(composed["lengths"][0].item())
    return composed["input_ids"][0, :length]


@torch.no_grad()
def _judge_candidate(
    verifier_model: torch.nn.Module,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    eos_token_id: int,
    max_new_tokens: int,
) -> float:
    input_ids = _to_2d_ids(prompt_ids).to(next(verifier_model.parameters()).device)
    sampled = _sample_with_kv_cache(
        model=verifier_model,
        prompt_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=0.0,
        top_p=1.0,
        num_samples=1,
    )[0]
    pred = _decode_ids(tokenizer, sampled).strip().lower()
    if pred.startswith("right"):
        return 1.0
    if pred.startswith("wrong"):
        return 0.0
    return 0.0


def _resolve_row_text(batch: dict[str, Any], key: str, idx: int) -> str:
    values = batch.get(key)
    if isinstance(values, list) and idx < len(values):
        return str(values[idx])
    return ""


def _train_log_prefix() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"[\033[34mTrain\033[0m] {now}"


def _format_cuda_memory(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "cuda_mem=N/A"
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(dev_idx) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(dev_idx) / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 3)
    return (
        f"cuda_mem(GB): allocated={allocated:.3f}, "
        f"reserved={reserved:.3f}, max_allocated={max_allocated:.3f}"
    )


def _log_builder_debug(
    *,
    enabled: bool,
    tokenizer: Any,
    builder_input_ids: torch.Tensor,
    candidate_ids: list[torch.Tensor],
    device: torch.device,
    row_idx: int,
    step_idx: int,
) -> None:
    if not enabled:
        return

    input_text = _decode_ids(tokenizer, builder_input_ids.squeeze(0))
    output_texts = [_decode_ids(tokenizer, ids) for ids in candidate_ids]
    mem_state = _format_cuda_memory(device)

    print(
        f"{_train_log_prefix()} builder_call row={row_idx} step={step_idx} "
        f"num_candidates={len(output_texts)}",
        flush=True,
    )
    print(f"{_train_log_prefix()} input: {input_text}", flush=True)
    for idx, text in enumerate(output_texts):
        print(f"{_train_log_prefix()} output[{idx}]: {text}", flush=True)
    print(f"{_train_log_prefix()} {mem_state}", flush=True)


def _log_builder_pre_sample_memory(
    *,
    enabled: bool,
    device: torch.device,
    row_idx: int,
    step_idx: int,
) -> None:
    if not enabled:
        return
    mem_state = _format_cuda_memory(device)
    print(
        f"{_train_log_prefix()} builder_pre_sample row={row_idx} step={step_idx} {mem_state}",
        flush=True,
    )


def train_one_batch(
    *,
    batch: dict[str, Any],
    builder_engine: Any,
    shared_model: torch.nn.Module,
    tokenizer: Any,
    planner_composer: Composer,
    builder_composer: Composer,
    verifier_composer: Composer,
    debug_rank0: bool,
    device: torch.device,
    cfg: dict[str, Any],
) -> dict[str, float]:
    algo_cfg = cfg.get("algorithm", {})
    planner_cfg = algo_cfg.get("planner", {})
    builder_cfg = algo_cfg.get("builder", {})
    verifier_cfg = algo_cfg.get("verifier", {})
    div_num = int(algo_cfg.get("div_num", 1))

    max_plan_steps = int(planner_cfg.get("max_plan_steps", 4))
    planner_max_new_tokens = int(planner_cfg.get("max_new_tokens", 96))
    builder_max_new_tokens = int(builder_cfg.get("max_new_tokens", 96))
    builder_k = int(builder_cfg.get("k_samples", 2))
    builder_temp = float(builder_cfg.get("temperature", 0.8))
    builder_top_p = float(builder_cfg.get("top_p", 0.95))
    select_tau = float(builder_cfg.get("select_tau", 1.0))
    kl_beta = float(builder_cfg.get("kl_beta", 0.02))
    verifier_threshold = float(verifier_cfg.get("threshold", 0.5))
    verifier_max_new_tokens = int(verifier_cfg.get("max_new_tokens", 6))

    if div_num < 1:
        raise ValueError(f"algorithm.div_num must be >= 1, got {div_num}")

    eos_id = int(tokenizer.eos_token_id or tokenizer.pad_token_id or 0)
    builder_model = builder_engine.module
    builder_anchor_param = next((param for param in builder_model.parameters() if param.requires_grad), None)
    if builder_anchor_param is None:
        raise ValueError("builder model must have trainable parameters")
    batch_size = int(batch["question_input_ids"].shape[0])

    metrics = {
        "loss/builder": 0.0,
        "reward/pass_rate": 0.0,
        "reward/mean": 0.0,
        "reward/kl": 0.0,
        "pbv/plans_nonempty": 0.0,
        "pbv/num_steps": 0.0,
    }

    total_builder_loss = torch.zeros((), device=device)
    reward_count = 0

    for row in range(batch_size):
        question_ids = _trim_padding_1d(batch["question_input_ids"][row], pad_token_id=eos_id)
        answer_ids = _trim_padding_1d(batch["answer_input_ids"][row], pad_token_id=eos_id)

        question_text = _resolve_row_text(batch, "question", row)
        answer_text = _resolve_row_text(batch, "answer", row)
        if not question_text:
            question_text = _decode_ids(tokenizer, question_ids)
        if not answer_text:
            answer_text = _decode_ids(tokenizer, answer_ids)

        question_text_ids = _encode_text_1d(tokenizer, question_text, device=device)
        answer_text_ids = _encode_text_1d(tokenizer, answer_text, device=device)

        planner_input = _compose_single_input_ids(
            planner_composer,
            outputs_1d=[question_text_ids],
        ).unsqueeze(0)
        plan_ids = _sample_with_kv_cache(
            model=shared_model,
            prompt_ids=planner_input,
            max_new_tokens=planner_max_new_tokens,
            eos_token_id=eos_id,
            temperature=0.8,
            top_p=0.95,
            num_samples=1,
        )[0]
        plan_text = _decode_ids(tokenizer, plan_ids)
        plan_steps = _split_plan_steps(plan_text, max_steps=max_plan_steps)

        if plan_steps:
            metrics["pbv/plans_nonempty"] += 1.0
        else:
            plan_steps = ["Generate a concise next reasoning step that moves to the final answer."]

        plan_divs = _split_plan_divs(plan_steps, div_num=div_num)

        accepted_prefix = ""

        for div_idx, plan_div in enumerate(plan_divs):
            if not plan_div:
                zero_loss = builder_anchor_param.reshape(-1)[0] * 0.0
                builder_engine.backward(zero_loss)
                continue

            plan_div_text = "\n".join(plan_div).strip()
            plan_div_ids = _encode_text_1d(tokenizer, plan_div_text, device=device)
            accepted_prefix_ids = _encode_text_1d(tokenizer, accepted_prefix, device=device)
            builder_prompt_ids = _compose_single_input_ids(
                builder_composer,
                outputs_1d=[question_text_ids, plan_div_ids, accepted_prefix_ids],
            ).unsqueeze(0)

            _log_builder_pre_sample_memory(
                enabled=debug_rank0,
                device=device,
                row_idx=row,
                step_idx=div_idx,
            )
            candidates = _sample_with_kv_cache(
                model=builder_model,
                prompt_ids=builder_prompt_ids,
                max_new_tokens=builder_max_new_tokens,
                eos_token_id=eos_id,
                temperature=builder_temp,
                top_p=builder_top_p,
                num_samples=builder_k,
            )
            _log_builder_debug(
                enabled=debug_rank0,
                tokenizer=tokenizer,
                builder_input_ids=builder_prompt_ids,
                candidate_ids=candidates,
                device=device,
                row_idx=row,
                step_idx=div_idx,
            )

            cand_logp = _completion_logprob_batch(
                builder_model,
                builder_prompt_ids,
                candidates,
                require_grad=True,
            )
            cand_logp_ref = _completion_logprob_batch(
                shared_model,
                builder_prompt_ids,
                candidates,
                require_grad=False,
            )

            verifier_scores: list[float] = []
            candidate_texts: list[str] = []
            for cand in candidates:
                cand_text = _decode_ids(tokenizer, cand)
                candidate_texts.append(cand_text)
                candidate_ids = _encode_text_1d(tokenizer, cand_text, device=device)
                verifier_prompt_ids = _compose_single_input_ids(
                    verifier_composer,
                    outputs_1d=[
                        question_text_ids,
                        plan_div_ids,
                        accepted_prefix_ids,
                        candidate_ids,
                        answer_text_ids,
                    ],
                )
                verifier_scores.append(
                    _judge_candidate(
                        verifier_model=shared_model,
                        tokenizer=tokenizer,
                        prompt_ids=verifier_prompt_ids,
                        eos_token_id=eos_id,
                        max_new_tokens=verifier_max_new_tokens,
                    )
                )

            verifier_tensor = torch.tensor(verifier_scores, dtype=torch.float32, device=device)
            is_pass = (verifier_tensor >= verifier_threshold).to(dtype=torch.float32)

            kl_est = cand_logp - cand_logp_ref.to(device=device, dtype=cand_logp.dtype)
            reward = is_pass - kl_beta * kl_est.detach()
            advantage = reward - reward.mean()
            step_loss = -(advantage.detach() * cand_logp).mean()

            builder_engine.backward(step_loss)
            total_builder_loss = total_builder_loss + step_loss.detach()

            metrics["reward/pass_rate"] += float(is_pass.mean().item())
            metrics["reward/mean"] += float(reward.mean().item())
            metrics["reward/kl"] += float(kl_est.mean().item())
            reward_count += 1

            pass_indices = torch.nonzero(is_pass > 0.5, as_tuple=False).squeeze(-1)
            if pass_indices.numel() > 0:
                selected_logp = cand_logp.detach()[pass_indices]
                probs = torch.softmax(selected_logp / max(select_tau, 1.0e-6), dim=0)
                local_pick = int(torch.multinomial(probs, num_samples=1).item())
                chosen_idx = int(pass_indices[local_pick].item())
            else:
                chosen_idx = int(torch.argmax(verifier_tensor).item())

            chosen_text = candidate_texts[chosen_idx].strip()
            if chosen_text:
                accepted_prefix = f"{accepted_prefix}\n{chosen_text}".strip()

        metrics["pbv/num_steps"] += float(len(plan_steps))

    if reward_count > 0:
        metrics["loss/builder"] = float(total_builder_loss.item() / reward_count)
        metrics["reward/pass_rate"] /= reward_count
        metrics["reward/mean"] /= reward_count
        metrics["reward/kl"] /= reward_count
    metrics["pbv/plans_nonempty"] /= max(batch_size, 1)
    metrics["pbv/num_steps"] /= max(batch_size, 1)

    return metrics


def main() -> None:
    config_path = Path(__file__).parent / "config_pbv.yaml"

    manager = Manager()
    manager.load_config(config_path)
    config = manager.config

    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})

    if str(runtime_cfg.get("attn_implementation", "")).lower() != "flash_attention_2":
        raise ValueError("PBV requires runtime.attn_implementation=flash_attention_2")
    if not bool(runtime_cfg.get("gradient_checkpointing", False)):
        raise ValueError("PBV requires runtime.gradient_checkpointing=true")

    zero_stage = int(manager.deepspeed_config.get("zero_optimization", {}).get("stage", -1))
    if zero_stage != 2:
        raise ValueError(f"PBV requires DeepSpeed ZeRO stage 2, got stage={zero_stage}")

    dist_ctx = init_distributed(backend="nccl")
    is_rank0 = is_main_process(dist_ctx)

    if is_rank0:
        print("[pbv] config loaded", flush=True)
        print(f"[pbv] distributed: rank={dist_ctx.rank} world_size={dist_ctx.world_size}", flush=True)

    tokenizer = load_tokenizer(config, imp_model="builder")
    pad_token_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
    composers = manager.load_composers()
    planner_composer = composers.get("planner")
    builder_composer = composers.get("builder")
    verifier_composer = composers.get("verifier")
    if planner_composer is None or builder_composer is None or verifier_composer is None:
        raise ValueError("prompting.composers must include planner, builder, verifier")

    processed = process_all_sources(config, tokenizer)
    train_records = processed.get("train")
    if train_records is None:
        raise ValueError("No 'train' source found in config")

    train_loader = build_dataloader(
        train_records,
        config,
        dist_ctx,
        shuffle=True,
        pad_token_id=pad_token_id,
    )

    toolbox = ToolBox(manager)
    models = toolbox.load_models("builder")
    builder_engine = toolbox.engine
    if builder_engine is None:
        raise ValueError("DeepSpeed engine not initialized for builder")

    shared_model = models.get("shared_frozen")
    if shared_model is None:
        raise ValueError("config.models.shared_frozen is required")
    if not isinstance(shared_model, torch.nn.Module):
        raise ValueError("shared_frozen must be a torch.nn.Module")

    shared_model = shared_model.to(dist_ctx.device)
    shared_model.eval()
    shared_model.requires_grad_(False)

    metrics_cfg = runtime_cfg.get("metrics", {})
    perf_logger = StepMetricsLogger.from_config(metrics_cfg)
    metrics_emitter = MetricsEmitter.from_config(metrics_cfg)

    log_every = int(train_cfg.get("log_every_steps", 10))
    total_steps = int(train_cfg.get("max_steps", 1000))

    if is_rank0:
        print("[pbv] engine initialized (builder trainable, shared frozen)", flush=True)

    step = 0
    try:
        for epoch_idx in range(10_000):
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch_idx)

            for batch in train_loader:
                if step >= total_steps:
                    break

                step += 1
                t0 = time.perf_counter()
                device_batch = move_to_device(batch, dist_ctx.device)

                pbv_metrics = train_one_batch(
                    batch=device_batch,
                    builder_engine=builder_engine,
                    shared_model=shared_model,
                    tokenizer=tokenizer,
                    planner_composer=planner_composer,
                    builder_composer=builder_composer,
                    verifier_composer=verifier_composer,
                    debug_rank0=is_rank0,
                    device=dist_ctx.device,
                    cfg=config,
                )

                builder_engine.step()

                elapsed = time.perf_counter() - t0
                metrics = {
                    **pbv_metrics,
                    "train/step": float(step),
                    "train/epoch": float(epoch_idx),
                }
                perf_metrics = perf_logger.update(
                    step_time_sec=elapsed,
                    batch=device_batch,
                    device=dist_ctx.device,
                    sync_global=(step % log_every == 0),
                )
                metrics.update(perf_metrics)

                if step % log_every == 0:
                    reduced = reduce_metrics(metrics, dist_ctx)
                    if is_rank0:
                        metrics_emitter.emit(step_id=step, metrics=reduced)

            if step >= total_steps:
                break
    finally:
        cleanup_distributed()

    if is_rank0:
        print(f"[pbv] training finished: total_steps={step}", flush=True)


if __name__ == "__main__":
    main()
