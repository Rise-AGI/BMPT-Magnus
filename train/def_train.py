from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

from bmpt.util import Composer

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for yaml configs.")
        loaded = yaml.safe_load(text)
        return loaded or {}
    raise ValueError(f"Unsupported config format: {config_path}")


def _require_transformers() -> None:
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError("PBV requires `transformers` to be installed.")


def _safe_temp(value: float) -> float:
    return max(value, 1e-6)


def _load_tokenizer(model_path: str) -> Any:
    _require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


def _load_model(model_path: str, attn_implementation: str = "auto") -> nn.Module:
    _require_transformers()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    return model


def _tokenize_text(tokenizer: Any, text: str, max_len: int | None = None) -> list[int]:
    encoded = tokenizer(str(text), add_special_tokens=False, return_attention_mask=False)
    token_ids = [int(token) for token in encoded.get("input_ids", [])]
    if max_len is not None and max_len > 0:
        token_ids = token_ids[:max_len]
    return token_ids


def _get_ids_from_batch(
    batch: dict[str, Any],
    field: str,
    index: int,
    max_len: int | None = None,
) -> list[int]:
    ids_key = f"{field}_input_ids"
    if ids_key in batch and isinstance(batch[ids_key], torch.Tensor):
        ids_tensor = batch[ids_key][index]
        ids = [int(tok) for tok in ids_tensor.tolist()]
        if max_len is not None and max_len > 0:
            ids = ids[:max_len]
        return ids
    return []


def _decode_text(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _to_single_batch_tensor(token_ids: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def _compose_ids(
    composer: Composer,
    ids_list: list[list[int]],
    device: torch.device,
) -> list[int]:
    outputs: list[torch.Tensor] = []
    for ids in ids_list:
        outputs.append(_to_single_batch_tensor(ids, device=device))
    composed = composer.compose(outputs=outputs)
    valid_len = int(composed["lengths"][0].item())
    return [int(item) for item in composed["input_ids"][0, :valid_len].tolist()]


class QwenProcessVerifier(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_new_tokens: int = 6,
        temperature: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max(max_new_tokens, 1)
        self.temperature = max(float(temperature), 0.0)

    def _decode_judgement(self, text: str) -> float:
        normalized = text.strip().lower()
        if normalized.startswith("right"):
            return 1.0
        if normalized.startswith("wrong"):
            return 0.0
        return 0.0

    def judge_ids(
        self,
        prompt_ids: list[int],
        plan_step_ids: list[int],
        prefix_ids: list[int],
        candidate_ids: list[int],
        composer: Composer,
        device: torch.device,
    ) -> float:
        composed = composer.compose(
            outputs=[
                _to_single_batch_tensor(prompt_ids, device=device),
                _to_single_batch_tensor(plan_step_ids, device=device),
                _to_single_batch_tensor(prefix_ids, device=device),
                _to_single_batch_tensor(candidate_ids, device=device),
            ]
        )

        input_ids = composed["input_ids"]
        attention_mask = composed["attention_mask"]
        prompt_len = int(composed["lengths"][0].item())
        do_sample = self.temperature > 0
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=_safe_temp(self.temperature),
                pad_token_id=int(self.tokenizer.pad_token_id),
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )

        generated_tail = generated[0, prompt_len:]
        text = self.tokenizer.decode(generated_tail.tolist(), skip_special_tokens=True)
        return self._decode_judgement(text)


def build_models_from_config(config: dict[str, Any], loader_fn: Any = None) -> dict[str, Any]:
    _ = loader_fn
    _require_transformers()

    model_cfg = config.get("models") or {}
    planner_cfg = model_cfg.get("planner") or {}
    builder_cfg = model_cfg.get("builder") or {}
    verifier_cfg = model_cfg.get("verifier") or {}

    runtime_cfg = config.get("runtime") or {}
    attn_implementation = runtime_cfg.get("attn_implementation", "auto")

    tokenizer_path = str(
        verifier_cfg.get("path")
        or planner_cfg.get("path")
        or builder_cfg.get("path")
        or ""
    )
    if not tokenizer_path:
        raise ValueError("models.verifier.path (or planner/builder path) is required.")

    tokenizer = _load_tokenizer(tokenizer_path)

    planner = _load_model(
        str(planner_cfg.get("path", tokenizer_path)),
        attn_implementation=attn_implementation,
    )
    builder = _load_model(
        str(builder_cfg.get("path", tokenizer_path)),
        attn_implementation=attn_implementation,
    )
    verifier_model = _load_model(
        str(verifier_cfg.get("path", tokenizer_path)),
        attn_implementation=attn_implementation,
    )
    verifier = QwenProcessVerifier(
        model=verifier_model,
        tokenizer=tokenizer,
        max_new_tokens=int(verifier_cfg.get("max_new_tokens", 6)),
        temperature=float(verifier_cfg.get("temperature", 0.0)),
    )
    verifier.requires_grad_(False)
    verifier.eval()

    planner_ref = copy.deepcopy(planner)
    builder_ref = copy.deepcopy(builder)
    planner_ref.requires_grad_(False)
    builder_ref.requires_grad_(False)

    enable_gradient_ckpt = bool(runtime_cfg.get("gradient_checkpointing", False))
    if enable_gradient_ckpt:
        for model in [planner, builder]:
            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

    planner_trainable = bool(planner_cfg.get("trainable", False))
    builder_trainable = bool(builder_cfg.get("trainable", False))
    if not planner_trainable:
        planner.requires_grad_(False)
    if not builder_trainable:
        builder.requires_grad_(False)

    return {
        "planner": planner,
        "builder": builder,
        "verifier": verifier,
        "planner_ref": planner_ref,
        "builder_ref": builder_ref,
    }


def build_optimizers_from_config(models: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    optimizer_cfg = config.get("optimizer") or {}
    lr = float(optimizer_cfg.get("lr", 2e-5))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.01))
    betas = optimizer_cfg.get("betas") or [0.9, 0.95]
    eps = float(optimizer_cfg.get("eps", 1e-8))

    planner_params = list(models["planner"].parameters())
    builder_params = list(models["builder"].parameters())
    all_params = planner_params + builder_params

    optimizer = torch.optim.AdamW(
        all_params,
        lr=lr,
        betas=(float(betas[0]), float(betas[1])),
        eps=eps,
        weight_decay=weight_decay,
    )
    return {"policy": optimizer}


def _sample_with_logprob(
    model: nn.Module,
    tokenizer: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    require_grad: bool = False,
) -> tuple[list[int], torch.Tensor]:
    if not prompt_ids:
        prompt_ids = [0]

    input_ids = _to_single_batch_tensor(prompt_ids, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    do_sample = temperature > 0

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=_safe_temp(temperature),
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = generated["sequences"][0].tolist()
    new_ids = generated_ids[len(prompt_ids):]

    if not new_ids:
        return new_ids, torch.zeros((), device=device, requires_grad=require_grad)

    if require_grad:
        full_ids = prompt_ids + new_ids
        full_input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        outputs = model(full_input_ids)
        logits = outputs.logits[:, len(prompt_ids) - 1 : len(full_ids) - 1, :]
        logprobs = torch.log_softmax(logits[0], dim=-1)
        total_logprob = torch.zeros((), device=device)
        for idx, token_id in enumerate(new_ids):
            total_logprob = total_logprob + logprobs[idx, token_id]
    else:
        if "scores" in generated and len(generated["scores"]) > 0:
            scores_stack = torch.stack(generated["scores"], dim=1)
            logprobs = torch.log_softmax(scores_stack[0], dim=-1)
            total_logprob = torch.zeros((), device=device)
            for idx, token_id in enumerate(new_ids):
                if idx < logprobs.size(0):
                    total_logprob = total_logprob + logprobs[idx, token_id]
        else:
            total_logprob = torch.zeros((), device=device)

    return new_ids, total_logprob


def _completion_logprob(
    model: nn.Module,
    prompt_ids: list[int],
    completion_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    if not completion_ids:
        return torch.zeros((), device=device)

    context_ids = prompt_ids.copy() if prompt_ids else [0]
    total_logprob = torch.zeros((), device=device)

    model.eval()
    with torch.no_grad():
        for token_id in completion_ids:
            input_ids = _to_single_batch_tensor(context_ids, device=device)
            outputs = model(input_ids)
            logits = outputs.logits[0, -1]
            logprobs = torch.log_softmax(logits, dim=-1)
            total_logprob = total_logprob + logprobs[token_id]
            context_ids.append(token_id)

    return total_logprob


def _split_plan_steps(plan_text: str, max_steps: int) -> list[str]:
    text = plan_text.replace("\r", "\n")
    chunks = []
    for raw in text.split("\n"):
        part = raw.strip()
        if not part:
            continue
        for piece in part.split(";"):
            item = piece.strip()
            if item:
                chunks.append(item)
    if not chunks:
        stripped = plan_text.strip()
        chunks = [stripped] if stripped else []
    if max_steps > 0:
        return chunks[:max_steps]
    return chunks


def _discounted_returns(step_rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0 for _ in step_rewards]
    running = 0.0
    for index in range(len(step_rewards) - 1, -1, -1):
        running = step_rewards[index] + gamma * running
        returns[index] = running
    return returns


def _verifier_prob(
    verifier: QwenProcessVerifier,
    prompt_ids: list[int],
    plan_step_ids: list[int],
    prefix_ids: list[int],
    candidate_ids: list[int],
    composer: Composer,
    device: torch.device,
) -> torch.Tensor:
    score = verifier.judge_ids(
        prompt_ids=prompt_ids,
        plan_step_ids=plan_step_ids,
        prefix_ids=prefix_ids,
        candidate_ids=candidate_ids,
        composer=composer,
        device=device,
    )
    return torch.tensor(score, dtype=torch.float32, device=device)


def _weighted_pick(indices: list[int], weights: list[float]) -> int:
    if not indices:
        return -1
    total = sum(weights)
    if total <= 0:
        return random.choice(indices)
    threshold = random.random() * total
    acc = 0.0
    for idx, weight in zip(indices, weights):
        acc += max(weight, 0.0)
        if acc >= threshold:
            return idx
    return indices[-1]


def step(models: dict[str, Any], input: dict[str, Any]) -> dict[str, Any]:
    config = input.get("_merged_config") or {}
    algorithm_cfg = config.get("algorithm") or {}
    schedule_cfg = algorithm_cfg.get("training_schedule") or {}
    planner_cfg = algorithm_cfg.get("planner") or {}
    builder_cfg = algorithm_cfg.get("builder") or {}

    planner_steps = int(schedule_cfg.get("planner_steps", 1))
    builder_steps = int(schedule_cfg.get("builder_steps", 2))
    cycle = max(planner_steps + builder_steps, 1)
    global_step = int(input.get("global_step", 0))
    phase = "planner" if (global_step % cycle) < planner_steps else "builder"

    max_prompt_tokens = int(algorithm_cfg.get("max_prompt_tokens", 256))
    max_plan_tokens = int(planner_cfg.get("max_plan_tokens", 128))
    max_plan_steps = int(planner_cfg.get("max_plan_steps", 10))
    planner_gamma = float(planner_cfg.get("gamma", 0.95))
    planner_temp = float(planner_cfg.get("temperature", 0.8))
    planner_step_token_budget = int(planner_cfg.get("step_token_budget", 64))
    planner_reward_mix_final = float(planner_cfg.get("reward_mix_final", 0.1))

    builder_k = int(builder_cfg.get("k_samples", 8))
    builder_step_tokens = int(builder_cfg.get("step_max_tokens", 64))
    builder_temp = float(builder_cfg.get("temperature", 1.0))
    builder_kl_beta = float(builder_cfg.get("kl_beta", 0.02))
    builder_pass_tau = float(builder_cfg.get("pass_select_tau", 0.7))
    builder_verifier_threshold = float(builder_cfg.get("verifier_threshold", 0.5))

    batch = input.get("batch") or {}
    prompts = batch.get("prompt") or batch.get("prompts") or []
    final_labels = batch.get("labels")
    if final_labels is None:
        targets = batch.get("response") or batch.get("targets") or []
        if targets:
            final_labels = [1.0 if str(item).strip() else 0.0 for item in targets]
        else:
            final_labels = [0.0 for _ in prompts]

    if not prompts:
        dummy = torch.zeros((), requires_grad=True)
        return {
            "loss": dummy,
            "metrics": {
                "loss/total": 0.0,
                "phase/is_planner": 1.0 if phase == "planner" else 0.0,
            },
            "aux": {"plans": [], "selected_steps": []},
        }

    planner: nn.Module = models["planner"]
    builder: nn.Module = models["builder"]
    verifier: QwenProcessVerifier = models["verifier"]
    planner_ref: nn.Module = models["planner_ref"]
    builder_ref: nn.Module = models["builder_ref"]
    tokenizer = input.get("tokenizer")
    if tokenizer is None:
        raise ValueError("tokenizer is required in input for step function")

    composer_map = input.get("composers") or {}
    planner_composer = composer_map.get("planner_context")
    builder_composer = composer_map.get("builder_context")
    verifier_composer = composer_map.get("verifier_judge")
    if planner_composer is None or builder_composer is None or verifier_composer is None:
        raise ValueError("planner_context, builder_context and verifier_judge composers are required in input['composers'].")

    device = next(planner.parameters()).device
    verifier_device = next(verifier.model.parameters()).device

    planner_losses: list[torch.Tensor] = []
    builder_losses: list[torch.Tensor] = []
    rewards_all: list[float] = []
    pass_rate_all: list[float] = []
    sampled_steps_all: list[int] = []
    plan_texts: list[str] = []
    selected_steps_text: list[list[str]] = []

    for sample_idx, (prompt, final_label) in enumerate(zip(prompts, final_labels)):
        prompt_text = str(prompt)
        prompt_ids_planner = _get_ids_from_batch(
            batch=batch,
            field="prompt",
            index=sample_idx,
            max_len=max_prompt_tokens,
        )

        plan_ids, _ = _sample_with_logprob(
            model=planner,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids_planner,
            max_new_tokens=max_plan_tokens,
            temperature=planner_temp,
            device=device,
        )
        plan_text = _decode_text(tokenizer, plan_ids)
        plan_steps = _split_plan_steps(plan_text, max_steps=max_plan_steps)
        plan_texts.append(plan_text)

        if not plan_steps:
            continue

        prefix_ids: list[int] = []
        step_rewards_for_planner: list[float] = []
        local_selected_text: list[str] = []

        for plan_step in plan_steps:
            plan_step_ids = _tokenize_text(tokenizer, plan_step, max_len=planner_step_token_budget)
            step_prompt_ids = _compose_ids(
                composer=builder_composer,
                ids_list=[prompt_ids_planner, plan_step_ids, prefix_ids],
                device=device,
            )

            group_logps: list[torch.Tensor] = []
            group_rewards: list[float] = []
            group_pass: list[int] = []
            group_probs: list[float] = []
            group_samples: list[list[int]] = []

            for _ in range(max(builder_k, 1)):
                cand_ids, cand_logp = _sample_with_logprob(
                    model=builder,
                    tokenizer=tokenizer,
                    prompt_ids=step_prompt_ids,
                    max_new_tokens=builder_step_tokens,
                    temperature=builder_temp,
                    device=device,
                    require_grad=True,
                )
                with torch.no_grad():
                    cand_logp_ref = _completion_logprob(
                        model=builder_ref,
                        prompt_ids=step_prompt_ids,
                        completion_ids=cand_ids,
                        device=device,
                    )
                    verifier_prob = _verifier_prob(
                        verifier=verifier,
                        prompt_ids=prompt_ids_planner,
                        plan_step_ids=plan_step_ids,
                        prefix_ids=prefix_ids,
                        candidate_ids=cand_ids,
                        composer=verifier_composer,
                        device=verifier_device,
                    )
                is_pass = 1 if float(verifier_prob.item()) >= builder_verifier_threshold else 0
                kl_est = float((cand_logp.detach() - cand_logp_ref.detach()).item())
                reward = float(is_pass) - builder_kl_beta * kl_est

                group_logps.append(cand_logp)
                group_rewards.append(reward)
                group_pass.append(is_pass)
                group_probs.append(float(verifier_prob.detach().item()))
                group_samples.append(cand_ids)

            reward_tensor = torch.tensor(group_rewards, device=device)
            advantage = reward_tensor - reward_tensor.mean()
            logp_tensor = torch.stack(group_logps)
            if float(advantage.abs().sum().item()) > 0:
                builder_losses.append(-(advantage.detach() * logp_tensor).mean())

            pass_rate = float(sum(group_pass)) / float(len(group_pass))
            pass_rate_all.append(pass_rate)
            sampled_steps_all.append(len(group_pass))

            passed_indices = [idx for idx, item in enumerate(group_pass) if item == 1]
            if passed_indices:
                logits = [float(logp_tensor[idx].detach().item()) / _safe_temp(builder_pass_tau) for idx in passed_indices]
                logits_max = max(logits)
                weights = [pow(2.718281828, value - logits_max) for value in logits]
                chosen_local = _weighted_pick(passed_indices, weights)
            else:
                best_idx = max(range(len(group_probs)), key=lambda idx: group_probs[idx])
                chosen_local = best_idx

            chosen_ids = group_samples[chosen_local]
            chosen_reward = 1.0 if group_pass[chosen_local] == 1 else 0.0
            step_rewards_for_planner.append(chosen_reward)
            rewards_all.append(chosen_reward)
            step_text = _decode_text(tokenizer, chosen_ids)
            local_selected_text.append(step_text)
            prefix_ids = prefix_ids + chosen_ids

        selected_steps_text.append(local_selected_text)
        planner_returns = _discounted_returns(step_rewards_for_planner, gamma=planner_gamma)
        if planner_returns:
            last_return = planner_returns[-1]
            final_label_float = float(final_label)
            planner_returns[-1] = (1.0 - planner_reward_mix_final) * last_return + planner_reward_mix_final * final_label_float

        if phase == "planner" and planner_returns:
            planner_step_logps: list[torch.Tensor] = []
            plan_prefix: list[int] = prompt_ids_planner.copy()
            for step_text in plan_steps:
                step_ids = _tokenize_text(tokenizer, step_text + "\n", max_len=planner_step_token_budget)
                if not step_ids:
                    continue
                step_logp = _completion_logprob(
                    model=planner,
                    prompt_ids=plan_prefix,
                    completion_ids=step_ids,
                    device=device,
                )
                planner_step_logps.append(step_logp)
                plan_prefix = plan_prefix + step_ids

            upper = min(len(planner_returns), len(planner_step_logps))
            if upper > 0:
                returns_tensor = torch.tensor(planner_returns[:upper], device=device)
                centered = returns_tensor - returns_tensor.mean()
                logp_tensor = torch.stack(planner_step_logps[:upper])
                planner_losses.append(-(centered.detach() * logp_tensor).mean())

        if phase == "builder":
            continue

    if phase == "planner":
        if planner_losses:
            loss = torch.stack(planner_losses).mean()
        else:
            loss = torch.zeros((), device=device, requires_grad=True)
    else:
        if builder_losses:
            loss = torch.stack(builder_losses).mean()
        else:
            loss = torch.zeros((), device=device, requires_grad=True)

    metrics = {
        "loss/total": float(loss.detach().item()),
        "phase/is_planner": 1.0 if phase == "planner" else 0.0,
        "phase/is_builder": 1.0 if phase == "builder" else 0.0,
        "reward/step_pass": float(sum(rewards_all) / len(rewards_all)) if rewards_all else 0.0,
        "builder/pass_rate": float(sum(pass_rate_all) / len(pass_rate_all)) if pass_rate_all else 0.0,
        "builder/k_effective": float(sum(sampled_steps_all) / len(sampled_steps_all)) if sampled_steps_all else 0.0,
    }
    aux = {
        "plans": plan_texts,
        "selected_steps": selected_steps_text,
        "phase": phase,
    }
    return {"loss": loss, "metrics": metrics, "aux": aux}


def evaluate(models: dict[str, Any], input: dict[str, Any]) -> dict[str, Any]:
    config = input.get("_merged_config") or {}
    model_dict = models
    tokenizer = model_dict.get("tokenizer")
    verifier: QwenProcessVerifier = model_dict.get("verifier")

    val_iterable = input.get("val_iterable")
    phase = str(input.get("phase", "eval"))

    if val_iterable is None:
        return {"metrics": {f"{phase}/num_batches": 0.0}, "aux": {}}

    verifier_device = next(verifier.model.parameters()).device

    pass_count = 0
    total_count = 0
    batch_count = 0

    verifier.eval()
    verifier_composer = model_dict.get("verifier_composer")

    with torch.no_grad():
        for batch in val_iterable:
            prompts = batch.get("prompt") or batch.get("prompts") or []
            responses = batch.get("response") or batch.get("targets") or []

            for idx, (prompt_text, response_text) in enumerate(zip(prompts, responses)):
                prompt_ids = _get_ids_from_batch(
                    batch=batch,
                    field="prompt",
                    index=idx,
                )
                response_ids = _get_ids_from_batch(
                    batch=batch,
                    field="response",
                    index=idx,
                )

                verifier_prob = _verifier_prob(
                    verifier=verifier,
                    prompt_ids=prompt_ids,
                    plan_step_ids=[],
                    prefix_ids=[],
                    candidate_ids=response_ids,
                    composer=verifier_composer,
                    device=verifier_device,
                )
                if float(verifier_prob.item()) >= 0.5:
                    pass_count += 1
                total_count += 1

            batch_count += 1

    pass_rate = float(pass_count) / float(total_count) if total_count > 0 else 0.0

    return {
        "metrics": {
            f"{phase}/pass_rate": pass_rate,
            f"{phase}/num_samples": float(total_count),
            f"{phase}/num_batches": float(batch_count),
        },
        "aux": {},
    }