from __future__ import annotations

import random
from typing import Any

import torch
from torch import nn

from bmpt.util import _debug_print
from bmpt.util import Composer

from pbv_common import (
    BUILDER_STEP_CONSTRAINT,
    PLANNER_PROMPT_SUFFIX,
    VERIFIER_JUDGE_INSTRUCTION,
    _compose_ids,
    _decode_text,
    _get_ids_from_batch,
    _safe_temp,
    _tokenize_text,
)
from pbv_sampling import (
    _completion_logprob,
    _completion_logprob_batch,
    _sample_with_logprob,
    _sample_with_logprob_batch,
)
from pbv_verifier import QwenProcessVerifier


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


def step(models: dict[str, Any], input: dict[str, Any], engine=None) -> dict[str, Any]:
    """
    Step 函数训练 PBV (Planner-Builder-Verifier) 算法。

    Args:
        models: 模型字典（包含 planner/builder/verifier 等）
        input: 输入字典（包含 batch、config、global_step 等）
        engine: DeepSpeed Engine（被训练的模型）。如果提供，step 内部会多次调用
                engine.backward() 并在完成后返回 backward_done=True

    Returns:
        dict: {
            "loss": 标量值（无计算图）或无，
            "backward_done": True（如果 engine 提供且已内部 backward），
            "step_done": False,
            "metrics": dict 训练指标,
            "aux": dict 辅助输出,
        }
    """
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
    _ = planner_ref
    tokenizer = input.get("tokenizer")
    if tokenizer is None:
        raise ValueError("tokenizer is required in input for step function")

    composer_map = input.get("composers") or {}
    planner_composer = composer_map.get("planner_context")
    builder_composer = composer_map.get("builder_context")
    verifier_composer = composer_map.get("verifier_judge")
    if planner_composer is None or builder_composer is None or verifier_composer is None:
        raise ValueError(
            "planner_context, builder_context and verifier_judge composers are required in input['composers']."
        )

    _ = planner_composer
    device = next(planner.parameters()).device
    verifier_device = next(verifier.model.parameters()).device

    planner_losses: list[torch.Tensor] = []
    builder_losses: list[torch.Tensor] = []
    rewards_all: list[float] = []
    pass_rate_all: list[float] = []
    sampled_steps_all: list[int] = []
    plan_texts: list[str] = []
    selected_steps_text: list[list[str]] = []

    for prompt, final_label in zip(prompts, final_labels):
        prompt_text = str(prompt)
        planner_input_text = f"题目：\n{prompt_text}\n\n{PLANNER_PROMPT_SUFFIX}"
        prompt_ids_planner = _tokenize_text(tokenizer, planner_input_text, max_len=max_prompt_tokens)

        _debug_print(config, "[\033[34m训练\033[0m] 开始采样 Planner")

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

        _debug_print(config, "[\033[34m训练\033[0m] 完成采样 Planner")
        _debug_print(config, "[\033[34m训练\033[0m] " + plan_text)

        prefix_ids: list[int] = []
        step_rewards_for_planner: list[float] = []
        local_selected_text: list[str] = []

        for step_idx in range(max(max_plan_steps, 0)):
            if step_idx >= len(plan_steps):
                if engine is not None:
                    loss_builder = torch.zeros((), device=device, requires_grad=True)
                    engine.backward(loss_builder)
                continue

            plan_step = plan_steps[step_idx]
            builder_step_text = f"{plan_step}\n\n{BUILDER_STEP_CONSTRAINT}"
            builder_plan_step_ids = _tokenize_text(
                tokenizer,
                builder_step_text,
                max_len=planner_step_token_budget,
            )
            verifier_step_text = f"{plan_step}\n\n{VERIFIER_JUDGE_INSTRUCTION}"
            verifier_plan_step_ids = _tokenize_text(
                tokenizer,
                verifier_step_text,
                max_len=planner_step_token_budget,
            )
            step_prompt_ids = _compose_ids(
                composer=builder_composer,
                ids_list=[prompt_ids_planner, builder_plan_step_ids, prefix_ids],
                device=device,
            )

            _debug_print(config, "[\033[34m训练\033[0m] 开始采样 Builder")
            _debug_print(config, "[\033[33m监测\033[0m] 申请的显存：" + str(torch.cuda.memory_allocated() / 1e9) + "GB")
            _debug_print(config, "[\033[33m监测\033[0m] 保留的显存：" + str(torch.cuda.memory_reserved() / 1e9) + "GB")

            batch_results = _sample_with_logprob_batch(
                model=builder,
                tokenizer=tokenizer,
                prompt_ids=step_prompt_ids,
                num_samples=max(builder_k, 1),
                max_new_tokens=builder_step_tokens,
                temperature=builder_temp,
                device=device,
                require_grad=True,
            )

            _debug_print(config, "[\033[34m训练\033[0m] 完成采样 Builder")
            _debug_print(config, "[\033[33m监测\033[0m] 申请的显存：" + str(torch.cuda.memory_allocated() / 1e9) + "GB")
            _debug_print(config, "[\033[33m监测\033[0m] 保留的显存：" + str(torch.cuda.memory_reserved() / 1e9) + "GB")

            group_samples = [ids for ids, _ in batch_results]
            group_logps = [logp for _, logp in batch_results]

            with torch.no_grad():
                cand_logp_refs = _completion_logprob_batch(
                    model=builder_ref,
                    prompt_ids_list=[step_prompt_ids] * len(group_samples),
                    completion_ids_list=group_samples,
                    device=device,
                )

                _debug_print(config, "[\033[34m训练\033[0m] 开始批改")
                _debug_print(config, "[\033[33m监测\033[0m] 申请的显存：" + str(torch.cuda.memory_allocated() / 1e9) + "GB")
                _debug_print(config, "[\033[33m监测\033[0m] 保留的显存：" + str(torch.cuda.memory_reserved() / 1e9) + "GB")

                verifier_probs = verifier.judge_ids_batch(
                    prompt_ids_list=[prompt_ids_planner] * len(group_samples),
                    plan_step_ids_list=[verifier_plan_step_ids] * len(group_samples),
                    prefix_ids_list=[prefix_ids] * len(group_samples),
                    candidate_ids_list=group_samples,
                    composer=verifier_composer,
                    device=verifier_device,
                )
                _debug_print(config, "[\033[34m训练\033[0m] 完成批改")
                _debug_print(config, "[\033[33m监测\033[0m] 申请的显存：" + str(torch.cuda.memory_allocated() / 1e9) + "GB")
                _debug_print(config, "[\033[33m监测\033[0m] 保留的显存：" + str(torch.cuda.memory_reserved() / 1e9) + "GB")

            group_rewards: list[float] = []
            group_pass: list[int] = []
            group_probs: list[float] = []
            for i, (cand_logp, cand_logp_ref) in enumerate(zip(group_logps, cand_logp_refs)):
                verifier_prob = verifier_probs[i]
                is_pass = 1 if verifier_prob >= builder_verifier_threshold else 0
                kl_est = float((cand_logp.detach() - cand_logp_ref.detach()).item())
                reward = float(is_pass) - builder_kl_beta * kl_est
                group_rewards.append(reward)
                group_pass.append(is_pass)
                group_probs.append(verifier_prob)

            reward_tensor = torch.tensor(group_rewards, device=device)
            advantage = reward_tensor - reward_tensor.mean()
            logp_tensor = torch.stack(group_logps)
            has_advantage = float(advantage.abs().sum().item()) > 0
            if has_advantage:
                loss_builder = -(advantage.detach() * logp_tensor).mean()
            else:
                loss_builder = torch.zeros((), device=device, requires_grad=True)

            if engine is not None:
                engine.backward(loss_builder)
            else:
                if has_advantage:
                    builder_losses.append(loss_builder)

            pass_rate = float(sum(group_pass)) / float(len(group_pass))
            pass_rate_all.append(pass_rate)
            sampled_steps_all.append(len(group_pass))

            passed_indices = [idx for idx, item in enumerate(group_pass) if item == 1]
            if passed_indices:
                logits = [
                    float(logp_tensor[idx].detach().item()) / _safe_temp(builder_pass_tau)
                    for idx in passed_indices
                ]
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
                loss_planner = -(centered.detach() * logp_tensor).mean()
                if engine is not None:
                    engine.backward(loss_planner)
                else:
                    planner_losses.append(loss_planner)

        if phase == "builder":
            continue

    if engine is not None:
        if phase == "planner":
            loss_scalar = sum(planner_losses) / len(planner_losses) if planner_losses else 0.0
        else:
            loss_scalar = sum(builder_losses) / len(builder_losses) if builder_losses else 0.0
    else:
        if phase == "planner":
            if planner_losses:
                loss = torch.stack(planner_losses).mean()
                loss_scalar = float(loss.detach().item())
            else:
                loss = torch.zeros((), device=device, requires_grad=True)
                loss_scalar = 0.0
        else:
            if builder_losses:
                loss = torch.stack(builder_losses).mean()
                loss_scalar = float(loss.detach().item())
            else:
                loss = torch.zeros((), device=device, requires_grad=True)
                loss_scalar = 0.0

    metrics = {
        "loss/total": loss_scalar,
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

    if engine is not None:
        return {
            "loss": None,
            "backward_done": True,
            "step_done": False,
            "metrics": metrics,
            "aux": aux,
        }
    return {"loss": loss, "metrics": metrics, "aux": aux}


def evaluate(models: dict[str, Any], input: dict[str, Any]) -> dict[str, Any]:
    model_dict = models
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

            for idx, _ in enumerate(zip(prompts, responses)):
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
