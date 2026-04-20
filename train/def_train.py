from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


@dataclass
class TextCodec:
    vocab_size: int = 256

    def encode(self, text: str, max_len: int) -> list[int]:
        if max_len <= 0:
            return []
        return [ord(ch) % self.vocab_size for ch in text][:max_len]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token % 95 + 32) for token in token_ids)

class ProcessVerifier(nn.Module):
    """Placeholder verifier head.

    Production setup should replace this with think-mode LLM API call.
    This local head keeps BMPT `step` contract runnable in offline tests.
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pooled = self.embed(input_ids).mean(dim=1)
        logits = self.net(pooled)
        return torch.sigmoid(logits).squeeze(-1)


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


def build_models_from_config(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("models") or {}
    planner_cfg = model_cfg.get("planner") or {}
    builder_cfg = model_cfg.get("builder") or {}
    verifier_cfg = model_cfg.get("verifier") or {}

    vocab_size = int(planner_cfg.get("vocab_size", 4096))
    planner_hidden = int(planner_cfg.get("hidden_size", 1024))
    builder_hidden = int(builder_cfg.get("hidden_size", 1536))
    verifier_hidden = int(verifier_cfg.get("hidden_size", 1024))

    planner = ToyCausalPolicy(vocab_size=vocab_size, hidden_size=planner_hidden)
    builder = ToyCausalPolicy(vocab_size=vocab_size, hidden_size=builder_hidden)
    verifier = ProcessVerifier(vocab_size=vocab_size, hidden_size=verifier_hidden)
    verifier.requires_grad_(False)

    planner_ref = copy.deepcopy(planner)
    builder_ref = copy.deepcopy(builder)
    planner_ref.requires_grad_(False)
    builder_ref.requires_grad_(False)

    return {
        "planner": planner,
        "builder": builder,
        "verifier": verifier,
        "planner_ref": planner_ref,
        "builder_ref": builder_ref,
        "codec": TextCodec(vocab_size=vocab_size),
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


def _safe_temp(value: float) -> float:
    return max(value, 1e-6)


def _sample_with_logprob(
    model: ToyCausalPolicy,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> tuple[list[int], torch.Tensor]:
    if not prompt_ids:
        prompt_ids = [0]
    token_ids = prompt_ids.copy()
    generated: list[int] = []
    total_logprob = torch.zeros((), device=device)
    for _ in range(max_new_tokens):
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        logits = model(input_tensor)[0, -1] / _safe_temp(temperature)
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        token_id = int(sampled.item())
        token_logp = torch.log(probs[sampled]).squeeze(0)
        generated.append(token_id)
        token_ids.append(token_id)
        total_logprob = total_logprob + token_logp
    return generated, total_logprob


def _completion_logprob(
    model: ToyCausalPolicy,
    prompt_ids: list[int],
    completion_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    if not completion_ids:
        return torch.zeros((), device=device)
    context = prompt_ids.copy() if prompt_ids else [0]
    total = torch.zeros((), device=device)
    for token in completion_ids:
        input_tensor = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(input_tensor)[0, -1]
        logprobs = torch.log_softmax(logits, dim=-1)
        total = total + logprobs[token]
        context.append(token)
    return total


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
    verifier: ProcessVerifier,
    prompt_ids: list[int],
    plan_step_ids: list[int],
    prefix_ids: list[int],
    candidate_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    merged = prompt_ids + [2] + plan_step_ids + [3] + prefix_ids + [4] + candidate_ids
    if not merged:
        merged = [0]
    input_tensor = torch.tensor([merged], dtype=torch.long, device=device)
    return verifier(input_tensor).squeeze(0)


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
    prompts = batch.get("prompts") or []
    final_labels = batch.get("labels")
    if final_labels is None:
        targets = batch.get("targets") or []
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

    planner: ToyCausalPolicy = models["planner"]
    builder: ToyCausalPolicy = models["builder"]
    verifier: ProcessVerifier = models["verifier"]
    planner_ref: ToyCausalPolicy = models["planner_ref"]
    builder_ref: ToyCausalPolicy = models["builder_ref"]
    codec: TextCodec = models["codec"]
    device = next(planner.parameters()).device

    planner_losses: list[torch.Tensor] = []
    builder_losses: list[torch.Tensor] = []
    rewards_all: list[float] = []
    pass_rate_all: list[float] = []
    sampled_steps_all: list[int] = []
    plan_texts: list[str] = []
    selected_steps_text: list[list[str]] = []

    for prompt, final_label in zip(prompts, final_labels):
        prompt_ids = codec.encode(str(prompt), max_prompt_tokens)

        plan_ids, _ = _sample_with_logprob(
            model=planner,
            prompt_ids=prompt_ids,
            max_new_tokens=max_plan_tokens,
            temperature=planner_temp,
            device=device,
        )
        plan_text = codec.decode(plan_ids)
        plan_steps = _split_plan_steps(plan_text, max_steps=max_plan_steps)
        plan_texts.append(plan_text)

        if not plan_steps:
            continue

        prefix_ids: list[int] = []
        step_rewards_for_planner: list[float] = []
        local_selected_text: list[str] = []

        for plan_step in plan_steps:
            plan_step_ids = codec.encode(plan_step, planner_step_token_budget)
            step_prompt_ids = prompt_ids + [1] + plan_step_ids + [5] + prefix_ids

            group_logps: list[torch.Tensor] = []
            group_rewards: list[float] = []
            group_pass: list[int] = []
            group_probs: list[float] = []
            group_samples: list[list[int]] = []

            for _ in range(max(builder_k, 1)):
                cand_ids, cand_logp = _sample_with_logprob(
                    model=builder,
                    prompt_ids=step_prompt_ids,
                    max_new_tokens=builder_step_tokens,
                    temperature=builder_temp,
                    device=device,
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
                        prompt_ids=prompt_ids,
                        plan_step_ids=plan_step_ids,
                        prefix_ids=prefix_ids,
                        candidate_ids=cand_ids,
                        device=device,
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
            step_text = codec.decode(chosen_ids)
            local_selected_text.append(step_text)
            prefix_ids = prefix_ids + [6] + chosen_ids

        selected_steps_text.append(local_selected_text)
        planner_returns = _discounted_returns(step_rewards_for_planner, gamma=planner_gamma)
        if planner_returns:
            last_return = planner_returns[-1]
            final_label_float = float(final_label)
            planner_returns[-1] = (1.0 - planner_reward_mix_final) * last_return + planner_reward_mix_final * final_label_float

        if phase == "planner" and planner_returns:
            planner_step_logps: list[torch.Tensor] = []
            plan_prefix: list[int] = prompt_ids.copy()
            for step_text in plan_steps:
                step_ids = codec.encode(step_text + "\n", planner_step_token_budget)
                if not step_ids:
                    continue
                step_logp = _completion_logprob(
                    model=planner,
                    prompt_ids=plan_prefix + [1],
                    completion_ids=step_ids,
                    device=device,
                )
                planner_step_logps.append(step_logp)
                plan_prefix = plan_prefix + [1] + step_ids

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
