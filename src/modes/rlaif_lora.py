from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as f

from src.core.types import StepContext


ForwardFn = Callable[[dict[str, torch.nn.Module], dict[str, Any], StepContext], dict[str, Any]]
RewardFn = Callable[[dict[str, Any], dict[str, Any], StepContext], torch.Tensor]


def _extract_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim != logits.ndim - 1:
        raise ValueError("`labels` rank must be logits rank - 1")
    log_probs = f.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return gathered


def run_rlaif_lora_step(
    models: dict[str, torch.nn.Module],
    batch: dict[str, Any],
    ctx: StepContext,
    forward_fn: ForwardFn,
    reward_fn: RewardFn,
) -> dict[str, Any]:
    outputs = forward_fn(models, batch, ctx)

    policy_logits = outputs.get("policy_logits")
    labels = outputs.get("labels", batch.get("labels"))
    if policy_logits is None or labels is None:
        raise ValueError("RLAIF requires `policy_logits` and `labels`")

    policy_logp = _extract_log_probs(policy_logits, labels)
    policy_logp_mean = policy_logp.mean(dim=-1)

    if "reference_logits" in outputs and outputs["reference_logits"] is not None:
        reference_logp = _extract_log_probs(outputs["reference_logits"], labels).mean(dim=-1)
        kl = (policy_logp_mean - reference_logp).mean()
    else:
        kl = torch.tensor(0.0, device=policy_logits.device, dtype=policy_logits.dtype)

    reward = reward_fn(outputs, batch, ctx)
    if not isinstance(reward, torch.Tensor):
        reward = torch.tensor(reward, device=policy_logits.device, dtype=policy_logits.dtype)

    reward = reward.to(device=policy_logits.device, dtype=policy_logits.dtype)
    if reward.ndim == 0:
        reward = reward.unsqueeze(0)

    reward_scale = float(ctx.full_config.get("rlaif", {}).get("reward_scale", 1.0))
    if ctx.full_config.get("rlaif", {}).get("normalize_reward", True):
        reward = (reward - reward.mean()) / (reward.std(unbiased=False) + 1.0e-6)

    kl_coef = float(ctx.full_config.get("rlaif", {}).get("kl_coef", 0.0))
    objective = reward_scale * reward - kl_coef * (policy_logp_mean - policy_logp_mean.detach())
    loss = -(objective.mean()) + kl_coef * kl

    metrics = {
        "loss/rlaif": float(loss.detach().item()),
        "reward/mean": float(reward.detach().mean().item()),
        "kl": float(kl.detach().item()),
    }

    return {
        "loss": loss,
        "metrics": metrics,
        "aux": {"mode": ctx.mode},
    }
