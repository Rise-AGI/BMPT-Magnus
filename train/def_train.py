from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as f

from src.core.types import StepContext
from util.train_utils import (
    build_models_from_config as _build_models_from_config,
    load_config_cached,
    resolve_callbacks,
    resolve_models,
    resolve_step_config,
)


_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    target = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    return load_config_cached(target)


def build_models_from_config(config: dict[str, Any], loader_fn):
    return _build_models_from_config(config, loader_fn=loader_fn)


def default_forward(
    models: dict[str, torch.nn.Module],
    batch: dict[str, Any],
    _ctx: StepContext,
) -> Any:
    policy_model = models["policy"]
    return policy_model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch.get("labels"),
    )


def default_reward(
    _outputs: dict[str, Any],
    batch: dict[str, Any],
    _ctx: StepContext,
) -> torch.Tensor:
    batch_size = batch["input_ids"].shape[0]
    device = batch["input_ids"].device
    return torch.zeros(batch_size, device=device)


def _build_step_context(mode: str, global_step: int, merged_config: dict[str, Any]) -> StepContext:
    runtime_cfg = merged_config.get("runtime", {})
    return StepContext(
        mode=mode,
        global_step=global_step,
        runtime_config=runtime_cfg,
        full_config=merged_config,
    )


def _extract_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = f.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def sft_step(
    models: dict[str, torch.nn.Module],
    batch: dict[str, Any],
    ctx: StepContext,
    forward_fn: Callable[..., Any],
) -> dict[str, Any]:
    """User-owned SFT logic. Modify this function for custom SFT training."""
    outputs = forward_fn(models, batch, ctx)
    loss = outputs["loss"].mean()

    return {
        "loss": loss,
        "metrics": {"loss/sft": float(loss.detach().item())},
        "aux": {"mode": ctx.mode},
    }


def rlaif_lora_step(
    models: dict[str, torch.nn.Module],
    batch: dict[str, Any],
    ctx: StepContext,
    forward_fn: Callable[..., Any],
    reward_fn: Callable[..., torch.Tensor],
) -> dict[str, Any]:
    """User-owned RLAIF-LoRA logic. Modify this function for custom RL training."""
    outputs = forward_fn(models, batch, ctx)
    policy_logits = outputs["policy_logits"]
    labels = outputs["labels"]

    policy_logp = _extract_log_probs(policy_logits, labels).mean(dim=-1)

    reference_logits = outputs.get("reference_logits")
    if reference_logits is None:
        kl = torch.zeros((), device=policy_logits.device, dtype=policy_logits.dtype)
    else:
        reference_logp = _extract_log_probs(reference_logits, labels).mean(dim=-1)
        kl = (policy_logp - reference_logp).mean()

    reward = reward_fn(outputs, batch, ctx)
    reward = reward.to(device=policy_logits.device, dtype=policy_logits.dtype)

    rlaif_cfg = ctx.full_config.get("rlaif", {})
    reward_scale = float(rlaif_cfg.get("reward_scale", 1.0))
    if rlaif_cfg.get("normalize_reward", True):
        reward = (reward - reward.mean()) / (reward.std(unbiased=False) + 1.0e-6)

    kl_coef = float(rlaif_cfg.get("kl_coef", 0.0))
    objective = reward_scale * reward - kl_coef * (policy_logp - policy_logp.detach())
    loss = -(objective.mean()) + kl_coef * kl

    return {
        "loss": loss,
        "metrics": {
            "loss/rlaif": float(loss.detach().item()),
            "reward/mean": float(reward.detach().mean().item()),
            "kl": float(kl.detach().item()),
        },
        "aux": {"mode": ctx.mode},
    }


def step(models, input):
    """Run one train step with unified protocol.

    Args:
        models: dict[str, torch.nn.Module] | torch.nn.Module
        input: dict[str, Any]

    Returns:
        dict with keys: loss, metrics, aux
    """

    merged_config = resolve_step_config(input, _DEFAULT_CONFIG_PATH)

    mode = input.get("mode", merged_config.get("mode", "sft"))
    global_step = int(input.get("global_step", 0))
    ctx = _build_step_context(mode=mode, global_step=global_step, merged_config=merged_config)

    model_dict = resolve_models(models, merged_config, input)

    batch = input["batch"]
    forward_fn, reward_fn = resolve_callbacks(input, default_forward, default_reward)

    if mode == "sft":
        return sft_step(model_dict, batch, ctx, forward_fn=forward_fn)

    if mode == "rlaif_lora":
        return rlaif_lora_step(
            model_dict,
            batch,
            ctx,
            forward_fn=forward_fn,
            reward_fn=reward_fn,
        )

    raise ValueError(f"Unsupported mode: {mode}")
