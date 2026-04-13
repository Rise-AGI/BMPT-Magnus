from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as f

from src.core.types import StepContext
from util.train_utils import (
    build_models_from_config as _build_models_from_config,
    get_cached_top_level,
    load_config_cached,
    resolve_config_path,
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


def _build_step_context(
    mode: str,
    global_step: int,
    merged_config: dict[str, Any],
    cached_config: dict[str, Any],
) -> StepContext:
    runtime_cfg = merged_config.get("runtime", {})
    return StepContext(
        mode=mode,
        global_step=global_step,
        runtime_config=runtime_cfg,
        full_config=merged_config,
        cached_config=cached_config,
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
    reward_fns: dict[str, Callable[..., torch.Tensor]],
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

    weighted_cfg = ctx.full_config["weighted"]
    weights = weighted_cfg["weights"]
    reward_names = [name for name in weights if name != "kl"]

    reward_scale = float(ctx.full_config.get("rlaif", {}).get("reward_scale", 1.0))
    normalize_reward = bool(ctx.full_config.get("rlaif", {}).get("normalize_reward", True))
    reward_losses: dict[str, torch.Tensor] = {}
    for reward_name in reward_names:
        reward = reward_fns[reward_name](outputs, batch, ctx)
        reward = reward.to(device=policy_logits.device, dtype=policy_logits.dtype)
        if normalize_reward:
            reward = (reward - reward.mean()) / (reward.std(unbiased=False) + 1.0e-6)
        reward_losses[reward_name] = -(reward_scale * reward * policy_logp).mean()

    kl_loss = kl
    total_loss = torch.zeros((), device=policy_logits.device, dtype=policy_logits.dtype)
    metrics: dict[str, float] = {}
    for reward_name in reward_names:
        reward_weight = float(weights[reward_name])
        reward_loss = reward_losses[reward_name]
        total_loss = total_loss + reward_weight * reward_loss
        metrics[f"loss/{reward_name}"] = float(reward_loss.detach().item())
        metrics[f"weight/{reward_name}"] = reward_weight

    kl_weight = float(weights["kl"])
    total_loss = total_loss + kl_weight * kl_loss
    metrics["loss/kl"] = float(kl_loss.detach().item())
    metrics["weight/kl"] = kl_weight
    metrics["loss/total"] = float(total_loss.detach().item())

    return {
        "loss": total_loss,
        "metrics": metrics,
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

    config_path = resolve_config_path(input, _DEFAULT_CONFIG_PATH)
    cached_config = get_cached_top_level(config_path)
    merged_config = resolve_step_config(input, _DEFAULT_CONFIG_PATH)

    mode = input.get("mode", merged_config.get("mode", "sft"))
    global_step = int(input.get("global_step", 0))
    ctx = _build_step_context(
        mode=mode,
        global_step=global_step,
        merged_config=merged_config,
        cached_config=cached_config,
    )

    model_dict = resolve_models(models, merged_config, input)

    batch = input["batch"]
    forward_fn, reward_fn = resolve_callbacks(input, default_forward, default_reward)
    reward_fns = input.get("reward_fns")
    if reward_fns is None:
        reward_fns = {"reward": reward_fn}

    if mode == "sft":
        return sft_step(model_dict, batch, ctx, forward_fn=forward_fn)

    if mode == "rlaif_lora":
        return rlaif_lora_step(
            model_dict,
            batch,
            ctx,
            forward_fn=forward_fn,
            reward_fns=reward_fns,
        )

    raise ValueError(f"Unsupported mode: {mode}")
