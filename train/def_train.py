from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from util.templates import default_forward, default_reward, gather_log_probs
from util.train_utils import (
    build_models_from_config as _build_models_from_config,
    build_step_context,
    get_cached_top_level,
    load_config_cached,
    resolve_config_path,
    resolve_callbacks,
    resolve_global_step,
    resolve_models,
    resolve_reward_fns,
    resolve_step_config,
)


_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    target = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    return load_config_cached(target)


def build_models_from_config(config: dict[str, Any], loader_fn):
    return _build_models_from_config(config, loader_fn=loader_fn)


def sft_step(
    models: dict[str, torch.nn.Module],
    batch: dict[str, Any],
    ctx,
    forward_fn: Callable[..., Any],
) -> dict[str, Any]:
    """User-owned SFT logic. Modify this function for custom SFT training."""
    outputs = forward_fn(models, batch, ctx)
    loss = outputs["loss"].mean()

    return {
        "loss": loss,
        "metrics": {"loss/sft": float(loss.detach().item())},
        "aux": {},
    }


def rlaif_lora_step(
    models: dict[str, torch.nn.Module],
    batch: dict[str, Any],
    ctx,
    forward_fn: Callable[..., Any],
    reward_fns: dict[str, Callable[..., torch.Tensor]],
) -> dict[str, Any]:
    """User-owned RLAIF-LoRA logic. Modify this function for custom RL training."""
    outputs = forward_fn(models, batch, ctx)
    policy_logits = outputs["policy_logits"]
    labels = outputs["labels"]

    policy_logp = gather_log_probs(policy_logits, labels).mean(dim=-1)

    reference_logits = outputs.get("reference_logits")
    if reference_logits is None:
        kl = torch.zeros((), device=policy_logits.device, dtype=policy_logits.dtype)
    else:
        reference_logp = gather_log_probs(reference_logits, labels).mean(dim=-1)
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
        "aux": {},
    }


def _default_step_impl(
    models: dict[str, torch.nn.Module],
    batch: dict[str, Any],
    ctx: Any,
    forward_fn: Callable[..., Any],
    reward_fns: dict[str, Callable[..., torch.Tensor]],
) -> dict[str, Any]:
    _ = reward_fns
    return sft_step(models=models, batch=batch, ctx=ctx, forward_fn=forward_fn)


def step(models, input):
    """Run one train step with unified protocol.

    Args:
        models: dict[str, torch.nn.Module] | torch.nn.Module
        input: dict[str, Any]

    Returns:
        dict with keys: loss, metrics, aux
    """

    config_path = resolve_config_path(input, _DEFAULT_CONFIG_PATH)
    cached_config = input.get("_cached_config")
    if cached_config is None:
        cached_config = get_cached_top_level(config_path)

    merged_config = input.get("_merged_config")
    if merged_config is None:
        merged_config = resolve_step_config(input, _DEFAULT_CONFIG_PATH)

    global_step = resolve_global_step(input)
    ctx = build_step_context(
        global_step=global_step,
        merged_config=merged_config,
        cached_config=cached_config,
    )

    model_dict = resolve_models(models, merged_config, input)

    batch = input["batch"]
    forward_fn, reward_fn = resolve_callbacks(input, default_forward, default_reward)
    reward_fns = resolve_reward_fns(input, reward_fn)
    step_impl = input.get("step_impl", _default_step_impl)

    return step_impl(
        model_dict,
        batch,
        ctx,
        forward_fn=forward_fn,
        reward_fns=reward_fns,
    )
