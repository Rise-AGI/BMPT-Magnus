from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from src.core.config import deep_merge_dict, load_yaml_config
from src.core.types import StepContext
from src.modes.rlaif_lora import run_rlaif_lora_step
from src.modes.sft import run_sft_step


_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    target = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    return load_yaml_config(target)


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


def _normalize_models(models: Any) -> dict[str, torch.nn.Module]:
    if isinstance(models, dict):
        if "policy" not in models:
            raise ValueError("`models` dict must contain `policy`")
        return models
    return {"policy": models}


def _resolve_callbacks(
    input_payload: dict[str, Any],
) -> tuple[Callable[..., Any], Callable[..., torch.Tensor]]:
    forward_fn = input_payload.get("forward_fn", default_forward)
    reward_fn = input_payload.get("reward_fn", default_reward)
    return forward_fn, reward_fn


def _build_step_context(mode: str, global_step: int, merged_config: dict[str, Any]) -> StepContext:
    runtime_cfg = merged_config.get("runtime", {})
    return StepContext(
        mode=mode,
        global_step=global_step,
        runtime_config=runtime_cfg,
        full_config=merged_config,
    )


def step(models, input):
    """Run one train step with unified protocol.

    Args:
        models: dict[str, torch.nn.Module] | torch.nn.Module
        input: dict[str, Any]

    Returns:
        dict with keys: loss, metrics, aux
    """

    if not isinstance(input, dict):
        raise TypeError("`input` must be a dict")
    if "batch" not in input:
        raise ValueError("`input` must contain `batch`")

    file_config = load_config(input.get("config_path"))
    override_config = input.get("config", {})
    if override_config and not isinstance(override_config, dict):
        raise TypeError("`input[\"config\"]` must be a dict")
    merged_config = deep_merge_dict(file_config, override_config)

    mode = input.get("mode", merged_config.get("mode", "sft"))
    global_step = int(input.get("global_step", 0))
    ctx = _build_step_context(mode=mode, global_step=global_step, merged_config=merged_config)

    model_dict = _normalize_models(models)
    batch = input["batch"]
    forward_fn, reward_fn = _resolve_callbacks(input)

    if mode == "sft":
        return run_sft_step(model_dict, batch, ctx, forward_fn=forward_fn)

    if mode == "rlaif_lora":
        return run_rlaif_lora_step(
            model_dict,
            batch,
            ctx,
            forward_fn=forward_fn,
            reward_fn=reward_fn,
        )

    raise ValueError(f"Unsupported mode: {mode}")
