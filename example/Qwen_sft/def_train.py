from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from util.templates import default_forward
from util.train_utils import (
    build_models_from_config as _build_models_from_config,
    build_step_context,
    get_cached_top_level,
    load_config_cached,
    resolve_config_path,
    resolve_global_step,
    resolve_models,
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
    ctx: Any,
    forward_fn: Callable[..., Any],
) -> dict[str, Any]:
    outputs = forward_fn(models, batch, ctx)
    loss = outputs["loss"].mean()
    return {
        "loss": loss,
        "metrics": {"loss/sft": float(loss.detach().item())},
        "aux": {},
    }


def step(models, input):
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
    forward_fn = input.get("forward_fn", default_forward)
    return sft_step(model_dict, batch, ctx, forward_fn=forward_fn)
