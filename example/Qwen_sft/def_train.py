from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

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
    policy_model = model_dict["policy"]
    outputs = policy_model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch.get("labels"),
    )

    loss = outputs.loss.mean()
    reward = -loss.detach()
    return {
        "loss": loss,
        "reward": reward,
        "metrics": {
            "loss/sft": float(loss.detach().item()),
            "reward/sft": float(reward.item()),
        },
        "aux": {},
    }
