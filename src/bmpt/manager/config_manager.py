from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class LoadedConfig:
    config_path: Path
    deepspeed_config_path: Path
    config: dict[str, Any]
    deepspeed_config: dict[str, Any]


def load_mapping_file(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path).expanduser().resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")

    suffix = path_obj.suffix.lower()
    with path_obj.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            data = json.load(handle) or {}
        else:
            data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Top-level config must be a mapping: {path_obj}")
    return data


def resolve_deepspeed_config_path(config: dict[str, Any], config_path: str | Path) -> Path:
    runtime_cfg = config.get("runtime", {})
    ds_path = runtime_cfg.get("deepspeed_config_path")
    if ds_path is None:
        raise ValueError("`runtime.deepspeed_config_path` is required")

    ds_path_obj = Path(str(ds_path))
    if not ds_path_obj.is_absolute():
        ds_path_obj = Path(config_path).expanduser().resolve().parent / ds_path_obj
    resolved = ds_path_obj.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {resolved}")
    return resolved


def _to_deepspeed_optimizer(optimizer_cfg: dict[str, Any]) -> dict[str, Any]:
    optimizer_type = str(optimizer_cfg.get("type", "adamw")).lower()
    if optimizer_type != "adamw":
        raise ValueError(f"Unsupported optimizer.type: {optimizer_type}")

    return {
        "type": "AdamW",
        "params": {
            "lr": float(optimizer_cfg.get("lr", 2.0e-5)),
            "weight_decay": float(optimizer_cfg.get("weight_decay", 0.0)),
            "betas": list(optimizer_cfg.get("betas", [0.9, 0.999])),
            "eps": float(optimizer_cfg.get("eps", 1.0e-8)),
        },
    }


def _to_deepspeed_scheduler(scheduler_cfg: dict[str, Any], train_cfg: dict[str, Any]) -> dict[str, Any] | None:
    scheduler_type = str(scheduler_cfg.get("type", "none")).lower()
    if scheduler_type == "none":
        return None

    if scheduler_type != "cosine":
        raise ValueError(f"Unsupported scheduler.type: {scheduler_type}")

    warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
    min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.0))
    total_num_steps = int(train_cfg.get("max_steps", 1))
    if total_num_steps <= warmup_steps:
        total_num_steps = warmup_steps + 1

    return {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_num_steps": warmup_steps,
            "total_num_steps": total_num_steps,
            "warmup_min_ratio": 0.0,
            "cos_min_ratio": min_lr_ratio,
        },
    }


def build_runtime_deepspeed_config(config: dict[str, Any], deepspeed_config: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(deepspeed_config)
    train_cfg = config.get("train", {})
    optimizer_cfg = config.get("optimizer", {})
    scheduler_cfg = config.get("scheduler", {})

    if "zero_optimization" not in merged:
        merged["zero_optimization"] = {}
    merged["zero_optimization"]["elastic_checkpoint"] = False

    merged["train_micro_batch_size_per_gpu"] = int(train_cfg.get("per_device_batch_size", 1))
    merged["gradient_accumulation_steps"] = int(train_cfg.get("gradient_accumulation_steps", 1))
    merged["gradient_clipping"] = float(train_cfg.get("grad_clip_norm", 1.0))

    if isinstance(optimizer_cfg, dict) and optimizer_cfg:
        merged["optimizer"] = _to_deepspeed_optimizer(optimizer_cfg)

    scheduler = None
    if isinstance(scheduler_cfg, dict):
        scheduler = _to_deepspeed_scheduler(scheduler_cfg, train_cfg)
    if scheduler is None:
        merged.pop("scheduler", None)
    else:
        merged["scheduler"] = scheduler

    mixed_precision = str(train_cfg.get("mixed_precision", "bf16")).lower()
    if mixed_precision == "bf16":
        merged["bf16"] = {"enabled": True}
        merged["fp16"] = {"enabled": False}
    elif mixed_precision == "fp16":
        merged["bf16"] = {"enabled": False}
        merged["fp16"] = {"enabled": True}
    elif mixed_precision == "no":
        merged["bf16"] = {"enabled": False}
        merged["fp16"] = {"enabled": False}
    else:
        raise ValueError(f"Unsupported train.mixed_precision: {mixed_precision}")

    return merged


def strip_optimizer_scheduler(config: dict[str, Any]) -> dict[str, Any]:
    stripped = deepcopy(config)
    stripped.pop("optimizer", None)
    stripped.pop("scheduler", None)
    return stripped


def load_config_bundle(config_path: str | Path) -> LoadedConfig:
    resolved_config_path = Path(config_path).expanduser().resolve()
    config = load_mapping_file(resolved_config_path)
    deepspeed_config_path = resolve_deepspeed_config_path(config, resolved_config_path)
    base_deepspeed_config = load_mapping_file(deepspeed_config_path)
    runtime_deepspeed_config = build_runtime_deepspeed_config(config, base_deepspeed_config)
    stripped_config = strip_optimizer_scheduler(config)

    return LoadedConfig(
        config_path=resolved_config_path,
        deepspeed_config_path=deepspeed_config_path,
        config=stripped_config,
        deepspeed_config=runtime_deepspeed_config,
    )
