from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from src.core.config import deep_merge_dict, load_yaml_config


_CONFIG_CACHE: dict[str, Any] = {
    "path": None,
    "config": None,
}


def reset_config_cache() -> None:
    _CONFIG_CACHE["path"] = None
    _CONFIG_CACHE["config"] = None


def load_config_cached(config_path: str | Path) -> dict[str, Any]:
    normalized = str(Path(config_path).resolve())
    cached_path = _CONFIG_CACHE.get("path")
    cached_config = _CONFIG_CACHE.get("config")
    if cached_path == normalized and isinstance(cached_config, dict):
        return cached_config

    loaded = load_yaml_config(normalized)
    _CONFIG_CACHE["path"] = normalized
    _CONFIG_CACHE["config"] = loaded
    return loaded


def resolve_config_path(input_payload: dict[str, Any], default_config_path: str | Path) -> str | Path:
    return input_payload.get("config_path", default_config_path)


def get_cached_top_level(config_path: str | Path) -> dict[str, Any]:
    return load_config_cached(config_path)


def resolve_step_config(
    input_payload: dict[str, Any],
    default_config_path: str | Path,
) -> dict[str, Any]:
    config_path = resolve_config_path(input_payload, default_config_path)
    base_config = load_config_cached(config_path)
    override_config = input_payload.get("config", {})
    return deep_merge_dict(base_config, override_config)


def normalize_models(models: Any) -> dict[str, torch.nn.Module]:
    return models if isinstance(models, dict) else {"policy": models}


def expected_model_keys(config: dict[str, Any]) -> list[str]:
    model_cfg = config.get("models", {})

    keys: list[str] = []
    for label, spec in model_cfg.items():
        if label == "policy":
            keys.append(label)
            continue
        if spec.get("enabled", False):
            keys.append(label)
    return keys


def validate_models_by_config(models: dict[str, torch.nn.Module], config: dict[str, Any]) -> None:
    _ = models
    _ = config


def default_model_loader(
    model_label: str,
    model_spec: dict[str, Any],
    _config: dict[str, Any],
) -> torch.nn.Module:
    raise NotImplementedError(
        "No default model loader is implemented. "
        "Please pass `loader_fn` in `input`, or build `models` outside and pass it into `step`. "
        f"Missing loader for label: {model_label}, path: {model_spec.get('path')}"
    )


def build_models_from_config(
    config: dict[str, Any],
    loader_fn: Callable[[str, dict[str, Any], dict[str, Any]], torch.nn.Module] = default_model_loader,
) -> dict[str, torch.nn.Module]:
    model_cfg = config.get("models", {})
    required_keys = expected_model_keys(config)
    built: dict[str, torch.nn.Module] = {}
    for label in required_keys:
        spec = model_cfg.get(label)
        built[label] = loader_fn(label, spec, config)
    return built


def resolve_models(
    models: Any,
    merged_config: dict[str, Any],
    input_payload: dict[str, Any],
) -> dict[str, torch.nn.Module]:
    if models is None:
        loader_fn = input_payload.get("loader_fn", default_model_loader)
        return build_models_from_config(merged_config, loader_fn=loader_fn)

    model_dict = normalize_models(models)
    validate_models_by_config(model_dict, merged_config)
    return model_dict


def resolve_callbacks(
    input_payload: dict[str, Any],
    default_forward: Callable[..., Any],
    default_reward: Callable[..., torch.Tensor],
) -> tuple[Callable[..., Any], Callable[..., torch.Tensor]]:
    forward_fn = input_payload.get("forward_fn", default_forward)
    reward_fn = input_payload.get("reward_fn", default_reward)
    return forward_fn, reward_fn
