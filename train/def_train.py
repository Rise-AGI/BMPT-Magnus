from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from util.train_utils import (
    build_models_from_config,
    build_step_context,
    get_cached_top_level,
    load_config as _load_config,
    resolve_config_path,
    resolve_global_step,
    resolve_models,
    resolve_step_config,
)


_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    return _load_config(config_path, _DEFAULT_CONFIG_PATH)


def _as_long_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.long)
    return torch.tensor(value, device=device, dtype=torch.long)


def step(models, input):
    print("DEBUG step: enter")
    
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
    _ = ctx
    print("DEBUG step: context built")

    model_dict = resolve_models(models, merged_config, input)
    print(f"DEBUG step: models resolved, keys={list(model_dict.keys())}")
    
    batch = input["batch"]
    policy_model = model_dict["policy"]
    print("DEBUG step: policy_model obtained")

    model_device = next(policy_model.parameters()).device
    print(f"DEBUG step: model_device={model_device}")
    
    input_ids = _as_long_tensor(batch["input_ids"], model_device)
    print(f"DEBUG step: input_ids shape={input_ids.shape}, dtype={input_ids.dtype}")
    
    attention_mask_value = batch.get("attention_mask")
    if attention_mask_value is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    else:
        attention_mask = _as_long_tensor(attention_mask_value, model_device)
    print(f"DEBUG step: attention_mask shape={attention_mask.shape}")

    labels_value = batch.get("labels")
    if labels_value is None:
        labels = input_ids.clone()
    else:
        labels = _as_long_tensor(labels_value, model_device)
    labels = labels.masked_fill(attention_mask == 0, -100)
    print(f"DEBUG step: labels shape={labels.shape}, num_valid={(labels != -100).sum()}")

    print("DEBUG step: calling policy_model forward")
    outputs = policy_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    print(f"DEBUG step: forward done, loss={outputs.loss}")

    loss = outputs.loss.mean()
    print(f"DEBUG step: final loss={loss.item()}")
    
    return {
        "loss": loss,
        "metrics": {
            "loss/sft": float(loss.detach().item()),
        },
        "aux": {},
    }
