from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from bmpt.train_utils import (
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


def _is_debug_enabled(
    input_payload: dict[str, Any], merged_config: dict[str, Any] | None = None
) -> bool:
    if merged_config is not None:
        return bool(merged_config.get("runtime", {}).get("debug", False))
    return bool(input_payload.get("_debug", False))


def _debug_print(enabled: bool, message: str) -> None:
    if not enabled:
        return
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    print(message, flush=True)


def step(models, input):
    debug = _is_debug_enabled(input)
    _debug_print(debug, "DEBUG step: enter")

    config_path = resolve_config_path(input, _DEFAULT_CONFIG_PATH)
    cached_config = input.get("_cached_config")
    if cached_config is None:
        cached_config = get_cached_top_level(config_path)

    merged_config = input.get("_merged_config")
    if merged_config is None:
        merged_config = resolve_step_config(input, _DEFAULT_CONFIG_PATH)
    debug = _is_debug_enabled(input, merged_config)

    global_step = resolve_global_step(input)
    ctx = build_step_context(
        global_step=global_step,
        merged_config=merged_config,
        cached_config=cached_config,
    )
    _ = ctx
    _debug_print(debug, "DEBUG step: context built")

    model_dict = resolve_models(models, merged_config, input)
    _debug_print(debug, f"DEBUG step: models resolved, keys={list(model_dict.keys())}")

    batch = input["batch"]
    policy_model = model_dict["policy"]
    _debug_print(debug, "DEBUG step: policy_model obtained")

    model_device = next(policy_model.parameters()).device
    _debug_print(debug, f"DEBUG step: model_device={model_device}")

    input_ids = _as_long_tensor(batch["input_ids"], model_device)
    _debug_print(
        debug, f"DEBUG step: input_ids shape={input_ids.shape}, dtype={input_ids.dtype}"
    )

    attention_mask_value = batch.get("attention_mask")
    if attention_mask_value is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    else:
        attention_mask = _as_long_tensor(attention_mask_value, model_device)
    _debug_print(debug, f"DEBUG step: attention_mask shape={attention_mask.shape}")

    labels_value = batch.get("labels")
    if labels_value is None:
        labels = input_ids.clone()
    else:
        labels = _as_long_tensor(labels_value, model_device)
    labels = labels.masked_fill(attention_mask == 0, -100)
    _debug_print(
        debug,
        f"DEBUG step: labels shape={labels.shape}, num_valid={(labels != -100).sum()}",
    )

    _debug_print(debug, "DEBUG step: calling policy_model forward")
    outputs = policy_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    _debug_print(debug, f"DEBUG step: forward done, loss={outputs.loss}")

    loss = outputs.loss.mean()
    _debug_print(debug, f"DEBUG step: final loss={loss.item()}")

    return {
        "loss": loss,
        "metrics": {
            "loss/sft": float(loss.detach().item()),
        },
        "aux": {},
    }


def evaluate(models, input):
    config_path = resolve_config_path(input, _DEFAULT_CONFIG_PATH)
    cached_config = input.get("_cached_config")
    if cached_config is None:
        cached_config = get_cached_top_level(config_path)

    merged_config = input.get("_merged_config")
    if merged_config is None:
        merged_config = resolve_step_config(input, _DEFAULT_CONFIG_PATH)

    model_dict = resolve_models(models, merged_config, input)
    policy_model = model_dict["policy"]
    val_iterable = input.get("val_iterable")
    phase = str(input.get("phase", "eval"))

    if val_iterable is None:
        return {"metrics": {f"{phase}/num_batches": 0.0}, "aux": {}}

    model_device = next(policy_model.parameters()).device
    was_training = bool(policy_model.training)
    policy_model.eval()

    loss_token_sum = torch.zeros(1, device=model_device, dtype=torch.float64)
    token_sum = torch.zeros(1, device=model_device, dtype=torch.float64)
    sample_sum = torch.zeros(1, device=model_device, dtype=torch.float64)
    batch_sum = torch.zeros(1, device=model_device, dtype=torch.float64)

    with torch.no_grad():
        for batch in val_iterable:
            input_ids = _as_long_tensor(batch["input_ids"], model_device)

            attention_mask_value = batch.get("attention_mask")
            if attention_mask_value is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            else:
                attention_mask = _as_long_tensor(attention_mask_value, model_device)

            labels_value = batch.get("labels")
            if labels_value is None:
                labels = input_ids.clone()
            else:
                labels = _as_long_tensor(labels_value, model_device)
            labels = labels.masked_fill(attention_mask == 0, -100)

            outputs = policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss.mean().detach().to(dtype=torch.float64)
            valid_tokens = (labels != -100).sum().to(dtype=torch.float64)

            loss_token_sum += loss * valid_tokens
            token_sum += valid_tokens
            sample_sum += float(input_ids.shape[0])
            batch_sum += 1.0

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_token_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_sum, op=dist.ReduceOp.SUM)

    safe_tokens = token_sum.clamp_min(1.0)
    avg_loss = float((loss_token_sum / safe_tokens).item())
    ppl = float(torch.exp(torch.tensor(min(avg_loss, 20.0))).item())

    if was_training:
        policy_model.train()

    return {
        "metrics": {
            f"{phase}/loss_token_avg": avg_loss,
            f"{phase}/ppl": ppl,
            f"{phase}/num_tokens": float(token_sum.item()),
            f"{phase}/num_samples": float(sample_sum.item()),
            f"{phase}/num_batches": float(batch_sum.item()),
        },
        "aux": {},
    }
