from __future__ import annotations

import copy
from typing import Any

import torch

from pbv_common import _load_model, _load_tokenizer, _require_transformers
from pbv_verifier import QwenProcessVerifier


def build_models_from_config(config: dict[str, Any], loader_fn: Any = None) -> dict[str, Any]:
    _ = loader_fn
    _require_transformers()

    model_cfg = config.get("models") or {}
    planner_cfg = model_cfg.get("planner") or {}
    builder_cfg = model_cfg.get("builder") or {}
    verifier_cfg = model_cfg.get("verifier") or {}

    runtime_cfg = config.get("runtime") or {}
    attn_implementation = runtime_cfg.get("attn_implementation", "auto")

    algorithm_cfg = config.get("algorithm") or {}
    schedule_cfg = algorithm_cfg.get("training_schedule") or {}
    planner_trainable = bool(planner_cfg.get("trainable", False))
    builder_trainable = bool(builder_cfg.get("trainable", False))

    tokenizer_path = str(
        verifier_cfg.get("path") or planner_cfg.get("path") or builder_cfg.get("path") or ""
    )
    if not tokenizer_path:
        raise ValueError("models.verifier.path (or planner/builder path) is required.")

    tokenizer = _load_tokenizer(tokenizer_path)

    planner_path = str(planner_cfg.get("path", tokenizer_path))
    builder_path = str(builder_cfg.get("path", tokenizer_path))
    verifier_path = str(verifier_cfg.get("path", tokenizer_path))

    builder = _load_model(builder_path, attn_implementation=attn_implementation)
    verifier_model = _load_model(verifier_path, attn_implementation=attn_implementation)
    verifier = QwenProcessVerifier(
        model=verifier_model,
        tokenizer=tokenizer,
        max_new_tokens=int(verifier_cfg.get("max_new_tokens", 6)),
        temperature=float(verifier_cfg.get("temperature", 0.0)),
    )
    verifier.requires_grad_(False)
    verifier.eval()

    if not planner_trainable and planner_path == verifier_path:
        planner = verifier_model
    else:
        planner = _load_model(planner_path, attn_implementation=attn_implementation)

    if not builder_trainable:
        builder.requires_grad_(False)
        builder_ref = builder
    elif builder_path == verifier_path:
        builder_ref = verifier_model
    else:
        builder_ref = copy.deepcopy(builder)
        builder_ref.requires_grad_(False)

    planner_steps = int(schedule_cfg.get("planner_steps", 1))
    if planner_steps > 0 and planner_trainable:
        planner_ref = copy.deepcopy(planner)
        planner_ref.requires_grad_(False)
    else:
        planner_ref = planner

    if not planner_trainable:
        planner.requires_grad_(False)

    enable_gradient_ckpt = bool(runtime_cfg.get("gradient_checkpointing", False))
    if enable_gradient_ckpt:
        trainable_models = []
        if planner_trainable:
            trainable_models.append(planner)
        if builder_trainable:
            trainable_models.append(builder)
        for model in trainable_models:
            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

    return {
        "planner": planner,
        "builder": builder,
        "verifier": verifier,
        "planner_ref": planner_ref,
        "builder_ref": builder_ref,
    }


def build_optimizers_from_config(models: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    optimizer_cfg = config.get("optimizer") or {}
    lr = float(optimizer_cfg.get("lr", 2e-5))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.01))
    betas = optimizer_cfg.get("betas") or [0.9, 0.95]
    eps = float(optimizer_cfg.get("eps", 1e-8))

    planner_params = list(models["planner"].parameters())
    builder_params = list(models["builder"].parameters())
    all_params = planner_params + builder_params

    optimizer = torch.optim.AdamW(
        all_params,
        lr=lr,
        betas=(float(betas[0]), float(betas[1])),
        eps=eps,
        weight_decay=weight_decay,
    )
    return {"policy": optimizer}
