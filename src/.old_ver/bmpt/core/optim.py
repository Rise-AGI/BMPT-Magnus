from __future__ import annotations

import math
from typing import Any

import torch


def build_optimizer(models: dict[str, torch.nn.Module], config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_cfg = config["optimizer"]
    optimizer_type = optimizer_cfg["type"].lower()
    lr = float(optimizer_cfg["lr"])
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
    betas = tuple(optimizer_cfg.get("betas", [0.9, 0.999]))
    eps = float(optimizer_cfg.get("eps", 1.0e-8))

    trainable_params = []
    for model in models.values():
        if isinstance(model, torch.nn.Module):
            trainable_params.extend(param for param in model.parameters() if param.requires_grad)

    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    total_training_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.LambdaLR:
    scheduler_cfg = config.get("scheduler", {})
    scheduler_type = scheduler_cfg.get("type", "none").lower()

    if scheduler_type == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)

    if scheduler_type == "cosine":
        warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
        min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.0))
        total_steps = max(total_training_steps, warmup_steps + 1)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step + 1) / float(max(warmup_steps, 1))
            progress = (current_step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
            cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
            return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
