from __future__ import annotations

from typing import Any, Callable

import torch

from src.core.types import StepContext


ForwardFn = Callable[[dict[str, torch.nn.Module], dict[str, Any], StepContext], Any]


def run_sft_step(
    models: dict[str, torch.nn.Module],
    batch: dict[str, Any],
    ctx: StepContext,
    forward_fn: ForwardFn,
) -> dict[str, Any]:
    outputs = forward_fn(models, batch, ctx)

    loss = getattr(outputs, "loss", None)
    if loss is None:
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        else:
            raise ValueError("SFT forward output must provide `loss`")

    if not isinstance(loss, torch.Tensor):
        raise TypeError("SFT `loss` must be torch.Tensor")
    if loss.ndim != 0:
        loss = loss.mean()

    metrics = {
        "loss/sft": float(loss.detach().item()),
    }

    return {
        "loss": loss,
        "metrics": metrics,
        "aux": {"mode": ctx.mode},
    }
