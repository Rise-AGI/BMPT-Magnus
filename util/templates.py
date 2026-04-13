from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as f

from src.core.types import StepContext


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


def gather_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = f.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
