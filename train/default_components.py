from __future__ import annotations

from typing import Any, Iterable

import torch


class TinyPolicy(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 32) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _ = attention_mask
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden)
        if labels is None:
            return {"logits": logits}
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        return {"loss": loss, "logits": logits}


def load_model(label: str, spec: dict[str, Any], config: dict[str, Any]) -> torch.nn.Module:
    _ = label
    _ = spec
    _ = config
    return TinyPolicy()


def build_dataloader(config: dict[str, Any], _dist_ctx: Any) -> Iterable[dict[str, Any]]:
    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("per_device_batch_size", 2))
    max_seq_len = int(train_cfg.get("max_seq_len", 128))
    max_steps = int(train_cfg.get("max_steps", 20))
    if max_steps <= 0:
        max_steps = 20

    seq_len = min(max_seq_len, 128)
    for _ in range(max_steps):
        yield {
            "input_ids": torch.randint(0, 32, (batch_size, seq_len), dtype=torch.long),
            "labels": torch.randint(0, 32, (batch_size, seq_len), dtype=torch.long),
        }
