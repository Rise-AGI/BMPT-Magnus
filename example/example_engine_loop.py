from __future__ import annotations

import torch

from bmpt.algorithms.def_train import step
from bmpt.core.engine import TrainingEngine
from bmpt.core.optim import build_optimizer, build_scheduler


class TinyPolicy(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, attention_mask=None):
        _ = attention_mask
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden)
        if labels is None:
            return {"logits": logits}
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        return {"loss": loss, "logits": logits}


def main() -> None:
    torch.manual_seed(0)

    models: dict[str, torch.nn.Module] = {"policy": TinyPolicy()}
    config = {
        "optimizer": {
            "type": "adamw",
            "lr": 1.0e-4,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
        },
        "scheduler": {
            "type": "cosine",
            "warmup_steps": 1,
            "min_lr_ratio": 0.1,
        },
    }

    optimizer = build_optimizer(models, config)
    scheduler = build_scheduler(optimizer, config, total_training_steps=10)
    engine = TrainingEngine(
        step_fn=step,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_accum_steps=2,
        grad_clip_norm=1.0,
    )

    for _ in range(4):
        batch = {
            "input_ids": torch.randint(0, 32, (2, 8), dtype=torch.long),
            "labels": torch.randint(0, 32, (2, 8), dtype=torch.long),
        }
        output = engine.run_micro_step(
            models=models,
            batch=batch,
            extra_input={
                "config": {
                    "weighted": {
                        "enabled": True,
                        "normalize_weights": False,
                        "weights": {"reward": 1.0, "kl": 0.0},
                    }
                },
            },
        )
        print(output["metrics"])


if __name__ == "__main__":
    main()
