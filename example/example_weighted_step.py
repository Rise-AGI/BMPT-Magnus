from __future__ import annotations

import torch

from bmpt.algorithms.def_train import step


class TinyPolicy(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16) -> None:
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


def main() -> None:
    torch.manual_seed(42)

    models = {"policy": TinyPolicy(vocab_size=32, hidden_size=16)}

    batch = {
        "input_ids": torch.randint(0, 32, (2, 8), dtype=torch.long),
        "labels": torch.randint(0, 32, (2, 8), dtype=torch.long),
    }

    result = step(
        models=models,
        input={
            "batch": batch,
            "config": {
                "runtime": {
                    "debug": False,
                }
            },
        },
    )

    print("loss:", float(result["loss"].detach().item()))
    print("metrics:", result["metrics"])


if __name__ == "__main__":
    main()
