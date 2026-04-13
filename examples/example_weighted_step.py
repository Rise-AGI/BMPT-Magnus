from __future__ import annotations

import torch

from train.def_train import step
from train.def_train import rlaif_lora_step


class TinyPolicy(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        return self.lm_head(hidden)


def main() -> None:
    torch.manual_seed(42)

    model = TinyPolicy(vocab_size=32, hidden_size=16)
    models = {"policy": model}

    batch = {
        "input_ids": torch.randint(0, 32, (2, 8), dtype=torch.long),
        "labels": torch.randint(0, 32, (2, 8), dtype=torch.long),
    }

    snapshots: dict[str, dict[str, float]] = {}

    def forward_fn(model_dict, batch_dict, ctx):
        policy_logits = model_dict["policy"](batch_dict["input_ids"])
        snapshots["cached_weights"] = dict(ctx.cached_config["weighted"]["weights"])
        snapshots["full_weights"] = dict(ctx.full_config["weighted"]["weights"])
        return {
            "policy_logits": policy_logits,
            "reference_logits": policy_logits.detach(),
            "labels": batch_dict["labels"],
        }

    def reward_helpfulness(outputs, batch_dict, ctx):
        _ = outputs
        _ = ctx
        return -batch_dict["labels"].float().mean(dim=1) / 100.0

    def reward_safety(outputs, batch_dict, ctx):
        _ = batch_dict
        _ = ctx
        return torch.tanh(outputs["policy_logits"].mean(dim=(1, 2)))

    result = step(
        models=models,
        input={
            "batch": batch,
            "step_impl": rlaif_lora_step,
            "forward_fn": forward_fn,
            "reward_fns": {
                "reward_helpfulness": reward_helpfulness,
                "reward_safety": reward_safety,
            },
            "config": {
                "weighted": {
                    "enabled": True,
                    "normalize_weights": False,
                    "weights": {
                        "reward_helpfulness": 1.0,
                        "reward_safety": 0.5,
                        "kl": 0.02,
                    },
                },
                "rlaif": {
                    "reward_scale": 1.0,
                    "normalize_reward": False,
                },
            },
        },
    )

    print("loss:", float(result["loss"].detach().item()))
    print("metrics:", result["metrics"])
    print("cached weighted:", snapshots["cached_weights"])
    print("full weighted:", snapshots["full_weights"])


if __name__ == "__main__":
    main()
