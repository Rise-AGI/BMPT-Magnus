from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from bmpt.data.dataset import PreprocessedDataset


def build_dataloader(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    dist_ctx: Any,
    shuffle: bool = True,
) -> DataLoader:
    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("per_device_batch_size", 1))

    dataset = PreprocessedDataset(records)

    sampler = None
    if getattr(dist_ctx, "is_distributed", False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(dist_ctx.world_size),
            rank=int(dist_ctx.rank),
            shuffle=shuffle,
            drop_last=False,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None and shuffle,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate_fn,
    )


def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    all_keys: set[str] = set()
    for item in batch:
        all_keys.update(item.keys())

    for key in all_keys:
        values = []
        for item in batch:
            val = item.get(key)
            if val is None:
                values.append(None)
            elif isinstance(val, list):
                values.append(torch.tensor(val, dtype=torch.long))
            elif isinstance(val, torch.Tensor):
                values.append(val)
            else:
                values.append(val)

        non_none_values = [v for v in values if v is not None]
        if non_none_values and isinstance(non_none_values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values

    return result