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
    pad_token_id: int = 0,
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
        collate_fn=lambda batch: _collate_fn(batch, pad_token_id),
    )


def _collate_fn(batch: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
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
            shapes = [tuple(v.shape) for v in non_none_values]
            if len(set(shapes)) == 1:
                result[key] = torch.stack(values)
            else:
                is_input_ids = key == "input_ids" or key.endswith("_input_ids")
                pad_val = pad_token_id if is_input_ids else 0
                max_len = max(v.shape[0] for v in non_none_values)
                padded = torch.full(
                    (len(non_none_values), max_len),
                    fill_value=pad_val,
                    dtype=non_none_values[0].dtype,
                )
                for i, v in enumerate(non_none_values):
                    padded[i, : v.shape[0]] = v
                result[key] = padded
        else:
            result[key] = values

    return result