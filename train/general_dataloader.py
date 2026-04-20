from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class GeneralJsonlDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        return_fields: list[str],
    ) -> None:
        self.records = records
        self.return_fields = return_fields

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.records[idx]
        result = {}
        for field in self.return_fields:
            if field in row:
                result[field] = row[field]
        return result


def _collate_general(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        return {}
    result = {}
    for key in batch[0].keys():
        values = [item.get(key) for item in batch]
        if any(v is None for v in values):
            continue
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values, dtype=torch.float32)
        else:
            result[key] = values
    return result


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_general_dataloader(
    config: dict[str, Any],
    dist_ctx: Any,
    path_key: str = "train_path",
    shuffle: bool = True,
) -> DataLoader | None:
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    path = data_cfg.get(path_key)
    if path is None:
        return None

    records = _read_jsonl(path)
    return_fields = data_cfg.get("return_fields", ["prompt"])
    batch_size = int(train_cfg.get("per_device_batch_size", 1))

    dataset = GeneralJsonlDataset(records=records, return_fields=return_fields)

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
        collate_fn=_collate_general,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )