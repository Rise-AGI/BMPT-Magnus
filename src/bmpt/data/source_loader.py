from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class JsonlSourceDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path) -> None:
        path_obj = Path(path).expanduser().resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Data source not found: {path_obj}")
        if not path_obj.is_file():
            raise ValueError(f"Data source is not a file: {path_obj}")

        self.path = path_obj
        self.records = _load_jsonl(path_obj)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            if not isinstance(row, dict):
                raise ValueError(f"Each JSONL row must be object mapping: {path}")
            rows.append(row)
    return rows


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    keys: set[str] = set()
    for item in batch:
        keys.update(item.keys())

    for key in keys:
        values = [item.get(key) for item in batch]
        if values and all(isinstance(v, torch.Tensor) for v in values if v is not None):
            valid = [v for v in values if v is not None]
            if valid and len({tuple(v.shape) for v in valid}) == 1:
                output[key] = torch.stack(valid)
            else:
                output[key] = values
        else:
            output[key] = values
    return output


def _resolve_world_size_and_rank() -> tuple[int, int]:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    if world_size < 1:
        world_size = 1
    if rank < 0:
        rank = 0
    if rank >= world_size:
        rank = 0
    return world_size, rank


def build_single_source_dataloader(
    source_cfg: dict[str, Any],
    config: dict[str, Any],
    *,
    shuffle: bool = True,
) -> DataLoader:
    data_path = source_cfg.get("path")
    if data_path is None:
        raise ValueError("Each data source requires `path`")

    dataset = JsonlSourceDataset(data_path)
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    loader_cfg = data_cfg.get("loader", {})

    batch_size = int(train_cfg.get("per_device_batch_size", 1))
    num_workers = int(loader_cfg.get("num_workers", 0))
    world_size, rank = _resolve_world_size_and_rank()

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": sampler is None and shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(loader_cfg.get("pin_memory", True)),
        "drop_last": bool(loader_cfg.get("drop_last", False)),
        "collate_fn": _collate_batch,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(loader_cfg.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = bool(loader_cfg.get("persistent_workers", True))

    return DataLoader(**kwargs)


def build_source_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    data_cfg = config.get("data", {})
    sources = data_cfg.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("config.data.sources must be a non-empty list")

    loaders: dict[str, DataLoader] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each item in config.data.sources must be mapping")
        source_path = str(source.get("path", ""))
        source_name = str(source.get("name", source_path)).strip()
        if not source_name:
            raise ValueError("Unable to resolve source name")
        if source_name in loaders:
            raise ValueError(f"Duplicate source name in config.data.sources: {source_name}")

        shuffle = bool(source.get("shuffle", True))
        loaders[source_name] = build_single_source_dataloader(source, config, shuffle=shuffle)

    return loaders
