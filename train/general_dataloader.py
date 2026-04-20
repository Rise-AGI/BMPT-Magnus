from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


class GeneralJsonlDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        return_fields: list[str],
        tokenizer: Any | None = None,
        tokenize_fields: list[str] | None = None,
        max_seq_len: int = 512,
    ) -> None:
        self.records = records
        self.return_fields = return_fields
        self.tokenizer = tokenizer
        self.tokenize_fields = tokenize_fields or []
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.records[idx]
        result = {}

        for field in self.return_fields:
            if field not in row:
                continue

            value = row[field]

            if field in self.tokenize_fields and self.tokenizer is not None:
                text = str(value)
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_seq_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                result[f"{field}_ids"] = encoded["input_ids"].squeeze(0)
                result[f"{field}_mask"] = encoded["attention_mask"].squeeze(0)
                result[field] = text
            else:
                result[field] = value

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


def _load_tokenizer_from_config(config: dict[str, Any]) -> Any | None:
    if AutoTokenizer is None:
        return None

    model_cfg = config.get("models") or {}
    verifier_cfg = model_cfg.get("verifier") or {}
    planner_cfg = model_cfg.get("planner") or {}
    builder_cfg = model_cfg.get("builder") or {}

    tokenizer_path = str(
        verifier_cfg.get("path")
        or planner_cfg.get("path")
        or builder_cfg.get("path")
        or ""
    )

    if not tokenizer_path:
        data_cfg = config.get("data") or {}
        tokenizer_path = str(data_cfg.get("tokenizer_path", ""))

    if not tokenizer_path:
        return None

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


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
    tokenize_fields = data_cfg.get("tokenize_fields", [])
    max_seq_len = int(train_cfg.get("max_seq_len", 512))
    batch_size = int(train_cfg.get("per_device_batch_size", 1))

    tokenizer = None
    if tokenize_fields:
        tokenizer = _load_tokenizer_from_config(config)

    dataset = GeneralJsonlDataset(
        records=records,
        return_fields=return_fields,
        tokenizer=tokenizer,
        tokenize_fields=tokenize_fields,
        max_seq_len=max_seq_len,
    )

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