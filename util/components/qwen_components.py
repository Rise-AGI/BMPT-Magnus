from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class JsonlSFTDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        prompt_key: str,
        response_key: str,
        max_seq_len: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.records[idx]

        if "input_ids" in row and "attention_mask" in row:
            input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(row["attention_mask"], dtype=torch.long)
            labels = torch.tensor(row.get("labels", row["input_ids"]), dtype=torch.long)
            labels = labels.masked_fill(attention_mask == 0, -100)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        prompt = str(row[self.prompt_key])
        response = str(row[self.response_key])
        text = f"{prompt}\n{response}"

        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels = labels.masked_fill(attention_mask == 0, -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _require_hf() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise ImportError(
            "`util.components.qwen_components` requires `transformers` and `peft`. "
            "Please install them in your uv environment."
        ) from exc

    return AutoModelForCausalLM, AutoTokenizer, LoraConfig, TaskType, get_peft_model


def _load_tokenizer(config: dict[str, Any]) -> Any:
    _, AutoTokenizer, _, _, _ = _require_hf()
    model_cfg = config["models"]["policy"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _apply_lora_if_needed(model: Any, spec: dict[str, Any]) -> Any:
    _, _, LoraConfig, TaskType, get_peft_model = _require_hf()
    lora_cfg = spec.get("lora", {})
    if not lora_cfg.get("enabled", False):
        return model

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg.get("r", 64)),
        lora_alpha=int(lora_cfg.get("alpha", 128)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", [])),
        bias="none",
    )
    return get_peft_model(model, config)


def load_model(label: str, spec: dict[str, Any], _config: dict[str, Any]) -> torch.nn.Module:
    AutoModelForCausalLM, _, _, _, _ = _require_hf()
    model = AutoModelForCausalLM.from_pretrained(
        spec["path"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if label == "policy":
        model = _apply_lora_if_needed(model, spec)

    trainable = bool(spec.get("trainable", False))
    if not trainable:
        model.requires_grad_(False)
        model.eval()
    else:
        model.train()

    return model


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path_obj = Path(path)
    rows: list[dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_dataloader(config: dict[str, Any], dist_ctx: Any) -> DataLoader:
    tokenizer = _load_tokenizer(config)

    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    records = _read_jsonl(data_cfg["train_path"])

    dataset = JsonlSFTDataset(
        records=records,
        tokenizer=tokenizer,
        prompt_key=str(data_cfg.get("prompt_key", "prompt")),
        response_key=str(data_cfg.get("response_key", "response")),
        max_seq_len=int(train_cfg.get("max_seq_len", 4096)),
    )

    sampler = None
    if getattr(dist_ctx, "is_distributed", False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(dist_ctx.world_size),
            rank=int(dist_ctx.rank),
            shuffle=True,
            drop_last=True,
        )

    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("per_device_batch_size", 1)),
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
