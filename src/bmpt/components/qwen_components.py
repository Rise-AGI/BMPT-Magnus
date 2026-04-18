from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
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
            "`bmpt.components.qwen_components` requires `transformers` and `peft`. "
            "Please install them in your environment."
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


def _is_rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _resolve_attn_implementation(config: dict[str, Any]) -> str:
    runtime_cfg = config.get("runtime", {})
    explicit = runtime_cfg.get("attn_implementation")
    if explicit is not None:
        value = str(explicit).strip()
        if value:
            return value

    if bool(runtime_cfg.get("flash_attention", False)):
        return "flash_attention_2"
    return "auto"


def _load_with_attn(
    loader_cls: Any,
    model_path: str,
    requested_attn: str,
) -> tuple[Any, str, bool]:
    base_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    attn_value = requested_attn.strip()
    attn_lower = attn_value.lower()
    if attn_lower == "flash_attention":
        attn_value = "flash_attention_2"
        attn_lower = attn_value

    if attn_lower == "default":
        model = loader_cls.from_pretrained(model_path, **base_kwargs)
        return model, "default", False

    if attn_lower == "auto":
        try:
            model = loader_cls.from_pretrained(
                model_path,
                attn_implementation="flash_attention_2",
                **base_kwargs,
            )
            return model, "flash_attention_2", False
        except Exception as exc:
            if _is_rank0():
                warnings.warn(
                    f"FlashAttention auto probe failed, fallback to default attention: {exc}",
                    RuntimeWarning,
                )
            model = loader_cls.from_pretrained(model_path, **base_kwargs)
            return model, "default", True

    try:
        model = loader_cls.from_pretrained(
            model_path,
            attn_implementation=attn_value,
            **base_kwargs,
        )
        return model, attn_value, False
    except Exception as exc:
        if attn_lower == "flash_attention_2":
            if _is_rank0():
                warnings.warn(
                    f"Requested flash_attention_2 unavailable, fallback to default attention: {exc}",
                    RuntimeWarning,
                )
            model = loader_cls.from_pretrained(model_path, **base_kwargs)
            return model, "default", True
        raise RuntimeError(f"Unsupported or unavailable attn_implementation={attn_value}: {exc}") from exc


def load_model(label: str, spec: dict[str, Any], config: dict[str, Any]) -> torch.nn.Module:
    AutoModelForCausalLM, _, _, _, _ = _require_hf()
    requested_attn = _resolve_attn_implementation(config)
    model, actual_attn, did_fallback = _load_with_attn(
        loader_cls=AutoModelForCausalLM,
        model_path=spec["path"],
        requested_attn=requested_attn,
    )

    runtime_cfg = config.get("runtime", {})
    enable_gradient_ckpt = bool(runtime_cfg.get("gradient_checkpointing", False))
    if enable_gradient_ckpt:
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        elif _is_rank0():
            warnings.warn("gradient_checkpointing is enabled in config but model has no gradient_checkpointing_enable()", RuntimeWarning)

    if label == "policy":
        model = _apply_lora_if_needed(model, spec)

    if _is_rank0():
        print(
            "[bmpt] model_init "
            f"requested_attn={requested_attn} "
            f"actual_attn={actual_attn} "
            f"fallback={int(did_fallback)} "
            f"gradient_checkpointing={int(enable_gradient_ckpt)}",
            flush=True,
        )

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


def build_dataloader(config: dict[str, Any], dist_ctx: Any, path_key: str = "train_path", shuffle: bool = True) -> DataLoader | None:
    tokenizer = _load_tokenizer(config)

    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    path = data_cfg.get(path_key)
    if path is None:
        return None

    records = _read_jsonl(path)

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
            shuffle=shuffle,
            drop_last=False,
        )

    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("per_device_batch_size", 1)),
        sampler=sampler,
        shuffle=sampler is None and shuffle,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
