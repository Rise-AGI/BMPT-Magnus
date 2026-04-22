from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from bmpt.util import Composer

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None


PLANNER_PROMPT_SUFFIX = '把题目拆成可批改的小目标，形如"根据 xyz 计算 abc"。\n现在开始拆分，每行一个'
BUILDER_STEP_CONSTRAINT = "你不需要解出题目，完成下一步后立刻结束，否则即使解答正确也判定为失败"
VERIFIER_JUDGE_INSTRUCTION = "请判断候选是否与题目和步骤一致。仅输出 Right 或 Wrong。"


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for yaml configs.")
        loaded = yaml.safe_load(text)
        return loaded or {}
    raise ValueError(f"Unsupported config format: {config_path}")


def _require_transformers() -> None:
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError("PBV requires `transformers` to be installed.")


def _safe_temp(value: float) -> float:
    return max(value, 1e-6)


def _load_tokenizer(model_path: str) -> Any:
    _require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


def _load_model(model_path: str, attn_implementation: str = "auto") -> nn.Module:
    _require_transformers()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    return model


def _tokenize_text(tokenizer: Any, text: str, max_len: int | None = None) -> list[int]:
    encoded = tokenizer(str(text), add_special_tokens=False, return_attention_mask=False)
    token_ids = [int(token) for token in encoded.get("input_ids", [])]
    if max_len is not None and max_len > 0:
        token_ids = token_ids[:max_len]
    return token_ids


def _get_ids_from_batch(
    batch: dict[str, Any],
    field: str,
    index: int,
    max_len: int | None = None,
) -> list[int]:
    ids_key = f"{field}_input_ids"
    if ids_key in batch and isinstance(batch[ids_key], torch.Tensor):
        ids_tensor = batch[ids_key][index]
        ids = [int(tok) for tok in ids_tensor.tolist()]
        if max_len is not None and max_len > 0:
            ids = ids[:max_len]
        return ids
    return []


def _decode_text(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _to_single_batch_tensor(token_ids: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def _compose_ids(
    composer: Composer,
    ids_list: list[list[int]],
    device: torch.device,
) -> list[int]:
    outputs: list[torch.Tensor] = []
    for ids in ids_list:
        outputs.append(_to_single_batch_tensor(ids, device=device))
    composed = composer.compose(outputs=outputs)
    valid_len = int(composed["lengths"][0].item())
    return [int(item) for item in composed["input_ids"][0, :valid_len].tolist()]
