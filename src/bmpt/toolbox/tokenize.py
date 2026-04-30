from __future__ import annotations

from typing import Any

import torch


def tokenize_batch(
    tokenizer: Any,
    inputs: list[str] | torch.Tensor,
    padding_token: int = -1,
    max_length: int | None = None,
    truncation: bool = True,
) -> dict[str, torch.Tensor]:
    """对输入进行批处理 tokenize。

    Args:
        tokenizer: transformers Tokenizer 对象
        inputs: 输入文本列表或已编码的 tensor
        padding_token: 填充 token ID，-1 表示使用 tokenizer 的 pad_token_id
        max_length: 最大序列长度，None 表示不限制
        truncation: 是否截断超长序列

    Returns:
        包含 input_ids 和 attention_mask 的字典
    """
    if isinstance(inputs, torch.Tensor):
        # 如果输入已经是 tensor，直接返回
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        if inputs.dim() != 2:
            raise ValueError(f"Input tensor must be 1D or 2D, got {inputs.dim()}D")

        batch_size, seq_len = inputs.shape
        attention_mask = torch.ones_like(inputs, dtype=torch.long)
        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
        }

    if not isinstance(inputs, (list, tuple)):
        raise TypeError(f"inputs must be list[str] or torch.Tensor, got {type(inputs)}")

    if not inputs:
        raise ValueError("inputs list cannot be empty")

    # 确定 pad token
    pad_token_id = tokenizer.pad_token_id if padding_token == -1 else padding_token
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # 使用 tokenizer 的批量编码并统一 pad 到 batch 最大长度
    encode_kwargs: dict[str, Any] = {
        "add_special_tokens": False,
        "return_attention_mask": True,
        "return_tensors": "pt",
        "padding": True,
    }

    if max_length is not None:
        encode_kwargs["max_length"] = max_length
        encode_kwargs["truncation"] = truncation

    encoded = tokenizer(inputs, **encode_kwargs)
    input_ids: torch.Tensor = encoded["input_ids"]
    attention_mask: torch.Tensor = encoded["attention_mask"]

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"Tokenizer output shape mismatch: input_ids={tuple(input_ids.shape)} "
            f"attention_mask={tuple(attention_mask.shape)}"
        )

    # 如 tokenizer 使用了非目标 pad_token_id，则在 padding 位置归一化
    if getattr(tokenizer, "pad_token_id", None) is not None and tokenizer.pad_token_id != pad_token_id:
        input_ids = input_ids.clone()
        input_ids = input_ids.masked_fill(attention_mask == 0, int(pad_token_id))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
