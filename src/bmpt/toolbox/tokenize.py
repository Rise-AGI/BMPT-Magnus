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

    # 使用 tokenizer 的批量编码
    encode_kwargs: dict[str, Any] = {
        "add_special_tokens": False,
        "return_attention_mask": True,
        "return_tensors": "pt",
        "padding": False,
    }

    if max_length is not None:
        encode_kwargs["max_length"] = max_length
        encode_kwargs["truncation"] = truncation

    encoded = tokenizer(inputs, **encode_kwargs)
    input_ids: torch.Tensor = encoded["input_ids"]
    attention_mask: torch.Tensor = encoded["attention_mask"]

    # 手动 padding 到 batch 内最大长度
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

    batch_size = input_ids.size(0)
    seq_lengths = attention_mask.sum(dim=1)
    max_seq_len = int(seq_lengths.max().item())

    if max_seq_len < input_ids.size(1):
        # 需要 padding
        padded_input_ids = torch.full(
            (batch_size, max_seq_len),
            fill_value=pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        padded_attention_mask = torch.zeros(
            (batch_size, max_seq_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        for i in range(batch_size):
            length = int(seq_lengths[i].item())
            if length > 0:
                padded_input_ids[i, :length] = input_ids[i, :length]
                padded_attention_mask[i, :length] = attention_mask[i, :length]

        input_ids = padded_input_ids
        attention_mask = padded_attention_mask

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
