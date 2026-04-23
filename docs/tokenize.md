# `bmpt.toolbox.tokenize` 函数级文档

本文档描述 tokenize 工具模块的函数职责。

## `tokenize_batch(tokenizer, inputs, padding_token=-1, max_length=None, truncation=True)`

- 作用：对输入进行批处理 tokenize。
- 输入：
  - `tokenizer`：transformers Tokenizer 对象
  - `inputs: list[str] | torch.Tensor`：输入文本列表或已编码 tensor
  - `padding_token: int`：填充 token ID，-1 表示使用 tokenizer 的 pad_token_id
  - `max_length: int | None`：最大序列长度，None 表示不限制
  - `truncation: bool`：是否截断超长序列
- 输出：`dict[str, torch.Tensor]`
  - `input_ids`: `[B, L]` 输入 ID
  - `attention_mask`: `[B, L]` 注意力掩码
- 行为：
  - 输入为 `list[str]` 时：
    1. 使用 tokenizer 批量编码（无 padding）
    2. 按 batch 内最大长度手动 padding
  - 输入为 `torch.Tensor` 时：
    1. 确保为 2D，若为 1D 则添加 batch 维度
    2. 直接返回，attention_mask 全 1
- pad token 选择：
  1. `padding_token != -1` 时使用该值
  2. 否则使用 `tokenizer.pad_token_id`
  3. 否则使用 `tokenizer.eos_token_id`
  4. 否则使用 0
- 异常：
  - `TypeError`：inputs 类型不支持
  - `ValueError`：inputs 为空列表
  - `ValueError`：tensor 维度不为 1D 或 2D
