# `bmpt.prompt.composer_manager` 函数级文档

本文档描述提示词组合器管理模块的类与函数职责。

## `Composer`

- 类型：`dataclass(slots=True)`
- 作用：提示词组合器，将多个模型输出与预设提示拼接。
- 字段：
  - `name: str`：composer 名称
  - `prompt_token_ids: list[torch.Tensor]`：预 tokenize 的提示段列表
  - `pad_token_id: int`：填充 token ID
  - `max_total_len: int`：最大总长度限制（<=0 表示无限制）
  - `truncate_side: str`：截断方向，"left" 或 "right"
  - `pad_to_multiple_of: int | None`：可选，将长度填充到该数的倍数
  - `output_pad_token_id: int | None`：可选，识别输出有效长度的 pad token

## `Composer.compose(outputs, output_masks=None)`

- 作用：将输出片段与提示拼接成完整输入。
- 输入：
  - `outputs: list[torch.Tensor]`：模型输出列表，每个形状 `[B, T]`
  - `output_masks: list[torch.Tensor] | None`：可选，每个输出的有效位置掩码
- 输出：`dict[str, torch.Tensor]`
  - `input_ids`: `[B, L]` 拼接后的输入 ID
  - `attention_mask`: `[B, L]` 注意力掩码
  - `lengths`: `[B]` 实际长度
- 拼接模式：
  - `[prompt_0] + output_0 + [prompt_1] + output_1 + ... + [prompt_n]`
- 异常：
  - `ValueError`：outputs 数量与 prompts 不匹配
  - `ValueError`：tensor 形状或设备不一致

## `_load_tokenizer_for_prompting(config)`

- 作用：加载用于 prompting 的 tokenizer。
- 输入：`config: dict[str, Any]`
- 输出：`(tokenizer, pad_token_id)`
- 来源优先级：
  1. `config.prompting.tokenizer_source`
  2. 顶层 `tokenizer_source`
  3. `models.policy.path`

## `_tokenize_prompt(tokenizer, text, add_bos, add_eos)`

- 作用：对单个提示文本进行 tokenize。
- 输入：
  - `tokenizer`：transformers Tokenizer
  - `text: str`：文本
  - `add_bos: bool`：是否添加 BOS
  - `add_eos: bool`：是否添加 EOS
- 输出：`torch.Tensor`，形状 `[seq_len]`

## `build_composers_from_config(config)`

- 作用：从配置构建所有 composer。
- 输入：`config: dict[str, Any]`
- 输出：`dict[str, Composer]`，key 为 composer 名称
- 配置读取：
  - `config.prompting.composers`：composer 定义字典
  - 每个 composer 配置字段：
    - `prompts: list[str]`（至少 2 个）
    - `add_bos`, `add_eos: bool`（可选）
    - `pad_token_id`, `max_total_len`, `truncate_side`（可选）
    - `pad_to_multiple_of`, `output_pad_token_id`（可选）
- 异常：
  - `ValueError`：composers 配置格式错误
  - `ValueError`：prompts 数量不足
- 注意：若未配置 `prompting.composers`，返回空字典
