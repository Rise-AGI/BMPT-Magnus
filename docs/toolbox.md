# `bmpt.toolbox.toolbox` 函数级文档

本文档描述 `ToolBox` 的核心方法与行为。

## `ToolBox.__init__(manager)`

- 作用：构造工具箱并绑定 `Manager`。
- 输入：`manager: Manager`
- 初始化字段：
  - `self.models: dict[str, torch.nn.Module]`
  - `self.tokenizers: dict[str, Any]`
  - `self.engine: Any | None`
  - `self.optimizer: Any | None`
  - `self.scheduler: Any | None`

## `ToolBox.load_models(model_name)`

- 作用：加载 `config.models` 下全部模型，并将指定模型初始化为 DeepSpeed Engine。
- 输入：`model_name: str`
- 输出：`dict[str, torch.nn.Module]`
- 关键约束：
  - `name` 严格使用 `config.models` 的键名。
  - 调用前必须先执行 `Manager.load_config(...)`。
  - `model_name` 必须存在于 `config.models`。
- 内部流程：
  1. 校验配置与目标模型名
  2. 逐模型调用 `bmpt.model.loader.load_model(...)`
  3. 逐模型加载 tokenizer 到 `self.tokenizers[name]`
  4. 调用 `_initialize_deepspeed_engine(model_name)` 初始化主训练引擎
  5. 将 `self.models[model_name]` 替换为 engine 对象

## `ToolBox._initialize_deepspeed_engine(model_name)`

- 作用：内部方法，使用 `manager.deepspeed_config` 完成 DeepSpeed 初始化。
- 输入：`model_name: str`
- 输出：无（更新成员状态）
- 内部行为：
  - 获取目标模型可训练参数
  - 调用 `deepspeed.initialize(...)`
  - 保存返回对象到 `self.engine/optimizer/scheduler`
- 异常：
  - `ValueError`：Manager 未加载配置 / 模型无可训练参数
  - `ImportError`：缺少 `deepspeed`

## `ToolBox.optim_step()`

- 作用：封装一次 DeepSpeed 优化步。
- 输入：无
- 输出：无
- 行为：调用 `self.engine.step()`。
- 异常：`ValueError`（engine 未初始化）

## `ToolBox.tokenize_batch(name, inputs, padding_token=-1, max_length=None, truncation=True)`

- 作用：使用指定名称的 tokenizer 对输入进行批处理 tokenize。
- 输入：
  - `name: str`：tokenizer 名称，对应 config.models 中的键
  - `inputs: list[str] | torch.Tensor`：输入文本列表或已编码 tensor
  - `padding_token: int`：填充 token ID，-1 使用 tokenizer 自带的 padding
  - `max_length: int | None`：最大序列长度
  - `truncation: bool`：是否截断
- 输出：`dict[str, torch.Tensor]`
  - `input_ids`: `[B, L]` 输入 ID
  - `attention_mask`: `[B, L]` 注意力掩码
- 关键约束：
  - 必须先调用 `load_models(...)` 加载对应 tokenizer
- 行为：
  - 文本输入：tokenizer 编码后按 batch 最大长度 padding
  - Tensor 输入：确保 2D 后直接返回
- pad token 选择：
  1. `padding_token != -1` 时优先使用
  2. 否则 `tokenizer.pad_token_id`
  3. 否则 `tokenizer.eos_token_id`
  4. 否则 0
- 异常：
  - `ValueError`：指定 name 的 tokenizer 未加载
  - `TypeError`：inputs 类型不支持
  - `ValueError`：inputs 为空
