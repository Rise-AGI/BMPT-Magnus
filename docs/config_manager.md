# `bmpt.manager.config_manager` 函数级文档

本文档描述第一阶段配置管理模块的函数职责、输入输出与异常。

## `LoadedConfig`

- 类型：`dataclass`
- 字段：
  - `config_path: Path`：用户训练配置绝对路径
  - `deepspeed_config_path: Path`：DeepSpeed 配置绝对路径
  - `config: dict[str, Any]`：移除 `optimizer/scheduler` 后的业务配置
  - `deepspeed_config: dict[str, Any]`：合并后的运行时 DeepSpeed 配置

## `load_mapping_file(path)`

- 作用：读取 JSON/YAML 配置文件并返回字典。
- 输入：`path: str | Path`
- 输出：`dict[str, Any]`
- 规则：
  - 扩展名为 `.json` 时按 JSON 解析；其他扩展名按 YAML 解析。
  - 顶层必须是 mapping，否则报错。
- 异常：
  - `FileNotFoundError`：文件不存在
  - `ValueError`：顶层不是 mapping

## `resolve_deepspeed_config_path(config, config_path)`

- 作用：从 `runtime.deepspeed_config_path` 解析 DeepSpeed 配置绝对路径。
- 输入：
  - `config: dict[str, Any]`
  - `config_path: str | Path`
- 输出：`Path`
- 规则：
  - 仅使用 `runtime.deepspeed_config_path`。
  - 相对路径相对 `config_path` 所在目录解析。
- 异常：
  - `ValueError`：缺少 `runtime.deepspeed_config_path`
  - `FileNotFoundError`：路径解析后文件不存在

## `_to_deepspeed_optimizer(optimizer_cfg)`

- 作用：将 BMPT 的 `optimizer` 段转换为 DeepSpeed `optimizer` 格式。
- 输入：`optimizer_cfg: dict[str, Any]`
- 输出：`dict[str, Any]`
- 当前支持：`type=adamw`
- 异常：`ValueError`（不支持的 optimizer 类型）

## `_to_deepspeed_scheduler(scheduler_cfg, train_cfg)`

- 作用：将 BMPT 的 `scheduler` 段转换为 DeepSpeed `scheduler` 格式。
- 输入：
  - `scheduler_cfg: dict[str, Any]`
  - `train_cfg: dict[str, Any]`
- 输出：`dict[str, Any] | None`
- 当前支持：
  - `none` -> 返回 `None`
  - `cosine` -> 返回 `WarmupCosineLR`
- 规则：
  - `total_num_steps` 使用 `train.max_steps`，并保证 `> warmup_steps`。
- 异常：`ValueError`（不支持的 scheduler 类型）

## `build_runtime_deepspeed_config(config, deepspeed_config)`

- 作用：基于业务配置生成运行时 DeepSpeed 配置。
- 输入：
  - `config: dict[str, Any]`
  - `deepspeed_config: dict[str, Any]`
- 输出：`dict[str, Any]`
- 处理项：
  - 强制 `zero_optimization.elastic_checkpoint=False`
  - 写入批大小与梯度相关字段：
    - `train_micro_batch_size_per_gpu`
    - `gradient_accumulation_steps`
    - `gradient_clipping`
  - 注入 `optimizer`
  - 注入或移除 `scheduler`
  - 按 `train.mixed_precision` 写入 `bf16/fp16`
- 异常：`ValueError`（不支持的混精模式）

## `strip_optimizer_scheduler(config)`

- 作用：从业务配置中移除顶层 `optimizer` 和 `scheduler`。
- 输入：`config: dict[str, Any]`
- 输出：`dict[str, Any]`（深拷贝后的新字典）

## `load_config_bundle(config_path)`

- 作用：第一阶段配置加载主入口，返回完整配置包。
- 输入：`config_path: str | Path`
- 输出：`LoadedConfig`
- 流程：
  1. 加载主配置
  2. 解析并加载 DeepSpeed 配置
  3. 合并 runtime DeepSpeed 配置
  4. 剥离业务配置中的 `optimizer/scheduler`
  5. 组织为 `LoadedConfig` 返回
