# `bmpt.model.loader` 函数级文档

## `_require_hf()`
- 校验并导入 `transformers/peft` 依赖。

## `_is_rank0()`
- 判断当前进程是否 rank0（未初始化分布式时视为 rank0）。

## `_resolve_attn_implementation(config)`
- 按优先级解析 attention 实现：`runtime.attn_implementation` > `runtime.flash_attention` > `auto`。

## `_load_with_attn(loader_cls, model_path, requested_attn)`
- 负责按 attention 配置加载 HF 模型，并在不可用时执行兼容回退。

## `_apply_lora_if_needed(model, spec)`
- 当 `spec.lora.enabled=true` 时注入 LoRA。

## `load_model(label, spec, config)`
- 对外模型加载入口。
- 支持：
  - attention 自动探测/回退
  - 可选 gradient checkpointing
  - `policy` 模型可选 LoRA
  - `trainable` 决定是否冻结参数
