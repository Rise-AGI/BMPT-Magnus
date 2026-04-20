# Interface Contract（重构版）

本文件定义 BMPT 训练接口的强约束。

## 1. 算法可插拔机制

- `bmpt-train` 通过 `--def-train` 动态导入训练定义模块（默认 `bmpt.algorithms.def_train`）。
- 默认 `step` 实现为 SFT。
- 训练定义模块必须暴露四个可调用对象：
  - `load_config(config_path)`
  - `build_models_from_config(config, loader_fn=...)`
  - `step(models, input)`
  - `evaluate(models, input)`
- 主循环只依赖以上接口，不依赖算法类型字段。

## 2. 训练入口

- 命令行入口：`bmpt-train`
- Python 模块入口：`python -m bmpt.cli.train`
- 默认配置：`src/bmpt/algorithms/config.yaml`

## 3. 模型加载接口

模块路径：`bmpt.model.loader`

导出函数：
- `load_model(label: str, spec: dict, config: dict) -> torch.nn.Module`

说明：
- `label`：模型标识（如 `policy`）
- `spec`：模型配置（`path`、`lora` 等）
- `config`：完整训练配置

## 4. Tokenizer 加载接口

模块路径：`bmpt.tokenizer.loader`

导出函数：
- `resolve_tokenizer_source(config, local_source=None) -> str`：通用 tokenizer 来源解析
- `load_tokenizer(config) -> Any`：加载 tokenizer
- `get_vocab_hash(tokenizer) -> str`

说明：
- `resolve_tokenizer_source` 支持多级优先级解析：
  1. `local_source`（调用方传入，如从 `data.tokenizer_source` 或 `prompting.tokenizer_source` 提取）
  2. 顶层 `tokenizer_source`
  3. 默认 `models.policy.path`
- 支持值：models key（如 `policy`/`reference`）或直接路径
- `load_tokenizer` 使用 `data.tokenizer_source` 作为局部来源
- 自动设置 `pad_token = eos_token`

## 5. 数据预处理接口

模块路径：`bmpt.data.processor`

核心函数：
- `load_jsonl(path) -> list[dict]`
- `save_jsonl(records, path)`
- `validate_required_keys(records, required_keys, source_path)`
- `tokenize_records(records, tokenizer, tokenize_keys, max_seq_len) -> list[dict]`
- `process_source(source_config, tokenizer, max_seq_len, cache_dir) -> list[dict]`
- `process_all_sources(config, tokenizer) -> dict[str, list[dict]]`

预处理输出格式：
```json
{
  "prompt": "原始文本",
  "response": "原始文本",
  "prompt_input_ids": [1, 2, 3, ...],
  "response_input_ids": [4, 5, 6, ...]
}
```

说明：
- 保留原始字段
- 每个 `tokenize_key` 生成 `{key}_input_ids` 字段
- 不生成 `attention_mask`、`labels`

## 6. DataLoader 构建接口

模块路径：`bmpt.data.dataloader`

导出函数：
- `build_dataloader(records, config, dist_ctx, shuffle=True, pad_token_id=0) -> DataLoader`

说明：
- `records`：预处理后的记录列表
- `pad_token_id`：用于动态 padding 的 pad token ID（`*_input_ids` 字段使用此值填充）
- 返回的 DataLoader 每个批次为 `dict[str, Any]`，包含 `{key}_input_ids` 字段
- 支持变长序列动态 padding：batch 内 `*_input_ids` 自动 pad 到最大长度

## 7. `step(models, input)` 返回协议

`step` 返回字典，至少包含：

- `loss`: `torch.Tensor`
- `metrics`: `dict[str, float]`
- `aux`: `dict[str, Any]`（可选扩展）

说明：训练框架会在日志阶段为 `metrics` 追加 `perf/*` 指标（如滑动窗口 step 耗时与吞吐），并复用同一 metrics 输出通道。

示例：

```python
return {
    "loss": loss,
    "metrics": {"loss/sft": float(loss.detach().item())},
    "aux": {},
}
```

## 8. `step(models, input)` 输入格式

`input["batch"]` 包含预处理后的数据：

```python
{
    "prompt_input_ids": torch.Tensor,  # shape: [batch_size, seq_len]
    "response_input_ids": torch.Tensor,
    # 其他 *_input_ids 字段...
}
```

默认 `step` 函数逻辑：

1. 从 batch 中提取所有 `*_input_ids` 字段
2. 按字段名排序组合成合并的 `input_ids`
3. 生成 `attention_mask`（全 1）
4. 生成 `labels`：第一个 tokenize_key 对应 -100，其余保持原值

## 9. `step(models, input)` 可选输入

当配置了 `config.prompting.composers` 时，训练主程序会在启动阶段完成 prompt tokenize 并注入：

- `input["composers"]: dict[str, Composer]`

`Composer` 提供：

- `compose(outputs, output_masks=None) -> {"input_ids", "attention_mask", "lengths"}`

说明：

- 返回的 `input_ids` 与 `attention_mask` 可直接作为下一个模型前向输入。
- `compose` 当前不返回 `labels`；若要做监督训练，需要在 `step` 内按任务策略构造 `labels`（常见为非监督位填 `-100`）。

用于在 `step` 内执行 token 级组合：

- `[prompt] + [model1 out] + [prompt] + [model2 out] + ...`

## 10. `evaluate(models, input)` 约定

- `evaluate` 由用户在 `def_train.py` 内完全实现。
- 训练框架不提供默认 eval 聚合逻辑，不会回退到 `step`。
- 训练框架会在验证阶段调用 `evaluate(models, input)`，并将其返回的 `metrics` 直接输出。

推荐输入字段：

- `input["val_iterable"]`：验证集迭代器
- `input["phase"]`：阶段标识（`before_train` / `after_train`）
- `input["dist_ctx"]`：分布式上下文
- `input["config_path"]`、`input["_cached_config"]`、`input["_merged_config"]`
- `input["composers"]`

推荐返回字段：

- `metrics`: `dict[str, float]`
- `aux`: `dict[str, Any]`

## 11. 分布式约定

- `bmpt-train` 支持 launcher 模式（`--nproc-per-node` 等参数）与 worker 模式（由环境变量 `RANK/WORLD_SIZE/LOCAL_RANK` 触发）。
- worker 进程分布式初始化由 `bmpt.core.distributed.init_distributed` 统一处理。

## 12. Checkpoint 恢复契约

- 训练入口使用统一 `.pt` checkpoint payload（`format_version=2`）。
- `train.load_ckpt_mode=full`：恢复 model/optimizer/scheduler/step，并覆盖配置键：`optimizer.*`、`scheduler.*`、`train.gradient_accumulation_steps`、`train.mixed_precision`、`runtime.training_backend`。
- `train.load_ckpt_mode=weights_only`：仅恢复 model；其余相关配置只提示差异，不执行覆盖。
- 恢复失败（文件缺失、损坏、shape/key 不匹配等）默认 fail-fast，不静默回退到从头训练。