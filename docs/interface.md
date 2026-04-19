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

## 3. `step(models, input)` 返回协议

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

## 4. `step(models, input)` 可选输入

当配置了 `config.prompting.composers` 时，训练主程序会在启动阶段完成 prompt tokenize 并注入：

- `input["composers"]: dict[str, Composer]`

`Composer` 提供：

- `compose(outputs, output_masks=None) -> {"input_ids", "attention_mask", "lengths"}`

说明：

- 返回的 `input_ids` 与 `attention_mask` 可直接作为下一个模型前向输入。
- `compose` 当前不返回 `labels`；若要做监督训练，需要在 `step` 内按任务策略构造 `labels`（常见为非监督位填 `-100`）。

用于在 `step` 内执行 token 级组合：

- `[prompt] + [model1 out] + [prompt] + [model2 out] + ...`

## 5. `evaluate(models, input)` 约定

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

## 6. 分布式约定

- `bmpt-train` 支持 launcher 模式（`--nproc-per-node` 等参数）与 worker 模式（由环境变量 `RANK/WORLD_SIZE/LOCAL_RANK` 触发）。
- worker 进程分布式初始化由 `bmpt.core.distributed.init_distributed` 统一处理。
