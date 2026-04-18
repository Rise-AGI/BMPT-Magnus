# Interface Contract（重构版）

本文件定义 BMPT 训练接口的强约束。

## 1. 算法可插拔机制

- `bmpt-train` 通过 `--def-train` 动态导入训练定义模块（默认 `bmpt.algorithms.def_train`）。
- 默认 `step` 实现为 SFT。
- 训练定义模块必须暴露三个可调用对象：
  - `load_config(config_path)`
  - `build_models_from_config(config, loader_fn=...)`
  - `step(models, input)`
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

示例：

```python
return {
    "loss": loss,
    "metrics": {"loss/sft": float(loss.detach().item())},
    "aux": {},
}
```

## 4. 分布式约定

- `bmpt-train` 支持 launcher 模式（`--nproc-per-node` 等参数）与 worker 模式（由环境变量 `RANK/WORLD_SIZE/LOCAL_RANK` 触发）。
- worker 进程分布式初始化由 `bmpt.core.distributed.init_distributed` 统一处理。
