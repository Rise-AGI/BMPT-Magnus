# `step(models, input)` 接口约定

本文档定义算法层 `step` 的最小契约，供 AI Agent 编写/修改 `def_train.py` 时遵循。

## 1. 函数签名

推荐签名：

```python
def step(models, input):
    ...
```

要求：

- `models`：模型对象或模型字典（常见为 `{"policy": model}`）
- `input`：包含训练批次与上下文字段的字典

## 2. 输入字段（常用）

`input` 常见字段：

- `batch`：训练 batch（必需）
- `global_step`：全局步数（可选）
- `config_path`：配置路径（可选）
- `_cached_config`：缓存配置（可选）
- `_merged_config`：合并后配置（可选）
- `_debug`：调试开关（可选）

`batch` 常见字段：

- `input_ids`
- `attention_mask`（可选）
- `labels`（可选）

## 3. 返回字段（最小必需）

`step` 必须返回 `dict`，且至少包含：

- `loss`：`torch.Tensor`
- `metrics`：`dict[str, float]`
- `aux`：`dict[str, Any]`（可为空字典）

示例：

```python
return {
    "loss": loss,
    "metrics": {
        "loss/sft": float(loss.detach().item()),
    },
    "aux": {},
}
```

## 4. 与训练框架的协作约束

- 不在 `step` 内直接执行 `optimizer.step()`。
- 不在 `step` 内直接处理 DDP/DeepSpeed 初始化。
- 训练后端切换由 CLI + runtime 配置完成，`step` 只负责算法逻辑。

## 5. 兼容建议

- 优先复用 `bmpt.train_utils`：
  - `resolve_step_config`
  - `resolve_models`
  - `build_step_context`
- 保持 `metrics` 键稳定，便于日志与监控聚合。
