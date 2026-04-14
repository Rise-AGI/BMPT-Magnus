# Interface Contract

本文件定义训练接口的强约束。`train/def_train.py` 按本协议直接执行，不包含数据格式兜底判断。

## 1. step 接口

```python
step(models, input) -> dict[str, Any]
```

- `models`: `dict[str, torch.nn.Module] | None`
- `input`: `dict[str, Any]`
- 返回：
  - `loss`: `torch.Tensor`，标量，`requires_grad=True`
  - `metrics`: `dict[str, float | int]`
  - `aux`: `dict[str, Any]`

## 2. models 协议

- key 由 `config.yaml` 的 `models` 节点决定。
- 当 `models is None` 时，调用 `loader_fn(label, spec, config)` 构建模型。

## 3. input 协议

- `input["batch"]`: `dict[str, torch.Tensor]`
- `input["config_path"]`: 配置路径（可选）
- `input["config"]`: step 级配置覆盖（可选）
- `input["loader_fn"]`: 构建模型函数（当 `models is None` 时可选）

说明：forward/reward/loss 都在 `train/def_train.py` 的 `step` 中实现。

## 4. step 行为约定

签名：

`step` 在 `def_train.py` 中应自行完成：

- forward
- reward 计算
- loss 计算（含 KL 或其他项）

返回约束：

- `loss`: 必填，`torch.Tensor` 标量
- `reward`: 可选，`torch.Tensor` 或可转标量
- `metrics`: 可选，日志化标量字典

## 5. 维度约束

- `B`: batch size
- `T`: 序列长度
- `V`: 词表大小
- 在包含 KL/策略对比的算法中：
  - `labels.shape == policy_logits.shape[:-1]`
  - 若存在 `reference_logits`，需与 `policy_logits.shape` 完全一致

## 6. 运行时行为

- 配置按路径全局缓存；同一路径不会重复读取磁盘。
- `StepContext.cached_config` 暴露缓存中的原始配置（未叠加 step override）。
- `StepContext.full_config` 暴露当前 step 生效配置（已叠加 `input["config"]`）。
- 训练流程不使用顶层 `mode` 或 `algorithm` 开关；算法由 `def_train.py` 决定。
- 不做输入结构兜底判断，调用方必须严格遵守本协议。
- 协议变更时必须先更新本文件，再更新实现。

## 7. 顶层 weighted 配置

```yaml
weighted:
  enabled: true
  normalize_weights: false
  weights:
    reward: 1.0
    kl: 0.02
```

- `weights` 是绝对系数，不做自动归一化。
- `weights.kl` 作用于 KL loss。
- 如使用 weighted 多奖励策略，权重由该配置提供。

## 8. 最小运行示例

- 示例脚本：`examples/example_weighted_step.py`
- 运行方式：`uv run python examples/example_weighted_step.py`
