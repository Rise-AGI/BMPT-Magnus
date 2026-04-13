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
- `policy` 必须存在。
- 其他标签（如 `reference`, `reward`）在 `enabled: true` 时应提供。
- 当 `models is None` 时，调用 `loader_fn(label, spec, config)` 构建模型。

## 3. input 协议

- `input["batch"]`: `dict[str, torch.Tensor]`
- `input["mode"]`: `"sft" | "rlaif_lora"`（可选，默认取配置）
- `input["config_path"]`: 配置路径（可选）
- `input["config"]`: step 级配置覆盖（可选）
- `input["forward_fn"]`: 前向函数（可选）
- `input["reward_fns"]`: 多奖励函数字典（`rlaif_lora` 推荐）
- `input["reward_fn"]`: 单奖励函数（兼容字段）
- `input["loader_fn"]`: 构建模型函数（当 `models is None` 时可选）

## 4. forward_fn 协议

签名：

```python
forward_fn(models, batch, ctx) -> dict[str, torch.Tensor]
```

### 4.1 SFT 模式

必须返回：

- `loss`: `torch.Tensor`
  - 形状：`() | (B,) | (B, T)`
  - `def_train.py` 中统一执行 `loss.mean()`

### 4.2 RLAIF-LoRA 模式

必须返回：

- `policy_logits`: `torch.Tensor`，形状 `(B, T, V)`
- `labels`: `torch.LongTensor`，形状 `(B, T)`

可选返回：

- `reference_logits`: `torch.Tensor | None`，形状 `(B, T, V)`

## 5. reward_fn / reward_fns 协议

签名：

```python
reward_fn(outputs, batch, ctx) -> torch.Tensor
```

返回约束：

- `torch.Tensor`，形状 `(B,)`
- dtype 建议 `float32` 或 `bfloat16`（最终会对齐到 `policy_logits.dtype`）
- 与 batch 一一对应：第 `i` 个 reward 对应第 `i` 个样本

`reward_fns` 协议：

- 类型：`dict[str, Callable]`
- key 必须与 `weighted.weights` 除 `kl` 外的 key 严格一致
- 聚合公式：`loss_total = Σ(weight_i * loss_i) + weight_kl * loss_kl`
- `weighted.normalize_weights` 必须为 `false`（当前实现不做权重归一化）

## 6. 维度约束

- `B`: batch size
- `T`: 序列长度
- `V`: 词表大小
- 在 `rlaif_lora` 中：
  - `labels.shape == policy_logits.shape[:-1]`
  - 若存在 `reference_logits`，需与 `policy_logits.shape` 完全一致

## 7. 运行时行为

- 配置按路径全局缓存；同一路径不会重复读取磁盘。
- `StepContext.cached_config` 暴露缓存中的原始配置（未叠加 step override）。
- `StepContext.full_config` 暴露当前 step 生效配置（已叠加 `input["config"]`）。
- 不做输入结构兜底判断，调用方必须严格遵守本协议。
- 协议变更时必须先更新本文件，再更新实现。

## 8. 顶层 weighted 配置

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
- 当前联合目标策略仅支持 `weighted`。

## 9. 最小运行示例

- 示例脚本：`train/example_weighted_step.py`
- 运行方式：`uv run python train/example_weighted_step.py`
