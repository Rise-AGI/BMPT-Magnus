# Brisk Post-Training Guide

本文档定义 `train/def_train.py` 入口的 `step(models, input)` 协议，作为 SFT / RLAIF-LoRA / 多模型联合训练的统一接口。

## 1. 设计目标

- 用户只关心任务逻辑：自定义 `forward` 与 `reward`。
- 底层统一管理训练细节：模式切换、loss 聚合、性能优化开关。
- 同一 `step` 协议可覆盖单模型与多模型，避免重复实现。

## 2. 目录边界

- `src/`: 内核实现（训练引擎、模式插件、多模型调度、性能工具）。
- `train/`: 用户逻辑（数据处理、forward/reward 定义、训练入口与配置）。

## 3. step 协议

在 `train/def_train.py` 中实现：

```python
def step(models, input):
    """Run one train step.

    Args:
        models: dict[str, torch.nn.Module] | torch.nn.Module
        input: dict[str, Any]

    Returns:
        dict with keys:
          - loss: torch.Tensor (scalar, requires_grad)
          - metrics: dict[str, float | int]
          - aux: dict[str, Any] (optional)
    """
```

### 3.1 `models` 约定

- 单模型：可直接传 `nn.Module`，内部转换为 `{"policy": model}`。
- 多模型：推荐使用字典，常见键：
  - `policy`: 待优化主模型
  - `reference`: 参考模型（如 KL 约束）
  - `reward`: 奖励模型（RLAIF）

### 3.2 `input` 约定

`input` 至少应包含一组 batch 张量，推荐字段：

- `batch`: `dict[str, Tensor]`，例如 `input_ids`, `attention_mask`, `labels`
- `mode`: `str`，如 `"sft"`, `"rlaif_lora"`
- `config`: `dict[str, Any]`，本 step 的超参覆盖（可选）

## 4. 返回值约定

- `loss`: 必须是标量张量，用于 `backward()`。
- `metrics`: 仅放可日志化标量（例如 `loss/sft`, `reward/mean`, `kl`）。
- `aux`: 非日志必需信息（如中间 token、调试信息）。

> 要求：`metrics` 与 `aux` 不应影响计算图主路径，避免额外显存与性能开销。

## 5. 推荐执行流程

`step` 内部推荐按以下顺序组织：

1. 解析 `mode`，构造统一上下文。
2. 调用用户自定义 `forward` 生成 logits / sequences。
3. 按模式计算损失：
   - SFT: supervised loss
   - RLAIF-LoRA: reward + KL + policy objective
4. 聚合多模型损失（如加权求和）。
5. 输出 `loss`, `metrics`, `aux`。

## 6. 用户扩展点

建议在 `train/` 下提供以下函数，并由 `step` 调用：

```python
def custom_forward(models, batch, ctx):
    ...

def reward_fn(outputs, batch, ctx):
    ...
```

其中 `ctx` 可包含：`mode`, `global_step`, `device`, `dtype`, `runtime_config`。

## 7. 性能基线约束

- 相比朴素实现，吞吐下降不超过 5%。
- 观测指标：
  - `tokens_per_sec`
  - `step_latency_ms`
  - `max_memory_mb`

为保证该约束，`step` 设计应遵循：

- 避免重复 tokenize / 数据搬运。
- 避免在 step 内频繁创建大对象。
- 将非必要统计移出主路径，按间隔记录。

## 8. 配置协议（`train/config.yaml`）

训练入口默认从 `train/config.yaml` 读取基础参数，`input["config"]` 可在 step 级别覆盖。

### 8.1 顶层字段

- `mode`: `sft` 或 `rlaif_lora`
- `seed`: 随机种子
- `models`: 模型加载与可训练性
- `optimizer`: 优化器参数（含学习率）
- `scheduler`: 学习率调度参数
- `train`: 通用训练超参
- `rlaif`: RLAIF 专用超参（非 RLAIF 模式可忽略）
- `runtime`: 运行时后端与性能开关
- `data`: 数据集路径与字段映射

### 8.2 `models` 字段协议

- `models.policy.path`: 主模型路径（必填）
- `models.policy.trainable`: 是否更新参数
- `models.policy.lora.enabled`: 是否启用 LoRA
- `models.reference.enabled`: 是否启用 reference model
- `models.reward.enabled`: 是否启用 reward model

约束：

- `mode == "sft"` 时，`policy` 必须存在。
- `mode == "rlaif_lora"` 时，`policy` 必须开启 LoRA，且建议开启 `reference` 与 `reward`。

### 8.3 `optimizer` 字段协议

- `optimizer.type`: 当前推荐 `adamw`
- `optimizer.lr`: 学习率（必填）
- `optimizer.weight_decay`, `optimizer.betas`, `optimizer.eps`: 可选高级参数

### 8.4 `train` 字段协议

- `per_device_batch_size`
- `gradient_accumulation_steps`
- `max_seq_len`
- `mixed_precision`（`bf16` / `fp16` / `no`）
- `grad_clip_norm`
- `log_every_steps`
- `checkpoint_every_steps`

### 8.5 `runtime` 字段协议

- `backend`: `accelerate`（首版默认）
- `compile`: 是否启用 `torch.compile`
- `gradient_checkpointing`: 是否启用梯度检查点
- `flash_attention`: 是否启用 flash attention（若模型/环境支持）

### 8.6 覆盖优先级

同一字段冲突时，按以下优先级生效：

1. `input["config"]`（当前 step 动态覆盖）
2. `train/config.yaml`（全局默认）

### 8.7 示例

参见：`train/config.yaml`

RLAIF-LoRA 参考：`train/config.example.rlaif.yaml`

## 9. 最小示例（SFT）

```python
def step(models, input):
    model = models["policy"] if isinstance(models, dict) else models
    batch = input["batch"]

    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch.get("labels"),
    )
    loss = out.loss

    return {
        "loss": loss,
        "metrics": {"loss/sft": float(loss.detach())},
        "aux": {},
    }
```

---

后续实现顺序：

1. 增加多模型调度策略（交替/加权）。
2. 接入真实 reward model 推理流程。
3. 增加性能 AB 基准，验证 <=5% 目标。

## 10. 入口调用说明（`train/def_train.py`）

默认入口已提供：

- `load_config(config_path=None)`: 读取配置文件（默认 `train/config.yaml`）
- `step(models, input)`: 统一 step 执行接口

`input` 可选扩展字段：

- `config_path`: 指定配置文件路径
- `forward_fn`: 自定义 forward 回调
- `reward_fn`: 自定义 reward 回调（RLAIF 模式）
- `global_step`: 当前全局 step（用于上下文）

示例：

```python
result = step(
    models=models,
    input={
        "batch": batch,
        "mode": "sft",
        "config_path": "train/config.yaml",
        "config": {"optimizer": {"lr": 1e-5}},
        "forward_fn": custom_forward,
    },
)
loss = result["loss"]
metrics = result["metrics"]
```
