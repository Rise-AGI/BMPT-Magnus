# Brisk Post-Training Guide

快速上手请先看：`docs/guide.md`

本文档定义 `train/def_train.py` 入口的 `step(models, input)` 协议，作为 SFT / RLAIF-LoRA / 多模型联合训练的统一接口。
严格字段与维度约束请以 `docs/interface.md` 为准。

## 1. 设计目标

- 用户只关心任务逻辑：自定义 `forward` 与多 reward。
- 底层统一管理训练细节：模式切换、loss 聚合、性能优化开关。
- 同一 `step` 协议可覆盖单模型与多模型，避免重复实现。

## 2. 目录边界

- `src/`: 内核实现（配置加载、上下文结构、调度/性能工具等通用能力）。
- `util/`: 通用工具（配置缓存、模型校验/构建、回调解析）。
- `train/`: 用户逻辑（SFT/RLAIF 算法、forward/reward 定义、训练入口与配置）。

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
- 多模型：推荐使用字典，key 默认由 `config.yaml` 的 `models` 标签决定，常见键：
  - `policy`: 待优化主模型
  - `reference`: 参考模型（如 KL 约束）
  - `reward`: 奖励模型（RLAIF）

约束：

- `step` 会按配置校验传入的 `models` key。
- 当传入 `models=None` 时，`step` 会尝试按配置标签构建模型（需提供 `loader_fn`）。

### 3.2 `input` 约定

`input` 至少应包含一组 batch 张量，推荐字段：

- `batch`: `dict[str, Tensor]`，例如 `input_ids`, `attention_mask`, `labels`
- `step_impl`: 自定义训练步骤实现函数（可选，不传则走默认 SFT 模板）
- `config`: `dict[str, Any]`，本 step 的超参覆盖（可选）

## 4. 返回值约定

- `loss`: 必须是标量张量，用于 `backward()`。
- `metrics`: 仅放可日志化标量（例如 `loss/sft`, `reward/mean`, `kl`）。
- `aux`: 非日志必需信息（如中间 token、调试信息）。

> 要求：`metrics` 与 `aux` 不应影响计算图主路径，避免额外显存与性能开销。

## 5. 推荐执行流程

`step` 内部推荐按以下顺序组织（在 `train/def_train.py` 中实现）：

1. 构造统一上下文。
2. 调用用户自定义 `forward` 生成 logits / sequences。
3. 在 `step_impl` 中计算损失（可复用 `sft_step` 或 `rlaif_lora_step` 模板）。
4. 聚合多模型损失（如 weighted）。
5. 输出 `loss`, `metrics`, `aux`。

## 6. 用户扩展点

建议在 `train/def_train.py` 中直接维护以下函数，并由 `step` 调用：

```python
def custom_forward(models, batch, ctx):
    ...

def reward_fn(outputs, batch, ctx):
    ...

def reward_fns(outputs, batch, ctx):
    ...
```

其中 `ctx` 可包含：`global_step`, `runtime_config`, `full_config`, `cached_config`。

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
配置文件会被全局缓存；只有当 `config_path` 改变时才重新加载。
完整字段说明见：`docs/config.md`

### 8.1 顶层字段

- `seed`: 随机种子
- `models`: 模型加载与可训练性
- `optimizer`: 优化器参数（含学习率）
- `scheduler`: 学习率调度参数
- `train`: 通用训练超参
- `rlaif`: RLAIF 专用超参（非 RLAIF 模式可忽略）
- `weighted`: 顶层加权配置（reward/kl 系数）
- `runtime`: 运行时后端与性能开关
- `data`: 数据集路径与字段映射

### 8.2 `models` 字段协议

- `models.policy.path`: 主模型路径（必填）
- `models.policy.trainable`: 是否更新参数
- `models.policy.lora.enabled`: 是否启用 LoRA

约束：

- `models` 仅用于声明要加载的 key。
- 训练入口按 key 调用 `loader_fn` 构建 `{key: model}` 并传入 `step`。

### 8.3 `optimizer` 字段协议

- `optimizer.type`: 当前推荐 `adamw`
- `optimizer.lr`: 学习率（必填）
- `optimizer.weight_decay`, `optimizer.betas`, `optimizer.eps`: 可选高级参数

### 8.4 `weighted` 字段协议

- `weighted.enabled`: 是否启用加权聚合
- `weighted.normalize_weights`: 固定为 `false`
- `weighted.weights`: 加权系数，要求包含 `kl`

约束：

- 除 `kl` 外，其余 key 必须与 `input["reward_fns"]` key 严格一致。
- 系数按绝对值使用，不做归一化。

### 8.5 `train` 字段协议

- `per_device_batch_size`
- `gradient_accumulation_steps`
- `max_seq_len`
- `mixed_precision`（`bf16` / `fp16` / `no`）
- `grad_clip_norm`
- `log_every_steps`
- `checkpoint_every_steps`
- `checkpoint_dir`

### 8.6 `runtime` 字段协议

- `backend`: `accelerate`（首版默认）
- `compile`: 是否启用 `torch.compile`
- `gradient_checkpointing`: 是否启用梯度检查点
- `flash_attention`: 是否启用 flash attention（若模型/环境支持）

### 8.7 覆盖优先级

同一字段冲突时，按以下优先级生效：

1. `input["config"]`（当前 step 动态覆盖）
2. `train/config.yaml`（全局默认）

### 8.8 示例

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

1. 持续完善 weighted 多 reward 训练策略。
2. 接入真实 reward model 推理流程。
3. 增加性能 AB 基准，验证 <=5% 目标。

## 10. 入口调用说明（`train/def_train.py`）

默认入口已提供：

- `load_config(config_path=None)`: 读取配置文件（默认 `train/config.yaml`）
- `step(models, input)`: 统一 step 执行接口
- `sft_step(...)`: SFT 算法逻辑（用户可直接修改）
- `rlaif_lora_step(...)`: RLAIF-LoRA 算法逻辑（用户可直接修改）
- `build_models_from_config(config, loader_fn)`: 按配置标签构建模型字典
- `ctx.cached_config`: 缓存中的原始配置
- `ctx.full_config`: 当前 step 生效配置（含 override）

底层工具位于：`util/train_utils.py`

`input` 可选扩展字段：

- `config_path`: 指定配置文件路径
- `forward_fn`: 自定义 forward 回调
- `reward_fns`: 多奖励回调字典（RLAIF 模式）
- `global_step`: 当前全局 step（用于上下文）
- `loader_fn`: 当 `models=None` 时用于加载模型

示例：

```python
def loader_fn(label, spec, config):
    # 例如: 使用 transformers 加载 spec["path"]
    return loaded_model

result = step(
    models=None,
    input={
        "batch": batch,
        "step_impl": sft_step,
        "config_path": "train/config.yaml",
        "config": {"optimizer": {"lr": 1e-5}},
        "loader_fn": loader_fn,
        "forward_fn": custom_forward,
        "reward_fns": {
            "reward": custom_reward_fn,
        },
    },
)
loss = result["loss"]
metrics = result["metrics"]
```

## 11. 最小可运行示例

- 双 reward weighted 示例：`examples/example_weighted_step.py`
- 示例展示：
  - `reward_fns` 与 `weighted.weights`（除 `kl`）严格同名
  - `ctx.cached_config` 与 `ctx.full_config` 的差异

## 12. 计算核心（src/core）

- `src/core/engine.py`: 统一训练执行器（反传、梯度累计、优化器步进、调度器步进）
- `src/core/optim.py`: 优化器与学习率调度器构建
- `src/core/distributed.py`: 分布式初始化、DDP 包装、跨 rank 指标聚合
- `examples/example_engine_loop.py`: 使用 `TrainingEngine` 驱动 `train/def_train.py::step` 的最小示例

## 13. 正式训练入口（多 GPU / 集群）

- 根入口：`main.py`
- 默认执行：`python main.py --entry train`
- 支持 `torchrun` 启动多 GPU / 多机训练
- 支持 `runtime.training_backend` 在 `pytorch` / `deepspeed` 间切换

单机多卡示例：

```bash
torchrun --nproc_per_node=8 main.py --entry train --config train/config.yaml
```

双机示例（节点 0）：

```bash
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr=<master_ip> --master_port=29500 main.py --entry train --config train/config.yaml
```

双机示例（节点 1）：

```bash
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 --master_addr=<master_ip> --master_port=29500 main.py --entry train --config train/config.yaml
```

入口参数：

- `--loader module:function`: 模型加载函数（默认 `util.components.qwen_components:load_model`）
- `--dataloader module:function`: 数据构建函数（默认 `util.components.qwen_components:build_dataloader`）
- `--max-steps`: 覆盖配置中的训练步数
- `--backend pytorch|deepspeed`: 覆盖 `config.yaml` 中的后端设置
- `--save-final`: 训练结束额外保存 `latest` checkpoint

后端切换配置（`train/config.yaml`）：

```yaml
runtime:
  training_backend: pytorch
  deepspeed_config_path: ../configs/deepspeed_zero2.json
```

DeepSpeed 路线（读取外部 JSON）示例：

```bash
torchrun --nproc_per_node=8 main.py --entry train --backend deepspeed --config train/config.yaml
```

Checkpoint 配置：

- `train.checkpoint_every_steps`: 每隔多少 step 保存一次（`<=0` 表示不保存）
- `train.checkpoint_dir`: checkpoint 目录（默认 `checkpoints`）

Qwen 7B + LoRA（单机多卡）示例：

```bash
torchrun --nproc_per_node=8 main.py --entry train --config train/config.yaml --loader util.components.qwen_components:load_model --dataloader util.components.qwen_components:build_dataloader
```
