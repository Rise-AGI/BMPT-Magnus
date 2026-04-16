# Interface Contract（重构版）

本文件定义 BMPT 训练接口的强约束，并解释“算法层可插拔”的实现原理。
核心目标是：你只需改 `train/def_train.py` 和 `train/config.yaml`，无需改训练框架主循环。

---

## 1. 算法层可插拔：机制与实现原理

### 1.1 设计目标

- 把“算法逻辑”与“工程执行逻辑”解耦。
- 算法层负责：forward / reward / loss。
- 框架层负责：分布式、优化器、梯度累积、checkpoint、日志与后端切换。

### 1.2 可插拔点（实际生效位置）

- `main.py` 通过 `--def-train` 动态导入训练定义模块（默认 `train.def_train`）。
- 该模块必须暴露三个可调用对象：
  - `load_config(config_path)`
  - `build_models_from_config(config, loader_fn=...)`
  - `step(models, input)`
- 主循环只调用这三个接口，不关心你在 `step` 里实现的是 SFT、RLAIF、DPO 还是其它目标。

### 1.3 为什么不需要 `mode/algorithm` 开关

- 训练流程不通过顶层 `mode` 或 `algorithm` 字段选算法。
- 算法由 `def_train.py` 的 `step` 实现直接决定。
- 因此同一套训练引擎可以复用到不同训练目标，接口稳定、迁移成本低。

### 1.4 训练主循环执行流程图

```text
CLI(main.py)
  |
  +--> parse_args
  |
  +--> import def_train module (--def-train)
  |       |
  |       +--> load_config(config_path)
  |       +--> build_models_from_config(config, loader_fn)
  |
  +--> init_distributed
  +--> build dataloader(train/val)
  +--> resolve total_steps (step/epoch)
  |
  +--> select backend
          |
          +--> PyTorch backend:
          |      for each batch:
          |        move_to_device
          |        engine.run_micro_step(...)
          |          -> step(models, input)
          |          -> backward / grad_accum / optimizer.step / scheduler.step
          |        log / checkpoint
          |
          +--> DeepSpeed backend:
                 for each batch:
                   move_to_device
                   step(models, input)
                   ds_engine.backward(loss)
                   ds_engine.step()
                 log / checkpoint
  |
  +--> optional final validation
  +--> cleanup_distributed
```

### 1.5 数据流向示意图（含 `step`）

```text
config.yaml
   |
   +--> models.* ------------------------------+
   |                                           |
   +--> train/runtime/optimizer/scheduler ----+|          +-----------------------+
                                               ||          | train/def_train.py   |
dataloader(batch) --> move_to_device ----------+----------> | step(models, input) |
                                                           |  - forward           |
models(dict[str, nn.Module]) -----------------------------> |  - reward(optional)  |
                                                           |  - loss              |
                                                           +-----+-----------------+
                                                                 |
                                                                 +--> {loss, metrics, aux}
                                                                         |
                                                                         +--> engine/deepspeed
                                                                              backward + step
```

---

## 2. `config.yaml` 参数详解（按实际训练入口）

说明：以下字段来自 `train/config.yaml` 与运行时解析逻辑；CLI 参数可覆盖部分配置。

### 2.1 顶层结构

```yaml
seed: 42

models: {...}
optimizer: {...}
scheduler: {...}
train: {...}
weighted: {...}   # 可选
rlaif: {...}      # 可选
runtime: {...}
data: {...}
```

### 2.2 `models`

用于声明要加载的模型键（key）及规格，训练入口据此组装 `dict[str, nn.Module]`。

```yaml
models:
  policy:
    path: Qwen/Qwen2.5-7B-Instruct
    trainable: true
    lora:
      enabled: true
      r: 64
      alpha: 128
      dropout: 0.05
      target_modules: [q_proj, k_proj, v_proj, o_proj]

  reference:
    path: Qwen/Qwen2.5-7B-Instruct
    trainable: false

  reward:
    path: null
    trainable: false
```

- `models.<key>.path`: 模型路径或 HuggingFace 名称。
- `models.<key>.trainable`: 是否参与梯度更新。
- `models.<key>.lora.*`: LoRA 参数（由具体 loader 是否支持决定）。
- 约束：`models` 仅声明“加载哪些 key”；算法如何使用这些 key，由 `step` 决定。

### 2.3 `optimizer`

```yaml
optimizer:
  type: adamw
  lr: 2.0e-5
  weight_decay: 0.1
  betas: [0.9, 0.95]
  eps: 1.0e-8
```

- `type`: 当前支持 `adamw`。
- `lr`: 学习率。
- `weight_decay`: 权重衰减。
- `betas`: Adam 一阶/二阶动量系数。
- `eps`: 数值稳定项。

### 2.4 `scheduler`

```yaml
scheduler:
  type: cosine   # cosine | none
  warmup_steps: 100
  min_lr_ratio: 0.1
```

- `type=none`: 常数学习率。
- `type=cosine`: 预热后余弦衰减。
- `warmup_steps`: 预热步数。
- `min_lr_ratio`: 余弦末端比例（最小 lr = `lr * min_lr_ratio`）。

### 2.5 `train`

```yaml
train:
  control_mode: step  # step | epoch
  epochs: 1
  max_steps: -1
  per_device_batch_size: 2
  gradient_accumulation_steps: 8
  max_seq_len: 4096
  mixed_precision: bf16
  grad_clip_norm: 1.0
  checkpoint_every_steps: 500
  checkpoint_dir: checkpoints
  log_every_steps: 10
```

- `control_mode`:
  - `step`: 按 `max_steps` 截断（可被 CLI `--max-steps` 覆盖）。
  - `epoch`: 按 `epochs * len(dataloader)` 计算总步数，忽略 `max_steps`。
- `epochs`: 仅 `control_mode=epoch` 时生效。
- `max_steps`: 仅 `control_mode=step` 时生效；`<=0` 时使用内部默认上限。
- `per_device_batch_size`: 每卡 micro-batch 大小。
- `gradient_accumulation_steps`: 梯度累积步数。
- `max_seq_len`: 序列最大长度（通常由 dataloader/tokenizer 使用）。
- `mixed_precision`: `bf16 | fp16 | no`（DeepSpeed 会映射为 bf16/fp16 配置）。
- `grad_clip_norm`: 梯度裁剪阈值。
- `checkpoint_every_steps`: 步间隔保存；`<=0` 表示不按间隔保存。
- `checkpoint_dir`: checkpoint 输出目录。
- `log_every_steps`: 日志打印频率。

### 2.6 `weighted`（可选）

```yaml
weighted:
  enabled: true
  normalize_weights: false
  weights:
    reward: 1.0
    kl: 0.02
```

- `enabled`: 是否启用 weighted 多项损失/奖励加权逻辑（由 `step` 解释）。
- `normalize_weights`: 是否做权重归一化（若你的算法实现支持）。
- `weights`: 各子目标绝对系数；`weights.kl` 通常作用于 KL 项。
- 约束：`weights` 中除 `kl` 外的 key，需与你的奖励项名称一致。

### 2.7 `rlaif`（可选）

```yaml
rlaif:
  reward_scale: 1.0
  normalize_reward: true
```

- 仅在你在 `step` 内实现 RLAIF 相关逻辑时生效。

### 2.8 `runtime`

```yaml
runtime:
  backend: accelerate
  training_backend: pytorch   # pytorch | deepspeed
  distributed_backend: nccl
  deepspeed_config_path: deepspeed_zero2.json
  compile: false
  gradient_checkpointing: true
  flash_attention: true
  debug: false
```

- `training_backend`: 正式训练后端选择。
- `distributed_backend`: 分布式通信后端（如 `nccl`）。
- `deepspeed_config_path`: DeepSpeed JSON 配置路径（可相对 `config.yaml`）。
- `compile/gradient_checkpointing/flash_attention`: 运行期特性开关，是否生效取决于组件实现。
- `debug`: 打开调试日志。

DeepSpeed 下，训练入口会用 `config.yaml` 覆盖 JSON 关键项：

- `train.per_device_batch_size -> train_micro_batch_size_per_gpu`
- `train.gradient_accumulation_steps -> gradient_accumulation_steps`
- `train.grad_clip_norm -> gradient_clipping`
- `optimizer -> optimizer`（当前支持 `adamw`）
- `train.mixed_precision -> bf16/fp16`
- `scheduler -> scheduler`（当前支持 `cosine`、`none`）

### 2.9 `data`

```yaml
data:
  train_path: data/train.jsonl
  val_path: null
  prompt_key: prompt
  response_key: response
```

- `train_path` / `val_path`: 训练与验证数据路径。
- `prompt_key` / `response_key`: 文本字段名（默认 Qwen 组件使用）。

### 2.10 CLI 覆盖优先级

优先级（高 -> 低）：

1. CLI 参数（如 `--backend`、`--max-steps`）
2. `step(..., input={"config": ...})` 的 step 级覆盖
3. `train/config.yaml` 默认值

---

## 3. `step(models, input)` 详细契约（含张量维度）

### 3.1 函数签名

```python
step(models, input) -> dict[str, Any]
```

- `models`: `dict[str, torch.nn.Module] | None`
- `input`: `dict[str, Any]`

返回值约束：

- `loss`: 必填，`torch.Tensor` 标量，`requires_grad=True`
- `metrics`: 可选，`dict[str, float | int]`，用于日志聚合
- `aux`: 可选，`dict[str, Any]`，承载额外调试/分析信息
- `reward`: 可选，`torch.Tensor` 或可转标量

### 3.2 `models` 参数说明

- key 来源于 `config.yaml` 的 `models` 节点。
- 典型结构：

```python
models = {
    "policy": policy_model,
    "reference": reference_model,   # 可选
    "reward": reward_model,         # 可选
}
```

- 当 `models is None` 时，可通过 `input["loader_fn"]` 按 `loader_fn(label, spec, config)` 构建。

### 3.3 `input` 参数说明

必选字段：

- `input["batch"]`: `dict[str, torch.Tensor]`

常见可选字段：

- `input["global_step"]`: `int`，当前全局步。
- `input["config_path"]`: 配置文件路径。
- `input["config"]`: step 级配置覆盖（会与缓存配置深合并）。
- `input["loader_fn"]`: 当 `models is None` 时用于构建模型。
- `input["forward_fn"]` / `input["reward_fn"]` / `input["reward_fns"]`: 自定义回调（按算法需要）。

### 3.4 `batch` 张量字段契约（逐项）

统一记号：

- `B`: batch size
- `T`: sequence length
- `V`: vocab size

常见输入（Causal LM 场景）：

- `batch["input_ids"]`: `torch.LongTensor`，形状 `[B, T]`
  - 每个元素是 token id。
- `batch["attention_mask"]`: `torch.LongTensor | torch.BoolTensor`，形状 `[B, T]`
  - `1/True` 表示有效 token，`0/False` 表示 padding。
- `batch["labels"]`: `torch.LongTensor`，形状 `[B, T]`
  - 监督目标 token id。
  - 被 mask 的位置应置为 `-100`（供交叉熵忽略）。

模型输出常见张量：

- `policy_logits`: `torch.FloatTensor`，形状 `[B, T, V]`
- `reference_logits`（可选）: `torch.FloatTensor`，形状 `[B, T, V]`
- `loss`: `torch.FloatTensor` 标量，形状 `[]`

奖励相关常见张量（可选）：

- `reward`: `torch.FloatTensor`，形状可为 `[]` 或 `[B]`
- 多奖励项：如 `reward_helpfulness`, `reward_safety`，通常为 `[B]`

### 3.5 维度一致性硬约束

- `labels.shape == policy_logits.shape[:-1]`，即 `[B, T] == [B, T, V][:-1]`。
- 若存在 `reference_logits`，其形状必须与 `policy_logits` 完全一致。
- `loss` 必须是单标量 tensor（不是向量），且可反传。

### 3.6 `step` 内部职责边界

`step` 在 `def_train.py` 中应自行完成：

- forward
- reward 计算
- loss 计算（含 KL 或其他项）

框架层不会替你补齐这些算法步骤。

---

## 4. 运行时行为与协议规则

- 配置按路径全局缓存；同一路径不会重复读取磁盘。
- `StepContext.cached_config` 是缓存中的原始配置（未叠加 step override）。
- `StepContext.full_config` 是当前 step 生效配置（已叠加 `input["config"]`）。
- 训练流程不使用顶层 `mode` 或 `algorithm` 开关；算法由 `def_train.py` 决定。
- 不做输入结构兜底判断，调用方必须严格遵守本协议。
- 协议变更时，必须先更新本文件，再更新实现。

---
