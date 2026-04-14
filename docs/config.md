# 配置说明（`train/config.yaml`）

本文档说明项目当前支持的配置项、用途和默认语义。

## 顶层结构

```yaml
seed: 42

models: {...}
optimizer: {...}
scheduler: {...}
train: {...}
weighted: {...}   # 可选
runtime: {...}
data: {...}
```

## `models`

用于声明需要加载的模型标签与参数。训练入口只按 key 加载并组装 `{key: model}` 传入 `step`。

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

- `models` 只用于声明“要加载哪些 key”。
- 训练入口会按 key 逐一调用 `loader_fn` 构建 `{key: model}`。

## `optimizer`

```yaml
optimizer:
  type: adamw
  lr: 2.0e-5
  weight_decay: 0.1
  betas: [0.9, 0.95]
  eps: 1.0e-8
```

## `scheduler`

```yaml
scheduler:
  type: cosine   # cosine | none
  warmup_steps: 100
  min_lr_ratio: 0.1
```

## `train`

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

- `control_mode` 控制训练截断方式：
  - `step`：按 `max_steps` 截断（或 CLI `--max-steps` 覆盖）。
  - `epoch`：按 `epochs * len(dataloader)` 计算总步数，忽略 `max_steps`。
- `max_steps <= 0` 且 `control_mode=step` 时，训练入口会使用内部默认上限。
- `checkpoint_every_steps <= 0` 表示不按间隔保存。

## `rlaif`（可选）

默认 `train/config.yaml` 不包含该字段；仅在你在 `def_train.py` 中实现 RLAIF 逻辑时按需添加。

```yaml
rlaif:
  reward_scale: 1.0
  normalize_reward: true
```

## `weighted`

```yaml
weighted:
  enabled: true
  normalize_weights: false
  weights:
    reward: 1.0
    kl: 0.02
```

- 该字段为可选，仅在你在 `def_train.py` 中实现 weighted 多奖励策略时需要。
- `weights` 为绝对系数，不做归一化。
- `weights` 中除 `kl` 外的键，应与你在 `def_train.py` 中实现的奖励项名称一致。

## `runtime`

```yaml
runtime:
  backend: accelerate
  training_backend: pytorch   # pytorch | deepspeed
  distributed_backend: nccl
  deepspeed_config_path: deepspeed_zero2.json
  compile: false
  gradient_checkpointing: true
  flash_attention: true
```

- `training_backend` 控制正式训练入口走 PyTorch 或 DeepSpeed。
- 使用 DeepSpeed 时会读取 `deepspeed_config_path` 指向的外部 JSON。
- 训练入口会用 `train/config.yaml` 覆盖 DeepSpeed JSON 的关键训练参数：
  - `train.per_device_batch_size` -> `train_micro_batch_size_per_gpu`
  - `train.gradient_accumulation_steps` -> `gradient_accumulation_steps`
  - `train.grad_clip_norm` -> `gradient_clipping`
  - `optimizer` -> `optimizer`（当前支持 `adamw`）
  - `train.mixed_precision` -> `bf16/fp16`
  - `scheduler` -> `scheduler`（当前支持 `cosine`、`none`）

## `data`

```yaml
data:
  train_path: data/train.jsonl
  val_path: null
  prompt_key: prompt
  response_key: response
```

- `util.components.qwen_components` 默认按 JSONL 读取该结构。

## CLI 覆盖关系

优先级从高到低：

1. CLI 参数（如 `--backend`, `--max-steps`）
2. `step(..., input={"config": ...})` 的 step 级覆盖
3. `train/config.yaml` 默认值
