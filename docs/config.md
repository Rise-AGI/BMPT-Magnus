# 配置说明（`train/config.yaml`）

本文档说明项目当前支持的配置项、用途和默认语义。

## 顶层结构

```yaml
mode: sft | rlaif_lora
seed: 42

models: {...}
optimizer: {...}
scheduler: {...}
train: {...}
rlaif: {...}
weighted: {...}
runtime: {...}
data: {...}
```

## `mode`

- `sft`: 监督微调模式
- `rlaif_lora`: RLAIF + LoRA 模式

## `models`

用于声明需要加载的模型标签与参数。

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
    enabled: false
    path: Qwen/Qwen2.5-7B-Instruct
    trainable: false

  reward:
    enabled: false
    path: null
    trainable: false
```

- `policy` 为必需模型标签。
- `enabled: true` 的标签会被视为期望加载的模型。

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

- `max_steps <= 0` 时，训练入口会使用内部默认上限。
- `checkpoint_every_steps <= 0` 表示不按间隔保存。

## `rlaif`

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

- 当前实现固定使用加权聚合。
- `weights` 为绝对系数，不做归一化。
- `weights` 中除 `kl` 外的键，应与 `reward_fns` 键一致。

## `runtime`

```yaml
runtime:
  backend: accelerate
  training_backend: pytorch   # pytorch | deepspeed
  distributed_backend: nccl
  deepspeed_config_path: ../configs/deepspeed_zero2.json
  compile: false
  gradient_checkpointing: true
  flash_attention: true
```

- `training_backend` 控制正式训练入口走 PyTorch 或 DeepSpeed。
- 使用 DeepSpeed 时会读取 `deepspeed_config_path` 指向的外部 JSON。

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
