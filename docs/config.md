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
  load_ckpt_path: null           # 可选；支持相对 config 的路径
  load_ckpt_mode: full           # full | weights_only
  load_ckpt_strict: true         # 传给 model.load_state_dict(strict=...)
  log_every_steps: 10
```

- `control_mode` 控制训练截断方式：
  - `step`：按 `max_steps` 截断（或 CLI `--max-steps` 覆盖）。
  - `epoch`：按 `epochs * len(dataloader)` 计算总步数，忽略 `max_steps`。
- `max_steps <= 0` 且 `control_mode=step` 时，训练入口会使用内部默认上限。
- `checkpoint_every_steps <= 0` 表示不按间隔保存。
- checkpoint 文件统一为 `.pt`：间隔保存 `step_<global_step>[.rank_<rank>].pt`，结束保存 `latest[.rank_<rank>].pt`。
- `load_ckpt_mode=full`：恢复 model + optimizer + scheduler + step，并用 checkpoint 内 `resume_config` 覆盖相关配置。
- `load_ckpt_mode=weights_only`：仅恢复 model；其余配置仅打印差异不覆盖。
- `load_ckpt_strict`：传给 `model.load_state_dict(strict=...)`。
  - `true`（默认）：checkpoint 与当前模型参数名/shape 需严格匹配，缺键/多键/shape 不一致会直接报错并终止。
  - `false`：允许非严格匹配（常用于只加载部分权重或结构轻微变动时），但不匹配参数不会被恢复。
- `full` 覆盖键：`optimizer.*`、`scheduler.*`、`train.gradient_accumulation_steps`、`train.mixed_precision`、`runtime.training_backend`。

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
  attn_implementation: auto   # auto | flash_attention_2 | sdpa | eager
  metrics:
    enabled: true
    window_size: 20
    global_throughput: true
    output: []                 # [] 不输出；可选 stdout / file:/path/to.log / 二者同时
  deepspeed_config_path: deepspeed_zero2.json
  compile: false
  gradient_checkpointing: true
  flash_attention: true       # 兼容字段，等价于优先尝试 flash_attention_2
```

- `training_backend` 控制正式训练入口走 PyTorch 或 DeepSpeed。
- `attn_implementation` 控制模型 attention 实现。
  - 推荐默认：`auto`
  - 可显式指定：`flash_attention_2`、`sdpa`、`eager`
  - 若指定/自动探测到 `flash_attention_2` 但环境不支持，会自动回退并告警。
- `metrics` 控制训练性能日志：
  - `window_size`：滑动窗口长度（用于 step 耗时与吞吐平均值）
  - `global_throughput`：是否按全局口径统计吞吐（sum tokens/samples + max step time）；为降低通信开销，全局归约仅在 `log_every_steps` 命中时执行
  - `output`：输出目标列表
    - `[]`：默认不输出
    - `stdout`：输出到控制台
    - `file:/abs/path.log`：追加写入文件
    - 支持同时配置 `stdout` 与 `file:...`
- 使用 DeepSpeed 时会读取 `deepspeed_config_path` 指向的外部 JSON。
- `gradient_checkpointing: true` 时，模型会设置 `use_cache=false` 并启用 `gradient_checkpointing_enable()`。
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

- `bmpt.components.qwen_components` 默认按 JSONL 读取该结构。

## `prompting`（可选）

用于注册多个 composer。训练主程序在启动阶段加载并预 tokenize prompt，然后将 `composers` 注入 `step(..., input)`。

```yaml
prompting:
  tokenizer_source: policy
  composers:
    chain_a:
      prompts:
        - "请先总结："
        - "请继续判断："
        - "最终结论："
      max_total_len: 4096
      truncate_side: left          # left | right
      pad_to_multiple_of: 8
      add_bos: false
      add_eos: false
      output_pad_token_id: null
```

- `tokenizer_source` 默认绑定 `models.policy.path`。
- `composers.<name>.prompts` 长度必须为 `N+1`，对应 `compose(outputs=[...])` 的 `N` 组 batched output。
- `compose` 支持动态 padding，并受 `max_total_len` 限制。
- 性能关键点：prompt 仅在启动时 tokenize 一次，step 内只做 token 级拼接。
- `compose` 返回 `input_ids/attention_mask/lengths`，可直接喂给下一个模型；如需训练 loss，请在 `step` 内自行构造 `labels`。

## CLI 覆盖关系

优先级从高到低：

1. CLI 参数（如 `--backend`, `--max-steps`）
2. `step(..., input={"config": ...})` 的 step 级覆盖
3. `train/config.yaml` 默认值

其中 `attn_implementation` 的来源优先级为：

1. CLI `--attn-implementation`
2. `runtime.attn_implementation`
3. `runtime.flash_attention=true`
4. 默认 `auto`
