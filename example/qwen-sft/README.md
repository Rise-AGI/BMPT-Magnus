# BMPT SFT 示例：微调 Qwen2.5-7B

本示例展示如何使用 BMPT 工具箱进行 SFT（Supervised Fine-Tuning）微调 Qwen2.5-7B 模型。

## 文件结构

```
example/qwen-sft/
├── train.py              # 训练脚本（用户入口）
├── config.yaml           # 训练配置
├── deepspeed_zero2.json  # DeepSpeed Zero2 配置
├── data/
│   └── train.jsonl       # 训练数据
└── README.md             # 本文件
```

## 依赖安装

```bash
pip install -e .[deepspeed]
```

## 快速开始

### 单卡训练

```bash
cd example/qwen-sft
python train.py
```

### 多卡训练（推荐）

```bash
cd example/qwen-sft
torchrun --nproc_per_node=8 train.py
```

## 数据格式

训练数据为 JSONL 格式，每行包含：

- `prompt`: 输入文本
- `response`: 输出文本（目标）

示例：
```json
{"prompt": "问题文本", "response": "回答文本"}
```

预处理后自动生成 `prompt_input_ids` 和 `response_input_ids` 字段。

## 配置说明

### config.yaml 关键字段

| 字段 | 说明 |
|------|------|
| `models.policy.path` | 模型路径（HuggingFace 或本地） |
| `models.policy.lora.enabled` | 是否启用 LoRA |
| `data.sources` | 数据源列表（name, path, tokenize_keys） |
| `train.max_steps` | 最大训练步数 |
| `train.per_device_batch_size` | 单卡批次大小 |
| `train.gradient_accumulation_steps` | 梯度累积步数 |
| `optimizer.lr` | 学习率 |
| `runtime.deepspeed_config_path` | DeepSpeed 配置文件路径 |

### DeepSpeed 配置

本示例使用 ZeRO-Stage2：
- `stage: 2` - ZeRO Stage 2 优化
- `overlap_comm: true` - 通信重叠
- `contiguous_gradients: true` - 连续梯度

## 训练脚本结构

```python
# 1. 加载配置
manager = Manager()
manager.load_config("config.yaml")

# 2. 初始化分布式
dist_ctx = init_distributed()

# 3. 加载 tokenizer 和预处理数据
tokenizer = load_tokenizer(config)
processed_data = process_all_sources(config, tokenizer)

# 4. 构建 DataLoader
train_loader = build_dataloader(train_records, config, dist_ctx)

# 5. 加载模型并初始化 DeepSpeed
toolbox = ToolBox(manager)
models = toolbox.load_models("policy")
engine = toolbox.engine

# 6. 训练循环
for batch in train_loader:
    input_ids, attention_mask, labels = build_sft_batch(batch, tokenizer, device)
    outputs = engine.module(input_ids, attention_mask, labels=labels)
    engine.backward(outputs.loss)
    engine.step()

# 7. 清理
cleanup_distributed()
```

## 输出

训练日志输出到 stdout，包含：
- `loss/sft`: SFT 损失
- `perf/tokens_per_sec`: 吞吐量
- `train/step`: 当前步数

Checkpoint 保存到 `checkpoints/step_*.pt`。

## 扩展

### 使用 LoRA

修改 config.yaml：
```yaml
models:
  policy:
    path: Qwen/Qwen2.5-7B-Instruct
    trainable: true
    lora:
      enabled: true
      r: 64
      alpha: 128
      target_modules: ["q_proj", "v_proj"]
```

### 添加验证集

在 data.sources 中添加：
```yaml
data:
  sources:
    - name: train
      path: data/train.jsonl
      tokenize_keys: [prompt, response]
    - name: val
      path: data/val.jsonl
      tokenize_keys: [prompt, response]
```

训练脚本中：
```python
val_records = processed_data.get("val")
if val_records:
    val_loader = build_dataloader(val_records, config, dist_ctx, shuffle=False)
    # 定期验证...
```