# BMPT 训练上手教程

这份文档面向“有算法想法，但不想深挖 Torch/DeepSpeed 细节”的使用者。

## 1. 三层结构（先记住这个）

- 算法层：`train/def_train.py`（你写训练目标和 loss）。
- 执行层：`main.py` + `src/core/`（框架负责分布式、反传、优化器、checkpoint）。
- 配置层：`train/config.yaml`（你调学习率、后端和路径）。

你通常只需要改两处：

1. `train/def_train.py`
2. `train/config.yaml`

## 2. 环境准备

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install torch transformers peft deepspeed pyyaml
```

如果你先跑 PyTorch 路线，可先不装 `deepspeed`。

## 3. 配置核心项

在 `train/config.yaml` 中重点看这些字段：

- `optimizer.lr`
- `train.per_device_batch_size`
- `train.gradient_accumulation_steps`
- `runtime.training_backend`: `pytorch` 或 `deepspeed`
- `runtime.deepspeed_config_path`
- `weighted.weights`（多目标权重，包含 `kl`）

完整字段说明见：`docs/config.md`

## 4. 实现你的算法

在 `train/def_train.py` 里实现你的 `step_impl(...)`（可复用 `sft_step/rlaif_lora_step` 作为模板）。

接口约定：

- `forward_fn(models, batch, ctx)`
  - SFT 返回 `{"loss": tensor}`
  - RLAIF 返回 `{"policy_logits": ..., "labels": ..., "reference_logits": ...}`
- `reward_fns` 是字典，key 必须与 `weighted.weights`（除 `kl`）一致。
- 训练入口会把 `models` 按 `{key: model}` 传给 `step_impl`，不再通过顶层 `mode` 做路由。

## 5. 启动训练

单卡快速验证：

```bash
torchrun --nproc_per_node=1 main.py --entry train --backend pytorch --config train/config.yaml --max-steps 20
```

DeepSpeed 路线：

```bash
torchrun --nproc_per_node=1 main.py --entry train --backend deepspeed --config train/config.yaml --max-steps 20
```

## 6. 多 GPU / 集群

单机 8 卡：

```bash
torchrun --nproc_per_node=8 main.py --entry train --backend pytorch --config train/config.yaml
```

双机示例（每机 8 卡）：

```bash
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr=<master_ip> --master_port=29500 main.py --entry train --backend deepspeed --config train/config.yaml
```

```bash
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 --master_addr=<master_ip> --master_port=29500 main.py --entry train --backend deepspeed --config train/config.yaml
```

## 7. 保存与恢复

配置项：

- `train.checkpoint_every_steps`
- `train.checkpoint_dir`

训练结束额外保存 `latest`：

```bash
torchrun --nproc_per_node=8 main.py --entry train --backend pytorch --config train/config.yaml --save-final
```

## 8. 组件选择

默认组件（快速 smoke test）：

- `util.components.default_components:load_model`
- `util.components.default_components:build_dataloader`

Qwen 组件（生产训练）：

- `util.components.qwen_components:load_model`
- `util.components.qwen_components:build_dataloader`

示例：

```bash
torchrun --nproc_per_node=8 main.py --entry train --config train/config.yaml --loader util.components.qwen_components:load_model --dataloader util.components.qwen_components:build_dataloader
```
