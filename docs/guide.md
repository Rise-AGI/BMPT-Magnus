# BMPT 训练上手教程

这份文档面向“有算法想法，但不想深挖训练底层细节”的使用者。

## 1. 三层结构

- 算法层：`src/bmpt/algorithms/def_train.py`（训练目标与 loss）。
- 执行层：`bmpt-train` + `src/bmpt/core/`（分布式、优化器、checkpoint）。
- 配置层：`src/bmpt/algorithms/config.yaml`（学习率、路径等）。

## 2. 安装

```bash
pip install -e .
```

可选依赖：

```bash
pip install -e .[deepspeed]
```

## 3. 单机训练

```bash
bmpt-train --config src/bmpt/algorithms/config.yaml --max-steps 20
```

也可以用 workspace 自动发现配置与 `def_train.py`：

```bash
bmpt-train --workspace /path/to/workspace --max-steps 20
```

## 4. 训练主路线（DeepSpeed）

```bash
bmpt-train --config src/bmpt/algorithms/config.yaml --max-steps 20
```

## 5. 分布式启动（torchrun 风格）

单机 8 卡：

```bash
bmpt-train --nproc-per-node 8 --config src/bmpt/algorithms/config.yaml
```

双机示例（每机 8 卡）：

```bash
bmpt-train --nnodes 2 --node-rank 0 --nproc-per-node 8 --master-addr <master_ip> --master-port 29500 --config src/bmpt/algorithms/config.yaml
```

```bash
bmpt-train --nnodes 2 --node-rank 1 --nproc-per-node 8 --master-addr <master_ip> --master-port 29500 --config src/bmpt/algorithms/config.yaml
```

## 6. 组件选择

默认组件（生产训练）：

- `bmpt.components.qwen_components:load_model`
- `bmpt.components.qwen_components:build_dataloader`

快速 smoke 组件：

- `bmpt.components.default_components:load_model`
- `bmpt.components.default_components:build_dataloader`

示例：

```bash
bmpt-train --config src/bmpt/algorithms/config.yaml --loader bmpt.components.qwen_components:load_model --dataloader bmpt.components.qwen_components:build_dataloader
```

## 7. Attention 实现选择

默认推荐：

```bash
bmpt-train --config src/bmpt/algorithms/config.yaml --attn-implementation auto
```

手动指定：

```bash
bmpt-train --config src/bmpt/algorithms/config.yaml --attn-implementation sdpa
```

说明：

- `auto` 会优先尝试 `flash_attention_2`。
- 若当前环境不支持 FlashAttention，程序会自动回退并打印 warning。

## 8. 断点继续训练

恢复训练由配置驱动（不是 CLI 参数）：

```yaml
train:
  load_ckpt_path: checkpoints/latest.pt
  load_ckpt_mode: full
  load_ckpt_strict: true
```

然后正常启动训练即可：

```bash
bmpt-train --config src/bmpt/algorithms/config.yaml
```

常用说明：

- `load_ckpt_mode=full`：恢复 model + optimizer + scheduler + step。
- `load_ckpt_mode=weights_only`：仅恢复 model（常用于迁移初始化）。
- `load_ckpt_path` 相对路径按 `--config` 文件目录解析。
- 多卡场景下，若 `load_ckpt_path` 对应的基础文件不存在，会自动按 rank 尝试 `*.rank_<rank>.pt`。

## 9. 示例脚本

- 引擎示例：`python example/example_engine_loop.py`
- 算法 step 示例：`python example/example_weighted_step.py`
