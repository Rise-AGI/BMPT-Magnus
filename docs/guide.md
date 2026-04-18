# BMPT 训练上手教程

这份文档面向“有算法想法，但不想深挖 Torch/DeepSpeed 细节”的使用者。

## 1. 三层结构

- 算法层：`src/bmpt/algorithms/def_train.py`（训练目标与 loss）。
- 执行层：`bmpt-train` + `src/bmpt/core/`（分布式、优化器、checkpoint）。
- 配置层：`src/bmpt/algorithms/config.yaml`（学习率、后端、路径）。

## 2. 安装

```bash
pip install -e .
```

可选依赖：

```bash
pip install -e .[torch]
pip install -e .[deepspeed]
```

## 3. 单机训练

```bash
bmpt-train --backend pytorch --config src/bmpt/algorithms/config.yaml --max-steps 20
```

也可以用 workspace 自动发现配置与 `def_train.py`：

```bash
bmpt-train --workspace /path/to/workspace --backend pytorch --max-steps 20
```

## 4. DeepSpeed 路线

```bash
bmpt-train --backend deepspeed --config src/bmpt/algorithms/config.yaml --max-steps 20
```

## 5. 分布式启动（torchrun 风格）

单机 8 卡：

```bash
bmpt-train --nproc-per-node 8 --backend pytorch --config src/bmpt/algorithms/config.yaml
```

双机示例（每机 8 卡）：

```bash
bmpt-train --nnodes 2 --node-rank 0 --nproc-per-node 8 --master-addr <master_ip> --master-port 29500 --backend deepspeed --config src/bmpt/algorithms/config.yaml
```

```bash
bmpt-train --nnodes 2 --node-rank 1 --nproc-per-node 8 --master-addr <master_ip> --master-port 29500 --backend deepspeed --config src/bmpt/algorithms/config.yaml
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

## 7. 示例脚本

- 引擎示例：`python example/example_engine_loop.py`
- 算法 step 示例：`python example/example_weighted_step.py`
