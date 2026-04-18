# BMPT 任务配方

本文档提供面向 AI Agent 的标准操作配方。

## 配方 1：快速跑默认训练

命令：

```bash
bmpt-train --config src/bmpt/algorithms/config.yaml --backend pytorch --max-steps 20
```

预期信号：

- 周期性打印 `step=... metrics=...`

## 配方 2：使用 workspace 自动发现

命令：

```bash
bmpt-train --workspace /path/to/workspace --backend pytorch --max-steps 20
```

预期信号：

- 启动阶段不出现 config / def_train 导入错误

注意：

- 同时传 `--workspace` 与 `--config` 时，以 `--config` 为准
- 同时传 `--workspace` 与 `--def-train` 时，以 `--def-train` 为准

## 配方 3：用文件路径覆盖算法模块

命令：

```bash
bmpt-train --config /path/to/config.yaml --def-train /path/to/def_train.py --backend pytorch
```

预期信号：

- 不出现 `load_config/build_models_from_config/step` 缺失错误

## 配方 4：单机分布式

命令：

```bash
bmpt-train --nproc-per-node 8 --config src/bmpt/algorithms/config.yaml --backend pytorch
```

预期信号：

- 多个 worker 拉起，rank-0 持续输出训练日志

## 配方 5：多机分布式

节点 0：

```bash
bmpt-train --nnodes 2 --node-rank 0 --nproc-per-node 8 --master-addr <master_ip> --master-port 29500 --config src/bmpt/algorithms/config.yaml --backend deepspeed
```

节点 1：

```bash
bmpt-train --nnodes 2 --node-rank 1 --nproc-per-node 8 --master-addr <master_ip> --master-port 29500 --config src/bmpt/algorithms/config.yaml --backend deepspeed
```

预期信号：

- 进程组初始化成功并进入训练循环

## 配方 6：DeepSpeed 配置路径规则

规则：

- 在训练配置里设置 `runtime.deepspeed_config_path`
- 不依赖 CLI 的 deepspeed 配置路径参数

预期信号：

- DeepSpeed JSON 按“相对 `--config` 所在目录”正确解析

## 配方 7：自定义 attention 实现

命令：

```bash
bmpt-train --config src/bmpt/algorithms/config.yaml --attn-implementation auto
```

```bash
bmpt-train --config src/bmpt/algorithms/config.yaml --attn-implementation sdpa
```

预期信号：

- 启动日志打印 `requested_attn=... actual_attn=...`
- 当 FlashAttention 不可用时，出现 warning 且继续训练

## 配方 8：使用 prompting composer

配置：

- 在 `config.prompting.composers` 注册多个 composer。

预期信号：

- 启动日志打印 `loaded_composers=[...]`
- `step` 内可从 `input["composers"]` 读取并调用 `compose(...)`
