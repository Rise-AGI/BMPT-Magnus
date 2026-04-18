# BMPT AI 快速上手

本文档面向需要快速、安全使用 BMPT 的 AI Agent。

## 1. 基础事实

- 主命令：`bmpt-train`
- 默认算法模块：`bmpt.algorithms.def_train`
- 默认 `step` 行为：SFT
- 默认配置路径：`src/bmpt/algorithms/config.yaml`

## 2. 快速启动

安装开发模式包：

```bash
pip install -e .
```

单进程运行：

```bash
bmpt-train --config src/bmpt/algorithms/config.yaml --backend pytorch --max-steps 20
```

单机分布式：

```bash
bmpt-train --nproc-per-node 8 --config src/bmpt/algorithms/config.yaml --backend pytorch
```

## 3. `--workspace` 自动发现

`--workspace <path>` 仅自动发现：

- `def_train.py`
- `config.json` / `config.yaml` / `config.yml`

示例：

```bash
bmpt-train --workspace /path/to/workspace --backend pytorch
```

## 4. 参数优先级

### `--config`

1. 手动指定 `--config`
2. `--workspace` 自动发现 config
3. 内置默认配置

### `--def-train`

1. 手动指定 `--def-train`
2. `--workspace` 自动发现 `def_train.py`
3. 内置默认模块 `bmpt.algorithms.def_train`

`--def-train` 同时支持：

- Python 模块路径
- Python 文件路径

### DeepSpeed 配置来源

DeepSpeed 配置路径只来自：

- 当前训练配置内的 `runtime.deepspeed_config_path`

不使用 `--deepspeed-config` 这类 CLI 选项。

### Attention 配置来源

优先级从高到低：

1. CLI `--attn-implementation`
2. `runtime.attn_implementation`
3. `runtime.flash_attention=true`（兼容字段）
4. 默认 `auto`

推荐使用 `auto`，并允许在不支持 FlashAttention 时自动回退。

## 5. 继续阅读

- 算法接口：`agents/step_contract.md`
- 配置结构：`agents/config.schema.json`
