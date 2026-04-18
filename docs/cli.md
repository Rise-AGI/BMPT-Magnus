# BMPT CLI 说明

本文档说明 `bmpt-train` 的参数、优先级与 `--workspace` 自动发现规则。

## 1. 基本用法

```bash
bmpt-train --config src/bmpt/algorithms/config.yaml
```

查看帮助：

```bash
bmpt-train --help
```

## 2. 分布式启动（torchrun 风格）

单机多卡：

```bash
bmpt-train --nproc-per-node 8 --config src/bmpt/algorithms/config.yaml
```

多机：

```bash
bmpt-train --nnodes 2 --node-rank 0 --nproc-per-node 8 --master-addr <master_ip> --master-port 29500 --config src/bmpt/algorithms/config.yaml
```

## 3. `--workspace` 自动发现

`--workspace <path>` 会在目录下自动发现：

- `def_train.py`
- `config.json` / `config.yaml` / `config.yml`

注意：`--workspace` 不会搜索 `deepspeed*.json`。

## 4. 参数优先级

### `--config`

1. 手动指定 `--config`
2. `--workspace` 自动发现 config
3. 默认 `src/bmpt/algorithms/config.yaml`

### `--def-train`

1. 手动指定 `--def-train`
2. `--workspace` 自动发现 `def_train.py`
3. 默认 `bmpt.algorithms.def_train`

`--def-train` 支持两种形式：

- Python 模块路径（如 `bmpt.algorithms.def_train`）
- 文件路径（如 `/path/to/def_train.py`）

### DeepSpeed 配置

`runtime.deepspeed_config_path` 仍是最高优先级来源。

当使用 deepspeed 后端时，程序只读取当前训练配置里的：

- `runtime.deepspeed_config_path`

并按“相对 `--config` 文件目录”解析相对路径。

### `attn_implementation`

优先级从高到低：

1. CLI `--attn-implementation`
2. `runtime.attn_implementation`
3. 兼容字段 `runtime.flash_attention=true`（等价于尝试 `flash_attention_2`）
4. 默认 `auto`

常见取值：

- `auto`
- `flash_attention_2`
- `sdpa`
- `eager`

当请求 `flash_attention_2` 但环境不支持时，程序会自动回退到默认 attention，并打印 warning。

## 5. 常见示例

使用 workspace 自动发现：

```bash
bmpt-train --workspace /path/to/workspace
```

手动覆盖 config，保留 workspace 的 def_train 自动发现：

```bash
bmpt-train --workspace /path/to/workspace --config /path/to/another/config.yaml
```

手动覆盖 def_train（文件路径）：

```bash
bmpt-train --config src/bmpt/algorithms/config.yaml --def-train /path/to/def_train.py
```

显式指定 attention 实现：

```bash
bmpt-train --config src/bmpt/algorithms/config.yaml --attn-implementation sdpa
```
