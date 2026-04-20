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

### `--loader`

模型加载函数，默认 `bmpt.model.loader:load_model`。

支持自定义模型加载逻辑，只需提供模块路径和函数名。

### DeepSpeed 配置

`runtime.deepspeed_config_path` 仍是最高优先级来源。

当使用 deepspeed 后端时，程序只读取当前训练配置里的：

- `runtime.deepspeed_config_path`

并按"相对 `--config` 文件目录"解析相对路径。

### Checkpoint 恢复（配置驱动）

`bmpt-train` 不提供额外恢复 CLI 参数，恢复行为完全由 `train` 配置控制：

- `train.load_ckpt_path`
- `train.load_ckpt_mode`（`full` / `weights_only`）
- `train.load_ckpt_strict`

其中 `load_ckpt_path` 相对路径按 `--config` 文件目录解析。
多卡恢复时，若基础文件不存在，会按当前 rank 自动尝试 `*.rank_<rank>.pt`。

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

## 5. 数据配置

### `data.sources`

定义数据源列表，每个数据源包含以下字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | 是 | JSONL 文件路径 |
| `required_keys` | list[string] | 是 | 必须存在的字段，缺失则报错 |
| `tokenize_keys` | list[string] | 是 | 需要进行 tokenize 的字段 |
| `name` | string | 否 | 数据源名称，用于区分 train/val |

**`name` 字段约定**：

- `name: train` — **必需**，训练数据源
- `name: val` — **可选**，验证数据源（训练前后自动调用 evaluate）

示例：

```yaml
data:
  sources:
    - path: data/train.jsonl
      required_keys: [prompt, response]
      tokenize_keys: [prompt, response]
      name: train
    - path: data/val.jsonl
      required_keys: [prompt, response]
      tokenize_keys: [prompt, response]
      name: val
  max_seq_len: 4096
  cache_dir: null
```

### 预处理流程

训练启动时，会在 tokenizer 加载后自动执行：

1. 加载 JSONL 文件
2. 验证 `required_keys` 是否存在
3. 对 `tokenize_keys` 指定的字段进行 tokenize
4. 生成 `{key}_input_ids` 字段（如 `prompt_input_ids`、`response_input_ids`）
5. 保存缓存文件

### 缓存机制

预处理后的数据会自动缓存到：

- 默认：`{source_dir}/{filename}.tokenized.jsonl`
- 自定义：`cache_dir/{filename}.tokenized.jsonl`

缓存文件附带 `.meta.json` 元数据，包含哈希值。当以下条件变化时，自动重新处理：

- 源文件内容或大小变化
- `required_keys` 变化
- `tokenize_keys` 变化
- `max_seq_len` 变化
- tokenizer vocab 变化

### max_seq_len 截断规则

`data.max_seq_len` 用于截断 tokenize 后的序列长度。**注意：截断长度是共用的**，所有 `tokenize_keys` 字段共享同一个 `max_seq_len` 值。

若需要更精细的截断控制，请使用 `config.prompting.composers`。

### labels 生成规则

默认 `step` 函数的 labels 生成规则：

- `tokenize_keys` 中第一个字段对应的 input_ids：labels = -100（不计算 loss）
- 其余字段对应的 input_ids：labels = 原 token id（计算 loss）

例如 `tokenize_keys: [prompt, response]`：
- `prompt_input_ids` 对应的 labels = -100
- `response_input_ids` 对应的 labels = 原 token id

## 6. 常见示例

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