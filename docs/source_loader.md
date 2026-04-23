# `bmpt.data.source_loader` 函数级文档

本文档描述数据源加载模块的类与函数职责。

## `JsonlSourceDataset`

- 类型：`Dataset[dict[str, Any]]`
- 作用：原始 JSONL 数据源的数据集封装，不做预先 tokenize。
- 构造函数：
  - `path: str | Path`：JSONL 文件路径
- 属性：
  - `path: Path`：解析后的绝对路径
  - `records: list[dict[str, Any]]`：加载的原始记录列表

## `_load_jsonl(path)`

- 作用：加载 JSONL 文件并返回记录列表。
- 输入：`path: Path`
- 输出：`list[dict[str, Any]]`
- 异常：
  - `ValueError`：某行不是有效的对象映射

## `_collate_batch(batch)`

- 作用：自定义 collate 函数，将 batch 内样本对齐为张量。
- 输入：`batch: list[dict[str, Any]]`
- 输出：`dict[str, Any]`
- 规则：
  - 如果某字段所有值都是同形状 Tensor，则 `torch.stack`
  - 否则保持为列表

## `_resolve_world_size_and_rank()`

- 作用：从环境变量解析分布式信息。
- 输出：`(world_size, rank)`
- 环境变量：
  - `WORLD_SIZE`：默认 1
  - `RANK`：默认 0

## `build_single_source_dataloader(source_cfg, config, *, shuffle=True)`

- 作用：为单个数据源构建 DataLoader。
- 输入：
  - `source_cfg: dict[str, Any]`：单个数据源配置（含 `path`, `name`, `shuffle`）
  - `config: dict[str, Any]`：完整训练配置
  - `shuffle: bool`：是否 shuffle（仅非分布式时有效）
- 输出：`DataLoader`
- 分布式行为：
  - `WORLD_SIZE > 1` 时自动使用 `DistributedSampler`
  - 单卡时根据 `shuffle` 参数决定
- 配置读取：
  - `train.per_device_batch_size`
  - `data.loader.num_workers`
  - `data.loader.pin_memory`
  - `data.loader.prefetch_factor`
  - `data.loader.persistent_workers`

## `build_source_dataloaders(config)`

- 作用：为 `config.data.sources` 中每个源构建独立的 DataLoader。
- 输入：`config: dict[str, Any]`
- 输出：`dict[str, DataLoader]`，key 为 source `name`（或 path）
- 约束：
  - 每个 source 必须有 `path`
  - source 名称必须唯一
- 异常：
  - `ValueError`：sources 为空或格式错误
  - `ValueError`：重复的 source name
