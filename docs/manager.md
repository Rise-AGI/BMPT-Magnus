# `bmpt.manager.manager` 函数级文档

本文档描述 `Manager` 的职责与方法语义。

## `Manager.__init__()`

- 作用：初始化管理器状态。
- 初始化字段：
  - `config_path: Path | None`
  - `deepspeed_config_path: Path | None`
  - `config: dict[str, Any]`
  - `deepspeed_config: dict[str, Any]`
  - `_shared_manager: mp.managers.SyncManager | None`：内部共享内存管理器
  - `_shared_config: Any | None`：共享配置 proxy

## `Manager.load_config(config_path)`

- 作用：加载并归一化训练配置。
- 输入：`config_path: str | Path`
- 输出：`dict[str, Any]`
- 内部行为：
  - 调用 `bmpt.manager.config_manager.load_config_bundle`
  - 将配置结果写入 Manager 成员
  - 返回 `self.config`
- 说明：
  - 返回的 `self.config` 已移除顶层 `optimizer/scheduler`
  - 对应参数已注入到 `self.deepspeed_config`

## `Manager.build_source_dataloaders()`

- 作用：根据 `config.data.sources` 为每个源构建 DataLoader。
- 输出：`dict[str, DataLoader]`，key 为 source 名称
- 约束：
  - 必须先调用 `Manager.load_config(...)`
  - 不预先 tokenize，返回原始文本 DataLoader
- 异常：`ValueError`（config 为空）

## `Manager.load_composers()`

- 作用：加载配置中定义的所有提示词 composer。
- 输出：`dict[str, Composer]`
- 约束：必须先调用 `Manager.load_config(...)`
- 说明：若未配置 `prompting.composers`，返回空字典
- 异常：`ValueError`（config 为空）

## `Manager.spawn_worker(def_worker, preserved_worker=None, worker_args=None, worker_kwargs=None)`

- 作用：启动训练 worker 和可选的保留 worker 进程。
- 输入：
  - `def_worker: Callable`：训练 worker 函数，签名为 `(rank, config, *args, **kwargs)`
  - `preserved_worker: tuple | list | None`：保留 worker 函数列表
  - `worker_args: tuple | None`：传给 worker 的额外位置参数
  - `worker_kwargs: dict | None`：传给 worker 的额外关键字参数
- 输出：`list[mp.Process]` 启动的进程列表
- GPU 分配：
  - 假设 N 张可见 GPU，K 个 preserved_worker
  - 训练 worker 使用前 `N-K` 张 GPU
  - preserved_worker[0] 绑定到 GPU N-1
  - preserved_worker[1] 绑定到 GPU N-2，以此类推
- 约束：必须先调用 `Manager.load_config(...)`
- 配置共享：
  - 使用 `multiprocessing.Manager()` 创建共享内存
  - config 以 proxy 形式传给子进程
- 异常：`ValueError`（config 为空）
