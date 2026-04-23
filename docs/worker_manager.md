# `bmpt.distributed.worker_manager` 函数级文档

本文档描述分布式 worker 管理模块的函数职责。

## `_get_visible_devices()`

- 作用：获取当前可见的 GPU 设备列表。
- 输出：`list[int]`
- 规则：
  - 优先读取 `CUDA_VISIBLE_DEVICES`
  - 未设置时返回 `[0, 1, ..., n-1]`（n 为可用 GPU 数）

## `_training_worker_entry(rank, world_size, local_device, config, def_worker, worker_args, worker_kwargs)`

- 作用：训练 worker 进程入口函数。
- 输入：
  - `rank: int`：全局 rank
  - `world_size: int`：总进程数
  - `local_device: int`：物理 GPU 设备号
  - `config: Any`：配置（支持共享内存 proxy）
  - `def_worker: Callable`：用户定义的 worker 函数
  - `worker_args: tuple | None`：额外位置参数
  - `worker_kwargs: dict | None`：额外关键字参数
- 环境设置：
  - `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `CUDA_VISIBLE_DEVICES`
- 调用：
  - `def_worker(rank, config_dict, *worker_args, **worker_kwargs)`

## `_preserved_worker_entry(device_id, config, preserved_func, worker_args, worker_kwargs)`

- 作用：保留 worker 进程入口函数。
- 输入：
  - `device_id: int`：物理 GPU 设备号
  - `config: Any`：配置
  - `preserved_func: Callable`：保留 worker 函数
  - `worker_args`, `worker_kwargs`：额外参数
- 环境设置：
  - `CUDA_VISIBLE_DEVICES` 设为 `device_id`
- 调用：
  - `preserved_func(config_dict, *worker_args, **worker_kwargs)`

## `spawn_worker_processes(def_worker, config, preserved_worker=None, worker_args=None, worker_kwargs=None)`

- 作用：启动训练 worker 和可选的保留 worker 进程。
- 输入：
  - `def_worker: Callable[..., Any]`：训练 worker 函数
  - `config: Any`：配置对象（支持 Manager proxy）
  - `preserved_worker: tuple | list | None`：保留 worker 函数列表
  - `worker_args: tuple | None`：传给 worker 的额外位置参数
  - `worker_kwargs: dict | None`：传给 worker 的额外关键字参数
- 输出：`list[mp.Process]` 启动的进程列表
- GPU 分配规则：
  - 假设有 N 张可见 GPU
  - 保留 K 个 preserved_worker（K = len(preserved_worker)）
  - 训练 worker 使用前 `N-K` 张 GPU（0 到 N-K-1）
  - preserved_worker 从后往前分配：
    - preserved_worker[0] -> GPU N-1
    - preserved_worker[1] -> GPU N-2
    - ...以此类推
- 异常：
  - `RuntimeError`：无可用的 CUDA 设备
  - `ValueError`：preserved_worker 数量 >= 可见 GPU 数
