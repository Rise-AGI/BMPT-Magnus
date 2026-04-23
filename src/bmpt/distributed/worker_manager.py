from __future__ import annotations

import os
import sys
import multiprocessing as mp
from typing import Any, Callable

import torch


def _get_visible_devices() -> list[int]:
    """获取可见的 GPU 设备列表。"""
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible is None or cuda_visible.strip() == "":
        count = torch.cuda.device_count()
        return list(range(count))
    devices = [int(x.strip()) for x in cuda_visible.split(",") if x.strip().isdigit()]
    if not devices:
        count = torch.cuda.device_count()
        return list(range(count))
    return devices


def _training_worker_entry(
    rank: int,
    world_size: int,
    local_device: int,
    config: Any,
    def_worker: Callable[..., Any],
    worker_args: tuple[Any, ...] | None,
    worker_kwargs: dict[str, Any] | None,
) -> None:
    """训练 worker 的入口函数。"""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_device)

    # 初始化该进程的 CUDA 设备
    torch.cuda.set_device(0)  # 因为设置了 CUDA_VISIBLE_DEVICES，本地就是 0

    args = worker_args if worker_args is not None else ()
    kwargs = worker_kwargs if worker_kwargs is not None else {}

    # 转换共享配置为普通 dict（如果是 Manager proxy 对象）
    config_dict = dict(config) if hasattr(config, "keys") else config

    def_worker(rank, config_dict, *args, **kwargs)


def _preserved_worker_entry(
    device_id: int,
    config: Any,
    preserved_func: Callable[..., Any],
    worker_args: tuple[Any, ...] | None,
    worker_kwargs: dict[str, Any] | None,
) -> None:
    """保留 worker 的入口函数。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(0)

    args = worker_args if worker_args is not None else ()
    kwargs = worker_kwargs if worker_kwargs is not None else {}

    config_dict = dict(config) if hasattr(config, "keys") else config

    preserved_func(config_dict, *args, **kwargs)


def spawn_worker_processes(
    def_worker: Callable[..., Any],
    config: Any,
    preserved_worker: tuple[Any, ...] | list[Any] | None = None,
    worker_args: tuple[Any, ...] | None = None,
    worker_kwargs: dict[str, Any] | None = None,
) -> list[mp.Process]:
    """启动训练 worker 和保留 worker 进程。

    Args:
        def_worker: 训练 worker 函数，签名应为 (rank, config, *args, **kwargs)
        config: 配置对象（支持 Manager proxy）
        preserved_worker: 保留 worker 函数列表，每个函数签名为 (config, *args, **kwargs)
        worker_args: 传给 worker 的额外位置参数
        worker_kwargs: 传给 worker 的额外关键字参数

    Returns:
        启动的进程列表
    """
    visible_devices = _get_visible_devices()
    if not visible_devices:
        raise RuntimeError("No visible CUDA devices found")

    num_preserved = len(preserved_worker) if preserved_worker else 0
    if num_preserved >= len(visible_devices):
        raise ValueError(
            f"Cannot preserve {num_preserved} workers with only {len(visible_devices)} visible GPUs"
        )

    # 分配设备：前面的给训练 worker，后面的从后往前分配给 preserved_worker
    training_devices = visible_devices[:-num_preserved] if num_preserved > 0 else visible_devices
    preserved_devices = list(reversed(visible_devices[-num_preserved:])) if num_preserved > 0 else []

    world_size = len(training_devices)
    processes: list[mp.Process] = []

    # 启动训练 worker
    for rank, device_id in enumerate(training_devices):
        p = mp.Process(
            target=_training_worker_entry,
            args=(
                rank,
                world_size,
                device_id,
                config,
                def_worker,
                worker_args,
                worker_kwargs,
            ),
        )
        p.start()
        processes.append(p)

    # 启动保留 worker
    if preserved_worker:
        for idx, preserved_func in enumerate(preserved_worker):
            device_id = preserved_devices[idx]
            p = mp.Process(
                target=_preserved_worker_entry,
                args=(
                    device_id,
                    config,
                    preserved_func,
                    worker_args,
                    worker_kwargs,
                ),
            )
            p.start()
            processes.append(p)

    return processes
