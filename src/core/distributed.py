from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(slots=True)
class DistributedContext:
    is_distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def init_distributed(backend: str = "nccl") -> DistributedContext:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return DistributedContext(
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(ctx: DistributedContext) -> bool:
    return ctx.rank == 0


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def wrap_models_for_ddp(
    models: dict[str, torch.nn.Module],
    ctx: DistributedContext,
    find_unused_parameters: bool = False,
) -> dict[str, torch.nn.Module]:
    wrapped: dict[str, torch.nn.Module] = {}
    for name, model in models.items():
        model = model.to(ctx.device)
        has_trainable = any(param.requires_grad for param in model.parameters())
        if ctx.is_distributed and has_trainable:
            if ctx.device.type == "cuda":
                wrapped[name] = DistributedDataParallel(
                    model,
                    device_ids=[ctx.local_rank],
                    output_device=ctx.local_rank,
                    find_unused_parameters=find_unused_parameters,
                )
            else:
                wrapped[name] = DistributedDataParallel(
                    model,
                    find_unused_parameters=find_unused_parameters,
                )
        else:
            wrapped[name] = model
    return wrapped


def reduce_metrics(metrics: dict[str, float], ctx: DistributedContext) -> dict[str, float]:
    if not ctx.is_distributed:
        return metrics

    reduced: dict[str, float] = {}
    for key, value in metrics.items():
        tensor = torch.tensor(float(value), device=ctx.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[key] = float((tensor / float(ctx.world_size)).item())
    return reduced
