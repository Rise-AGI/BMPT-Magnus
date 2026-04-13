from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable

import torch

from src.core.distributed import (
    cleanup_distributed,
    init_distributed,
    is_main_process,
    move_to_device,
    reduce_metrics,
    wrap_models_for_ddp,
)
from src.core.engine import TrainingEngine
from src.core.optim import build_optimizer, build_scheduler
from train.def_train import build_models_from_config, load_config, step


def _load_symbol(path: str) -> Callable[..., Any]:
    module_name, symbol_name = path.split(":", maxsplit=1)
    module = import_module(module_name)
    return getattr(module, symbol_name)


def _resolve_total_steps(config: dict[str, Any], max_steps_override: int | None) -> int:
    train_cfg = config.get("train", {})
    configured_max_steps = int(train_cfg.get("max_steps", -1))
    if max_steps_override is not None:
        return max_steps_override
    if configured_max_steps > 0:
        return configured_max_steps
    return 1000


def _resolve_training_backend(config: dict[str, Any], backend_override: str | None) -> str:
    if backend_override is not None:
        return backend_override
    return str(config.get("runtime", {}).get("training_backend", "pytorch"))


def _resolve_deepspeed_config_path(config: dict[str, Any], config_path: str | Path) -> str:
    runtime_cfg = config.get("runtime", {})
    ds_path = runtime_cfg.get("deepspeed_config_path")
    if ds_path is None:
        raise ValueError("`runtime.deepspeed_config_path` is required for deepspeed backend")

    ds_path_obj = Path(ds_path)
    if not ds_path_obj.is_absolute():
        ds_path_obj = Path(config_path).parent / ds_path_obj
    resolved = ds_path_obj.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {resolved}")
    return str(resolved)


def _run_pytorch_backend(
    models: dict[str, torch.nn.Module],
    data_iterable: Any,
    config: dict[str, Any],
    config_path: str | Path,
    dist_ctx: Any,
    total_steps: int,
) -> None:
    train_cfg = config.get("train", {})
    optimizer = build_optimizer(models, config)
    scheduler = build_scheduler(optimizer, config, total_training_steps=total_steps)
    engine = TrainingEngine(
        step_fn=step,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_accum_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
    )

    log_every = int(train_cfg.get("log_every_steps", 10))
    mode = config.get("mode", "sft")
    for idx, batch in enumerate(data_iterable):
        if idx >= total_steps:
            break

        device_batch = move_to_device(batch, dist_ctx.device)
        output = engine.run_micro_step(
            models=models,
            batch=device_batch,
            extra_input={
                "mode": mode,
                "config_path": str(config_path),
            },
        )

        if (idx + 1) % log_every == 0 and is_main_process(dist_ctx):
            reduced = reduce_metrics(output["metrics"], dist_ctx)
            print(f"step={idx + 1} metrics={reduced}")


def _run_deepspeed_backend(
    models: dict[str, torch.nn.Module],
    data_iterable: Any,
    config: dict[str, Any],
    config_path: str | Path,
    dist_ctx: Any,
    total_steps: int,
) -> None:
    try:
        deepspeed = import_module("deepspeed")
    except Exception as exc:
        raise ImportError("deepspeed backend requires `deepspeed` package") from exc

    train_cfg = config.get("train", {})
    ds_config_path = _resolve_deepspeed_config_path(config, config_path)

    policy_model = models["policy"].to(dist_ctx.device)
    trainable_parameters = [param for param in policy_model.parameters() if param.requires_grad]
    ds_engine, _, _, _ = deepspeed.initialize(
        model=policy_model,
        model_parameters=trainable_parameters,
        config=ds_config_path,
    )

    models["policy"] = ds_engine
    for name, model in models.items():
        if name == "policy":
            continue
        models[name] = model.to(dist_ctx.device)

    log_every = int(train_cfg.get("log_every_steps", 10))
    mode = config.get("mode", "sft")
    for idx, batch in enumerate(data_iterable):
        if idx >= total_steps:
            break

        device_batch = move_to_device(batch, dist_ctx.device)
        output = step(
            models,
            {
                "batch": device_batch,
                "global_step": idx,
                "mode": mode,
                "config_path": str(config_path),
            },
        )
        loss = output["loss"]
        ds_engine.backward(loss)
        ds_engine.step()

        metrics = dict(output.get("metrics", {}))
        metrics["engine/global_step"] = idx + 1
        if (idx + 1) % log_every == 0 and is_main_process(dist_ctx):
            reduced = reduce_metrics(metrics, dist_ctx)
            print(f"step={idx + 1} metrics={reduced}")


def run_train(
    config_path: str | Path,
    loader: str,
    dataloader: str,
    max_steps_override: int | None = None,
    backend_override: str | None = None,
) -> None:
    config = load_config(config_path)
    backend = config.get("runtime", {}).get("distributed_backend", "nccl")
    dist_ctx = init_distributed(backend=backend)

    try:
        loader_fn = _load_symbol(loader)
        dataloader_fn = _load_symbol(dataloader)

        models = build_models_from_config(config, loader_fn=loader_fn)
        training_backend = _resolve_training_backend(config, backend_override)
        if training_backend == "pytorch":
            models = wrap_models_for_ddp(models, dist_ctx)

        data_iterable = dataloader_fn(config, dist_ctx)
        total_steps = _resolve_total_steps(config, max_steps_override)

        if training_backend == "deepspeed":
            _run_deepspeed_backend(
                models=models,
                data_iterable=data_iterable,
                config=config,
                config_path=config_path,
                dist_ctx=dist_ctx,
                total_steps=total_steps,
            )
        else:
            _run_pytorch_backend(
                models=models,
                data_iterable=data_iterable,
                config=config,
                config_path=config_path,
                dist_ctx=dist_ctx,
                total_steps=total_steps,
            )
    finally:
        cleanup_distributed()
