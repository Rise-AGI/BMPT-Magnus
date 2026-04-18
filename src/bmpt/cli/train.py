from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, cast

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from bmpt.core.distributed import (
    cleanup_distributed,
    init_distributed,
    is_main_process,
    move_to_device,
    reduce_metrics,
    wrap_models_for_ddp,
)
from bmpt.core.engine import TrainingEngine
from bmpt.core.optim import build_optimizer, build_scheduler


def _default_config_path() -> str:
    return str(Path(__file__).resolve().parent.parent / "algorithms" / "config.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BMPT training entrypoint")
    parser.add_argument(
        "--config",
        default=_default_config_path(),
        help="Path to training config file",
    )
    parser.add_argument(
        "--loader",
        default="bmpt.components.qwen_components:load_model",
        help="Model loader symbol path module:function",
    )
    parser.add_argument(
        "--dataloader",
        default="bmpt.components.qwen_components:build_dataloader",
        help="Dataloader builder symbol path module:function",
    )
    parser.add_argument(
        "--def-train",
        default="bmpt.algorithms.def_train",
        help="Training definition module path, e.g. bmpt.algorithms.def_train",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max training steps override",
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "deepspeed"],
        default=None,
        help="Optional training backend override",
    )
    parser.add_argument(
        "--save-final",
        action="store_true",
        help="Save final checkpoint with tag `latest`",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logs",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="Launch distributed workers per node (torchrun style)",
    )
    parser.add_argument(
        "--nnodes",
        default=None,
        help="Number of nodes (supports torchrun syntax, e.g. 2 or 1:4)",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=None,
        help="Rank of current node in multi-node setup",
    )
    parser.add_argument(
        "--master-addr",
        default=None,
        help="Master node address for rendezvous",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="Master node port for rendezvous",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _is_debug_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("runtime", {}).get("debug", False))


def _is_rank0() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _debug_print(config: dict[str, Any], message: str) -> None:
    if _is_debug_enabled(config) and _is_rank0():
        print(message, flush=True)


def _load_symbol(path: str) -> Callable[..., Any]:
    module_name, symbol_name = path.split(":", maxsplit=1)
    module = import_module(module_name)
    return getattr(module, symbol_name)


def _load_def_train_functions(def_train_module: str) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    module = import_module(def_train_module)
    load_config_fn = getattr(module, "load_config", None)
    build_models_from_config_fn = getattr(module, "build_models_from_config", None)
    step_fn = getattr(module, "step", None)

    if not callable(load_config_fn):
        raise AttributeError(f"`{def_train_module}` must expose callable `load_config`")
    if not callable(build_models_from_config_fn):
        raise AttributeError(f"`{def_train_module}` must expose callable `build_models_from_config`")
    if not callable(step_fn):
        raise AttributeError(f"`{def_train_module}` must expose callable `step`")
    return load_config_fn, build_models_from_config_fn, step_fn


def _resolve_total_steps(config: dict[str, Any], max_steps_override: int | None) -> int:
    train_cfg = config.get("train", {})
    configured_max_steps = int(train_cfg.get("max_steps", -1))
    if max_steps_override is not None:
        return max_steps_override
    if configured_max_steps > 0:
        return configured_max_steps
    return 1000


def _resolve_train_control_mode(config: dict[str, Any]) -> str:
    train_cfg = config.get("train", {})
    mode = str(train_cfg.get("control_mode", "step")).lower()
    if mode not in {"step", "epoch"}:
        raise ValueError(f"Unsupported train.control_mode: {mode}")
    return mode


def _resolve_total_steps_by_mode(
    config: dict[str, Any],
    max_steps_override: int | None,
    data_iterable: Any,
) -> int:
    mode = _resolve_train_control_mode(config)
    if mode == "step":
        return _resolve_total_steps(config, max_steps_override)

    train_cfg = config.get("train", {})
    epochs = int(train_cfg.get("epochs", 1))
    if epochs <= 0:
        raise ValueError("`train.epochs` must be > 0 when train.control_mode is `epoch`")
    if not hasattr(data_iterable, "__len__"):
        raise ValueError("`train.control_mode=epoch` requires dataloader with __len__")
    return int(len(data_iterable)) * epochs


def _iter_training_batches(
    data_iterable: Any,
    config: dict[str, Any],
    total_steps: int,
):
    mode = _resolve_train_control_mode(config)
    if mode == "step":
        for step_idx, batch in enumerate(data_iterable):
            if step_idx >= total_steps:
                break
            yield step_idx, batch
        return

    train_cfg = config.get("train", {})
    epochs = int(train_cfg.get("epochs", 1))
    global_step = 0
    for epoch_idx in range(epochs):
        sampler = getattr(data_iterable, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch_idx)
        for batch in data_iterable:
            if global_step >= total_steps:
                return
            yield global_step, batch
            global_step += 1


def _resolve_training_backend(config: dict[str, Any], backend_override: str | None) -> str:
    if backend_override is not None:
        return backend_override
    return str(config.get("runtime", {}).get("training_backend", "pytorch"))


def _checkpoint_settings(config: dict[str, Any]) -> tuple[int, Path]:
    train_cfg = config.get("train", {})
    every = int(train_cfg.get("checkpoint_every_steps", 0))
    checkpoint_dir = Path(str(train_cfg.get("checkpoint_dir", "checkpoints")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return every, checkpoint_dir


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DistributedDataParallel):
        return cast(torch.nn.Module, model.module)
    return model


def _save_pytorch_checkpoint(
    models: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.LambdaLR | None,
    checkpoint_dir: Path,
    step_id: int,
) -> Path:
    save_path = checkpoint_dir / f"step_{step_id}.pt"
    state = {
        "global_step": step_id,
        "models": {name: _unwrap_model(model).state_dict() for name, model in models.items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(state, save_path)
    return save_path


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


def _load_deepspeed_config(config: dict[str, Any], config_path: str | Path, total_steps: int) -> dict[str, Any]:
    ds_path = _resolve_deepspeed_config_path(config, config_path)
    with Path(ds_path).open("r", encoding="utf-8") as handle:
        ds_config = json.load(handle)

    train_cfg = config.get("train", {})
    optimizer_cfg = config.get("optimizer", {})
    scheduler_cfg = config.get("scheduler", {})

    ds_config["train_micro_batch_size_per_gpu"] = int(train_cfg.get("per_device_batch_size", 1))
    ds_config["gradient_accumulation_steps"] = int(train_cfg.get("gradient_accumulation_steps", 1))
    ds_config["gradient_clipping"] = float(train_cfg.get("grad_clip_norm", 1.0))

    optimizer_type = str(optimizer_cfg.get("type", "adamw")).lower()
    if optimizer_type != "adamw":
        raise ValueError(f"Unsupported optimizer.type for deepspeed backend: {optimizer_type}")
    ds_config["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": float(optimizer_cfg.get("lr", 2.0e-5)),
            "weight_decay": float(optimizer_cfg.get("weight_decay", 0.0)),
            "betas": list(optimizer_cfg.get("betas", [0.9, 0.999])),
            "eps": float(optimizer_cfg.get("eps", 1.0e-8)),
        },
    }

    mixed_precision = str(train_cfg.get("mixed_precision", "bf16")).lower()
    if mixed_precision == "bf16":
        ds_config["bf16"] = {"enabled": True}
        ds_config["fp16"] = {"enabled": False}
    elif mixed_precision == "fp16":
        ds_config["bf16"] = {"enabled": False}
        ds_config["fp16"] = {"enabled": True}
    elif mixed_precision == "no":
        ds_config["bf16"] = {"enabled": False}
        ds_config["fp16"] = {"enabled": False}
    else:
        raise ValueError(f"Unsupported train.mixed_precision for deepspeed backend: {mixed_precision}")

    scheduler_type = str(scheduler_cfg.get("type", "none")).lower()
    if scheduler_type == "cosine":
        warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
        min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.0))
        total_num_steps = int(total_steps)
        if total_num_steps <= warmup_steps:
            total_num_steps = warmup_steps + 1
        ds_config["scheduler"] = {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_num_steps": warmup_steps,
                "total_num_steps": total_num_steps,
                "warmup_min_ratio": 0.0,
                "cos_min_ratio": min_lr_ratio,
            },
        }
    elif scheduler_type == "none":
        ds_config.pop("scheduler", None)
    else:
        raise ValueError(f"Unsupported scheduler.type for deepspeed backend: {scheduler_type}")

    return ds_config


def _run_validation(
    models: dict[str, torch.nn.Module],
    val_iterable: Any,
    config: dict[str, Any],
    config_path: str | Path,
    dist_ctx: Any,
    step_fn: Callable[..., Any],
    phase: str,
) -> None:
    _debug_print(config, f"DEBUG: _run_validation enter, phase={phase}")

    static_step_input = {
        "config_path": str(config_path),
        "_cached_config": config,
        "_merged_config": config,
    }

    total_loss = 0.0
    total_samples = 0
    metrics_sum: dict[str, float] = {}

    with torch.no_grad():
        for batch in val_iterable:
            device_batch = move_to_device(batch, dist_ctx.device)
            payload = {"batch": device_batch, "global_step": 0}
            payload.update(static_step_input)
            output = step_fn(models, payload)

            loss = float(output.get("loss", 0.0))
            total_loss += loss
            total_samples += 1

            for key, value in output.get("metrics", {}).items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value)

    if is_main_process(dist_ctx):
        avg_loss = total_loss / max(total_samples, 1)
        avg_metrics = {k: v / total_samples for k, v in metrics_sum.items()}
        print(f"[{phase}] val_loss={avg_loss:.4f} val_metrics={avg_metrics}")


def _run_pytorch_backend(
    models: dict[str, torch.nn.Module],
    data_iterable: Any,
    config: dict[str, Any],
    config_path: str | Path,
    dist_ctx: Any,
    total_steps: int,
    save_final: bool,
    step_fn: Callable[..., Any],
) -> None:
    train_cfg = config.get("train", {})
    optimizer = build_optimizer(models, config)
    scheduler = build_scheduler(optimizer, config, total_training_steps=total_steps)
    engine = TrainingEngine(
        step_fn=step_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_accum_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
    )

    log_every = int(train_cfg.get("log_every_steps", 10))
    checkpoint_every, checkpoint_dir = _checkpoint_settings(config)
    last_step = 0
    static_step_input = {
        "config_path": str(config_path),
        "_cached_config": config,
        "_merged_config": config,
        "_debug": _is_debug_enabled(config),
    }
    for idx, batch in _iter_training_batches(data_iterable, config, total_steps):
        last_step = idx + 1

        device_batch = move_to_device(batch, dist_ctx.device)
        output = engine.run_micro_step(
            models=models,
            batch=device_batch,
            extra_input=static_step_input,
        )

        if (idx + 1) % log_every == 0:
            reduced = reduce_metrics(output["metrics"], dist_ctx)
            if is_main_process(dist_ctx):
                print(f"step={idx + 1} metrics={reduced}")

        if checkpoint_every > 0 and (idx + 1) % checkpoint_every == 0 and is_main_process(dist_ctx):
            save_path = _save_pytorch_checkpoint(
                models=models,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_dir=checkpoint_dir,
                step_id=idx + 1,
            )
            print(f"checkpoint_saved={save_path}")

    if save_final and is_main_process(dist_ctx):
        final_step = max(last_step, 1)
        save_path = checkpoint_dir / "latest.pt"
        state = {
            "global_step": final_step,
            "models": {name: _unwrap_model(model).state_dict() for name, model in models.items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }
        torch.save(state, save_path)
        print(f"checkpoint_saved={save_path}")


def _run_deepspeed_backend(
    models: dict[str, torch.nn.Module],
    data_iterable: Any,
    before_train_val_iterable: Any | None,
    config: dict[str, Any],
    config_path: str | Path,
    dist_ctx: Any,
    total_steps: int,
    save_final: bool,
    step_fn: Callable[..., Any],
) -> None:
    try:
        deepspeed = import_module("deepspeed")
    except Exception as exc:
        raise ImportError("deepspeed backend requires `deepspeed` package") from exc

    train_cfg = config.get("train", {})
    ds_config = _load_deepspeed_config(config, config_path, total_steps=total_steps)

    policy_model = models["policy"].to(dist_ctx.device)
    trainable_parameters = [param for param in policy_model.parameters() if param.requires_grad]
    ds_engine, _, _, _ = deepspeed.initialize(
        model=policy_model,
        model_parameters=trainable_parameters,
        config=ds_config,
    )

    models["policy"] = ds_engine
    for name, model in models.items():
        if name == "policy":
            continue
        models[name] = model.to(dist_ctx.device)

    log_every = int(train_cfg.get("log_every_steps", 10))
    checkpoint_every, checkpoint_dir = _checkpoint_settings(config)
    last_step = 0
    static_step_input = {
        "config_path": str(config_path),
        "_cached_config": config,
        "_merged_config": config,
        "_debug": _is_debug_enabled(config),
    }

    if before_train_val_iterable is not None:
        _run_validation(
            models=models,
            val_iterable=before_train_val_iterable,
            config=config,
            config_path=config_path,
            dist_ctx=dist_ctx,
            step_fn=step_fn,
            phase="before_train",
        )

    for idx, batch in _iter_training_batches(data_iterable, config, total_steps):
        last_step = idx + 1

        device_batch = move_to_device(batch, dist_ctx.device)
        payload = {"batch": device_batch, "global_step": idx}
        payload.update(static_step_input)
        output = step_fn(models, payload)
        loss = output["loss"]
        ds_engine.backward(loss)
        ds_engine.step()

        metrics = dict(output.get("metrics", {}))
        metrics["engine/global_step"] = idx + 1
        if (idx + 1) % log_every == 0:
            reduced = reduce_metrics(metrics, dist_ctx)
            if is_main_process(dist_ctx):
                print(f"step={idx + 1} metrics={reduced}")

        if checkpoint_every > 0 and (idx + 1) % checkpoint_every == 0:
            tag = f"step_{idx + 1}"
            ds_engine.save_checkpoint(str(checkpoint_dir), tag=tag, client_state={"global_step": idx + 1})
            if is_main_process(dist_ctx):
                print(f"checkpoint_saved={checkpoint_dir / tag}")

    if save_final:
        final_step = max(last_step, 1)
        ds_engine.save_checkpoint(str(checkpoint_dir), tag="latest", client_state={"global_step": final_step})
        if is_main_process(dist_ctx):
            print(f"checkpoint_saved={checkpoint_dir / 'latest'}")


def run_train(
    config_path: str | Path,
    loader: str,
    dataloader: str,
    def_train_module: str,
    max_steps_override: int | None = None,
    backend_override: str | None = None,
    save_final: bool = False,
    debug_override: bool = False,
) -> None:
    load_config_fn, build_models_from_config_fn, step_fn = _load_def_train_functions(def_train_module)
    config = load_config_fn(config_path)
    if debug_override:
        config.setdefault("runtime", {})["debug"] = True
    backend = config.get("runtime", {}).get("distributed_backend", "nccl")
    dist_ctx = init_distributed(backend=backend)

    try:
        loader_fn = _load_symbol(loader)
        dataloader_fn = _load_symbol(dataloader)

        models = build_models_from_config_fn(config, loader_fn=loader_fn)
        training_backend = _resolve_training_backend(config, backend_override)
        if training_backend == "pytorch":
            models = wrap_models_for_ddp(models, dist_ctx)

        val_iterable = dataloader_fn(config, dist_ctx, path_key="val_path", shuffle=False)
        if val_iterable is not None and training_backend != "deepspeed":
            _run_validation(
                models=models,
                val_iterable=val_iterable,
                config=config,
                config_path=config_path,
                dist_ctx=dist_ctx,
                step_fn=step_fn,
                phase="before_train",
            )

        data_iterable = dataloader_fn(config, dist_ctx)
        total_steps = _resolve_total_steps_by_mode(config, max_steps_override, data_iterable)

        if training_backend == "deepspeed":
            _run_deepspeed_backend(
                models=models,
                data_iterable=data_iterable,
                before_train_val_iterable=val_iterable,
                config=config,
                config_path=config_path,
                dist_ctx=dist_ctx,
                total_steps=total_steps,
                save_final=save_final,
                step_fn=step_fn,
            )
        else:
            _run_pytorch_backend(
                models=models,
                data_iterable=data_iterable,
                config=config,
                config_path=config_path,
                dist_ctx=dist_ctx,
                total_steps=total_steps,
                save_final=save_final,
                step_fn=step_fn,
            )

        if val_iterable is not None:
            val_iterable = dataloader_fn(config, dist_ctx, path_key="val_path", shuffle=False)
            _run_validation(
                models=models,
                val_iterable=val_iterable,
                config=config,
                config_path=config_path,
                dist_ctx=dist_ctx,
                step_fn=step_fn,
                phase="after_train",
            )
    finally:
        cleanup_distributed()


def _is_worker_env() -> bool:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    return world_size > 1 and "RANK" in os.environ and "LOCAL_RANK" in os.environ


def _build_worker_args(args: argparse.Namespace) -> list[str]:
    worker_args = [
        "--worker",
        "--config",
        str(args.config),
        "--loader",
        args.loader,
        "--dataloader",
        args.dataloader,
        "--def-train",
        args.def_train,
    ]
    if args.max_steps is not None:
        worker_args.extend(["--max-steps", str(args.max_steps)])
    if args.backend is not None:
        worker_args.extend(["--backend", args.backend])
    if args.save_final:
        worker_args.append("--save-final")
    if args.debug:
        worker_args.append("--debug")
    return worker_args


def _should_launch(args: argparse.Namespace) -> bool:
    return args.nproc_per_node is not None


def _validate_launch_args(args: argparse.Namespace) -> None:
    has_aux_dist_arg = any(
        value is not None
        for value in [args.nnodes, args.node_rank, args.master_addr, args.master_port]
    )
    if args.nproc_per_node is None and has_aux_dist_arg:
        raise ValueError("Distributed launch args require --nproc-per-node")

    if args.nproc_per_node is not None and args.nproc_per_node < 1:
        raise ValueError("--nproc-per-node must be >= 1")

    nnodes = str(args.nnodes) if args.nnodes is not None else "1"
    is_multi_node = nnodes != "1"
    if is_multi_node:
        if args.node_rank is None:
            raise ValueError("Multi-node launch requires --node-rank")
        if args.master_addr is None:
            raise ValueError("Multi-node launch requires --master-addr")
        if args.master_port is None:
            raise ValueError("Multi-node launch requires --master-port")


def _launch_distributed(args: argparse.Namespace) -> None:
    _validate_launch_args(args)

    nnodes = str(args.nnodes) if args.nnodes is not None else "1"
    node_rank = str(args.node_rank) if args.node_rank is not None else "0"
    master_addr = args.master_addr if args.master_addr is not None else "127.0.0.1"
    master_port = str(args.master_port) if args.master_port is not None else "29500"

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(args.nproc_per_node),
        "--nnodes",
        nnodes,
        "--node-rank",
        node_rank,
        "--master-addr",
        master_addr,
        "--master-port",
        master_port,
        "-m",
        "bmpt.cli.train",
    ]
    command.extend(_build_worker_args(args))
    subprocess.run(command, check=True)


def _run_train_entry(args: argparse.Namespace) -> None:
    run_train(
        config_path=args.config,
        loader=args.loader,
        dataloader=args.dataloader,
        def_train_module=args.def_train,
        max_steps_override=args.max_steps,
        backend_override=args.backend,
        save_final=args.save_final,
        debug_override=args.debug,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker or _is_worker_env():
        _run_train_entry(args)
        return

    if _should_launch(args):
        _launch_distributed(args)
        return

    _validate_launch_args(args)
    _run_train_entry(args)


if __name__ == "__main__":
    main()
