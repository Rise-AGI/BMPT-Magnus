from __future__ import annotations

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import import_module
from importlib import util as importlib_util
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
from bmpt.core.async_checkpoint import AsyncCheckpointWriter
from bmpt.core.engine import TrainingEngine
from bmpt.core.logging import MetricsEmitter, StepMetricsLogger
from bmpt.core.optim import build_optimizer, build_scheduler
from bmpt.util import build_composers_from_config


def _default_config_path() -> str:
    return str(Path(__file__).resolve().parent.parent / "algorithms" / "config.yaml")


def _default_def_train_module() -> str:
    return "bmpt.algorithms.def_train"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BMPT training entrypoint")
    parser.add_argument(
        "--config",
        default=_default_config_path(),
        help="Path to training config file",
    )
    parser.add_argument(
        "--loader",
        default="bmpt.model.loader:load_model",
        help="Model loader symbol path module:function",
    )
    parser.add_argument(
        "--def-train",
        default=_default_def_train_module(),
        help="Training definition module path, e.g. bmpt.algorithms.def_train",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Workspace directory used to auto-discover config and def_train.py",
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
        "--attn-implementation",
        default=None,
        help="Override runtime attention backend, e.g. auto/eager/sdpa/flash_attention_2",
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


def _arg_provided(raw_argv: list[str], flag: str) -> bool:
    prefix = f"{flag}="
    return any(token == flag or token.startswith(prefix) for token in raw_argv)


def _find_workspace_candidates(workspace: Path, file_name: str) -> list[Path]:
    candidates = [
        path.resolve() for path in workspace.rglob(file_name) if path.is_file()
    ]
    return sorted(
        candidates, key=lambda path: (len(path.relative_to(workspace).parts), str(path))
    )


def _select_workspace_candidate(
    label: str, workspace: Path, candidates: list[Path]
) -> Path:
    chosen = candidates[0]
    if len(candidates) > 1:
        print(
            f"[workspace] Found multiple {label} files under {workspace}. "
            f"Using first candidate: {chosen}",
            flush=True,
        )
    return chosen


def _resolve_workspace_overrides(args: argparse.Namespace, raw_argv: list[str]) -> None:
    if args.workspace is None:
        return

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace}")
    if not workspace.is_dir():
        raise NotADirectoryError(f"Workspace is not a directory: {workspace}")

    config_is_manual = _arg_provided(raw_argv, "--config")
    def_train_is_manual = _arg_provided(raw_argv, "--def-train")

    if not config_is_manual:
        config_candidates: list[Path] = []
        for name in ("config.json", "config.yaml", "config.yml"):
            config_candidates.extend(_find_workspace_candidates(workspace, name))
        config_candidates = sorted(
            set(config_candidates),
            key=lambda path: (len(path.relative_to(workspace).parts), str(path)),
        )
        if config_candidates:
            args.config = str(
                _select_workspace_candidate("config", workspace, config_candidates)
            )

    if not def_train_is_manual:
        def_train_candidates = _find_workspace_candidates(workspace, "def_train.py")
        if def_train_candidates:
            args.def_train = str(
                _select_workspace_candidate(
                    "def_train", workspace, def_train_candidates
                )
            )


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


def _load_module_from_path(module_path: str | Path):
    path_obj = Path(module_path).expanduser().resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Training definition file not found: {path_obj}")
    if not path_obj.is_file():
        raise ValueError(f"Training definition path is not a file: {path_obj}")

    suffix = hashlib.sha1(str(path_obj).encode("utf-8")).hexdigest()[:12]
    module_name = f"_bmpt_user_def_train_{suffix}"
    spec = importlib_util.spec_from_file_location(module_name, path_obj)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from path: {path_obj}")

    module = importlib_util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_def_train_functions(
    def_train_module: str,
) -> tuple[
    Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]
]:
    module_path = Path(def_train_module)
    should_load_from_path = def_train_module.endswith(".py") or module_path.exists()
    if should_load_from_path:
        module = _load_module_from_path(def_train_module)
    else:
        module = import_module(def_train_module)
    load_config_fn = getattr(module, "load_config", None)
    build_models_from_config_fn = getattr(module, "build_models_from_config", None)
    step_fn = getattr(module, "step", None)
    evaluate_fn = getattr(module, "evaluate", None)

    if not callable(load_config_fn):
        raise AttributeError(f"`{def_train_module}` must expose callable `load_config`")
    if not callable(build_models_from_config_fn):
        raise AttributeError(
            f"`{def_train_module}` must expose callable `build_models_from_config`"
        )
    if not callable(step_fn):
        raise AttributeError(f"`{def_train_module}` must expose callable `step`")
    if not callable(evaluate_fn):
        raise AttributeError(f"`{def_train_module}` must expose callable `evaluate`")
    return load_config_fn, build_models_from_config_fn, step_fn, evaluate_fn


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
        raise ValueError(
            "`train.epochs` must be > 0 when train.control_mode is `epoch`"
        )
    if not hasattr(data_iterable, "__len__"):
        raise ValueError("`train.control_mode=epoch` requires dataloader with __len__")
    return int(len(data_iterable)) * epochs


def _iter_training_batches(
    data_iterable: Any,
    config: dict[str, Any],
    total_steps: int,
    start_step: int = 0,
):
    if start_step >= total_steps:
        return

    mode = _resolve_train_control_mode(config)
    if mode == "step":
        for step_idx, batch in enumerate(data_iterable):
            if step_idx < start_step:
                continue
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
            if global_step < start_step:
                global_step += 1
                continue
            yield global_step, batch
            global_step += 1


def _resolve_training_backend(
    config: dict[str, Any], backend_override: str | None
) -> str:
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


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _checkpoint_file_path(
    checkpoint_dir: Path, base_name: str, rank: int, world_size: int
) -> Path:
    if world_size <= 1:
        return checkpoint_dir / f"{base_name}.pt"
    return checkpoint_dir / f"{base_name}.rank_{rank}.pt"


def _extract_resume_config(config: dict[str, Any]) -> dict[str, Any]:
    train_cfg = config.get("train", {})
    runtime_cfg = config.get("runtime", {})
    return {
        "optimizer": copy.deepcopy(config.get("optimizer", {})),
        "scheduler": copy.deepcopy(config.get("scheduler", {})),
        "train": {
            "gradient_accumulation_steps": int(
                train_cfg.get("gradient_accumulation_steps", 1)
            ),
            "mixed_precision": str(train_cfg.get("mixed_precision", "bf16")),
        },
        "runtime": {
            "training_backend": str(runtime_cfg.get("training_backend", "pytorch"))
        },
    }


def _build_checkpoint_payload(
    *,
    backend: str,
    global_step: int,
    models: dict[str, torch.nn.Module],
    optimizer: Any,
    scheduler: Any,
    resume_config: dict[str, Any],
    rank: int,
    world_size: int,
    engine_state: dict[str, int] | None,
) -> dict[str, Any]:
    model_states: dict[str, Any] = {}
    for name, model in models.items():
        model_states[name] = _clone_to_cpu(_unwrap_model(model).state_dict())
    return {
        "format_version": 2,
        "backend": backend,
        "global_step": int(global_step),
        "models": model_states,
        "optimizer": _clone_to_cpu(optimizer.state_dict())
        if optimizer is not None
        else None,
        "scheduler": _clone_to_cpu(scheduler.state_dict())
        if scheduler is not None
        else None,
        "engine_state": engine_state,
        "resume_config": _clone_to_cpu(resume_config),
        "meta": {
            "rank": rank,
            "world_size": world_size,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def _resolve_load_ckpt_settings(
    config: dict[str, Any], config_path: str | Path
) -> tuple[Path | None, str, bool]:
    train_cfg = config.get("train", {})
    raw_path = train_cfg.get("load_ckpt_path")
    mode = str(train_cfg.get("load_ckpt_mode", "full")).lower()
    strict = bool(train_cfg.get("load_ckpt_strict", True))

    if mode not in {"full", "weights_only"}:
        raise ValueError(f"Unsupported train.load_ckpt_mode: {mode}")
    if raw_path in {None, ""}:
        return None, mode, strict

    ckpt_path = Path(str(raw_path))
    if not ckpt_path.is_absolute():
        ckpt_path = Path(config_path).parent / ckpt_path
    return ckpt_path.resolve(), mode, strict


def _resolve_ranked_load_path(path: Path, rank: int, world_size: int) -> Path:
    if path.exists() or world_size <= 1:
        return path
    ranked = path.with_name(f"{path.stem}.rank_{rank}{path.suffix}")
    if ranked.exists():
        return ranked
    return path


def _load_checkpoint_payload(
    path: Path, rank: int, world_size: int
) -> tuple[dict[str, Any], Path]:
    ranked_path = _resolve_ranked_load_path(path, rank, world_size)
    if not ranked_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ranked_path}")
    payload = torch.load(ranked_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be dict, got: {type(payload)}")
    return payload, ranked_path


def _flatten_leaf_values(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if len(value) == 0:
            return {prefix: {}}
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_leaf_values(child_key, item))
        return flattened
    return {prefix: value}


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def _log_resume_diff(
    path: str,
    old_value: Any,
    new_value: Any,
    override: bool,
    emit_logs: bool,
) -> None:
    if not emit_logs:
        return
    if old_value == new_value:
        print(
            f"[ckpt] override config: {path} {_format_value(old_value)} -> {_format_value(new_value)} (unchanged skipped)",
            flush=True,
        )
        return
    if override:
        print(
            f"[ckpt] override config: {path} {_format_value(old_value)} -> {_format_value(new_value)} (source=ckpt)",
            flush=True,
        )
        return
    print(
        f"[ckpt] config diff: {path} {_format_value(old_value)} -> {_format_value(new_value)} (weights_only not overridden)",
        flush=True,
    )


def _apply_or_report_resume_config(
    config: dict[str, Any], payload: dict[str, Any], mode: str, emit_logs: bool
) -> None:
    resume_cfg = payload.get("resume_config")
    if not isinstance(resume_cfg, dict):
        return

    should_override = mode == "full"

    old_optimizer = copy.deepcopy(config.get("optimizer", {}))
    new_optimizer = copy.deepcopy(resume_cfg.get("optimizer", {}))
    old_scheduler = copy.deepcopy(config.get("scheduler", {}))
    new_scheduler = copy.deepcopy(resume_cfg.get("scheduler", {}))

    merged_keys = sorted(
        set(_flatten_leaf_values("optimizer", old_optimizer)).union(
            _flatten_leaf_values("optimizer", new_optimizer)
        )
    )
    old_flat_optimizer = _flatten_leaf_values("optimizer", old_optimizer)
    new_flat_optimizer = _flatten_leaf_values("optimizer", new_optimizer)
    for key in merged_keys:
        _log_resume_diff(
            key,
            old_flat_optimizer.get(key),
            new_flat_optimizer.get(key),
            should_override,
            emit_logs,
        )

    merged_scheduler_keys = sorted(
        set(_flatten_leaf_values("scheduler", old_scheduler)).union(
            _flatten_leaf_values("scheduler", new_scheduler)
        )
    )
    old_flat_scheduler = _flatten_leaf_values("scheduler", old_scheduler)
    new_flat_scheduler = _flatten_leaf_values("scheduler", new_scheduler)
    for key in merged_scheduler_keys:
        _log_resume_diff(
            key,
            old_flat_scheduler.get(key),
            new_flat_scheduler.get(key),
            should_override,
            emit_logs,
        )

    old_grad_accum = config.get("train", {}).get("gradient_accumulation_steps")
    new_grad_accum = resume_cfg.get("train", {}).get("gradient_accumulation_steps")
    _log_resume_diff(
        "train.gradient_accumulation_steps",
        old_grad_accum,
        new_grad_accum,
        should_override,
        emit_logs,
    )

    old_precision = config.get("train", {}).get("mixed_precision")
    new_precision = resume_cfg.get("train", {}).get("mixed_precision")
    _log_resume_diff(
        "train.mixed_precision",
        old_precision,
        new_precision,
        should_override,
        emit_logs,
    )

    old_backend = config.get("runtime", {}).get("training_backend")
    new_backend = resume_cfg.get("runtime", {}).get("training_backend")
    _log_resume_diff(
        "runtime.training_backend",
        old_backend,
        new_backend,
        should_override,
        emit_logs,
    )

    if not should_override:
        return

    config["optimizer"] = new_optimizer
    config["scheduler"] = new_scheduler
    config.setdefault("train", {})["gradient_accumulation_steps"] = new_grad_accum
    config.setdefault("train", {})["mixed_precision"] = new_precision
    config.setdefault("runtime", {})["training_backend"] = new_backend


def _restore_model_states(
    models: dict[str, torch.nn.Module],
    model_states: dict[str, Any],
    strict: bool,
) -> None:
    for model_name, model in models.items():
        model_state = model_states.get(model_name)
        if model_state is None:
            if strict:
                raise KeyError(f"Missing model state for `{model_name}` in checkpoint")
            continue
        target_model = _unwrap_model(model)
        target_model.load_state_dict(model_state, strict=strict)


def _restore_pytorch_state(
    *,
    payload: dict[str, Any],
    models: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler
    | torch.optim.lr_scheduler.LambdaLR
    | None,
    engine: TrainingEngine,
    mode: str,
    strict: bool,
) -> int:
    model_states = payload.get("models")
    if not isinstance(model_states, dict):
        raise KeyError("Checkpoint missing `models` dict")
    _restore_model_states(models, model_states, strict=strict)

    if mode == "weights_only":
        return 0

    optimizer_state = payload.get("optimizer")
    if optimizer_state is None:
        raise KeyError("Checkpoint missing `optimizer` state for full resume")
    optimizer.load_state_dict(optimizer_state)

    scheduler_state = payload.get("scheduler")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    global_step = int(payload.get("global_step", 0))
    engine_state = payload.get("engine_state")
    if isinstance(engine_state, dict):
        engine.state.micro_step = int(engine_state.get("micro_step", global_step))
        engine.state.optimizer_step = int(
            engine_state.get("optimizer_step", engine.state.optimizer_step)
        )
    else:
        engine.state.micro_step = global_step
    engine.state.global_step = int(engine.state.optimizer_step)
    return global_step


def _restore_deepspeed_state(
    *,
    payload: dict[str, Any],
    models: dict[str, torch.nn.Module],
    ds_engine: Any,
    mode: str,
    strict: bool,
    rank: int = 0,
    world_size: int = 1,
) -> int:
    model_states = payload.get("models")
    if not isinstance(model_states, dict):
        raise KeyError("Checkpoint missing `models` dict")

    restored_models = dict(models)
    restored_models["policy"] = cast(torch.nn.Module, ds_engine.module)
    _restore_model_states(restored_models, model_states, strict=strict)

    if mode == "weights_only":
        return 0

    optimizer_state = payload.get("optimizer")
    if (
        optimizer_state is not None
        and getattr(ds_engine, "optimizer", None) is not None
    ):
        if world_size > 1:
            state_dict_list: list[dict[str, Any] | None] = [None] * world_size
            state_dict_list[rank] = optimizer_state
            ds_engine.optimizer.load_state_dict(state_dict_list)
        else:
            ds_engine.optimizer.load_state_dict(optimizer_state)

    scheduler_state = payload.get("scheduler")
    if (
        scheduler_state is not None
        and getattr(ds_engine, "lr_scheduler", None) is not None
    ):
        ds_engine.lr_scheduler.load_state_dict(scheduler_state)

    global_step = int(payload.get("global_step", 0))
    if hasattr(ds_engine, "global_steps"):
        setattr(ds_engine, "global_steps", global_step)
    if hasattr(ds_engine, "micro_steps"):
        setattr(ds_engine, "micro_steps", global_step)
    return global_step


def _resolve_deepspeed_config_path(
    config: dict[str, Any], config_path: str | Path
) -> str:
    runtime_cfg = config.get("runtime", {})
    ds_path = runtime_cfg.get("deepspeed_config_path")
    if ds_path is None:
        raise ValueError(
            "`runtime.deepspeed_config_path` is required for deepspeed backend"
        )

    ds_path_obj = Path(ds_path)
    if not ds_path_obj.is_absolute():
        ds_path_obj = Path(config_path).parent / ds_path_obj
    resolved = ds_path_obj.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {resolved}")
    return str(resolved)


def _load_deepspeed_config(
    config: dict[str, Any], config_path: str | Path, total_steps: int
) -> dict[str, Any]:
    ds_path = _resolve_deepspeed_config_path(config, config_path)
    with Path(ds_path).open("r", encoding="utf-8") as handle:
        ds_config = json.load(handle)

    zero_cfg = ds_config.get("zero_optimization", {})
    if zero_cfg.get("elastic_checkpoint", False) is True:
        raise ValueError(
            "BMPT does not support elastic_checkpoint=True in DeepSpeed config. "
            "Please set elastic_checkpoint=False or remove it from your config."
        )
    if "zero_optimization" not in ds_config:
        ds_config["zero_optimization"] = {}
    ds_config["zero_optimization"]["elastic_checkpoint"] = False

    train_cfg = config.get("train", {})
    optimizer_cfg = config.get("optimizer", {})
    scheduler_cfg = config.get("scheduler", {})

    ds_config["train_micro_batch_size_per_gpu"] = int(
        train_cfg.get("per_device_batch_size", 1)
    )
    ds_config["gradient_accumulation_steps"] = int(
        train_cfg.get("gradient_accumulation_steps", 1)
    )
    ds_config["gradient_clipping"] = float(train_cfg.get("grad_clip_norm", 1.0))

    optimizer_type = str(optimizer_cfg.get("type", "adamw")).lower()
    if optimizer_type != "adamw":
        raise ValueError(
            f"Unsupported optimizer.type for deepspeed backend: {optimizer_type}"
        )
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
        raise ValueError(
            f"Unsupported train.mixed_precision for deepspeed backend: {mixed_precision}"
        )

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
        raise ValueError(
            f"Unsupported scheduler.type for deepspeed backend: {scheduler_type}"
        )

    return ds_config


def _run_validation(
    models: dict[str, torch.nn.Module],
    val_iterable: Any,
    config: dict[str, Any],
    config_path: str | Path,
    dist_ctx: Any,
    composers: dict[str, Any],
    evaluate_fn: Callable[..., Any],
    phase: str,
) -> None:
    _debug_print(config, f"DEBUG: _run_validation enter, phase={phase}")

    payload = {
        "config_path": str(config_path),
        "_cached_config": config,
        "_merged_config": config,
        "composers": composers,
        "val_iterable": val_iterable,
        "dist_ctx": dist_ctx,
        "phase": phase,
    }
    output = evaluate_fn(models, payload)
    metrics = output.get("metrics", {}) if isinstance(output, dict) else {}
    if not isinstance(metrics, dict):
        raise TypeError("`evaluate` must return dict with `metrics: dict[str, float]`")

    if is_main_process(dist_ctx):
        print(f"[{phase}] eval_metrics={metrics}")


def _run_pytorch_backend(
    models: dict[str, torch.nn.Module],
    data_iterable: Any,
    config: dict[str, Any],
    config_path: str | Path,
    dist_ctx: Any,
    composers: dict[str, Any],
    total_steps: int,
    save_final: bool,
    step_fn: Callable[..., Any],
    load_ckpt_payload: dict[str, Any] | None,
    load_ckpt_mode: str,
    load_ckpt_strict: bool,
    tokenizer: Any | None = None,
) -> None:
    train_cfg = config.get("train", {})
    metrics_cfg = config.get("runtime", {}).get("metrics", {})
    perf_logger = StepMetricsLogger.from_config(metrics_cfg)
    metrics_emitter = MetricsEmitter.from_config(metrics_cfg)

    target_model_name = train_cfg.get("train_target_model")
    if target_model_name is not None:
        if target_model_name not in models:
            raise ValueError(f"train_target_model '{target_model_name}' not found in models")
        optimizer_models = {target_model_name: models[target_model_name]}
    else:
        optimizer_models = models

    optimizer = build_optimizer(optimizer_models, config)
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
    writer = AsyncCheckpointWriter(max_pending=1)
    resume_start_step = 0
    if load_ckpt_payload is not None:
        resume_start_step = _restore_pytorch_state(
            payload=load_ckpt_payload,
            models=models,
            optimizer=optimizer,
            scheduler=scheduler,
            engine=engine,
            mode=load_ckpt_mode,
            strict=load_ckpt_strict,
        )
        if _is_rank0():
            print(
                f"[ckpt] resumed from step={resume_start_step} mode={load_ckpt_mode}",
                flush=True,
            )

    resume_config = _extract_resume_config(config)
    last_step = resume_start_step
    static_step_input = {
        "config_path": str(config_path),
        "_cached_config": config,
        "_merged_config": config,
        "composers": composers,
        "_debug": _is_debug_enabled(config),
        "tokenizer": tokenizer,
    }
    try:
        for idx, batch in _iter_training_batches(
            data_iterable, config, total_steps, start_step=resume_start_step
        ):
            current_step = idx + 1
            last_step = current_step

            device_batch = move_to_device(batch, dist_ctx.device)
            step_start_time = time.perf_counter()
            output = engine.run_micro_step(
                models=models,
                batch=device_batch,
                extra_input=static_step_input,
            )
            step_time_sec = time.perf_counter() - step_start_time

            perf_metrics = perf_logger.update(
                step_time_sec=step_time_sec,
                batch=device_batch,
                device=dist_ctx.device,
                sync_global=(current_step % log_every == 0),
            )
            output["metrics"].update(perf_metrics)

            if current_step % log_every == 0:
                reduced = reduce_metrics(output["metrics"], dist_ctx)
                if is_main_process(dist_ctx):
                    metrics_emitter.emit(step_id=current_step, metrics=reduced)

            if checkpoint_every > 0 and current_step % checkpoint_every == 0:
                save_path = _checkpoint_file_path(
                    checkpoint_dir,
                    f"step_{current_step}",
                    rank=dist_ctx.rank,
                    world_size=dist_ctx.world_size,
                )
                payload = _build_checkpoint_payload(
                    backend="pytorch",
                    global_step=current_step,
                    models=models,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    resume_config=resume_config,
                    rank=dist_ctx.rank,
                    world_size=dist_ctx.world_size,
                    engine_state={
                        "micro_step": int(engine.state.micro_step),
                        "optimizer_step": int(engine.state.optimizer_step),
                    },
                )
                writer.enqueue(save_path, payload)
                if is_main_process(dist_ctx):
                    print(f"checkpoint_enqueued={save_path}")

        if save_final:
            final_step = max(last_step, resume_start_step, 1)
            save_path = _checkpoint_file_path(
                checkpoint_dir,
                "latest",
                rank=dist_ctx.rank,
                world_size=dist_ctx.world_size,
            )
            payload = _build_checkpoint_payload(
                backend="pytorch",
                global_step=final_step,
                models=models,
                optimizer=optimizer,
                scheduler=scheduler,
                resume_config=resume_config,
                rank=dist_ctx.rank,
                world_size=dist_ctx.world_size,
                engine_state={
                    "micro_step": int(engine.state.micro_step),
                    "optimizer_step": int(engine.state.optimizer_step),
                },
            )
            writer.enqueue(save_path, payload)
            if is_main_process(dist_ctx):
                print(f"checkpoint_enqueued={save_path}")

        writer.flush()
    finally:
        writer.close()


def _run_deepspeed_backend(
    models: dict[str, torch.nn.Module],
    data_iterable: Any,
    before_train_val_iterable: Any | None,
    config: dict[str, Any],
    config_path: str | Path,
    dist_ctx: Any,
    composers: dict[str, Any],
    total_steps: int,
    save_final: bool,
    step_fn: Callable[..., Any],
    evaluate_fn: Callable[..., Any],
    load_ckpt_payload: dict[str, Any] | None,
    load_ckpt_mode: str,
    load_ckpt_strict: bool,
    tokenizer: Any | None = None,
) -> None:
    try:
        deepspeed = import_module("deepspeed")
    except Exception as exc:
        raise ImportError("deepspeed backend requires `deepspeed` package") from exc

    train_cfg = config.get("train", {})
    metrics_cfg = config.get("runtime", {}).get("metrics", {})
    perf_logger = StepMetricsLogger.from_config(metrics_cfg)
    metrics_emitter = MetricsEmitter.from_config(metrics_cfg)

    ds_config = _load_deepspeed_config(config, config_path, total_steps=total_steps)

    target_model_name = train_cfg.get("train_target_model", "policy")
    if target_model_name not in models:
        raise ValueError(f"train_target_model '{target_model_name}' not found in models")

    target_model = models[target_model_name].to(dist_ctx.device)
    trainable_parameters = [
        param for param in target_model.parameters() if param.requires_grad
    ]
    if len(trainable_parameters) == 0:
        raise ValueError(f"Model '{target_model_name}' has no trainable parameters")

    ds_engine, _, _, _ = deepspeed.initialize(
        model=target_model,
        model_parameters=trainable_parameters,
        config=ds_config,
    )

    models[target_model_name] = ds_engine
    for name, model in models.items():
        if name == target_model_name:
            continue
        if isinstance(model, torch.nn.Module):
            models[name] = model.to(dist_ctx.device)

    log_every = int(train_cfg.get("log_every_steps", 10))
    checkpoint_every, checkpoint_dir = _checkpoint_settings(config)
    writer = AsyncCheckpointWriter(max_pending=1)
    resume_start_step = 0
    if load_ckpt_payload is not None:
        resume_start_step = _restore_deepspeed_state(
            payload=load_ckpt_payload,
            models=models,
            ds_engine=ds_engine,
            mode=load_ckpt_mode,
            strict=load_ckpt_strict,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
        )
        if _is_rank0():
            print(
                f"[ckpt] resumed from step={resume_start_step} mode={load_ckpt_mode}",
                flush=True,
            )

    resume_config = _extract_resume_config(config)
    last_step = resume_start_step
    static_step_input = {
        "config_path": str(config_path),
        "_cached_config": config,
        "_merged_config": config,
        "composers": composers,
        "_debug": _is_debug_enabled(config),
        "tokenizer": tokenizer,
    }

    if before_train_val_iterable is not None:
        _run_validation(
            models=models,
            val_iterable=before_train_val_iterable,
            config=config,
            config_path=config_path,
            dist_ctx=dist_ctx,
            composers=composers,
            evaluate_fn=evaluate_fn,
            phase="before_train",
        )

    try:
        for idx, batch in _iter_training_batches(
            data_iterable, config, total_steps, start_step=resume_start_step
        ):
            current_step = idx + 1
            last_step = current_step

            device_batch = move_to_device(batch, dist_ctx.device)
            step_start_time = time.perf_counter()
            payload = {"batch": device_batch, "global_step": idx}
            payload.update(static_step_input)
            output = step_fn(models, payload)
            loss = output["loss"]
            ds_engine.backward(loss)
            ds_engine.step()
            step_time_sec = time.perf_counter() - step_start_time

            metrics = dict(output.get("metrics", {}))
            metrics["engine/global_step"] = current_step
            metrics.update(
                perf_logger.update(
                    step_time_sec=step_time_sec,
                    batch=device_batch,
                    device=dist_ctx.device,
                    sync_global=(current_step % log_every == 0),
                )
            )
            if current_step % log_every == 0:
                reduced = reduce_metrics(metrics, dist_ctx)
                if is_main_process(dist_ctx):
                    metrics_emitter.emit(step_id=current_step, metrics=reduced)

            if checkpoint_every > 0 and current_step % checkpoint_every == 0:
                save_path = _checkpoint_file_path(
                    checkpoint_dir,
                    f"step_{current_step}",
                    rank=dist_ctx.rank,
                    world_size=dist_ctx.world_size,
                )
                models_for_save = dict(models)
                models_for_save["policy"] = cast(torch.nn.Module, ds_engine.module)
                payload = _build_checkpoint_payload(
                    backend="deepspeed",
                    global_step=current_step,
                    models=models_for_save,
                    optimizer=getattr(ds_engine, "optimizer", None),
                    scheduler=getattr(ds_engine, "lr_scheduler", None),
                    resume_config=resume_config,
                    rank=dist_ctx.rank,
                    world_size=dist_ctx.world_size,
                    engine_state={
                        "micro_step": int(current_step),
                        "optimizer_step": int(current_step),
                    },
                )
                writer.enqueue(save_path, payload)
                if is_main_process(dist_ctx):
                    print(f"checkpoint_enqueued={save_path}")

        if save_final:
            final_step = max(last_step, resume_start_step, 1)
            save_path = _checkpoint_file_path(
                checkpoint_dir,
                "latest",
                rank=dist_ctx.rank,
                world_size=dist_ctx.world_size,
            )
            models_for_save = dict(models)
            models_for_save["policy"] = cast(torch.nn.Module, ds_engine.module)
            payload = _build_checkpoint_payload(
                backend="deepspeed",
                global_step=final_step,
                models=models_for_save,
                optimizer=getattr(ds_engine, "optimizer", None),
                scheduler=getattr(ds_engine, "lr_scheduler", None),
                resume_config=resume_config,
                rank=dist_ctx.rank,
                world_size=dist_ctx.world_size,
                engine_state={
                    "micro_step": int(final_step),
                    "optimizer_step": int(final_step),
                },
            )
            writer.enqueue(save_path, payload)
            if is_main_process(dist_ctx):
                print(f"checkpoint_enqueued={save_path}")

        writer.flush()
    finally:
        writer.close()


def run_train(
    config_path: str | Path,
    loader: str,
    def_train_module: str,
    max_steps_override: int | None = None,
    backend_override: str | None = None,
    attn_implementation_override: str | None = None,
    save_final: bool = False,
    debug_override: bool = False,
) -> None:
    load_config_fn, build_models_from_config_fn, step_fn, evaluate_fn = (
        _load_def_train_functions(def_train_module)
    )
    config = load_config_fn(config_path)

    env_rank = int(os.getenv("RANK", "0"))
    env_world_size = int(os.getenv("WORLD_SIZE", "1"))
    load_ckpt_path, load_ckpt_mode, load_ckpt_strict = _resolve_load_ckpt_settings(
        config, config_path
    )
    load_ckpt_payload: dict[str, Any] | None = None
    if load_ckpt_path is not None:
        load_ckpt_payload, resolved_ckpt_path = _load_checkpoint_payload(
            load_ckpt_path,
            rank=env_rank,
            world_size=env_world_size,
        )
        if env_rank == 0:
            print(f"[ckpt] load checkpoint: {resolved_ckpt_path}", flush=True)
        _apply_or_report_resume_config(
            config,
            load_ckpt_payload,
            load_ckpt_mode,
            emit_logs=(env_rank == 0),
        )

    if debug_override:
        config.setdefault("runtime", {})["debug"] = True
    if attn_implementation_override is not None:
        config.setdefault("runtime", {})["attn_implementation"] = (
            attn_implementation_override
        )
    composers = build_composers_from_config(config)
    if len(composers) > 0 and int(os.getenv("RANK", "0")) == 0:
        print(f"[bmpt] loaded_composers={list(composers.keys())}", flush=True)
    backend = config.get("runtime", {}).get("distributed_backend", "nccl")
    dist_ctx = init_distributed(backend=backend)

    from bmpt.data import build_dataloader, process_all_sources
    from bmpt.tokenizer import load_tokenizer

    try:
        loader_fn = _load_symbol(loader)

        tokenizer = load_tokenizer(config)
        if env_rank == 0:
            print("[bmpt] tokenizer loaded", flush=True)

        processed_data = process_all_sources(config, tokenizer)
        if env_rank == 0:
            print(f"[bmpt] processed sources: {list(processed_data.keys())}", flush=True)

        models = build_models_from_config_fn(config, loader_fn=loader_fn)
        training_backend = _resolve_training_backend(config, backend_override)
        if training_backend == "pytorch":
            models = wrap_models_for_ddp(models, dist_ctx)

        val_records = processed_data.get("val")
        val_iterable = None
        if val_records is not None:
            val_iterable = build_dataloader(val_records, config, dist_ctx, shuffle=False)
            if training_backend != "deepspeed":
                _run_validation(
                    models=models,
                    val_iterable=val_iterable,
                    config=config,
                    config_path=config_path,
                    dist_ctx=dist_ctx,
                    composers=composers,
                    evaluate_fn=evaluate_fn,
                    phase="before_train",
                )

        train_records = processed_data.get("train")
        if train_records is None:
            raise ValueError("No 'train' source found in config['data']['sources']")
        data_iterable = build_dataloader(train_records, config, dist_ctx)
        total_steps = _resolve_total_steps_by_mode(
            config, max_steps_override, data_iterable
        )

        if training_backend == "deepspeed":
            _run_deepspeed_backend(
                models=models,
                data_iterable=data_iterable,
                before_train_val_iterable=val_iterable,
                config=config,
                config_path=config_path,
                dist_ctx=dist_ctx,
                composers=composers,
                total_steps=total_steps,
                save_final=save_final,
                step_fn=step_fn,
                evaluate_fn=evaluate_fn,
                load_ckpt_payload=load_ckpt_payload,
                load_ckpt_mode=load_ckpt_mode,
                load_ckpt_strict=load_ckpt_strict,
                tokenizer=tokenizer,
            )
        else:
            _run_pytorch_backend(
                models=models,
                data_iterable=data_iterable,
                config=config,
                config_path=config_path,
                dist_ctx=dist_ctx,
                composers=composers,
                total_steps=total_steps,
                save_final=save_final,
                step_fn=step_fn,
                load_ckpt_payload=load_ckpt_payload,
                load_ckpt_mode=load_ckpt_mode,
                load_ckpt_strict=load_ckpt_strict,
                tokenizer=tokenizer,
            )

        if val_records is not None:
            val_iterable = build_dataloader(val_records, config, dist_ctx, shuffle=False)
            _run_validation(
                models=models,
                val_iterable=val_iterable,
                config=config,
                config_path=config_path,
                dist_ctx=dist_ctx,
                composers=composers,
                evaluate_fn=evaluate_fn,
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
    if args.attn_implementation is not None:
        worker_args.extend(["--attn-implementation", args.attn_implementation])
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
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        _print_distributed_failure_hint(args, command, exc)
        raise


def _print_distributed_failure_hint(
    args: argparse.Namespace,
    command: list[str],
    exc: subprocess.CalledProcessError,
) -> None:
    print("[bmpt] Distributed launch failed.", file=sys.stderr, flush=True)
    print(f"[bmpt] returncode={exc.returncode}", file=sys.stderr, flush=True)
    print(
        f"[bmpt] command={' '.join(command)}",
        file=sys.stderr,
        flush=True,
    )
    print("[bmpt] Recommended checks:", file=sys.stderr, flush=True)
    print(
        "[bmpt] 1) Check host/container OOM kill logs: "
        "dmesg -T | egrep -i 'oom|killed process'",
        file=sys.stderr,
        flush=True,
    )
    print(
        "[bmpt] 2) Reduce worker count and retry: --nproc-per-node 1",
        file=sys.stderr,
        flush=True,
    )
    print(
        "[bmpt] 3) Reduce memory pressure in config: lower "
        "train_micro_batch_size_per_gpu or increase grad accumulation",
        file=sys.stderr,
        flush=True,
    )
    print(
        "[bmpt] 4) If using FlashAttention, retry with: --attn-implementation sdpa",
        file=sys.stderr,
        flush=True,
    )
    print(
        "[bmpt] 5) Resume from latest checkpoint if available.",
        file=sys.stderr,
        flush=True,
    )

    attn = args.attn_implementation
    if attn in (None, "auto", "flash_attention_2"):
        print(
            "[bmpt] Hint: current attention mode may increase memory pressure "
            "during distributed runs.",
            file=sys.stderr,
            flush=True,
        )


def _run_train_entry(args: argparse.Namespace) -> None:
    run_train(
        config_path=args.config,
        loader=args.loader,
        def_train_module=args.def_train,
        max_steps_override=args.max_steps,
        backend_override=args.backend,
        attn_implementation_override=args.attn_implementation,
        save_final=args.save_final,
        debug_override=args.debug,
    )


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    _resolve_workspace_overrides(args, raw_argv)

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
