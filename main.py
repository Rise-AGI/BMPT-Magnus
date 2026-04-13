from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, cast

import torch
from torch.nn.parallel import DistributedDataParallel

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brisk training entrypoint")
    parser.add_argument(
        "--entry",
        choices=["weighted", "engine", "train"],
        default="train",
        help="Which executable flow to run",
    )
    parser.add_argument(
        "--config",
        default="train/config.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--loader",
        default="util.components.qwen_components:load_model",
        help="Model loader symbol path module:function",
    )
    parser.add_argument(
        "--dataloader",
        default="util.components.qwen_components:build_dataloader",
        help="Dataloader builder symbol path module:function",
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
    return parser.parse_args()


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


def _load_deepspeed_config(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    ds_path = _resolve_deepspeed_config_path(config, config_path)
    with Path(ds_path).open("r", encoding="utf-8") as handle:
        ds_config = json.load(handle)

    train_cfg = config.get("train", {})
    ds_config["train_micro_batch_size_per_gpu"] = int(train_cfg.get("per_device_batch_size", 1))
    ds_config["gradient_accumulation_steps"] = int(train_cfg.get("gradient_accumulation_steps", 1))
    ds_config["gradient_clipping"] = float(train_cfg.get("grad_clip_norm", 1.0))
    return ds_config


def _run_pytorch_backend(
    models: dict[str, torch.nn.Module],
    data_iterable: Any,
    config: dict[str, Any],
    config_path: str | Path,
    dist_ctx: Any,
    total_steps: int,
    save_final: bool,
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
    checkpoint_every, checkpoint_dir = _checkpoint_settings(config)
    last_step = 0
    static_step_input = {
        "config_path": str(config_path),
        "_cached_config": config,
        "_merged_config": config,
    }
    for idx, batch in enumerate(data_iterable):
        if idx >= total_steps:
            break
        last_step = idx + 1

        device_batch = move_to_device(batch, dist_ctx.device)
        output = engine.run_micro_step(
            models=models,
            batch=device_batch,
            extra_input=static_step_input,
        )

        if (idx + 1) % log_every == 0 and is_main_process(dist_ctx):
            reduced = reduce_metrics(output["metrics"], dist_ctx)
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
    config: dict[str, Any],
    config_path: str | Path,
    dist_ctx: Any,
    total_steps: int,
    save_final: bool,
) -> None:
    try:
        deepspeed = import_module("deepspeed")
    except Exception as exc:
        raise ImportError("deepspeed backend requires `deepspeed` package") from exc

    train_cfg = config.get("train", {})
    ds_config = _load_deepspeed_config(config, config_path)

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
    }
    for idx, batch in enumerate(data_iterable):
        if idx >= total_steps:
            break
        last_step = idx + 1

        device_batch = move_to_device(batch, dist_ctx.device)
        payload = {"batch": device_batch, "global_step": idx}
        payload.update(static_step_input)
        output = step(models, payload)
        loss = output["loss"]
        ds_engine.backward(loss)
        ds_engine.step()

        metrics = dict(output.get("metrics", {}))
        metrics["engine/global_step"] = idx + 1
        if (idx + 1) % log_every == 0 and is_main_process(dist_ctx):
            reduced = reduce_metrics(metrics, dist_ctx)
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
    max_steps_override: int | None = None,
    backend_override: str | None = None,
    save_final: bool = False,
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
                save_final=save_final,
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
            )
    finally:
        cleanup_distributed()


def main() -> None:
    args = parse_args()
    if args.entry == "weighted":
        from examples.example_weighted_step import main as run_weighted_example

        run_weighted_example()
        return

    if args.entry == "engine":
        from examples.example_engine_loop import main as run_engine_example

        run_engine_example()
        return

    run_train(
        config_path=args.config,
        loader=args.loader,
        dataloader=args.dataloader,
        max_steps_override=args.max_steps,
        backend_override=args.backend,
        save_final=args.save_final,
    )


if __name__ == "__main__":
    main()
