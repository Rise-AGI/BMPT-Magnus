"""
BMPT SFT 训练示例：微调 Qwen2.5-7B 模型

使用方式：
    单卡：python train.py
    多卡：torchrun --nproc_per_node=8 train.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from bmpt.core.distributed import (
    DistributedContext,
    cleanup_distributed,
    init_distributed,
    is_main_process,
    move_to_device,
    reduce_metrics,
)
from bmpt.core.logging import MetricsEmitter, StepMetricsLogger
from bmpt.data.dataloader import build_dataloader
from bmpt.data.processor import process_all_sources
from bmpt.manager import Manager
from bmpt.tokenizer import load_tokenizer
from bmpt.toolbox import ToolBox


def build_sft_batch(
    batch: dict[str, torch.Tensor],
    tokenizer: object,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从预处理数据构建 SFT 训练输入。

    Args:
        batch: 包含 prompt_input_ids 和 response_input_ids 的批次
        tokenizer: Tokenizer 对象（用于获取 pad_token_id）
        device: 目标设备

    Returns:
        (input_ids, attention_mask, labels)
    """
    prompt_ids = batch.get("prompt_input_ids")
    response_ids = batch.get("response_input_ids")

    if prompt_ids is None or response_ids is None:
        raise ValueError("batch must contain prompt_input_ids and response_input_ids")

    prompt_ids = prompt_ids.to(device=device, dtype=torch.long)
    response_ids = response_ids.to(device=device, dtype=torch.long)

    input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    prompt_len = prompt_ids.shape[-1]
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    return input_ids, attention_mask, labels


def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"

    manager = Manager()
    manager.load_config(config_path)
    config = manager.config

    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})

    dist_ctx = init_distributed(backend="nccl")
    is_rank0 = is_main_process(dist_ctx)

    if is_rank0:
        print("[sft] config loaded", flush=True)
        print(f"[sft] distributed: rank={dist_ctx.rank} world_size={dist_ctx.world_size}", flush=True)

    tokenizer = load_tokenizer(config)
    if is_rank0:
        print("[sft] tokenizer loaded", flush=True)

    processed_data = process_all_sources(config, tokenizer)
    train_records = processed_data.get("train")
    if train_records is None:
        raise ValueError("No 'train' source found in config")

    if is_rank0:
        print(f"[sft] train records: {len(train_records)}", flush=True)

    pad_token_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
    train_loader = build_dataloader(
        train_records,
        config,
        dist_ctx,
        shuffle=True,
        pad_token_id=pad_token_id,
    )

    toolbox = ToolBox(manager)
    models = toolbox.load_models("policy")

    engine = toolbox.engine
    if engine is None:
        raise ValueError("DeepSpeed engine not initialized")

    if is_rank0:
        print("[sft] DeepSpeed engine initialized", flush=True)

    metrics_cfg = runtime_cfg.get("metrics", {})
    perf_logger = StepMetricsLogger.from_config(metrics_cfg)
    metrics_emitter = MetricsEmitter.from_config(metrics_cfg)

    log_every = int(train_cfg.get("log_every_steps", 10))
    total_steps = int(train_cfg.get("max_steps", 1000))
    checkpoint_every = int(train_cfg.get("checkpoint_every_steps", 0))
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))

    if checkpoint_every > 0 and is_rank0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"[sft] checkpoint_dir={checkpoint_dir}", flush=True)

    step = 0
    try:
        for epoch_idx in range(100):
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch_idx)

            for batch in train_loader:
                if step >= total_steps:
                    break

                step += 1
                start_time = time.perf_counter()

                device_batch = move_to_device(batch, dist_ctx.device)
                input_ids, attention_mask, labels = build_sft_batch(
                    device_batch, tokenizer, dist_ctx.device
                )

                outputs = engine.module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss.mean()

                engine.backward(loss)
                engine.step()

                elapsed = time.perf_counter() - start_time

                metrics = {
                    "loss/sft": float(loss.detach().item()),
                    "train/step": step,
                    "train/epoch": epoch_idx,
                }

                perf_metrics = perf_logger.update(
                    step_time_sec=elapsed,
                    batch=device_batch,
                    device=dist_ctx.device,
                    sync_global=(step % log_every == 0),
                )
                metrics.update(perf_metrics)

                if step % log_every == 0:
                    reduced = reduce_metrics(metrics, dist_ctx)
                    if is_rank0:
                        metrics_emitter.emit(step_id=step, metrics=reduced)

                if checkpoint_every > 0 and step % checkpoint_every == 0 and is_rank0:
                    ckpt_path = checkpoint_dir / f"step_{step}.pt"
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": engine.module.state_dict(),
                            "optimizer_state_dict": engine.optimizer.state_dict(),
                        },
                        ckpt_path,
                    )
                    print(f"[sft] checkpoint saved: {ckpt_path}", flush=True)

            if step >= total_steps:
                break

    finally:
        cleanup_distributed()

    if is_rank0:
        print(f"[sft] training finished: total_steps={step}", flush=True)


if __name__ == "__main__":
    main()