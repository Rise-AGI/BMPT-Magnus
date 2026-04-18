from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


@dataclass(slots=True)
class MetricsEmitter:
    targets: list[str]

    @classmethod
    def from_config(cls, metrics_cfg: dict[str, Any]) -> "MetricsEmitter":
        raw_output = metrics_cfg.get("output", [])
        targets: list[str] = []
        if raw_output is None:
            return cls(targets=[])
        if isinstance(raw_output, str):
            if raw_output.strip() != "":
                targets = [raw_output.strip()]
        elif isinstance(raw_output, list):
            for item in raw_output:
                text = str(item).strip()
                if text != "":
                    targets.append(text)
        else:
            raise ValueError("runtime.metrics.output must be string, list, or null")
        return cls(targets=targets)

    def emit(self, step_id: int, metrics: dict[str, float]) -> None:
        if len(self.targets) == 0:
            return
        for target in self.targets:
            if target == "stdout":
                print(f"step={step_id} metrics={metrics}")
                continue
            if target.startswith("file:"):
                file_path = Path(target[len("file:") :]).expanduser()
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with file_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"step={step_id} metrics={metrics}\n")
                continue
            raise ValueError(f"Unsupported runtime.metrics.output target: {target}")


@dataclass(slots=True)
class StepMetricsLogger:
    enabled: bool
    global_throughput: bool
    window_size: int
    step_time_ms_window: deque[float] = field(init=False, repr=False)
    tokens_per_sec_window: deque[float] = field(init=False, repr=False)
    samples_per_sec_window: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        maxlen = max(int(self.window_size), 1)
        self.step_time_ms_window = deque(maxlen=maxlen)
        self.tokens_per_sec_window = deque(maxlen=maxlen)
        self.samples_per_sec_window = deque(maxlen=maxlen)

    @classmethod
    def from_config(cls, metrics_cfg: dict[str, Any]) -> "StepMetricsLogger":
        return cls(
            enabled=bool(metrics_cfg.get("enabled", True)),
            global_throughput=bool(metrics_cfg.get("global_throughput", True)),
            window_size=int(metrics_cfg.get("window_size", 20)),
        )

    def _local_batch_stats(self, batch: dict[str, Any]) -> tuple[float, float]:
        input_ids = batch.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            return 0.0, 0.0

        local_samples = float(input_ids.shape[0])
        attention_mask = batch.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            local_tokens = float(attention_mask.to(dtype=torch.float32).sum().item())
        else:
            local_tokens = float(input_ids.numel())
        return local_tokens, local_samples

    def _global_stats(
        self,
        step_time_sec: float,
        local_tokens: float,
        local_samples: float,
        device: torch.device,
        sync_global: bool,
    ) -> tuple[float, float, float]:
        if not sync_global:
            return step_time_sec, local_tokens, local_samples
        if not self.global_throughput:
            return step_time_sec, local_tokens, local_samples
        if not dist.is_available() or not dist.is_initialized():
            return step_time_sec, local_tokens, local_samples

        time_tensor = torch.tensor(step_time_sec, dtype=torch.float64, device=device)
        tokens_tensor = torch.tensor(local_tokens, dtype=torch.float64, device=device)
        samples_tensor = torch.tensor(local_samples, dtype=torch.float64, device=device)

        dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)

        return (
            float(time_tensor.item()),
            float(tokens_tensor.item()),
            float(samples_tensor.item()),
        )

    def update(
        self,
        *,
        step_time_sec: float,
        batch: dict[str, Any],
        device: torch.device,
        sync_global: bool = True,
    ) -> dict[str, float]:
        if not self.enabled:
            return {}

        local_tokens, local_samples = self._local_batch_stats(batch)
        global_time_sec, global_tokens, global_samples = self._global_stats(
            step_time_sec=step_time_sec,
            local_tokens=local_tokens,
            local_samples=local_samples,
            device=device,
            sync_global=sync_global,
        )

        safe_time = max(global_time_sec, 1.0e-12)
        step_time_ms = global_time_sec * 1000.0
        tokens_per_sec = global_tokens / safe_time
        samples_per_sec = global_samples / safe_time

        self.step_time_ms_window.append(step_time_ms)
        self.tokens_per_sec_window.append(tokens_per_sec)
        self.samples_per_sec_window.append(samples_per_sec)

        return {
            "perf/step_time_ms": float(step_time_ms),
            "perf/step_time_ms_avg": float(sum(self.step_time_ms_window) / len(self.step_time_ms_window)),
            "perf/tokens_per_sec": float(tokens_per_sec),
            "perf/tokens_per_sec_avg": float(sum(self.tokens_per_sec_window) / len(self.tokens_per_sec_window)),
            "perf/samples_per_sec": float(samples_per_sec),
            "perf/samples_per_sec_avg": float(sum(self.samples_per_sec_window) / len(self.samples_per_sec_window)),
        }
