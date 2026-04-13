from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


StepFn = Callable[[Any, dict[str, Any]], dict[str, Any]]


@dataclass(slots=True)
class EngineState:
    global_step: int = 0
    micro_step: int = 0
    optimizer_step: int = 0


class TrainingEngine:
    def __init__(
        self,
        step_fn: StepFn,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.LambdaLR | None = None,
        grad_accum_steps: int = 1,
        grad_clip_norm: float | None = None,
        use_amp: bool = False,
        scaler: torch.cuda.amp.GradScaler | None = None,
    ) -> None:
        self.step_fn = step_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip_norm = grad_clip_norm
        self.use_amp = use_amp
        self.scaler = scaler
        self.state = EngineState()

    def _backward(self, loss: torch.Tensor) -> None:
        scaled_loss = loss / float(self.grad_accum_steps)
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
            return
        scaled_loss.backward()

    def _optimizer_step(self) -> None:
        if self.use_amp and self.scaler is not None:
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for group in self.optimizer.param_groups for p in group["params"]],
                    self.grad_clip_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in self.optimizer.param_groups for p in group["params"]],
                    self.grad_clip_norm,
                )
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        self.state.optimizer_step += 1

    def run_micro_step(self, models: Any, batch: dict[str, Any], extra_input: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "batch": batch,
            "global_step": self.state.global_step,
        }
        if extra_input:
            payload.update(extra_input)

        step_output = self.step_fn(models, payload)
        loss = step_output["loss"]
        self._backward(loss)

        self.state.micro_step += 1
        should_step = (self.state.micro_step % self.grad_accum_steps) == 0
        if should_step:
            self._optimizer_step()
            self.state.global_step += 1

        metrics = dict(step_output.get("metrics", {}))
        metrics["engine/should_step"] = 1 if should_step else 0
        metrics["engine/global_step"] = self.state.global_step
        metrics["engine/micro_step"] = self.state.micro_step
        metrics["engine/optimizer_step"] = self.state.optimizer_step

        return {
            "loss": loss,
            "metrics": metrics,
            "aux": step_output.get("aux", {}),
            "should_step": should_step,
        }
