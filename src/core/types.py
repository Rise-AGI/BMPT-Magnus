from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class StepContext:
    mode: str
    global_step: int
    runtime_config: dict[str, Any]
    full_config: dict[str, Any]
