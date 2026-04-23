"""Debug utilities for BMPT."""

from __future__ import annotations

from typing import Any

import torch.distributed as dist


def is_rank0() -> bool:
    """Check if current process is rank 0."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def debug_print(enabled: bool, message: str) -> None:
    """Print debug message if enabled and on rank 0.

    Args:
        enabled: Whether debug output is enabled.
        message: Message to print.
    """
    if not enabled:
        return
    if not is_rank0():
        return
    print(message, flush=True)


def _debug_print(config: dict[str, Any], message: str) -> None:
    """Print debug message if runtime.debug is enabled in config.

    Args:
        config: Configuration dictionary (expects runtime.debug field).
        message: Message to print.
    """
    enabled = bool(config.get("runtime", {}).get("debug", False))
    debug_print(enabled, message)
