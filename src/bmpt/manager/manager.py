from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from bmpt.data.source_loader import build_source_dataloaders
from bmpt.distributed.worker_manager import spawn_worker_processes
from bmpt.manager.config_manager import LoadedConfig, load_config_bundle
from bmpt.prompt.composer_manager import Composer, build_composers_from_config


class Manager:
    def __init__(self) -> None:
        self.config_path: Path | None = None
        self.deepspeed_config_path: Path | None = None
        self.config: dict[str, Any] = {}
        self.deepspeed_config: dict[str, Any] = {}
        self._shared_manager: mp.managers.SyncManager | None = None
        self._shared_config: Any | None = None

    def load_config(self, config_path: str | Path) -> dict[str, Any]:
        bundle: LoadedConfig = load_config_bundle(config_path)
        self.config_path = bundle.config_path
        self.deepspeed_config_path = bundle.deepspeed_config_path
        self.config = bundle.config
        self.deepspeed_config = bundle.deepspeed_config
        return self.config

    def build_source_dataloaders(self) -> dict[str, DataLoader]:
        if not self.config:
            raise ValueError("Manager config is empty. Call Manager.load_config(...) first")
        return build_source_dataloaders(self.config)

    def load_composers(self) -> dict[str, Composer]:
        if not self.config:
            raise ValueError("Manager config is empty. Call Manager.load_config(...) first")
        return build_composers_from_config(self.config)

    def spawn_worker(
        self,
        def_worker: Any,
        preserved_worker: tuple[Any, ...] | list[Any] | None = None,
        worker_args: tuple[Any, ...] | None = None,
        worker_kwargs: dict[str, Any] | None = None,
    ) -> list[mp.Process]:
        if not self.config:
            raise ValueError("Manager config is empty. Call Manager.load_config(...) first")

        if self._shared_manager is None:
            self._shared_manager = mp.Manager()
        self._shared_config = self._shared_manager.dict(self.config)

        return spawn_worker_processes(
            def_worker=def_worker,
            config=self._shared_config,
            preserved_worker=preserved_worker,
            worker_args=worker_args,
            worker_kwargs=worker_kwargs,
        )
