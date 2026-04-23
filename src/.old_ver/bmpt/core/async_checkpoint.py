from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, cast

import torch


class AsyncCheckpointWriter:
    def __init__(self, max_pending: int = 1) -> None:
        if max_pending < 1:
            raise ValueError("`max_pending` must be >= 1")
        self._queue: queue.Queue[tuple[Path, dict[str, Any]] | object] = queue.Queue(
            maxsize=max_pending
        )
        self._stop_sentinel = object()
        self._closed = False
        self._error: Exception | None = None
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _set_error(self, exc: Exception) -> None:
        with self._lock:
            if self._error is None:
                self._error = exc

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("asynchronous checkpoint writer failed") from self._error

    def _atomic_save(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._stop_sentinel:
                    return
                record = cast(tuple[Path, dict[str, Any]], item)
                save_path, payload = record
                start_time = time.perf_counter()
                self._atomic_save(save_path, payload)
                elapsed = time.perf_counter() - start_time
                print(f"[ckpt] saved {save_path} in {elapsed:.2f}s", flush=True)
            except Exception as exc:
                self._set_error(exc)
            finally:
                self._queue.task_done()

    def enqueue(self, path: str | Path, payload: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("asynchronous checkpoint writer has been closed")
        self._raise_if_error()
        item = (Path(path), payload)
        while True:
            try:
                self._queue.put_nowait(item)
                break
            except queue.Full:
                dropped = self._queue.get_nowait()
                self._queue.task_done()
                if dropped is self._stop_sentinel:
                    self._queue.put_nowait(dropped)
                    raise RuntimeError("asynchronous checkpoint writer is stopping")
        self._raise_if_error()

    def flush(self) -> None:
        self._raise_if_error()
        self._queue.join()
        self._raise_if_error()

    def close(self) -> None:
        if self._closed:
            self._raise_if_error()
            return
        self._closed = True
        flush_error: Exception | None = None
        try:
            self.flush()
        except Exception as exc:
            flush_error = exc

        while True:
            try:
                self._queue.put_nowait(self._stop_sentinel)
                break
            except queue.Full:
                self._queue.get_nowait()
                self._queue.task_done()

        self._worker.join()
        if flush_error is not None:
            raise flush_error
        self._raise_if_error()
