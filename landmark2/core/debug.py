# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Opt-in crash tracing for the isolated OA26 v9 training path."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch


def debug_enabled() -> bool:
    """Return whether v9 crash tracing is enabled."""
    return os.getenv("YOLO_V9_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


def debug_event(stage: str, **fields) -> None:
    """Flush a timestamped RAM/VRAM marker to stdout and the optional debug log."""
    if not debug_enabled():
        return
    if os.getenv("YOLO_V9_DEBUG_SYNC", "1").lower() not in {"0", "false", "no", "off"}:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    try:
        import psutil

        ram_gib = psutil.Process(os.getpid()).memory_info().rss / (1 << 30)
    except Exception:
        ram_gib = -1.0
    parts = [f"[v9-debug] {time.strftime('%H:%M:%S')}", f"pid={os.getpid()}", f"stage={stage}", f"ram={ram_gib:.3f}GiB"]
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        parts.extend(
            (
                f"cuda_alloc={torch.cuda.memory_allocated() / (1 << 30):.3f}GiB",
                f"cuda_reserved={torch.cuda.memory_reserved() / (1 << 30):.3f}GiB",
                f"cuda_peak={torch.cuda.max_memory_allocated() / (1 << 30):.3f}GiB",
            )
        )
    parts.extend(f"{key}={value}" for key, value in fields.items())
    message = " ".join(parts)
    print(message, flush=True)
    log_path = os.getenv("YOLO_V9_DEBUG_FILE")
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(message + "\n")
            file.flush()
            os.fsync(file.fileno())


def mark_backward(tensor: torch.Tensor, stage: str, **fields) -> torch.Tensor:
    """Attach a no-op autograd hook that records when backward reaches a tensor."""
    if debug_enabled() and tensor.requires_grad:
        tensor.register_hook(lambda gradient: (debug_event(stage, grad_shape=tuple(gradient.shape), **fields), gradient)[1])
    return tensor

