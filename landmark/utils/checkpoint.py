from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary")
    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("Could not locate a model state dict in checkpoint")


def _strip_prefix_if_present(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    for prefix in ("module.", "model."):
        state_dict = _strip_prefix_if_present(state_dict, prefix)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return {
        "checkpoint": checkpoint,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
    tracking: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric,
            "config": config,
            "tracking": tracking or {},
        },
        path,
    )
