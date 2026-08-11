"""Neural-network builder and pose modules."""

from __future__ import annotations


def __getattr__(name: str):
    if name in {"BaseModel", "DetectionModel", "PoseModel", "load_checkpoint"}:
        from . import tasks

        return getattr(tasks, name)
    raise AttributeError(name)


__all__ = ["BaseModel", "DetectionModel", "PoseModel", "load_checkpoint"]
