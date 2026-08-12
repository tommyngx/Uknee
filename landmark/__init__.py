"""Compact, self-contained MESKO4GF2 landmark pose package."""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "2.0.0"

if TYPE_CHECKING:
    from .utils.api import KneePose


def __getattr__(name: str):
    """Load the PyTorch runtime lazily so CLI GPU visibility can be set first."""
    if name == "KneePose":
        from .utils.api import KneePose

        return KneePose
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["KneePose", "__version__"]
