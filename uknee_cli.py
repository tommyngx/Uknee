"""Shared command-line normalization for Uknee training entry points."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable


_HW_PATTERN = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")


def parse_image_size(value: object) -> list[int]:
    """Return a canonical ``[height, width]`` pair.

    A scalar is a square canvas. Rectangular strings always use HxW order.
    """
    if isinstance(value, int):
        values = [value, value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
        if len(values) == 1:
            values *= 2
    else:
        raw = str(value).strip()
        match = _HW_PATTERN.match(raw)
        if match:
            values = [int(match.group(1)), int(match.group(2))]
        else:
            cleaned = raw.strip("[]() ")
            parts = [part.strip() for part in cleaned.split(",") if part.strip()]
            try:
                values = [int(raw)] * 2 if len(parts) == 1 else [int(part) for part in parts]
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid image size {value!r}; use 640 or HxW, for example 540x640."
                ) from exc
    if len(values) != 2 or any(int(item) <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            f"Invalid image size {value!r}; expected one positive value or [height, width]."
        )
    return [int(values[0]), int(values[1])]


def format_image_size(value: object) -> str:
    height, width = parse_image_size(value)
    return f"{height}x{width}"


def parse_gpu_ids(value: object) -> list[int]:
    """Accept ``[0,1]``, ``0,1``, a scalar, or an integer sequence."""
    if isinstance(value, int):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        cleaned = str(value).strip().strip("[]() ")
        if not cleaned:
            return []
        try:
            values = [int(part.strip()) for part in cleaned.split(",") if part.strip()]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid GPU list {value!r}; use [0], [0,1], or 0,1."
            ) from exc
    ids = [int(item) for item in values]
    if len(set(ids)) != len(ids) or any(item < -1 for item in ids) or (-1 in ids and len(ids) > 1):
        raise argparse.ArgumentTypeError(f"Invalid GPU list {value!r}.")
    return ids


def gpu_ids_to_device(value: object) -> str:
    ids = parse_gpu_ids(value)
    return "cpu" if ids == [-1] or not ids else ",".join(map(str, ids))


def resolve_project_root(value: str | Path | None, repository_root: str | Path) -> Path:
    path = Path(value).expanduser() if value else Path(repository_root)
    return path.resolve()


def resolve_dataset_path(
    value: str | Path,
    project_root: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve absolute/relative datasets and the ``/short-name`` convention."""
    raw = str(value).strip()
    if not raw:
        raise ValueError("Dataset path cannot be empty.")
    project_root = Path(project_root).expanduser().resolve()
    supplied = Path(raw).expanduser()
    candidates: list[Path] = []
    if supplied.is_absolute():
        candidates.append(supplied)
        # A single missing component such as /mesko is deliberate shorthand.
        if len(supplied.parts) == 2:
            candidates.append(project_root / "data" / supplied.name)
    else:
        candidates.extend((supplied, project_root / "data" / supplied, project_root / supplied))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    fallback = candidates[-1] if supplied.is_absolute() and len(supplied.parts) == 2 else candidates[0]
    if not supplied.is_absolute() and len(supplied.parts) == 1:
        fallback = project_root / "data" / supplied
    if must_exist:
        checked = ", ".join(str(item.resolve()) for item in candidates)
        raise FileNotFoundError(f"Dataset {value!r} was not found. Checked: {checked}")
    return fallback.resolve()


def safe_run_name(value: object) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return normalized or "run"


def first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


__all__ = [
    "first_existing",
    "format_image_size",
    "gpu_ids_to_device",
    "parse_gpu_ids",
    "parse_image_size",
    "resolve_dataset_path",
    "resolve_project_root",
    "safe_run_name",
]
