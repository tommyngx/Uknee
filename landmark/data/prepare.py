"""Validate MESKO YOLO-Pose data and create leakage-safe split manifests."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import yaml

from .schema import REGION_KEYPOINT_COUNTS, REGION_NAMES, validate_region_schema


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _atomic_text(path: Path, value: str) -> None:
    """Atomically publish deterministic manifests under multi-process launch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class PreparedDataset:
    yaml_path: Path
    source_yaml: Path
    root: Path
    train_images: tuple[Path, ...]
    val_images: tuple[Path, ...]


def _resolve_root(source_yaml: Path, configured: str | Path) -> Path:
    configured = Path(configured).expanduser()
    candidates = [configured]
    if not configured.is_absolute():
        candidates = [REPOSITORY_ROOT / configured, source_yaml.parent / configured]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Dataset root {configured!s} from {source_yaml} does not exist; "
        f"also checked relative to {REPOSITORY_ROOT}."
    )


def _resolve_images(root: Path, entries: str | Iterable[str]) -> list[Path]:
    entries = [entries] if isinstance(entries, str) else list(entries)
    images: list[Path] = []
    for entry in entries:
        path = Path(entry)
        path = path if path.is_absolute() else root / path
        if path.is_dir():
            images.extend(
                item.resolve()
                for item in sorted(path.iterdir())
                if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                value = line.strip()
                if value:
                    item = Path(value)
                    images.append((item if item.is_absolute() else root / item).resolve())
        elif path.is_file():
            images.append(path.resolve())
    if not images:
        raise ValueError(f"No images resolved from {entries!r} under {root}")
    return sorted(dict.fromkeys(images))


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" not in parts:
        raise ValueError(f"Image path must contain an 'images' directory: {image_path}")
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _group_key(path: Path) -> str:
    stem = path.stem.removesuffix("Flip")
    for suffix in ("_L", "_R"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _validate_label(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing YOLO-Pose label: {path}")
    rows: dict[int, list[str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) != 5 + 51 * 3:
            raise ValueError(
                f"{path}:{line_number} must contain class, box and 51x3 keypoints; "
                f"got {len(values)} values"
            )
        class_id = int(float(values[0]))
        if class_id not in range(4):
            raise ValueError(f"Unsupported class {class_id} at {path}:{line_number}")
        if class_id in rows:
            raise ValueError(f"Duplicate class {class_id} at {path}:{line_number}")
        rows[class_id] = values
        keypoints = torch.tensor([float(value) for value in values[5:]]).view(51, 3)
        visible = keypoints[:, 2] > 0
        required = REGION_KEYPOINT_COUNTS[class_id]
        if int(visible.sum()) != required or not visible[:required].all() or visible[required:].any():
            raise ValueError(
                f"{path}:{line_number} class {class_id} must expose exactly "
                f"{required} leading keypoints"
            )
        if not torch.isfinite(keypoints).all():
            raise ValueError(f"Non-finite keypoints at {path}:{line_number}")
    if tuple(sorted(rows)) != (0, 1, 2, 3):
        raise ValueError(f"{path} must contain exactly one row for each class 0..3")


def prepare_dataset(
    yaml_path: str | Path,
    *,
    cache_root: str | Path | None = None,
) -> PreparedDataset:
    """Return an absolute, validated dataset YAML accepted by Ultralytics."""
    source_yaml = Path(yaml_path).expanduser().resolve()
    metadata = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(metadata, dict):
        raise TypeError(f"Dataset YAML must contain a mapping: {source_yaml}")
    validate_region_schema(len(metadata.get("names", {})), metadata.get("kpt_shape", ()))
    names = tuple(metadata["names"][index] for index in range(4))
    if names != REGION_NAMES:
        raise ValueError(f"Expected class order {REGION_NAMES}, got {names}")
    counts = tuple(metadata.get("region_keypoint_counts", REGION_KEYPOINT_COUNTS))
    if counts != REGION_KEYPOINT_COUNTS:
        raise ValueError(f"Expected region_keypoint_counts={REGION_KEYPOINT_COUNTS}, got {counts}")

    root = _resolve_root(source_yaml, metadata.get("path", source_yaml.parent))
    train_images = _resolve_images(root, metadata.get("train", "images/train"))
    val_images = _resolve_images(root, metadata.get("val", "images/val"))
    for image in sorted(set(train_images) | set(val_images)):
        _validate_label(_label_path(image))

    if set(train_images) == set(val_images):
        groups: dict[str, list[Path]] = {}
        for image in train_images:
            groups.setdefault(_group_key(image), []).append(image)
        group_names = sorted(groups)
        seed = int(metadata.get("split_seed", 2006))
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(group_names), generator=generator).tolist()
        val_count = max(1, round(len(group_names) * float(metadata.get("val_fraction", 0.15))))
        val_groups = {group_names[index] for index in order[:val_count]}
        val_images = [image for group, paths in groups.items() if group in val_groups for image in paths]
        train_images = [image for group, paths in groups.items() if group not in val_groups for image in paths]

    train_groups = {_group_key(path) for path in train_images}
    val_groups = {_group_key(path) for path in val_images}
    overlap = train_groups & val_groups
    if overlap:
        raise ValueError(f"Train/validation case leakage detected, examples: {sorted(overlap)[:3]}")

    digest_input = "\n".join(map(str, train_images + val_images)) + repr(metadata)
    digest = hashlib.sha256(digest_input.encode()).hexdigest()[:16]
    cache_root = Path(cache_root) if cache_root else REPOSITORY_ROOT / "landmark" / ".cache" / "datasets"
    output = cache_root / digest
    output.mkdir(parents=True, exist_ok=True)
    train_manifest, val_manifest = output / "train.txt", output / "val.txt"
    _atomic_text(train_manifest, "\n".join(map(str, train_images)) + "\n")
    _atomic_text(val_manifest, "\n".join(map(str, val_images)) + "\n")

    resolved = dict(metadata)
    resolved.update(
        {
            "path": str(root),
            "train": str(train_manifest),
            "val": str(val_manifest),
            "test": str(val_manifest),
        }
    )
    resolved_yaml = output / "dataset_resolved.yaml"
    _atomic_text(resolved_yaml, yaml.safe_dump(resolved, sort_keys=False))
    return PreparedDataset(
        yaml_path=resolved_yaml,
        source_yaml=source_yaml,
        root=root,
        train_images=tuple(train_images),
        val_images=tuple(val_images),
    )
