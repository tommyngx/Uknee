"""Standalone training entry point for knee bounding-box detection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from collections import Counter, defaultdict
from copy import copy
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import yaml

# Support both ``python -m landmark.train_det`` and an absolute train_det.py path.
PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uknee_cli import gpu_ids_to_device, parse_gpu_ids, parse_image_size, safe_run_name


DEFAULT_CFG = PACKAGE_ROOT / "cfg" / "detect" / "kneelocation.yaml"
DEFAULT_DATA = PACKAGE_ROOT / "cfg" / "datasets" / "kneelocation.yaml"
MODEL_ROOT = PACKAGE_ROOT / "cfg" / "models"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


# Apply the requested GPU visibility before importing the torch-backed runtime.
_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument("--gpu", type=parse_gpu_ids, default=None)
_gpu_args, _ = _gpu_parser.parse_known_args()
if _gpu_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "" if _gpu_args.gpu == [-1] else ",".join(map(str, _gpu_args.gpu))
    )

import numpy as np
import torch
from PIL import Image

from landmark.core import RANK
from landmark.core.detect import DetectionTrainer, DetectionValidator
from landmark.core.torch_utils import unwrap_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an Uknee YOLO knee detector")
    parser.add_argument("--config", default=str(DEFAULT_CFG), help="Base training configuration")
    parser.add_argument("--model", default="yolo26-detect", help="Detection YAML or .pt checkpoint")
    parser.add_argument("--data", "--dataset", dest="data", default=str(DEFAULT_DATA), help="Detection dataset YAML")
    parser.add_argument("--project", default=str(REPOSITORY_ROOT), help="Output root; runs are written below it")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", "--imgz", "--img_size", dest="imgsz", type=parse_image_size, default=None)
    parser.add_argument("--batch", "--batch_size", dest="batch", type=int, default=None)
    parser.add_argument("--base_lr", type=float, default=None)
    parser.add_argument("--gpu", type=parse_gpu_ids, default=[0])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", default="")
    parser.add_argument("--exist_ok", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", nargs="?", const=True, default=False)
    parser.add_argument("--pretrained", nargs="?", const=True, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--cache", nargs="?", const=True, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--auto-export-onnx", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="Additional trainer overrides, e.g. optimizer=AdamW weight_decay=0.001",
    )
    return parser


def _load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    values = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise TypeError(f"Expected a YAML mapping in {source}")
    return values


def _parse_overrides(values: Iterable[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE override, got {value!r}")
        key, raw = value.split("=", 1)
        if not key or key.startswith("_"):
            raise ValueError(f"Invalid override key: {key!r}")
        overrides[key] = yaml.safe_load(raw)
    return overrides


def resolve_model_source(value: str | Path) -> Path:
    supplied = Path(value).expanduser()
    candidates = [supplied]
    if not supplied.suffix:
        candidates.extend((MODEL_ROOT / f"{supplied.name}.yaml", MODEL_ROOT / f"{supplied.name.replace('_', '-')}.yaml"))
    elif not supplied.is_absolute():
        candidates.append(MODEL_ROOT / supplied.name)
    model = next((candidate for candidate in candidates if candidate.is_file()), None)
    if model is None:
        raise FileNotFoundError(f"Detection model {value!r} not found; checked {', '.join(map(str, candidates))}")
    return model.resolve()


def _dataset_root(source_yaml: Path, configured: str | Path | None) -> Path:
    configured = Path(configured or source_yaml.parent).expanduser()
    candidates = [configured] if configured.is_absolute() else [REPOSITORY_ROOT / configured, source_yaml.parent / configured]
    root = next((candidate for candidate in candidates if candidate.is_dir()), None)
    if root is None:
        raise FileNotFoundError(f"Dataset root {configured} does not exist")
    return root.resolve()


def _images(root: Path, entries: str | Iterable[str]) -> list[Path]:
    entries = [entries] if isinstance(entries, str) else list(entries)
    resolved: list[Path] = []
    for entry in entries:
        source = Path(entry)
        source = source if source.is_absolute() else root / source
        if source.is_dir():
            resolved.extend(path.resolve() for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        elif source.is_file() and source.suffix.lower() == ".txt":
            for line in source.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    path = Path(line.strip())
                    resolved.append((path if path.is_absolute() else root / path).resolve())
        elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
            resolved.append(source.resolve())
    return sorted(dict.fromkeys(resolved))


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    if "images" not in parts:
        raise ValueError(f"Image path must contain an images directory: {image}")
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _identity_key(path: Path) -> str:
    """Join obvious alternate acquisitions while leaving unrelated cases separate."""
    stem = path.stem.upper().removesuffix("FLIP")
    stem = re.sub(r"KNEE\d+$", "", stem)
    stem = re.sub(r"^(\d+)P2([FM]\d+)$", r"\1\2", stem)
    return stem.rstrip("_-") or path.stem.upper()


def _validate_labels(images: list[Path], names: dict[int, str]) -> tuple[dict[Path, tuple[int, ...]], dict[str, Any]]:
    classes: Counter[int] = Counter()
    signatures: dict[Path, tuple[int, ...]] = {}
    boundary_overflow = 0
    missing: list[Path] = []
    for image in images:
        label = _label_path(image)
        if not label.is_file():
            missing.append(label)
            continue
        image_classes: list[int] = []
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label}:{line_number} must contain class + normalized xywh; got {len(fields)} values")
            try:
                values = [float(value) for value in fields]
            except ValueError as error:
                raise ValueError(f"Non-numeric label at {label}:{line_number}") from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite label at {label}:{line_number}")
            class_value, x, y, width, height = values
            if not class_value.is_integer() or int(class_value) not in names:
                raise ValueError(f"Unsupported class {class_value} at {label}:{line_number}; expected {sorted(names)}")
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                raise ValueError(f"Invalid normalized xywh at {label}:{line_number}: {values[1:]}")
            if x - width / 2 < 0 or y - height / 2 < 0 or x + width / 2 > 1 or y + height / 2 > 1:
                boundary_overflow += 1
            class_id = int(class_value)
            image_classes.append(class_id)
            classes[class_id] += 1
        if len(image_classes) != len(set(image_classes)):
            raise ValueError(f"Duplicate left/right class in {label}; expected at most one box per side")
        signatures[image] = tuple(sorted(image_classes))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} labels, examples: {missing[:3]}")
    absent = set(names) - set(classes)
    if absent:
        raise ValueError(f"Dataset has no instances for classes {sorted(absent)}")
    return signatures, {
        "images": len(images),
        "instances": sum(classes.values()),
        "class_instances": {names[index]: classes[index] for index in sorted(names)},
        "boundary_overflow_boxes": boundary_overflow,
    }


def _split_groups(images: list[Path], signatures: dict[Path, tuple[int, ...]], fraction: float, seed: int) -> tuple[list[Path], list[Path], int]:
    parent = {image: image for image in images}

    def find(image: Path) -> Path:
        while parent[image] != image:
            parent[image] = parent[parent[image]]
            image = parent[image]
        return image

    def union(left: Path, right: Path) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    tokens: dict[tuple[str, str], Path] = {}
    for image in images:
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        for token in (("case", _identity_key(image)), ("content", digest)):
            if token in tokens:
                union(image, tokens[token])
            else:
                tokens[token] = image

    groups: dict[Path, list[Path]] = defaultdict(list)
    for image in images:
        groups[find(image)].append(image)
    strata: dict[tuple[int, ...], list[list[Path]]] = defaultdict(list)
    for paths in groups.values():
        signature = tuple(sorted({class_id for path in paths for class_id in signatures[path]}))
        strata[signature].append(sorted(paths))

    validation: set[Path] = set()
    for signature, candidates in sorted(strata.items()):
        ordered = sorted(
            candidates,
            key=lambda paths: hashlib.sha256(f"{seed}:{signature}:{paths[0]}".encode()).hexdigest(),
        )
        count = min(len(ordered), max(1, round(len(ordered) * fraction)))
        for paths in ordered[:count]:
            validation.update(paths)
    train = [image for image in images if image not in validation]
    val = [image for image in images if image in validation]
    if not train or not val:
        raise ValueError("Unable to create non-empty train/validation splits")
    return train, val, len(groups)


def prepare_detection_dataset(data: str | Path, project: str | Path) -> tuple[Path, dict[str, Any]]:
    source_yaml = Path(data).expanduser().resolve()
    metadata = _load_yaml(source_yaml)
    raw_names = metadata.get("names")
    if isinstance(raw_names, list):
        names = dict(enumerate(map(str, raw_names)))
    elif isinstance(raw_names, dict):
        names = {int(index): str(name) for index, name in raw_names.items()}
    else:
        raise ValueError(f"{source_yaml} must define detection class names")
    if sorted(names) != list(range(len(names))):
        raise ValueError(f"Class indices must be contiguous from zero, got {sorted(names)}")
    root = _dataset_root(source_yaml, metadata.get("path"))
    train_source = _images(root, metadata.get("train", "images/train"))
    val_source = _images(root, metadata.get("val", "images/val"))
    if not train_source:
        raise ValueError(f"No training images found under {root}")

    all_images = sorted(set(train_source) | set(val_source))
    signatures, audit = _validate_labels(all_images, names)
    if not val_source or set(train_source) == set(val_source):
        train_images, val_images, groups = _split_groups(
            train_source,
            signatures,
            float(metadata.get("val_fraction", 0.15)),
            int(metadata.get("split_seed", 2026)),
        )
    else:
        train_images, val_images = train_source, val_source
        train_tokens = {_identity_key(path) for path in train_images}
        overlap = train_tokens & {_identity_key(path) for path in val_images}
        if overlap:
            raise ValueError(f"Train/validation case leakage detected: {sorted(overlap)[:5]}")
        groups = len({_identity_key(path) for path in all_images})

    cache_root = Path(project).expanduser().resolve() / ".uknee" / "datasets"
    digest = hashlib.sha256(
        (repr(metadata) + "\n" + "\n".join(map(str, train_images + val_images))).encode()
    ).hexdigest()[:16]
    output = cache_root / f"{safe_run_name(metadata.get('dataset_name', source_yaml.stem))}_{digest}"
    output.mkdir(parents=True, exist_ok=True)
    train_manifest, val_manifest = output / "train.txt", output / "val.txt"
    train_manifest.write_text("\n".join(map(str, train_images)) + "\n", encoding="utf-8")
    val_manifest.write_text("\n".join(map(str, val_images)) + "\n", encoding="utf-8")
    resolved = dict(metadata)
    resolved.update(path=str(root), train=str(train_manifest), val=str(val_manifest), test=str(val_manifest), names=names)
    resolved_yaml = output / "dataset_resolved.yaml"
    resolved_yaml.write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    audit.update(
        train_images=len(train_images),
        val_images=len(val_images),
        train_class_instances={
            names[index]: sum(index in signatures[image] for image in train_images) for index in sorted(names)
        },
        val_class_instances={
            names[index]: sum(index in signatures[image] for image in val_images) for index in sorted(names)
        },
        groups=groups,
        source_yaml=str(source_yaml),
        resolved_yaml=str(resolved_yaml),
    )
    return resolved_yaml, audit


def _xray_detection_defaults() -> dict[str, Any]:
    # Horizontal flip is intentionally disabled: classes encode anatomical side.
    return {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.25,
        "degrees": 2.0,
        "translate": 0.08,
        "scale": 0.25,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "bgr": 0.0,
        "mosaic": 0.5,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 15,
        "plots": True,
    }


class DetectionReportValidator(DetectionValidator):
    """Detection validator that writes one real four-image sample grid per epoch."""

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self._sample_records: list[dict[str, Any]] = []

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        super().update_metrics(preds, batch)
        if RANK not in {-1, 0} or len(self._sample_records) >= 4:
            return
        for index, pred in enumerate(preds):
            if len(self._sample_records) >= 4:
                break
            prepared = self._prepare_batch(index, batch)
            scaled = self.scale_preds(pred, prepared)
            try:
                with Image.open(prepared["im_file"]) as source:
                    image = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
            except Exception:
                image = (
                    batch["img"][index]
                    .detach()
                    .float()
                    .clamp(0, 1)
                    .cpu()
                    .permute(1, 2, 0)
                    .numpy()
                )
                scaled = pred
            confidence = scaled["conf"].detach().float().cpu()
            keep = confidence >= max(float(self.args.conf or 0.001), 0.25)
            boxes = scaled["bboxes"].detach().float().cpu()[keep]
            if boxes.numel():
                boxes[:, [0, 2]].clamp_(0, image.shape[1])
                boxes[:, [1, 3]].clamp_(0, image.shape[0])
            self._sample_records.append(
                {
                    "path": str(batch["im_file"][index]),
                    "image": image,
                    "boxes": boxes.numpy(),
                    "classes": scaled["cls"].detach().long().cpu()[keep].tolist(),
                    "scores": confidence[keep].tolist(),
                    "names": dict(self.names),
                }
            )

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        if RANK in {-1, 0} and self._sample_records:
            map50 = float(stats.get("metrics/mAP50(B)", 0.0))
            map5095 = float(stats.get("metrics/mAP50-95(B)", 0.0))
            for record in self._sample_records:
                record.update(map50=map50, map5095=map5095)
            from landmark.core.plotting_det import plot_detection_validation_samples

            epoch = max(int(getattr(self, "_report_epoch", 1)), 1)
            plot_detection_validation_samples(
                self._sample_records,
                self.save_dir / "samples" / f"detection_sample_e{epoch}.png",
                epoch=epoch,
            )
            self._sample_paths = [Path(record["path"]).name for record in self._sample_records]
        return stats


class DetectionReportTrainer(DetectionTrainer):
    """Detection trainer retaining the native loop while adding report artifacts."""

    def get_validator(self) -> DetectionReportValidator:
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        validator = DetectionReportValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
        validator._report_epoch = max(int(getattr(self, "epoch", 0)) + 1, 1)
        return validator

    def validate(self):
        self.validator._report_epoch = max(int(self.epoch) + 1, 1)
        return super().validate()


def _clean_row(row: dict[str, str]) -> dict[str, float | int]:
    cleaned: dict[str, float | int] = {}
    for key, raw in row.items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        cleaned[key.strip()] = int(value) if key.strip() == "epoch" else value
    return cleaned


def _scalar_metrics(values: Any) -> dict[str, float]:
    if not isinstance(values, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in values.items()
        if isinstance(value, (int, float)) or hasattr(value, "item")
    }


def write_detection_summary(
    save_dir: Path,
    model_path: Path,
    audit: dict[str, Any],
    onnx: Path | None = None,
    elapsed_seconds: float | None = None,
    trainer: DetectionTrainer | None = None,
) -> Path:
    csv_path = save_dir / "results.csv"
    rows: list[dict[str, float | int]] = []
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = [_clean_row(row) for row in csv.DictReader(stream)]
    metric = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda row: float(row.get(metric, float("-inf")))) if rows else {}
    final = rows[-1] if rows else {}

    if csv_path.is_file():
        try:
            from landmark.core.plotting_det import plot_dashboard_detection, plot_detection_metrics

            plot_dashboard_detection(
                csv_path,
                save_dir / "detection_dashboard.png",
                model_name=model_path.stem,
                elapsed_seconds=elapsed_seconds,
            )
            class_names = list(audit.get("class_instances", {}).keys()) or ["RightKnee", "LeftKnee"]
            plot_detection_metrics(
                getattr(getattr(trainer, "validator", None), "metrics", None),
                save_dir / "detection_metrics.png",
                model_name=model_path.stem,
                elapsed_seconds=elapsed_seconds,
                epochs_completed=len(rows),
                class_names=class_names,
            )
        except Exception as error:
            print(f"Warning: Failed to generate detection report plots: {error}")

    plot_files = sorted(
        str(path.relative_to(save_dir))
        for pattern in ("*.png", "*.jpg")
        for path in save_dir.glob(pattern)
        if path.is_file()
    )
    sample_paths = list(getattr(getattr(trainer, "validator", None), "_sample_paths", []))
    sample_files = sorted((save_dir / "samples").glob("detection_sample_e*.png"))
    args = getattr(trainer, "args", None)
    image_size = parse_image_size(getattr(args, "imgsz", [640, 640]))
    height, width = image_size
    model = unwrap_model(getattr(trainer, "model", None)) if trainer is not None else None
    parameters = sum(parameter.numel() for parameter in model.parameters()) if model is not None else None
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) if model is not None else None
    try:
        from landmark.core.torch_utils import get_flops

        gflops = float(get_flops(model, imgsz=image_size)) if model is not None else None
    except Exception:
        gflops = None
    epochs_completed = len(rows)
    duration = float(elapsed_seconds or 0.0)
    final_validation = _scalar_metrics(getattr(trainer, "metrics", {}))

    onnx_record: dict[str, Any] = {"status": "disabled"}
    if onnx and onnx.is_file():
        from landmark.core.exporter import onnx_sha256, read_onnx_metadata

        metadata = read_onnx_metadata(onnx)
        onnx_record = {
            "status": "ready",
            "path": str(onnx.relative_to(save_dir)) if onnx.is_relative_to(save_dir) else str(onnx),
            "format": "onnx",
            "sha256": onnx_sha256(onnx),
            "file_size_bytes": onnx.stat().st_size,
            "metadata": metadata,
        }
    artifacts = {
        key: value
        for key, value in {
            "best_checkpoint": "weights/best.pt" if (save_dir / "weights/best.pt").is_file() else None,
            "last_checkpoint": "weights/last.pt" if (save_dir / "weights/last.pt").is_file() else None,
            "metrics": "results.csv" if csv_path.is_file() else None,
            "dashboard": "detection_dashboard.png" if (save_dir / "detection_dashboard.png").is_file() else None,
            "metric_report": "detection_metrics.png" if (save_dir / "detection_metrics.png").is_file() else None,
            "training_plot": "results.png" if (save_dir / "results.png").is_file() else None,
            "labels_plot": "labels.jpg" if (save_dir / "labels.jpg").is_file() else None,
            "confusion_matrix": "confusion_matrix.png" if (save_dir / "confusion_matrix.png").is_file() else None,
            "plots": plot_files or None,
            "onnx_model": str(onnx.relative_to(save_dir)) if onnx and onnx.is_relative_to(save_dir) else str(onnx) if onnx else None,
        }.items()
        if value is not None
    }
    summary = {
        "schema_version": 2,
        "task": "knee_detection",
        "model": {
            "source": str(model_path),
            "name": model_path.stem,
            "parameters": parameters,
            "trainable_parameters": trainable,
            "gflops": gflops,
            "gflops_convention": "2 x MACs for one forward pass",
            "input_shape": [1, 3, height, width],
        },
        "dataset": {
            **audit,
            "config": audit.get("source_yaml"),
            "classes": {index: name for index, name in enumerate(audit.get("class_instances", {}))},
        },
        "preprocessing": {
            "schema_version": 1,
            "source_spatial_shape": "dynamic",
            "network_input_shape": [1, 3, height, width],
            "layout": "NCHW",
            "dtype": "float32",
            "color_space": "RGB",
            "value_range": [0.0, 1.0],
            "normalization": {"mode": "scale_0_1", "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
            "resize": {
                "mode": "letterbox",
                "keep_aspect_ratio": True,
                "target_height": height,
                "target_width": width,
                "pad_value": 114,
                "placement": "center",
                "stride": 32,
            },
        },
        "training": {
            "epochs_requested": int(getattr(args, "epochs", epochs_completed)),
            "epochs_completed": epochs_completed,
            "batch_size": int(getattr(args, "batch", 0)),
            "seed": int(getattr(args, "seed", 0)),
            "optimizer": type(getattr(trainer, "optimizer", None)).__name__ if trainer is not None else None,
            "initial_learning_rate": float(
                getattr(trainer, "optimizer", None).param_groups[0].get("initial_lr", getattr(args, "lr0", 0.0))
                if trainer is not None and getattr(trainer, "optimizer", None) is not None
                else getattr(args, "lr0", 0.0)
            ),
            "configured_initial_learning_rate": float(getattr(args, "lr0", 0.0)),
            "duration_seconds": round(duration, 3),
            "duration_hours": round(duration / 3600.0, 6),
            "seconds_per_epoch": round(duration / max(epochs_completed, 1), 3),
            "device": str(getattr(trainer, "device", "")),
            "torch_version": str(torch.__version__),
        },
        "performance": {
            "selection_metric": metric,
            "selection_mode": "max",
            "best_epoch": best.get("epoch"),
            "best": best,
            "final": final,
            "best_checkpoint_validation": final_validation,
        },
        "deployment": {
            "auto_export_onnx": bool(getattr(args, "auto_export_onnx", False)),
            "onnx": onnx_record,
        },
        "artifacts": artifacts,
    }
    summary["artifacts"].update(
        {
            "samples": "samples/detection_sample_e{epoch}.png" if sample_files else None,
            "samples_per_epoch": len(sample_paths),
            "sample_seed": int(getattr(args, "seed", 2026)),
            "sample_paths": sample_paths,
            "sample_output_width": 800,
        }
    )
    summary["artifacts"] = {key: value for key, value in summary["artifacts"].items() if value is not None}
    path = save_dir / "summary.yaml"
    path.write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> Any:
    import time

    parsed = vars(build_parser().parse_args(argv))
    project = Path(parsed.pop("project")).expanduser().resolve()
    model_path = resolve_model_source(parsed.pop("model"))
    dataset_yaml, audit = prepare_detection_dataset(parsed.pop("data"), project)
    overrides = _parse_overrides(parsed.pop("overrides"))
    config = {**_xray_detection_defaults(), **_load_yaml(parsed.pop("config"))}
    for key in ("task", "mode", "model", "data", "source", "save_dir"):
        config.pop(key, None)
    config.update({key: value for key, value in parsed.items() if value is not None})
    config.update(overrides)
    config.update(
        task="detect",
        data=str(dataset_yaml),
        project=str(project / "runs"),
        name=safe_run_name(parsed.get("name") or f"{model_path.stem}_kneelocation"),
        device=gpu_ids_to_device(parsed["gpu"]),
        gpu_ids=list(parsed["gpu"]),
        imgsz=parse_image_size(config.get("imgsz", 640)),
        auto_export_onnx=bool(config.get("auto_export_onnx", False)),
    )
    if parsed.get("base_lr") is not None:
        config["lr0"] = parsed["base_lr"]
        if str(config.get("optimizer", "auto")).lower() == "auto":
            config["optimizer"] = "AdamW"
    for key in ("base_lr", "gpu"):
        config.pop(key, None)
    # Non-square detection remains valid, but mosaic/multi-scale require a square canvas in this runtime.
    if config["imgsz"][0] != config["imgsz"][1]:
        config.update(mosaic=0.0, multi_scale=0.0, rect=False)

    if audit["boundary_overflow_boxes"]:
        print(
            f"Dataset audit: {audit['boundary_overflow_boxes']} boxes touch/cross an image edge; "
            "the loader will clip them during augmentation. Source labels were not modified."
        )
    print(
        f"Dataset audit: {audit['images']} images, {audit['instances']} boxes, "
        f"train={audit['train_images']}, val={audit['val_images']}, classes={audit['class_instances']}"
    )

    from landmark.core.model import YOLO

    start_time = time.time()
    model = YOLO(str(model_path), verbose=False)
    metrics = model.train(trainer=DetectionReportTrainer, **config)
    elapsed_seconds = time.time() - start_time
    save_dir = Path(model.trainer.save_dir)
    onnx_path: Path | None = None
    if config.get("auto_export_onnx") and model.trainer.best.is_file():
        from landmark.core.exporter import Exporter
        from landmark.utils.exporting import KneeDetectionExportWrapper

        onnx_path = save_dir / "weights" / f"{model_path.stem.replace('-', '_')}.onnx"
        # Export the checkpoint selected by mAP50-95, not the in-memory last epoch.
        best_model = YOLO(str(model.trainer.best), verbose=False)
        wrapper = KneeDetectionExportWrapper(best_model.model, confidence=0.25).eval()
        onnx_path = Exporter(
            {
                "format": "onnx",
                "imgsz": config["imgsz"],
                "batch": 1,
                "dynamic": bool(config.get("dynamic", False)),
                "simplify": bool(config.get("simplify", True)),
                "max_det": int(config.get("max_det", 300)),
                "path": onnx_path,
                "model_name": model_path.stem,
                "source_checkpoint": "weights/best.pt",
            }
        )(wrapper)
    summary = write_detection_summary(
        save_dir,
        model_path,
        audit,
        onnx_path,
        elapsed_seconds=elapsed_seconds,
        trainer=model.trainer,
    )
    print(f"Detection summary: {summary}")
    return metrics


if __name__ == "__main__":
    main()
