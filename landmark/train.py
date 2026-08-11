"""Command-line entry point for standalone Uknee landmark training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from uknee_cli import (
    first_existing,
    gpu_ids_to_device,
    parse_gpu_ids,
    parse_image_size,
    resolve_dataset_path,
    resolve_project_root,
    safe_run_name,
)

from .utils.api import KneePose


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DEFAULT_CFG = PACKAGE_ROOT / "cfg" / "default.yaml"
MODEL_ROOT = PACKAGE_ROOT / "cfg" / "models"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an Uknee landmark pose model")
    parser.add_argument("--config", default=str(DEFAULT_CFG), help="Default YAML; explicit CLI values win")
    parser.add_argument("--model", required=True, help="Model YAML name/path or .pt checkpoint")
    parser.add_argument("--project", default=str(REPOSITORY_ROOT), help="Project root; outputs go to <project>/runs/<name>")
    parser.add_argument("--dataset", "--data", dest="dataset", required=True, help="Dataset folder/YAML, absolute path, or /short-name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", "--imgz", "--img_size", dest="imgsz", type=parse_image_size)
    parser.add_argument("--batch", "--batch_size", dest="batch", type=int)
    parser.add_argument("--base_lr", type=float, help="Initial learning rate (maps to lr0)")
    parser.add_argument("--gpu", type=parse_gpu_ids, default=[0], help="GPU list such as [0] or [0,1]; [-1] selects CPU")
    parser.add_argument("--device", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--aug_strategy", choices=("auto", "none", "basic", "standard", "strong", "xray"), default="xray")
    parser.add_argument("--name", default="", help="Run folder under <project>/runs")
    parser.add_argument("--exist_ok", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", nargs="?", const=True, default=False)
    parser.add_argument("--pretrained", nargs="?", const=True, default=None)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--cache", nargs="?", const=True, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--auto-export-onnx", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="Additional trainer overrides, e.g. optimizer=AdamW weight_decay=0.001",
    )
    return parser


def _parse_overrides(values: list[str]) -> dict[str, Any]:
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
        candidates.extend(
            (
                MODEL_ROOT / f"{supplied.name}.yaml",
                MODEL_ROOT / f"{supplied.name.replace('_', '-')}.yaml",
            )
        )
    elif not supplied.is_absolute():
        candidates.append(MODEL_ROOT / supplied.name)
    model_path = first_existing(candidates)
    if model_path is None:
        raise FileNotFoundError(f"Model {value!r} was not found. Checked: {', '.join(map(str, candidates))}")
    return model_path.resolve()


def resolve_dataset_config(value: str | Path, project_root: str | Path) -> Path:
    """Resolve a YAML or make a portable runtime YAML for a dataset folder."""
    dataset_path = resolve_dataset_path(value, project_root, must_exist=True)
    if dataset_path.is_file():
        if dataset_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"Landmark dataset config must be YAML: {dataset_path}")
        return dataset_path

    preferred = [
        dataset_path / "data.yaml",
        dataset_path / "dataset.yaml",
        dataset_path / f"{dataset_path.name}.yaml",
    ]
    source_yaml = first_existing(preferred) or first_existing(sorted(dataset_path.glob("*.y*ml")))
    if source_yaml:
        metadata = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
        if not isinstance(metadata, dict):
            raise TypeError(f"Expected a YAML mapping in {source_yaml}")
    else:
        metadata = {
            "dataset_name": dataset_path.name,
            "train": "images/train",
            "val": "images/val" if (dataset_path / "images" / "val").is_dir() else "images/train",
            "test": "images/val" if (dataset_path / "images" / "val").is_dir() else "images/train",
            "channels": 3,
            "names": {0: "femur", 1: "tibia", 2: "fibula", 3: "patella"},
            "kpt_shape": [51, 3],
            "flip_idx": [],
            "region_keypoint_counts": [45, 51, 24, 9],
            "val_fraction": 0.15,
            "split_seed": 2006,
        }
    metadata["path"] = str(dataset_path.resolve())
    runtime_root = Path(project_root).resolve() / ".uknee" / "datasets"
    runtime_root.mkdir(parents=True, exist_ok=True)
    runtime_yaml = runtime_root / f"{safe_run_name(dataset_path.name)}_landmark.yaml"
    runtime_yaml.write_text(yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return runtime_yaml


def _load_training_config(path: str | Path) -> dict[str, Any]:
    values = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return values


def main(argv: list[str] | None = None) -> Any:
    parsed = vars(build_parser().parse_args(argv))
    project_root = resolve_project_root(parsed.pop("project"), REPOSITORY_ROOT)
    model_path = resolve_model_source(parsed.pop("model"))
    dataset_yaml = resolve_dataset_config(parsed.pop("dataset"), project_root)
    overrides = _parse_overrides(parsed.pop("overrides"))
    config_path = parsed.pop("config")

    train_args = _load_training_config(config_path)
    for key in ("task", "mode", "model", "data", "source", "save_dir"):
        train_args.pop(key, None)
    train_args.update({key: value for key, value in parsed.items() if value is not None})
    train_args["project"] = str(project_root / "runs")
    train_args["name"] = safe_run_name(parsed["name"] or f"{model_path.stem}_{dataset_yaml.stem}")
    train_args["device"] = parsed["device"] or gpu_ids_to_device(parsed["gpu"])
    train_args["gpu_ids"] = list(parsed["gpu"])
    train_args["imgsz"] = parse_image_size(train_args.get("imgsz", 640))
    if parsed.get("base_lr") is not None:
        train_args["lr0"] = parsed["base_lr"]
    train_args.pop("base_lr", None)
    train_args.pop("gpu", None)
    train_args.update(overrides)

    model = KneePose(model_path)
    return model.train(data=dataset_yaml, **train_args)


if __name__ == "__main__":
    main()
