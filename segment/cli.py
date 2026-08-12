"""CLI contract shared by segment training commands and tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from uknee_cli import parse_gpu_ids, parse_image_size, resolve_dataset_path, resolve_project_root


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "cfg" / "default.yaml"


def _str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, received: {value}")


def _load_defaults(config_path: str | Path) -> dict:
    path = Path(config_path).expanduser().resolve()
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return values


def build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    values = _load_defaults(DEFAULT_CONFIG)
    values.update(defaults or {})
    parser = argparse.ArgumentParser(description="Train an Uknee medical segmentation model")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Default YAML; explicit CLI values win")
    parser.add_argument("--model", default=values["model"], help="Name in segment.models.MODEL_REGISTRY")
    parser.add_argument("--project", default=values.get("project", ""), help="Project root; outputs go to <project>/runs/<name>")
    parser.add_argument("--dataset", "--base_dir", dest="dataset", default=values["dataset"], help="Dataset folder, absolute path, or /short-name")
    parser.add_argument("--dataset_name", default=values.get("dataset_name", ""))
    parser.add_argument("--train_file_dir", default=values["train_file_dir"])
    parser.add_argument("--val_file_dir", default=values["val_file_dir"])
    parser.add_argument("--base_lr", type=float, default=values["base_lr"])
    parser.add_argument("--batch", "--batch_size", dest="batch", type=int, default=values["batch"])
    parser.add_argument("--workers", type=int, default=values["workers"])
    parser.add_argument("--gpu", type=parse_gpu_ids, default=parse_gpu_ids(values["gpu"]))
    parser.add_argument("--epochs", "--max_epochs", dest="epochs", type=int, default=values["epochs"])
    parser.add_argument("--seed", type=int, default=values["seed"])
    parser.add_argument("--imgsz", "--imgz", "--img_size", dest="imgsz", type=parse_image_size, default=parse_image_size(values["imgsz"]))
    parser.add_argument("--num_classes", type=int, default=values["num_classes"])
    parser.add_argument("--input_channel", type=int, default=values["input_channel"])
    parser.add_argument("--aug_strategy", choices=("auto", "none", "basic", "standard", "strong", "xray"), default=values["aug_strategy"])
    parser.add_argument("--resume", action="store_true", default=values["resume"])
    parser.add_argument("--pretrained_path", default=values["pretrained_path"])
    parser.add_argument("--name", "--exp_name", dest="name", default=values["name"])
    parser.add_argument("--output_dir", default="", help="Legacy runs-root override")
    parser.add_argument("--exist_ok", action=argparse.BooleanOptionalAction, default=values.get("exist_ok", True))
    parser.add_argument("--pixel_spacing_mm", type=float, default=values["pixel_spacing_mm"])
    parser.add_argument("--auto_export_onnx", "--auto-export-onnx", dest="auto_export_onnx", action=argparse.BooleanOptionalAction, default=values["auto_export_onnx"])
    parser.add_argument("--onnx_export_interval", "--onnx-export-interval", dest="onnx_export_interval", type=int, default=values.get("onnx_export_interval", 10))
    parser.add_argument("--zero_shot_base_dir", default=values["zero_shot_base_dir"])
    parser.add_argument("--zero_shot_dataset_name", default=values["zero_shot_dataset_name"])
    parser.add_argument("--do_deeps", type=_str2bool, nargs="?", const=True, default=values["do_deeps"])
    parser.add_argument("--model_id", type=int, default=values["model_id"])
    parser.add_argument("--just_for_test", type=_str2bool, nargs="?", const=True, default=values["just_for_test"])
    parser.add_argument("--just_for_zero_shot", type=_str2bool, nargs="?", const=True, default=values["just_for_zero_shot"])
    return parser


def parse_segment_args(argv: list[str] | None = None):
    argv_values = list(argv) if argv is not None else None
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    preliminary, _ = pre_parser.parse_known_args(argv_values)
    defaults = _load_defaults(preliminary.config)
    args = build_parser(defaults).parse_args(argv_values)
    args.project = str(resolve_project_root(args.project, REPOSITORY_ROOT))
    dataset = resolve_dataset_path(args.dataset, args.project)
    args.dataset = str(dataset)
    args.base_dir = str(dataset)  # legacy dataloaders
    import sys

    effective_argv = argv_values if argv_values is not None else sys.argv[1:]
    dataset_name_is_explicit = any(
        token == "--dataset_name" or token.startswith("--dataset_name=") for token in effective_argv
    )
    args.dataset_name = args.dataset_name if dataset_name_is_explicit else dataset.name
    args.imgsz = parse_image_size(args.imgsz)
    args.img_size = list(args.imgsz)  # legacy model/dataloader name, always [H, W]
    args.gpu_ids = parse_gpu_ids(args.gpu)
    args.gpu = ",".join(map(str, args.gpu_ids)) if args.gpu_ids != [-1] else ""
    args.batch_size = int(args.batch)
    args.max_epochs = int(args.epochs)
    args.exp_name = args.name
    return args


__all__ = ["DEFAULT_CONFIG", "build_parser", "parse_segment_args"]
