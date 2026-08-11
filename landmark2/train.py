"""Command-line entry point for standalone Uknee pose training."""

from __future__ import annotations

import argparse
from typing import Any

import yaml

from .utils.api import KneePose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an Uknee landmark pose model")
    parser.add_argument("--model", required=True, help="Model YAML or .pt checkpoint")
    parser.add_argument("--data", required=True, help="YOLO-Pose dataset YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", nargs="?", const=True, default=False)
    parser.add_argument("--pretrained", nargs="?", const=True, default=False)
    parser.add_argument("--seed", type=int, default=2006)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--cache", nargs="?", const=True, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="Additional vendored Ultralytics overrides, e.g. optimizer=AdamW lr0=0.001",
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


def main(argv: list[str] | None = None) -> Any:
    args = vars(build_parser().parse_args(argv))
    model = KneePose(args.pop("model"))
    data = args.pop("data")
    args.update(_parse_overrides(args.pop("overrides")))
    if args.get("device") is None:
        args.pop("device")
    return model.train(data=data, **args)


if __name__ == "__main__":
    main()
