"""Compact configuration parsing for detect/pose workflows."""

from __future__ import annotations

import difflib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from landmark.core import (
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    FLOAT_OR_INT,
    LOGGER,
    RANK,
    ROOT,
    RUNS_DIR,
    STR_OR_PATH,
    TESTS_RUNNING,
    YAML,
    IterableSimpleNamespace,
)

TASKS = frozenset({"detect", "pose"})
MODES = frozenset({"train", "val", "predict", "export"})
TASK2DATA = {"detect": "coco8.yaml", "pose": "coco8-pose.yaml"}
TASK2METRIC = {"detect": "metrics/mAP50-95(B)", "pose": "metrics/mAP50-95(P)"}
_YOLO_CLI_COMMAND = [sys.executable, "-m", "landmark.train"]

CFG_FLOAT_KEYS = frozenset(
    {"warmup_epochs", "box", "cls", "cls_pw", "dfl", "time", "batch", "degrees", "shear"}
)
CFG_FRACTION_KEYS = frozenset(
    {
        "lr0", "lrf", "momentum", "weight_decay", "warmup_momentum", "warmup_bias_lr",
        "hsv_h", "hsv_s", "hsv_v", "translate", "perspective", "flipud", "fliplr",
        "bgr", "mosaic", "mixup", "cutmix", "copy_paste", "conf", "iou", "fraction", "multi_scale",
    }
)
CFG_INT_KEYS = frozenset(
    {"epochs", "patience", "workers", "seed", "close_mosaic", "mask_ratio", "max_det", "vid_stride", "line_width", "nbs", "save_period"}
)
CFG_BOOL_KEYS = frozenset(
    {
        "save", "exist_ok", "verbose", "deterministic", "single_cls", "rect", "cos_lr",
        "overlap_mask", "val", "save_json", "dnn", "plots", "show", "save_txt",
        "save_conf", "save_crop", "save_frames", "show_labels", "show_conf", "visualize",
        "augment", "agnostic_nms", "retina_masks", "show_boxes", "optimize", "dynamic",
        "simplify", "nms", "profile", "end2end",
    }
)


def cfg2dict(cfg: str | Path | dict | SimpleNamespace) -> dict[str, Any]:
    if isinstance(cfg, STR_OR_PATH):
        cfg = YAML.load(cfg)
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)
    if not isinstance(cfg, dict):
        raise TypeError(f"Configuration must be a mapping, got {type(cfg).__name__}")
    return dict(cfg)


def check_dict_alignment(base: dict, custom: dict, e: Exception | None = None, allowed_custom_keys=None) -> None:
    aliases = {
        "boxes": "show_boxes",
        "line_thickness": "line_width",
        "hide_labels": "show_labels",
        "hide_conf": "show_conf",
    }
    for old, new in aliases.items():
        if old in custom:
            value = custom.pop(old)
            custom[new] = not bool(value) if old.startswith("hide_") else value
    allowed = set(base) | set(allowed_custom_keys or ())
    invalid = set(custom) - allowed
    if invalid:
        lines = []
        for key in sorted(invalid):
            matches = difflib.get_close_matches(key, base, n=1)
            hint = f" Did you mean '{matches[0]}'?" if matches else ""
            lines.append(f"Unknown landmark argument '{key}'.{hint}")
        raise SyntaxError("\n".join(lines)) from e


def check_cfg(cfg: dict[str, Any], hard: bool = True) -> None:
    for key, value in cfg.items():
        if value is None:
            continue
        if key in CFG_FLOAT_KEYS and not isinstance(value, FLOAT_OR_INT):
            if hard:
                raise TypeError(f"'{key}' must be int or float, got {type(value).__name__}")
            cfg[key] = float(value)
        elif key == "scale":
            if isinstance(value, (list, tuple)):
                if len(value) != 2 or not all(isinstance(item, FLOAT_OR_INT) for item in value):
                    raise TypeError("'scale' must be a number or a two-number range")
            elif not isinstance(value, FLOAT_OR_INT):
                raise TypeError("'scale' must be a number or a two-number range")
        elif key in CFG_FRACTION_KEYS:
            if not isinstance(value, FLOAT_OR_INT):
                if hard:
                    raise TypeError(f"'{key}' must be int or float, got {type(value).__name__}")
                value = cfg[key] = float(value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"'{key}' must be between 0 and 1, got {value}")
        elif key in CFG_INT_KEYS and not isinstance(value, int):
            if hard:
                raise TypeError(f"'{key}' must be int, got {type(value).__name__}")
            cfg[key] = int(value)
        elif key in CFG_BOOL_KEYS and not isinstance(value, bool):
            if hard:
                raise TypeError(f"'{key}' must be bool, got {type(value).__name__}")
            cfg[key] = bool(value)
        elif key == "quantize" and str(value).lower() not in {"8", "16", "32", "int8", "fp16", "fp32"}:
            raise ValueError("'quantize' must be one of 8, 16, 32, int8, fp16 or fp32")


def get_cfg(
    cfg: str | Path | dict | SimpleNamespace = DEFAULT_CFG_DICT,
    overrides: dict | SimpleNamespace | None = None,
) -> IterableSimpleNamespace:
    values = cfg2dict(cfg)
    if overrides:
        custom = cfg2dict(overrides)
        check_dict_alignment(values, custom)
        values.update(custom)
    for key in ("project", "name"):
        if isinstance(values.get(key), FLOAT_OR_INT):
            values[key] = str(values[key])
    check_cfg(values)
    return IterableSimpleNamespace(**values)


def get_save_dir(args: SimpleNamespace, name: str | None = None) -> Path:
    if getattr(args, "save_dir", None):
        return Path(args.save_dir).expanduser().resolve()
    from landmark.core.files import increment_path

    project = Path(args.project) if args.project else RUNS_DIR / args.task
    if not project.is_absolute():
        base = ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else Path.cwd()
        worker = os.environ.get("PYTEST_XDIST_WORKER")
        if worker and TESTS_RUNNING:
            base /= worker
        project = base / project
    run_name = name or args.name or args.mode
    return increment_path(project / run_name, exist_ok=args.exist_ok if RANK in {-1, 0} else True).resolve()


__all__ = [
    "CFG_BOOL_KEYS", "CFG_FLOAT_KEYS", "CFG_FRACTION_KEYS", "CFG_INT_KEYS", "DEFAULT_CFG",
    "IterableSimpleNamespace", "MODES", "TASK2DATA", "TASK2METRIC", "TASKS", "cfg2dict",
    "check_cfg", "check_dict_alignment", "get_cfg", "get_save_dir",
]
