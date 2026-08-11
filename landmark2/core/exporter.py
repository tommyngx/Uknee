"""Pose-only TorchScript and ONNX exporter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from landmark2.core.export_utils import best_onnx_opset, trace_torchscript


def export_formats() -> dict[str, list[str]]:
    return {
        "Format": ["PyTorch", "TorchScript", "ONNX"],
        "Argument": ["-", "torchscript", "onnx"],
        "Suffix": [".pt", ".torchscript", ".onnx"],
    }


class Exporter:
    """Export a fixed-output knee-pose wrapper without optional platform backends."""

    def __init__(self, overrides: dict[str, Any] | None = None, _callbacks=None) -> None:
        self.args = dict(overrides or {})
        self.callbacks = _callbacks or {}

    def __call__(self, model: torch.nn.Module) -> Path:
        format_name = str(self.args.get("format", "torchscript")).lower()
        if format_name not in {"torchscript", "onnx"}:
            raise ValueError("landmark2 exports only 'torchscript' and 'onnx'")
        imgsz = self.args.get("imgsz", 640)
        height, width = ((imgsz, imgsz) if isinstance(imgsz, int) else tuple(imgsz))
        batch = int(self.args.get("batch", 1))
        device = next(model.parameters()).device
        image = torch.zeros(batch, 3, int(height), int(width), device=device)
        model = model.eval()
        for module in model.modules():
            if hasattr(module, "export"):
                module.export = True
                module.format = format_name
                module.dynamic = bool(self.args.get("dynamic"))
                module.max_det = int(self.args.get("max_det", 4))
        stem = Path(getattr(model, "yaml_file", getattr(model, "yaml", {}).get("yaml_file", "landmark2-pose"))).stem
        destination = Path(self.args.get("path") or f"{stem}.{format_name}").expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if format_name == "torchscript":
            metadata = json.dumps({"task": "pose", "stride": 32, "kpt_shape": [51, 3]})
            traced = trace_torchscript(model, image)
            torch.jit.save(traced, str(destination), _extra_files={"config.txt": metadata})
        else:
            torch.onnx.export(
                model,
                image,
                str(destination),
                input_names=["images"],
                output_names=["detections", "num_detections", "canonical"],
                opset_version=int(self.args.get("opset") or best_onnx_opset()),
                dynamic_axes={"images": {0: "batch"}} if self.args.get("dynamic") else None,
                dynamo=False,
            )
        return destination


__all__ = ["Exporter", "export_formats"]
