"""Pose-only TorchScript and ONNX exporter."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from landmark.core.export_utils import best_onnx_opset, trace_torchscript


def portable_model_stem(value: str) -> str:
    """Return a filesystem-safe model stem with separators normalized to underscores."""
    stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(value).stem).strip("_")
    return (stem or "landmark_model").lower()


def pose_onnx_metadata(model: torch.nn.Module, args: dict[str, Any], height: int, width: int) -> dict[str, str]:
    model_name = str(
        args.get("model_name")
        or Path(getattr(model, "yaml_file", getattr(model, "yaml", {}).get("yaml_file", "landmark-pose"))).stem
    )
    stride_value = getattr(model, "stride", 32)
    if isinstance(stride_value, torch.Tensor):
        stride_value = stride_value.max().item()
    elif isinstance(stride_value, (list, tuple)):
        stride_value = max(stride_value)
    return {
        "uknee.schema_version": "1",
        "uknee.task": "landmark_detection",
        "uknee.model_name": model_name,
        "uknee.source_checkpoint": str(args.get("source_checkpoint") or "weights/best.pt"),
        "uknee.opset": str(int(args.get("opset") or best_onnx_opset())),
        "uknee.preprocess": json.dumps(
            {
                "schema_version": 1,
                "source_spatial_shape": "dynamic",
                "network_input_shape": [1, 3, int(height), int(width)],
                "image_size_hw": [int(height), int(width)],
                "spatial_order": "height_width",
                "onnx_dynamic_axes": {"batch": bool(args.get("dynamic")), "height": False, "width": False},
                "layout": "NCHW",
                "dtype": "float32",
                "color_space": "RGB",
                "value_range": [0.0, 1.0],
                "normalization": {"mode": "scale_0_1", "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
                "resize": {
                    "mode": "letterbox",
                    "keep_aspect_ratio": True,
                    "target_height": int(height),
                    "target_width": int(width),
                    "pad_value": 114,
                    "placement": "center",
                    "stride": int(max(float(stride_value), 1.0)),
                },
            },
            separators=(",", ":"),
        ),
        "uknee.output": json.dumps(
            {
                "detections": ["batch", 4, 159],
                "num_detections": ["batch"],
                "canonical": ["batch", 129, 3],
                "keypoint_format": "x_y_confidence",
                "coordinate_space": "letterboxed_input_pixels",
            },
            separators=(",", ":"),
        ),
    }


def write_onnx_metadata(path: str | Path, metadata: dict[str, str]) -> None:
    import onnx

    path = Path(path)
    graph = onnx.load(str(path))
    retained = [(item.key, item.value) for item in graph.metadata_props if item.key not in metadata]
    del graph.metadata_props[:]
    for key, value in [*retained, *sorted(metadata.items())]:
        item = graph.metadata_props.add()
        item.key = key
        item.value = str(value)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))


def read_onnx_metadata(path: str | Path) -> dict[str, str]:
    import onnx

    graph = onnx.load(str(path), load_external_data=False)
    return {item.key: item.value for item in graph.metadata_props}


def onnx_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            raise ValueError("landmark exports only 'torchscript' and 'onnx'")
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
        stem = Path(getattr(model, "yaml_file", getattr(model, "yaml", {}).get("yaml_file", "landmark-pose"))).stem
        destination = Path(
            self.args.get("path") or f"{portable_model_stem(stem)}.{format_name}"
        ).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if format_name == "torchscript":
            metadata = json.dumps({"task": "pose", "stride": 32, "kpt_shape": [51, 3]})
            traced = trace_torchscript(model, image)
            torch.jit.save(traced, str(destination), _extra_files={"config.txt": metadata})
        else:
            try:
                with torch.no_grad():
                    reference = tuple(value.detach().cpu().numpy() for value in model(image))
                torch.onnx.export(
                    model,
                    image,
                    str(destination),
                    input_names=["images"],
                    output_names=["detections", "num_detections", "canonical"],
                    opset_version=int(self.args.get("opset") or best_onnx_opset()),
                    dynamic_axes={
                        "images": {0: "batch"},
                        "detections": {0: "batch"},
                        "num_detections": {0: "batch"},
                        "canonical": {0: "batch"},
                    }
                    if self.args.get("dynamic")
                    else None,
                    dynamo=False,
                )
                import onnxruntime as ort

                actual = ort.InferenceSession(
                    str(destination), providers=["CPUExecutionProvider"]
                ).run(None, {"images": image.detach().cpu().numpy()})
                differences = []
                for index, (expected, observed) in enumerate(zip(reference, actual)):
                    if index == 1:
                        np.testing.assert_array_equal(observed, expected.astype(observed.dtype, copy=False))
                    else:
                        np.testing.assert_allclose(observed, expected, rtol=2e-3, atol=2e-4)
                    differences.append(float(np.abs(observed.astype(float) - expected.astype(float)).max(initial=0.0)))
                metadata = pose_onnx_metadata(model, self.args, int(height), int(width))
                metadata["uknee.parity"] = json.dumps(
                    {
                        "validated": True,
                        "provider": "CPUExecutionProvider",
                        "max_abs_diff": max(differences, default=0.0),
                    },
                    separators=(",", ":"),
                )
                write_onnx_metadata(destination, metadata)
            except Exception:
                destination.unlink(missing_ok=True)
                raise
        return destination


__all__ = [
    "Exporter",
    "export_formats",
    "onnx_sha256",
    "pose_onnx_metadata",
    "portable_model_stem",
    "read_onnx_metadata",
    "write_onnx_metadata",
]
