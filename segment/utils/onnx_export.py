"""ONNX export contract for the supported RWKV segmentation models."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

from segment.utils.preprocessing import resolve_target_hw


SUPPORTED_AUTO_EXPORT_MODELS = frozenset({"RWKV_UNetV3", "RWKV_UNetV6"})
ONNX_OPSET = 17


class SegmentationONNXWrapper(nn.Module):
    """Expose one stable logits tensor regardless of the training head contract."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images)
        if isinstance(output, dict):
            return output["out"] if "out" in output else next(iter(output.values()))
        if isinstance(output, (list, tuple)):
            return output[-1]
        return output


def onnx_filename(model_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", str(model_name).strip()).strip("_")
    return f"{(stem or 'segment_model').lower()}.onnx"


def segment_preprocess_schema(args) -> dict:
    height, width = resolve_target_hw(getattr(args, "img_size", 256))
    channels = int(getattr(args, "input_channel", 3))
    return {
        "schema_version": 1,
        "source_spatial_shape": "dynamic",
        "network_input_shape": [1, channels, height, width],
        "image_size_hw": [height, width],
        "spatial_order": "height_width",
        "onnx_dynamic_axes": {"batch": True, "height": False, "width": False},
        "layout": "NCHW",
        "dtype": "float32",
        "color_space": "RGB" if channels == 3 else "GRAYSCALE",
        "value_range": [0.0, 1.0],
        "normalization": {"mode": "scale_0_1", "mean": [0.0] * channels, "std": [1.0] * channels},
        "resize": {
            "mode": "letterbox",
            "keep_aspect_ratio": True,
            "target_height": height,
            "target_width": width,
            "pad_value": 0,
            "placement": "center",
            "image_interpolation": "bilinear",
            "mask_interpolation": "nearest",
        },
    }


def build_segment_onnx_metadata(args, class_names=None) -> dict[str, str]:
    preprocess = segment_preprocess_schema(args)
    class_names = list(class_names or [])
    if not class_names:
        class_names = (
            ["foreground"]
            if int(args.num_classes) == 1
            else ["background", *[f"class_{index}" for index in range(1, int(args.num_classes))]]
        )
    return {
        "uknee.schema_version": "1",
        "uknee.task": "segmentation",
        "uknee.model_name": str(args.model),
        "uknee.source_checkpoint": "weights/best.pt",
        "uknee.opset": str(ONNX_OPSET),
        "uknee.preprocess": json.dumps(preprocess, separators=(",", ":")),
        "uknee.class_names": json.dumps(class_names, ensure_ascii=False, separators=(",", ":")),
        "uknee.output": json.dumps(
            {
                "name": "logits",
                "layout": "NCHW",
                "num_classes": int(args.num_classes),
                "postprocess": "sigmoid_threshold" if int(args.num_classes) == 1 else "argmax_channel",
                "label_map_encoding": "raw_class_id",
            },
            separators=(",", ":"),
        ),
    }


def _write_metadata(path: Path, metadata: dict[str, str]) -> None:
    import onnx

    graph = onnx.load(str(path))
    retained = [(item.key, item.value) for item in graph.metadata_props if item.key not in metadata]
    del graph.metadata_props[:]
    for key, value in retained:
        item = graph.metadata_props.add()
        item.key = key
        item.value = value
    for key, value in sorted(metadata.items()):
        item = graph.metadata_props.add()
        item.key = key
        item.value = str(value)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))


def read_onnx_metadata(path: str | Path) -> dict[str, str]:
    import onnx

    graph = onnx.load(str(path), load_external_data=False)
    return {item.key: item.value for item in graph.metadata_props}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_segment_onnx(
    model: nn.Module,
    args,
    output_path: str | Path,
    *,
    class_names=None,
    validate: bool = True,
) -> dict:
    """Export fixed-canvas logits and verify ONNX Runtime parity."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.module if isinstance(model, nn.DataParallel) else model
    wrapper = SegmentationONNXWrapper(model)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    preprocess = segment_preprocess_schema(args)
    _, channels, height, width = preprocess["network_input_shape"]
    dummy = torch.zeros(1, channels, height, width, dtype=torch.float32, device=device)

    try:
        with torch.no_grad():
            reference = wrapper(dummy).detach().float().cpu().numpy()
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            input_names=["images"],
            output_names=["logits"],
            opset_version=ONNX_OPSET,
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
            do_constant_folding=True,
            dynamo=False,
        )
        metadata = build_segment_onnx_metadata(args, class_names=class_names)
        _write_metadata(output_path, metadata)

        parity = {"validated": False, "provider": None, "max_abs_diff": None, "mean_abs_diff": None}
        if validate:
            import onnxruntime as ort

            session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
            actual = session.run(["logits"], {"images": dummy.detach().cpu().numpy()})[0]
            difference = np.abs(reference - actual)
            np.testing.assert_allclose(actual, reference, rtol=2e-3, atol=2e-4)
            parity = {
                "validated": True,
                "provider": "CPUExecutionProvider",
                "max_abs_diff": float(difference.max(initial=0.0)),
                "mean_abs_diff": float(difference.mean()) if difference.size else 0.0,
            }

        return {
            "status": "ready",
            "path": output_path.name,
            "format": "onnx",
            "opset": ONNX_OPSET,
            "sha256": _sha256(output_path),
            "file_size_bytes": output_path.stat().st_size,
            "metadata": metadata,
            "preprocess": preprocess,
            "parity": parity,
        }
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        model.train(was_training)


__all__ = [
    "ONNX_OPSET",
    "SUPPORTED_AUTO_EXPORT_MODELS",
    "SegmentationONNXWrapper",
    "build_segment_onnx_metadata",
    "export_segment_onnx",
    "onnx_filename",
    "read_onnx_metadata",
    "segment_preprocess_schema",
]
