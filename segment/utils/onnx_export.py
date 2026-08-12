"""ONNX export contract for the supported RWKV segmentation models."""

from __future__ import annotations

from copy import deepcopy
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


def _parity_statistics(expected: np.ndarray, actual: np.ndarray, num_classes: int) -> dict[str, float]:
    """Measure numerical and postprocessed agreement without over-penalizing near-zero logits."""
    difference = np.abs(expected - actual)
    if int(num_classes) == 1:
        expected_labels = expected[:, 0] >= 0.0
        actual_labels = actual[:, 0] >= 0.0
    else:
        expected_labels = expected.argmax(axis=1)
        actual_labels = actual.argmax(axis=1)
    return {
        "max_abs_diff": float(difference.max(initial=0.0)),
        "mean_abs_diff": float(difference.mean()) if difference.size else 0.0,
        "p99_abs_diff": float(np.percentile(difference, 99)) if difference.size else 0.0,
        "postprocess_agreement": float(np.mean(expected_labels == actual_labels)),
        "reference_mean_abs": float(np.mean(np.abs(expected))) if expected.size else 0.0,
        "reference_p99_abs": float(np.percentile(np.abs(expected), 99)) if expected.size else 0.0,
        "reference_max_abs": float(np.max(np.abs(expected), initial=0.0)),
    }


def _validate_parity_statistics(statistics: dict[str, float]) -> None:
    """Reject materially different exports while allowing normal backend interpolation drift."""
    limits = {
        "max_abs_diff": max(0.15, 0.03 * statistics["reference_max_abs"]),
        "mean_abs_diff": max(0.01, 0.01 * statistics["reference_mean_abs"]),
        "p99_abs_diff": max(0.05, 0.02 * statistics["reference_p99_abs"]),
        "postprocess_agreement": 0.995,
    }
    failed = [
        f"max_abs_diff={statistics['max_abs_diff']:.6g}>{limits['max_abs_diff']:.6g}"
        if statistics["max_abs_diff"] > limits["max_abs_diff"]
        else None,
        f"mean_abs_diff={statistics['mean_abs_diff']:.6g}>{limits['mean_abs_diff']:.6g}"
        if statistics["mean_abs_diff"] > limits["mean_abs_diff"]
        else None,
        f"p99_abs_diff={statistics['p99_abs_diff']:.6g}>{limits['p99_abs_diff']:.6g}"
        if statistics["p99_abs_diff"] > limits["p99_abs_diff"]
        else None,
        f"postprocess_agreement={statistics['postprocess_agreement']:.6g}<{limits['postprocess_agreement']:.6g}"
        if statistics["postprocess_agreement"] < limits["postprocess_agreement"]
        else None,
    ]
    failed = [item for item in failed if item]
    if failed:
        raise RuntimeError("ONNX parity validation failed: " + "; ".join(failed))


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
    preprocess = json.loads(metadata["uknee.preprocess"])
    output_contract = json.loads(metadata["uknee.output"])
    _, channels, height, width = preprocess["network_input_shape"]
    num_classes = int(output_contract["num_classes"])

    def set_shape(value_info, dimensions):
        tensor_shape = value_info.type.tensor_type.shape
        del tensor_shape.dim[:]
        for dimension in dimensions:
            dim = tensor_shape.dim.add()
            if isinstance(dimension, str):
                dim.dim_param = dimension
            else:
                dim.dim_value = int(dimension)

    # PyTorch 2.4 may incorrectly label V6 output spatial axes as symbolic while
    # tracing Resize. Publish the exact deployment contract explicitly.
    set_shape(graph.graph.input[0], ["batch", channels, height, width])
    set_shape(graph.graph.output[0], ["batch", num_classes, height, width])
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
    source_model = model.module if isinstance(model, nn.DataParallel) else model
    # Torch 2.4 can mix CUDA and CPU shape tensors while tracing F.interpolate.
    # Export an isolated CPU copy so ONNX creation never mutates or interrupts the live trainer.
    export_model = deepcopy(source_model).float().cpu().eval()
    wrapper = SegmentationONNXWrapper(export_model)
    device = torch.device("cpu")
    preprocess = segment_preprocess_schema(args)
    _, channels, height, width = preprocess["network_input_shape"]
    dummy = torch.zeros(1, channels, height, width, dtype=torch.float32, device=device)

    try:
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

        parity = {
            "validated": False,
            "validated_batches": [],
            "provider": None,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }
        if validate:
            import onnxruntime as ort

            session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
            batch_statistics = []
            for validation_batch in (1, 2):
                validation_input = dummy.expand(validation_batch, -1, -1, -1).contiguous()
                with torch.no_grad():
                    expected = wrapper(validation_input).detach().float().cpu().numpy()
                actual = session.run(["logits"], {"images": validation_input.cpu().numpy()})[0]
                if actual.shape != expected.shape:
                    raise RuntimeError(
                        f"ONNX output shape mismatch for batch={validation_batch}: "
                        f"expected={expected.shape}, actual={actual.shape}"
                    )
                statistics = _parity_statistics(expected, actual, int(args.num_classes))
                _validate_parity_statistics(statistics)
                batch_statistics.append(statistics)
            parity = {
                "validated": True,
                "validated_batches": [1, 2],
                "provider": "CPUExecutionProvider",
                "max_abs_diff": max(item["max_abs_diff"] for item in batch_statistics),
                "mean_abs_diff": max(item["mean_abs_diff"] for item in batch_statistics),
                "p99_abs_diff": max(item["p99_abs_diff"] for item in batch_statistics),
                "postprocess_agreement": min(item["postprocess_agreement"] for item in batch_statistics),
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
        del wrapper, export_model


__all__ = [
    "ONNX_OPSET",
    "SUPPORTED_AUTO_EXPORT_MODELS",
    "_parity_statistics",
    "_validate_parity_statistics",
    "SegmentationONNXWrapper",
    "build_segment_onnx_metadata",
    "export_segment_onnx",
    "onnx_filename",
    "read_onnx_metadata",
    "segment_preprocess_schema",
]
