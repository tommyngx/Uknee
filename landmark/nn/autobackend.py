"""Minimal inference backend for PyTorch, TorchScript and ONNX pose models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def check_class_names(names: list | dict) -> dict[int, str]:
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(key): str(value) for key, value in names.items()}
    if names and (min(names) < 0 or max(names) >= len(names)):
        raise KeyError(f"Invalid class indices for {len(names)} classes: {sorted(names)}")
    return names


def default_class_names(_data: str | Path | None = None) -> dict[int, str]:
    return {0: "femur", 1: "tibia", 2: "fibula", 3: "patella"}


class AutoBackend(nn.Module):
    """Small backend adapter used by the predictor; unsupported formats fail early."""

    @torch.no_grad()
    def __init__(
        self,
        model: str | Path | nn.Module,
        device: torch.device = torch.device("cpu"),
        dnn: bool = False,
        data: str | Path | None = None,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        if dnn:
            raise ValueError("landmark uses ONNX Runtime and does not expose OpenCV DNN")
        self.device = torch.device(device)
        self.fp16 = bool(fp16)
        self.nhwc = False
        self.dynamic = False
        self.batch = 1
        self.channels = 3
        self.imgsz = None
        self.end2end = False
        self.kpt_shape = [51, 3]

        if isinstance(model, nn.Module):
            self.format = "pt"
            self.model = model.to(self.device)
            if fuse and hasattr(self.model, "fuse"):
                fused = self.model.fuse(verbose=verbose)
                self.model = fused if fused is not None else self.model
            self.model.half() if self.fp16 else self.model.float()
            self.names = check_class_names(getattr(self.model, "names", default_class_names(data)))
            self.stride = max(int(getattr(self.model, "stride", torch.tensor([32])).max()), 32)
            self.channels = int(getattr(self.model, "yaml", {}).get("channels", 3))
            self.kpt_shape = getattr(self.model, "kpt_shape", self.kpt_shape)
            self.end2end = bool(getattr(self.model, "end2end", False))
            return

        path = Path(model)
        suffix = path.suffix.lower()
        if suffix in {".torchscript", ".ts"}:
            self.format = "torchscript"
            extra = {"config.txt": ""}
            self.model = torch.jit.load(str(path), map_location=self.device, _extra_files=extra)
            metadata = json.loads(extra["config.txt"]) if extra["config.txt"] else {}
            self.names = check_class_names(metadata.get("names", default_class_names(data)))
            self.stride = int(metadata.get("stride", 32))
            self.kpt_shape = metadata.get("kpt_shape", self.kpt_shape)
            return
        if suffix == ".pt":
            from landmark.nn.tasks import load_checkpoint

            self.format = "pt"
            self.model, _ = load_checkpoint(str(path), device=self.device, fuse=fuse)
            self.names = check_class_names(getattr(self.model, "names", default_class_names(data)))
            self.stride = max(int(getattr(self.model, "stride", torch.tensor([32])).max()), 32)
            self.kpt_shape = getattr(self.model, "kpt_shape", self.kpt_shape)
            self.end2end = bool(getattr(self.model, "end2end", False))
            return
        if suffix == ".onnx":
            import onnxruntime as ort

            self.format = "onnx"
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.names = default_class_names(data)
            self.stride = 32
            self.model = None
            return
        raise TypeError(f"Unsupported model format {suffix!r}; use .pt, .torchscript or .onnx")

    def forward(self, image: torch.Tensor, augment: bool = False, visualize: bool = False, embed=None, **kwargs: Any):
        if self.fp16 and image.dtype != torch.float16:
            image = image.half()
        if self.format == "onnx":
            outputs = self.session.run(None, {self.input_name: image.detach().cpu().numpy()})
            tensors = [torch.from_numpy(value).to(self.device) for value in outputs]
            return tensors[0] if len(tensors) == 1 else tensors
        if self.format == "pt":
            return self.model(image, augment=augment, visualize=visualize, embed=embed, **kwargs)
        return self.model(image)

    def warmup(self, imgsz=(1, 3, 640, 640)) -> None:
        if self.device.type != "cpu" and self.format != "onnx":
            image = torch.empty(*imgsz, device=self.device, dtype=torch.float16 if self.fp16 else torch.float32)
            self.forward(image)

    def set_head_attr(self, **kwargs: Any) -> None:
        if self.model is None or not hasattr(self.model, "model"):
            return
        head = self.model.model[-1]
        for key, value in kwargs.items():
            setattr(head, key, value)

