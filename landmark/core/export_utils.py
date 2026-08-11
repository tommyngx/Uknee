"""Shared helpers for the two supported deployment formats."""

from __future__ import annotations

import torch


def best_onnx_opset() -> int:
    """Return the conservative opset used by landmark exports."""
    return 17


def trace_torchscript(model: torch.nn.Module, image: torch.Tensor) -> torch.jit.ScriptModule:
    """Trace a model while allowing the fixed three-output pose contract."""
    return torch.jit.trace(model.eval(), image, strict=False, check_trace=False)


__all__ = ["best_onnx_opset", "trace_torchscript"]
