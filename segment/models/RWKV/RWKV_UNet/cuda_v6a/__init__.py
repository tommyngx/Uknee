"""Lazy loader and PyTorch registration for the RWKV-UNet V6a CUDA scan."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import torch

from ...cuda_utils import load_wkv_extension


_LOAD_LOCK = threading.Lock()
_EXTENSION = None
_LOAD_ERROR: Optional[BaseException] = None
_LIB_DEF = None
_LIB_IMPL = None


def _validate_cuda_inputs(
    r: torch.Tensor,
    decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    tensors = {"r": r, "decay": decay, "k": k, "v": v}
    if r.ndim != 4:
        raise ValueError(
            "RWKV V6a CUDA scan expects [B_seq, T, H, D] tensors; "
            f"received r.shape={tuple(r.shape)}"
        )
    for name, tensor in tensors.items():
        if tensor.shape != r.shape:
            raise ValueError(
                f"RWKV V6a CUDA scan requires identical shapes; "
                f"r.shape={tuple(r.shape)}, {name}.shape={tuple(tensor.shape)}"
            )
        if not tensor.is_cuda:
            raise RuntimeError(f"RWKV V6a CUDA scan received non-CUDA tensor '{name}'")
        if not tensor.is_contiguous():
            raise RuntimeError(f"RWKV V6a CUDA scan requires contiguous tensor '{name}'")
        if tensor.dtype != r.dtype:
            raise TypeError(
                f"RWKV V6a CUDA scan requires one dtype; r={r.dtype}, {name}={tensor.dtype}"
            )
    if r.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(
            "RWKV V6a CUDA scan supports float32, float16, and bfloat16; "
            f"received {r.dtype}"
        )


def _register_custom_op(extension) -> None:
    global _LIB_DEF, _LIB_IMPL
    if _LIB_DEF is not None:
        return

    _LIB_DEF = torch.library.Library("uknee_rwkv6a", "DEF")
    _LIB_DEF.define(
        "matrix_scan(Tensor r, Tensor decay, Tensor k, Tensor v) -> (Tensor, Tensor)"
    )

    def cuda_impl(r, decay, k, v):
        _validate_cuda_inputs(r, decay, k, v)
        return tuple(extension.forward(r, decay, k, v))

    _LIB_IMPL = torch.library.Library("uknee_rwkv6a", "IMPL")
    _LIB_IMPL.impl("matrix_scan", cuda_impl, "CUDA")

    @torch.library.register_fake("uknee_rwkv6a::matrix_scan")
    def fake_impl(r, decay, k, v):
        state_shape = (*r.shape, r.shape[-1])
        return r.new_empty(r.shape), r.new_empty(state_shape, dtype=torch.float32)

    def setup_context(ctx, inputs, output) -> None:
        r, decay, k, v = inputs
        _, states = output
        ctx.save_for_backward(r, decay, k, v, states)
        ctx.mark_non_differentiable(states)

    def backward(ctx, grad_y, _grad_states):
        r, decay, k, v, states = ctx.saved_tensors
        if grad_y is None:
            return tuple(torch.zeros_like(tensor) for tensor in (r, decay, k, v))
        return tuple(
            extension.backward(
                r,
                decay,
                k,
                v,
                states,
                grad_y.contiguous(),
            )
        )

    torch.library.register_autograd(
        "uknee_rwkv6a::matrix_scan",
        backward,
        setup_context=setup_context,
    )


def ensure_loaded():
    """Build/load the extension once and register its CUDA custom operator."""
    global _EXTENSION, _LOAD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_ERROR is not None:
        raise RuntimeError("RWKV V6a CUDA extension previously failed to load") from _LOAD_ERROR

    with _LOAD_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if not torch.cuda.is_available():
            raise RuntimeError(
                "RWKV V6a CUDA backend was requested, but torch.cuda.is_available() is False"
            )
        if torch.version.cuda is None:
            raise RuntimeError(
                "RWKV V6a CUDA backend requires a CUDA-enabled PyTorch build"
            )

        source_dir = Path(__file__).resolve().parent
        try:
            extension = load_wkv_extension(
                name="uknee_rwkv6a_matrix_cuda",
                sources=[
                    str(source_dir / "wkv6a_matrix_op.cpp"),
                    str(source_dir / "wkv6a_matrix_cuda.cu"),
                ],
                extra_cuda_cflags=["-O2"],
            )
            _register_custom_op(extension)
            _EXTENSION = extension
        except BaseException as exc:
            _LOAD_ERROR = exc
            raise RuntimeError(
                "Failed to build/load the RWKV V6a CUDA extension. Check that nvcc, "
                "a compatible C++ compiler, and the CUDA toolkit matching PyTorch "
                f"({torch.version.cuda}) are available. Set RWKV_VERBOSE_BUILD=1 for details."
            ) from exc
    return _EXTENSION


def matrix_scan(
    r: torch.Tensor,
    decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Run the update-first matrix recurrence and return only its public output."""
    ensure_loaded()
    _validate_cuda_inputs(r, decay, k, v)
    output, _states = torch.ops.uknee_rwkv6a.matrix_scan(r, decay, k, v)
    return output


def custom_op() -> torch._ops.OpOverload:
    """Return the registered overload for torch.library.opcheck."""
    ensure_loaded()
    return torch.ops.uknee_rwkv6a.matrix_scan.default


__all__ = ["custom_op", "ensure_loaded", "matrix_scan"]
