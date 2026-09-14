"""
RWKV_UNetV6a

Accelerated implementation of RWKV_UNetV6.

Architecture and trainable parameters are intended to remain checkpoint-compatible
with RWKV_UNetV6. The only difference is the Matrix-State scan execution backend.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .RWKV_UNetV6 import (
    AxialRWKV6SpatialMix,
    DEFAULT_DEPTHS,
    DEFAULT_EMBED_DIMS,
    DEFAULT_EXP_RATIOS,
    DEFAULT_MATRIX_STATE_STAGES,
    DEFAULT_NUM_HEADS,
    RWKV6MatrixStateScan,
    RWKV6SequenceMix,
    RWKV_UNetV6,
)


_VALID_BACKENDS = frozenset({"auto", "reference", "cuda"})


def _resolve_backend(requested: Optional[str]) -> str:
    backend = requested or os.environ.get("RWKV_V6A_BACKEND", "auto")
    backend = backend.strip().lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown RWKV V6a Matrix-State backend '{backend}'. "
            f"Expected one of {sorted(_VALID_BACKENDS)}."
        )
    if backend == "auto":
        return "cuda" if torch.cuda.is_available() else "reference"
    return backend


def reference_matrix_scan(
    r: torch.Tensor,
    decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Exact update-first pure-PyTorch recurrence used by frozen V6."""
    if r.ndim != 4:
        raise ValueError(f"Expected [B_seq, T, H, D], received {tuple(r.shape)}")
    if not (decay.shape == k.shape == v.shape == r.shape):
        raise ValueError("r, decay, k, and v must have identical shapes")

    batch_size, length, heads, head_dim = r.shape
    state = torch.zeros(
        batch_size,
        heads,
        head_dim,
        head_dim,
        device=r.device,
        dtype=state_dtype,
    )
    outputs = []
    for index in range(length):
        r_t = r[:, index].to(state_dtype)
        decay_t = decay[:, index].to(state_dtype)
        k_t = k[:, index].to(state_dtype)
        v_t = v[:, index].to(state_dtype)
        kv_t = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
        state = state * decay_t.unsqueeze(-1) + kv_t
        y_t = torch.einsum("bhd,bhde->bhe", r_t, state)
        outputs.append(y_t.to(r.dtype))
    return torch.stack(outputs, dim=1)


class RWKV6aMatrixStateScan(nn.Module):
    """Parameter-free V6 recurrence with an explicit reference/CUDA backend."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        state_dtype: torch.dtype = torch.float32,
        backend: str = "reference",
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if backend not in {"reference", "cuda"}:
            raise ValueError(f"Resolved backend must be reference or cuda, received '{backend}'")
        if backend == "cuda" and state_dtype != torch.float32:
            raise ValueError("RWKV V6a CUDA currently requires FP32 state accumulation")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.state_dtype = state_dtype
        self.backend = backend

    def forward(
        self,
        r: torch.Tensor,
        decay: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        batch_size, length, channels = r.shape
        if channels != self.dim:
            raise ValueError(f"Expected {self.dim} channels, received {channels}")

        shape = (batch_size, length, self.num_heads, self.head_dim)
        r = r.view(shape)
        decay = decay.view(shape)
        k = k.view(shape)
        v = v.view(shape)
        if reverse:
            r = torch.flip(r, dims=[1])
            decay = torch.flip(decay, dims=[1])
            k = torch.flip(k, dims=[1])
            v = torch.flip(v, dims=[1])

        if self.backend == "reference":
            output = reference_matrix_scan(r, decay, k, v, self.state_dtype)
        else:
            if not r.is_cuda:
                raise RuntimeError(
                    "RWKV V6a Matrix-State backend is CUDA, but its input is not on CUDA"
                )
            from .cuda_v6a import matrix_scan

            output = matrix_scan(
                r.contiguous(),
                decay.contiguous(),
                k.contiguous(),
                v.contiguous(),
            )

        if reverse:
            output = torch.flip(output, dims=[1])
        return output.reshape(batch_size, length, channels)


def _replace_v6_scans(module: nn.Module, backend: str) -> None:
    for child in module.modules():
        if not isinstance(child, RWKV6SequenceMix):
            continue
        for attribute in ("forward_scan", "backward_scan"):
            original = getattr(child, attribute)
            if not isinstance(original, RWKV6MatrixStateScan):
                raise TypeError(f"Unexpected V6 scan module at {attribute}: {type(original)!r}")
            setattr(
                child,
                attribute,
                RWKV6aMatrixStateScan(
                    dim=original.dim,
                    num_heads=original.num_heads,
                    state_dtype=original.state_dtype,
                    backend=backend,
                ),
            )


class RWKV6aSequenceMix(RWKV6SequenceMix):
    """V6 sequence mixer whose only substitution is the scan backend."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        low_rank_dim: Optional[int] = None,
        matrix_state_backend: str = "reference",
    ) -> None:
        backend = _resolve_backend(matrix_state_backend)
        if backend == "cuda":
            from .cuda_v6a import ensure_loaded

            ensure_loaded()
        super().__init__(dim=dim, num_heads=num_heads, low_rank_dim=low_rank_dim)
        _replace_v6_scans(self, backend)
        self.matrix_state_backend = backend


class AxialRWKV6aSpatialMix(AxialRWKV6SpatialMix):
    """V6 axial mixer with reference/CUDA V6a scans and identical weights."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        low_rank_dim: Optional[int] = None,
        matrix_state_backend: str = "reference",
    ) -> None:
        backend = _resolve_backend(matrix_state_backend)
        if backend == "cuda":
            from .cuda_v6a import ensure_loaded

            ensure_loaded()
        super().__init__(dim=dim, num_heads=num_heads, low_rank_dim=low_rank_dim)
        _replace_v6_scans(self, backend)
        self.matrix_state_backend = backend


class RWKV_UNetV6a(RWKV_UNetV6):
    """Checkpoint-compatible V6 model with an accelerated scan backend."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1,
        stem_dim: int = 24,
        depths: Tuple[int, ...] = DEFAULT_DEPTHS,
        embed_dims: Tuple[int, ...] = DEFAULT_EMBED_DIMS,
        exp_ratios: Tuple[float, ...] = DEFAULT_EXP_RATIOS,
        num_heads: Tuple[int, ...] = DEFAULT_NUM_HEADS,
        matrix_state_stages: Tuple[int, ...] = DEFAULT_MATRIX_STATE_STAGES,
        drop_path_rate: float = 0.1,
        matrix_state_backend: Optional[str] = None,
    ) -> None:
        backend = _resolve_backend(matrix_state_backend)
        if backend == "cuda":
            from .cuda_v6a import ensure_loaded

            ensure_loaded()

        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            stem_dim=stem_dim,
            depths=depths,
            embed_dims=embed_dims,
            exp_ratios=exp_ratios,
            num_heads=num_heads,
            matrix_state_stages=matrix_state_stages,
            drop_path_rate=drop_path_rate,
        )
        _replace_v6_scans(self, backend)
        self.matrix_state_backend = backend
        self._v6_reference_kwargs = {
            "input_channels": input_channels,
            "num_classes": num_classes,
            "stem_dim": stem_dim,
            "depths": tuple(depths),
            "embed_dims": tuple(embed_dims),
            "exp_ratios": tuple(exp_ratios),
            "num_heads": tuple(num_heads),
            "matrix_state_stages": tuple(matrix_state_stages),
            "drop_path_rate": drop_path_rate,
        }

        print("RWKV_UNetV6a")
        if backend == "cuda":
            print("Matrix-State backend: CUDA")
            print("CUDA extension: loaded successfully")
        else:
            print("Matrix-State backend: reference")

    def to_v6_reference(self) -> RWKV_UNetV6:
        """Create the checkpoint-identical CUDA-free model used for deployment."""
        reference = RWKV_UNetV6(**self._v6_reference_kwargs)
        reference.load_state_dict(self.state_dict(), strict=True)
        reference.train(self.training)
        return reference


def rwkv_unet_v6a(
    input_channel: int = 3,
    num_classes: int = 1,
    **kwargs,
) -> RWKV_UNetV6a:
    return RWKV_UNetV6a(
        input_channels=input_channel,
        num_classes=num_classes,
        **kwargs,
    )


__all__ = [
    "AxialRWKV6aSpatialMix",
    "RWKV6aMatrixStateScan",
    "RWKV6aSequenceMix",
    "RWKV_UNetV6a",
    "reference_matrix_scan",
    "rwkv_unet_v6a",
]
