"""Validate and benchmark RWKV_UNetV6a on an NVIDIA training machine.

Example:
    python -m segment.tools.benchmark_rwkv_v6a --batch 2 --imgsz 256 256
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass

import torch

from segment.models.RWKV.RWKV_UNet.RWKV_UNetV6 import RWKV_UNetV6
from segment.models.RWKV.RWKV_UNet.RWKV_UNetV6a import (
    RWKV_UNetV6a,
    reference_matrix_scan,
)
from segment.models.RWKV.RWKV_UNet.cuda_v6a import custom_op, matrix_scan


def _command_version(command: list[str]) -> str:
    if shutil.which(command[0]) is None:
        return "unavailable"
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        return f"unavailable: {exc}"
    text = (result.stdout or result.stderr).strip().splitlines()
    return text[-1] if text else f"exit={result.returncode}"


def environment_report() -> dict:
    properties = torch.cuda.get_device_properties(0)
    return {
        "gpu": properties.name,
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "vram_gib": properties.total_memory / 2**30,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "cuda_runtime": _command_version(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "nvcc": _command_version(["nvcc", "--version"]),
        "cxx": _command_version(["c++", "--version"]),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def error_statistics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    absolute = (actual.float() - expected.float()).abs()
    relative = absolute / expected.float().abs().clamp_min(1e-8)
    return {
        "max_abs_error": absolute.max().item(),
        "mean_abs_error": absolute.mean().item(),
        "max_relative_error": relative.max().item(),
    }


def parity_case(shape: tuple[int, int, int, int], dtype: torch.dtype) -> dict:
    torch.manual_seed(2006)
    source = (
        torch.randn(shape, device="cuda", dtype=dtype),
        torch.sigmoid(torch.randn(shape, device="cuda", dtype=dtype)),
        torch.randn(shape, device="cuda", dtype=dtype),
        torch.randn(shape, device="cuda", dtype=dtype),
    )
    reference_inputs = tuple(item.detach().clone().requires_grad_(True) for item in source)
    cuda_inputs = tuple(item.detach().clone().contiguous().requires_grad_(True) for item in source)
    reference = reference_matrix_scan(*reference_inputs)
    actual = matrix_scan(*cuda_inputs)
    generator = torch.Generator(device="cuda").manual_seed(2007)
    grad_output = torch.randn(reference.shape, device="cuda", dtype=dtype, generator=generator)
    reference_grads = torch.autograd.grad(reference, reference_inputs, grad_output)
    actual_grads = torch.autograd.grad(actual, cuda_inputs, grad_output)
    names = ("grad_r", "grad_decay", "grad_k", "grad_v")
    result = {"shape": list(shape), "dtype": str(dtype), "forward": error_statistics(actual, reference)}
    result.update(
        {
            name: error_statistics(actual_grad, reference_grad)
            for name, actual_grad, reference_grad in zip(names, actual_grads, reference_grads)
        }
    )
    return result


@dataclass
class Timing:
    mean_ms: float
    stdev_ms: float


def timed(callable_, warmup: int, iterations: int) -> Timing:
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        callable_()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return Timing(statistics.mean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0)


def scan_benchmark(length: int, warmup: int, iterations: int) -> dict:
    shape = (1, length, 6, 90)
    source = (
        torch.randn(shape, device="cuda"),
        torch.sigmoid(torch.randn(shape, device="cuda")),
        torch.randn(shape, device="cuda"),
        torch.randn(shape, device="cuda"),
    )

    def run_reference_forward():
        reference_matrix_scan(*source)

    def run_cuda_forward():
        matrix_scan(*source)

    def run_reference_training():
        inputs = tuple(item.detach().requires_grad_(True) for item in source)
        reference_matrix_scan(*inputs).sum().backward()

    def run_cuda_training():
        inputs = tuple(item.detach().requires_grad_(True) for item in source)
        matrix_scan(*inputs).sum().backward()

    return {
        "shape": list(shape),
        "reference_forward": vars(timed(run_reference_forward, warmup, iterations)),
        "cuda_forward": vars(timed(run_cuda_forward, warmup, iterations)),
        "reference_forward_backward": vars(timed(run_reference_training, warmup, iterations)),
        "cuda_forward_backward": vars(timed(run_cuda_training, warmup, iterations)),
    }


def model_benchmark(batch: int, height: int, width: int, warmup: int, iterations: int) -> dict:
    torch.manual_seed(2006)
    reference = RWKV_UNetV6().cuda().train()
    accelerated = RWKV_UNetV6a(matrix_state_backend="cuda").cuda().train()
    accelerated.load_state_dict(reference.state_dict(), strict=True)
    image = torch.randn(batch, 3, height, width, device="cuda")

    def measure(model):
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = model(image).square().mean()
            loss.backward()
            optimizer.step()

        torch.cuda.reset_peak_memory_stats()
        timing = timed(step, warmup, iterations)
        return {
            **vars(timing),
            "images_per_second": batch / (timing.mean_ms / 1000.0),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }

    return {
        "batch": batch,
        "image_size_hw": [height, width],
        "reference": measure(reference),
        "cuda": measure(accelerated),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(256, 256))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This validation/benchmark must run on an NVIDIA CUDA machine")
    extension = custom_op()  # Builds once, then reuses PyTorch's extension cache.
    opcheck_args = tuple(
        tensor.contiguous()
        for tensor in (
            torch.randn(1, 3, 2, 3, device="cuda", requires_grad=True),
            torch.sigmoid(torch.randn(1, 3, 2, 3, device="cuda")).requires_grad_(),
            torch.randn(1, 3, 2, 3, device="cuda", requires_grad=True),
            torch.randn(1, 3, 2, 3, device="cuda", requires_grad=True),
        )
    )
    result = {
        "environment": environment_report(),
        "opcheck": torch.library.opcheck(extension, opcheck_args),
        "fp32_parity": [parity_case((1, length, 6, 90), torch.float32) for length in (16, 28, 32, 45, 64)],
        "scan_benchmark": scan_benchmark(16, args.warmup, args.iterations),
        "model_benchmark": model_benchmark(
            args.batch, args.imgsz[0], args.imgsz[1], args.warmup, args.iterations
        ),
    }
    if torch.cuda.is_bf16_supported():
        result["bf16_parity"] = parity_case((1, 16, 6, 90), torch.bfloat16)
    rendered = json.dumps(result, indent=2, default=str)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(rendered + "\n")


if __name__ == "__main__":
    main()
