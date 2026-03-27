import os

import torch
from torch.utils.cpp_extension import load


def _is_verbose_build():
    value = os.environ.get("RWKV_VERBOSE_BUILD", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _ensure_cuda_arch_list():
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    if not torch.cuda.is_available():
        return

    try:
        archs = {
            f"{major}.{minor}"
            for major, minor in (
                torch.cuda.get_device_capability(index)
                for index in range(torch.cuda.device_count())
            )
        }
    except Exception:
        return

    if archs:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(sorted(archs))


def load_wkv_extension(name, sources, extra_cuda_cflags=None):
    _ensure_cuda_arch_list()
    verbose = _is_verbose_build()
    if verbose:
        for source in sources:
            print(source)

    return load(
        name=name,
        sources=sources,
        verbose=verbose,
        extra_cuda_cflags=list(extra_cuda_cflags or []),
    )
