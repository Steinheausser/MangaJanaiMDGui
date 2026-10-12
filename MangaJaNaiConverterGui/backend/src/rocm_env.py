"""
ROCm (AMD) runtime tuning. Must be imported before torch.

MIOpen benchmarks every applicable convolution solver the first time it sees a convolution shape,
compiling each candidate kernel on the CPU. With the full solver list this takes 70-330 s per new
image/tile size on RDNA4, while the winners are almost always the precompiled Winograd kernels
(3x3 convs) or rocBLAS GEMM (1x1 convs). Limiting the search to those solvers cuts the first-call
cost to ~2-4 s per new shape with no loss in steady-state speed. Users can override any of these by
setting the variable themselves.
"""

from __future__ import annotations

import os

ROCM_ENV_DEFAULTS = {
    # implicit-GEMM / composable-kernel solvers: compiled per shape, never fastest here,
    # and they print debug spam (a_grid_desc_...) to stdout
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM": "0",
    # direct OpenCL/HIP solvers: compiled per shape
    "MIOPEN_DEBUG_CONV_DIRECT": "0",
    "MIOPEN_DEBUG_CONV_FFT": "0",
    # benchmark the remaining candidates once per new shape and store the winner (find-db); new
    # shapes then cost ~2 s once instead of minutes
    "MIOPEN_FIND_MODE": "HYBRID",
    # only log MIOpen errors
    "MIOPEN_LOG_LEVEL": "3",
}


def configure_rocm_env() -> None:
    for key, value in ROCM_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


def is_rocm() -> bool:
    import torch

    return getattr(torch.version, "hip", None) is not None
