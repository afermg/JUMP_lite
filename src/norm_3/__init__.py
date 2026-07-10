"""norm_3: GPU-accelerated morphological profile normalization.

This module provides GPU-accelerated versions of normalization operations
using RAPIDS (cupy, cuml) for significantly faster processing.

Requires RAPIDS packages installed via pixi (see pixi.toml).

Usage:
    # Run pipeline
    pixi run python -m norm_3.pipeline input.path=data/profiles.parquet

    # Direct API usage
    from norm_3.core import RobustMAD, Spherize, TVN, PCATransform
    from norm_3.utils import to_gpu, to_cpu, is_gpu_available

Example:
    >>> import cupy as cp
    >>> from norm_3.core import RobustMAD
    >>> X = cp.random.randn(1000, 100)
    >>> scaler = RobustMAD()
    >>> X_normalized = scaler.fit_transform(X)
"""

from __future__ import annotations

__version__ = "0.1.0"

# Lazy imports to avoid loading GPU libraries unless needed
def __getattr__(name: str):
    if name == "RobustMAD":
        from norm_3.core import RobustMAD
        return RobustMAD
    elif name == "Spherize":
        from norm_3.core import Spherize
        return Spherize
    elif name == "TVN":
        from norm_3.core import TVN
        return TVN
    elif name == "TVN_EFAAR":
        from norm_3.core import TVN_EFAAR
        return TVN_EFAAR
    elif name == "PCATransform":
        from norm_3.core import PCATransform
        return PCATransform
    elif name == "is_gpu_available":
        from norm_3.utils import is_gpu_available
        return is_gpu_available
    elif name == "to_gpu":
        from norm_3.utils import to_gpu
        return to_gpu
    elif name == "to_cpu":
        from norm_3.utils import to_cpu
        return to_cpu
    elif name == "run_pipeline":
        from norm_3.pipeline import run_pipeline
        return run_pipeline
    raise AttributeError(f"module 'norm_3' has no attribute '{name}'")


__all__ = [
    "RobustMAD",
    "Spherize",
    "TVN",
    "TVN_EFAAR",
    "PCATransform",
    "is_gpu_available",
    "to_gpu",
    "to_cpu",
    "run_pipeline",
]
