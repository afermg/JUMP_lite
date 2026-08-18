"""Candidate-only full-JUMP JPEG XL compression tools."""

from __future__ import annotations

import os

# These must be set before CLI imports PyArrow, NumPy, imagecodecs, or
# Polars-adjacent dependencies. Candidate runtime fixes every cap to one.
_THREAD_CAPS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TBB_NUM_THREADS",
    "ARROW_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
)
for _name in _THREAD_CAPS:
    os.environ[_name] = "1"

__all__ = []
