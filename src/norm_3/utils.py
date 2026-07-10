"""GPU utilities for norm_3.

Provides GPU detection, array transfer, and memory management utilities.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import cupy as cp

# Global GPU availability flag (cached on first check)
_GPU_AVAILABLE: bool | None = None
_GPU_DEVICE_INFO: dict[str, Any] | None = None


def is_gpu_available() -> bool:
    """Check if GPU (CUDA via cupy) is available.

    Returns:
        True if cupy is installed and a GPU is accessible.
    """
    global _GPU_AVAILABLE, _GPU_DEVICE_INFO

    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    try:
        import cupy as cp

        # Try to get device info to verify GPU is actually accessible
        device = cp.cuda.Device()
        _GPU_DEVICE_INFO = {
            "name": getattr(device, "name", "Unknown"),
            "compute_capability": device.compute_capability,
            "total_memory_gb": device.mem_info[1] / (1024**3),
        }
        _GPU_AVAILABLE = True

    except ImportError:
        _GPU_AVAILABLE = False
        warnings.warn(
            "cupy not installed. Install via pixi for GPU acceleration: pixi run python ...",
            stacklevel=2
        )
    except Exception as e:
        _GPU_AVAILABLE = False
        warnings.warn(f"GPU not accessible: {e}", stacklevel=2)

    return _GPU_AVAILABLE


def get_gpu_info() -> dict[str, Any] | None:
    """Get GPU device information.

    Returns:
        Dictionary with GPU info (name, compute_capability, total_memory_gb),
        or None if GPU not available.
    """
    if not is_gpu_available():
        return None
    return _GPU_DEVICE_INFO


def to_gpu(arr: np.ndarray, dtype: Any | None = None) -> "cp.ndarray":
    """Transfer numpy array to GPU (cupy array).

    Args:
        arr: numpy array to transfer
        dtype: Optional dtype to cast to (e.g., cp.float32 for memory savings)

    Returns:
        cupy array on GPU

    Raises:
        RuntimeError: If GPU not available
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available. Check cupy installation and CUDA setup.")

    import cupy as cp

    if dtype is not None:
        return cp.asarray(arr, dtype=dtype)
    return cp.asarray(arr)


def to_cpu(arr: "cp.ndarray") -> np.ndarray:
    """Transfer cupy array to CPU (numpy array).

    Args:
        arr: cupy array to transfer

    Returns:
        numpy array on CPU
    """
    import cupy as cp

    if isinstance(arr, np.ndarray):
        return arr
    return cp.asnumpy(arr)


def ensure_gpu_array(arr: np.ndarray | "cp.ndarray") -> "cp.ndarray":
    """Ensure array is on GPU.

    Args:
        arr: numpy or cupy array

    Returns:
        cupy array on GPU
    """
    import cupy as cp

    if isinstance(arr, cp.ndarray):
        return arr
    return to_gpu(arr)


def ensure_cpu_array(arr: np.ndarray | "cp.ndarray") -> np.ndarray:
    """Ensure array is on CPU.

    Args:
        arr: numpy or cupy array

    Returns:
        numpy array on CPU
    """
    if isinstance(arr, np.ndarray):
        return arr
    return to_cpu(arr)


def get_array_module(arr: np.ndarray | "cp.ndarray"):
    """Get the array module (numpy or cupy) for the given array.

    Args:
        arr: numpy or cupy array

    Returns:
        numpy or cupy module
    """
    if is_gpu_available():
        import cupy as cp
        return cp.get_array_module(arr)
    return np


class GPUMemoryManager:
    """Context manager for GPU memory lifecycle.

    Automatically frees GPU memory when exiting the context.
    Useful for pipeline steps that allocate large temporary arrays.

    Example:
        with GPUMemoryManager() as gpu:
            X_gpu = gpu.transfer(df.to_numpy())
            # ... operations ...
            result = to_cpu(X_gpu)
        # GPU memory automatically freed here
    """

    def __init__(self, device: int = 0, memory_limit_gb: float | None = None):
        """Initialize GPU memory manager.

        Args:
            device: GPU device ID (default 0)
            memory_limit_gb: Optional memory limit in GB
        """
        self.device = device
        self.memory_limit_gb = memory_limit_gb
        self._arrays: list = []

    def __enter__(self) -> "GPUMemoryManager":
        if not is_gpu_available():
            warnings.warn("GPU not available, GPUMemoryManager will not manage GPU memory")
            return self

        import cupy as cp

        # Set device
        cp.cuda.Device(self.device).use()

        # Set memory limit if specified
        if self.memory_limit_gb is not None:
            limit_bytes = int(self.memory_limit_gb * 1024**3)
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=limit_bytes)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not is_gpu_available():
            return

        import cupy as cp

        # Clear tracked arrays
        for arr in self._arrays:
            del arr
        self._arrays.clear()

        # Free all GPU memory in the pool
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

        # Synchronize to ensure all operations complete
        cp.cuda.Stream.null.synchronize()

    def transfer(self, arr: np.ndarray, dtype: Any | None = None) -> "cp.ndarray":
        """Transfer array to GPU and track it for cleanup.

        Args:
            arr: numpy array to transfer
            dtype: Optional dtype to cast to

        Returns:
            cupy array on GPU
        """
        gpu_arr = to_gpu(arr, dtype=dtype)
        self._arrays.append(gpu_arr)
        return gpu_arr

    def track(self, arr: "cp.ndarray") -> "cp.ndarray":
        """Track an existing GPU array for cleanup.

        Args:
            arr: cupy array to track

        Returns:
            The same array (for chaining)
        """
        self._arrays.append(arr)
        return arr


@contextmanager
def gpu_context(device: int = 0, memory_limit_gb: float | None = None):
    """Context manager for GPU operations with automatic cleanup.

    Args:
        device: GPU device ID (default 0)
        memory_limit_gb: Optional memory limit in GB

    Yields:
        GPUMemoryManager instance

    Example:
        with gpu_context() as gpu:
            X = gpu.transfer(data)
            result = some_gpu_operation(X)
            output = to_cpu(result)
        # Memory freed here
    """
    manager = GPUMemoryManager(device=device, memory_limit_gb=memory_limit_gb)
    with manager:
        yield manager


def get_memory_info() -> dict[str, float] | None:
    """Get current GPU memory usage.

    Returns:
        Dictionary with 'used_gb', 'free_gb', 'total_gb',
        or None if GPU not available.
    """
    if not is_gpu_available():
        return None

    import cupy as cp

    mempool = cp.get_default_memory_pool()
    device = cp.cuda.Device()
    free, total = device.mem_info

    return {
        "used_gb": mempool.used_bytes() / (1024**3),
        "pool_total_gb": mempool.total_bytes() / (1024**3),
        "free_gb": free / (1024**3),
        "total_gb": total / (1024**3),
    }


def print_gpu_info():
    """Print GPU device and memory information."""
    if not is_gpu_available():
        print("GPU: Not available")
        return

    info = get_gpu_info()
    mem = get_memory_info()

    print(f"GPU: {info['name']}")
    print(f"  Compute Capability: {info['compute_capability']}")
    print(f"  Total Memory: {info['total_memory_gb']:.1f} GB")
    print(f"  Memory Used: {mem['used_gb']:.2f} GB")
    print(f"  Memory Free: {mem['free_gb']:.1f} GB")
