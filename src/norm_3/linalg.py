"""GPU-accelerated linear algebra operations for norm_3.

Provides cupy-based implementations of linear algebra operations
that don't have direct equivalents in cupy but are needed for
normalization (e.g., fractional_matrix_power from scipy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cupy as cp


def fractional_matrix_power(A: "cp.ndarray", p: float) -> "cp.ndarray":
    """Compute fractional matrix power A^p on GPU.

    Uses eigendecomposition for symmetric positive definite matrices.
    This is the GPU equivalent of scipy.linalg.fractional_matrix_power.

    For symmetric matrix A with eigendecomposition A = V @ D @ V.T:
        A^p = V @ D^p @ V.T

    Args:
        A: Square symmetric positive (semi-)definite matrix (n x n)
        p: Fractional power (e.g., 0.5 for sqrt, -0.5 for inverse sqrt)

    Returns:
        Matrix A raised to power p

    Note:
        - Uses cp.linalg.eigh which assumes symmetric input
        - Small/negative eigenvalues are clipped to avoid numerical issues
        - For non-symmetric matrices, consider using SVD-based approach
    """
    import cupy as cp

    # Use symmetric eigendecomposition (more numerically stable)
    eigenvalues, eigenvectors = cp.linalg.eigh(A)

    # Clip small/negative eigenvalues for numerical stability
    # This handles slightly non-positive-definite matrices
    eigenvalues = cp.maximum(eigenvalues, 1e-10)

    # Compute A^p = V @ diag(eigenvalues^p) @ V.T
    powered_eigenvalues = eigenvalues ** p

    # Efficient computation: V @ diag(d) @ V.T = (V * d) @ V.T
    result = (eigenvectors * powered_eigenvalues) @ eigenvectors.T

    return result


def matrix_sqrt(A: "cp.ndarray") -> "cp.ndarray":
    """Compute matrix square root A^(1/2) on GPU.

    Shorthand for fractional_matrix_power(A, 0.5).

    Args:
        A: Square symmetric positive (semi-)definite matrix

    Returns:
        Matrix square root such that result @ result = A
    """
    return fractional_matrix_power(A, 0.5)


def matrix_inv_sqrt(A: "cp.ndarray") -> "cp.ndarray":
    """Compute inverse matrix square root A^(-1/2) on GPU.

    Shorthand for fractional_matrix_power(A, -0.5).

    Args:
        A: Square symmetric positive definite matrix

    Returns:
        Inverse matrix square root such that result @ A @ result = I
    """
    return fractional_matrix_power(A, -0.5)


def covariance(X: "cp.ndarray", rowvar: bool = False, ddof: int = 1) -> "cp.ndarray":
    """Compute covariance matrix on GPU.

    Wrapper around cp.cov with default parameters for our use case.

    Args:
        X: Data matrix (n_samples x n_features if rowvar=False)
        rowvar: If True, each row is a variable. If False, each column is.
        ddof: Delta degrees of freedom (1 for sample covariance)

    Returns:
        Covariance matrix
    """
    import cupy as cp

    return cp.cov(X, rowvar=rowvar, ddof=ddof)


def correlation(X: "cp.ndarray", rowvar: bool = False) -> "cp.ndarray":
    """Compute correlation matrix on GPU.

    Args:
        X: Data matrix (n_samples x n_features if rowvar=False)
        rowvar: If True, each row is a variable. If False, each column is.

    Returns:
        Correlation matrix
    """
    import cupy as cp

    return cp.corrcoef(X, rowvar=rowvar)


def condition_number(A: "cp.ndarray") -> float:
    """Compute condition number of a matrix.

    Args:
        A: Square matrix

    Returns:
        Condition number (ratio of largest to smallest singular value)
    """
    import cupy as cp

    s = cp.linalg.svd(A, compute_uv=False)
    return float(s[0] / s[-1])


def is_ill_conditioned(A: "cp.ndarray", threshold: float = 1e10) -> bool:
    """Check if matrix is ill-conditioned.

    Args:
        A: Square matrix
        threshold: Condition number threshold (default 1e10)

    Returns:
        True if condition number exceeds threshold
    """
    return condition_number(A) > threshold


def svd_whitening(X: "cp.ndarray", epsilon: float = 1e-6) -> tuple["cp.ndarray", "cp.ndarray"]:
    """Compute SVD-based whitening transformation.

    Args:
        X: Centered data matrix (n_samples x n_features)
        epsilon: Regularization parameter for numerical stability

    Returns:
        Tuple of (W, W_inv) where:
        - W: Whitening matrix such that (X @ W) has identity covariance
        - W_inv: Inverse whitening (coloring) matrix
    """
    import cupy as cp

    n = len(X)

    # SVD of centered data
    _, s, Vt = cp.linalg.svd(X, full_matrices=False)

    # Whitening matrix: W = V @ diag(1/sqrt(s^2/(n-1) + eps))
    # Simplified: W = V @ diag(sqrt(n-1) / (s + eps))
    scale = cp.sqrt(n - 1) / (s + epsilon)
    W = (Vt.T * scale)

    # Inverse (coloring) matrix
    inv_scale = (s + epsilon) / cp.sqrt(n - 1)
    W_inv = (Vt.T * inv_scale)

    return W, W_inv


def zca_whitening(X: "cp.ndarray", epsilon: float = 1e-6) -> "cp.ndarray":
    """Compute ZCA whitening transformation.

    ZCA (Zero-phase Component Analysis) whitening produces transformed data
    that is maximally similar to the original while having identity covariance.

    Args:
        X: Centered data matrix (n_samples x n_features)
        epsilon: Regularization parameter for numerical stability

    Returns:
        ZCA whitening matrix W such that (X @ W) has identity covariance
        and is maximally similar to X
    """
    import cupy as cp

    n = len(X)

    # SVD of centered data
    _, s, Vt = cp.linalg.svd(X, full_matrices=False)

    # ZCA whitening: W = V @ diag(1/s) @ V.T * sqrt(n-1)
    scale = cp.sqrt(n - 1) / (s + epsilon)
    W = Vt.T @ (Vt * scale[:, None])

    return W


def regularized_inverse(A: "cp.ndarray", epsilon: float = 1e-6) -> "cp.ndarray":
    """Compute regularized matrix inverse.

    Computes (A + epsilon * I)^(-1) for numerical stability.

    Args:
        A: Square matrix
        epsilon: Regularization parameter

    Returns:
        Regularized inverse
    """
    import cupy as cp

    n = A.shape[0]
    regularized = A + epsilon * cp.eye(n, dtype=A.dtype)
    return cp.linalg.inv(regularized)


def ledoit_wolf_cov_gpu(X: "cp.ndarray") -> "cp.ndarray":
    """Compute regularized covariance using Ledoit-Wolf shrinkage.

    When n_samples is small relative to n_features, standard covariance
    estimates are noisy. Ledoit-Wolf shrinkage improves estimation by
    shrinking toward a structured target (scaled identity).

    Args:
        X: Input data (n_samples, n_features), does NOT need to be centered

    Returns:
        Shrinkage-regularized covariance matrix (n_features, n_features)
    """
    import cupy as cp
    from sklearn.covariance import LedoitWolf

    X_cpu = cp.asnumpy(X)
    lw = LedoitWolf().fit(X_cpu)
    return cp.asarray(lw.covariance_, dtype=X.dtype)


def fit_whiten(X: "cp.ndarray", k: int, epsilon: float = 1e-8) -> tuple["cp.ndarray", "cp.ndarray"]:
    """Fit PCA whitening transformation.

    Computes mean and whitening matrix W such that (X - mu) @ W
    has approximately identity covariance in k dimensions.

    Args:
        X: Input data (n_samples, n_features)
        k: Number of components to keep
        epsilon: Regularization for small eigenvalues

    Returns:
        mu: Mean vector (n_features,)
        W: Whitening matrix (n_features, k)
    """
    import cupy as cp

    n = len(X)
    mu = X.mean(axis=0)
    X_c = X - mu

    # Covariance matrix
    C = X_c.T @ X_c / (n - 1)

    # Eigendecompose (eigh returns ascending order)
    eigenvalues, eigenvectors = cp.linalg.eigh(C)

    # Sort descending and take top-k
    idx = cp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][:k]
    eigenvectors = eigenvectors[:, idx][:, :k]

    # Whitening matrix: W = V_k @ diag(1 / sqrt(lambda_k))
    scale = 1.0 / cp.sqrt(cp.maximum(eigenvalues, epsilon))
    W = eigenvectors * scale  # (n_features, k)

    return mu, W


def invsqrt(C: "cp.ndarray", epsilon: float = 1e-8) -> "cp.ndarray":
    """Compute symmetric inverse square root of positive semi-definite matrix.

    This is the CORAL alignment matrix: X @ invsqrt(C) transforms
    the covariance of X toward identity.

    Args:
        C: Covariance matrix (k, k)
        epsilon: Regularization for small eigenvalues

    Returns:
        Inverse square root matrix (k, k)
    """
    return fractional_matrix_power(C, -0.5)
