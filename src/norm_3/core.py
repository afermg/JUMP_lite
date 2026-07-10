"""GPU-accelerated transformer classes for morphological profile normalization.

This module provides cupy/cuml-based implementations of:
- RobustMAD: Median Absolute Deviation normalization
- Spherize: ZCA/PCA whitening transformation
- TVN: Typical Variation Normalization
- TVN_EFAAR: EFAAR-style TVN with CORAL
- PCATransform: PCA dimensionality reduction

All classes follow sklearn's fit/transform API pattern.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
from scipy.stats import median_abs_deviation as scipy_mad
from cuml.decomposition import PCA as cuPCA
from cuml.preprocessing import StandardScaler as cuStandardScaler

from norm_3.linalg import (
    covariance,
    fractional_matrix_power,
    is_ill_conditioned,
    condition_number,
    ledoit_wolf_cov_gpu,
    fit_whiten,
    invsqrt,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# TVN State Tracking (module-level for pipeline-wide tracking)
# =============================================================================

_tvn_ill_conditioned = False
_tvn_max_condition_number = 0.0


def reset_tvn_state():
    """Reset TVN ill-conditioning tracking state. Call at start of pipeline."""
    global _tvn_ill_conditioned, _tvn_max_condition_number
    _tvn_ill_conditioned = False
    _tvn_max_condition_number = 0.0


def get_tvn_state():
    """Get TVN ill-conditioning state. Returns (ill_conditioned, max_condition_number)."""
    return _tvn_ill_conditioned, _tvn_max_condition_number


def update_tvn_state(ill_conditioned: bool, cond_number: float):
    """Update global TVN state if ill-conditioned."""
    global _tvn_ill_conditioned, _tvn_max_condition_number
    if ill_conditioned:
        _tvn_ill_conditioned = True
    if cond_number > _tvn_max_condition_number:
        _tvn_max_condition_number = cond_number


# =============================================================================
# Spherize State Tracking (module-level for pipeline-wide tracking)
# =============================================================================

_spherize_ill_conditioned = False
_spherize_max_condition_number = 0.0


def reset_spherize_state():
    """Reset Spherize ill-conditioning tracking state. Call at start of pipeline."""
    global _spherize_ill_conditioned, _spherize_max_condition_number
    _spherize_ill_conditioned = False
    _spherize_max_condition_number = 0.0


def get_spherize_state():
    """Get Spherize ill-conditioning state. Returns (ill_conditioned, max_condition_number)."""
    return _spherize_ill_conditioned, _spherize_max_condition_number


def update_spherize_state(ill_conditioned: bool, cond_number: float):
    """Update global Spherize state if ill-conditioned."""
    global _spherize_ill_conditioned, _spherize_max_condition_number
    if ill_conditioned:
        _spherize_ill_conditioned = True
    if cond_number > _spherize_max_condition_number:
        _spherize_max_condition_number = cond_number


# =============================================================================
# Spherize Truncation State Tracking (module-level for metrics reporting)
# =============================================================================

_spherize_truncation_method: str | None = None
_spherize_truncation_k: int | None = None
_spherize_truncation_input_dims: int | None = None
_spherize_truncation_variance_removed: float | None = None


def reset_spherize_truncation_state():
    """Reset spherize truncation tracking. Call at start of pipeline."""
    global _spherize_truncation_method, _spherize_truncation_k
    global _spherize_truncation_input_dims, _spherize_truncation_variance_removed
    _spherize_truncation_method = None
    _spherize_truncation_k = None
    _spherize_truncation_input_dims = None
    _spherize_truncation_variance_removed = None


def get_spherize_truncation_state():
    """Get spherize truncation state for metrics reporting."""
    return {
        "spherize_truncation_method": _spherize_truncation_method,
        "spherize_truncation_k": _spherize_truncation_k,
        "spherize_truncation_input_dims": _spherize_truncation_input_dims,
        "spherize_truncation_k_pct": (
            round(_spherize_truncation_k / _spherize_truncation_input_dims * 100, 2)
            if _spherize_truncation_k is not None and _spherize_truncation_input_dims is not None
            else None
        ),
        "spherize_truncation_variance_removed": _spherize_truncation_variance_removed,
    }


def update_spherize_truncation_state(method: str, k: int, input_dims: int, variance_removed: float):
    """Record spherize truncation info for metrics."""
    global _spherize_truncation_method, _spherize_truncation_k
    global _spherize_truncation_input_dims, _spherize_truncation_variance_removed
    _spherize_truncation_method = method
    _spherize_truncation_k = k
    _spherize_truncation_input_dims = input_dims
    _spherize_truncation_variance_removed = variance_removed


# =============================================================================
# Helper Functions
# =============================================================================


def median_abs_deviation(arr, axis: int = 0):
    """Calculate median absolute deviation with scale factor for Gaussian consistency.

    Args:
        arr: Input array
        axis: Axis along which to compute MAD

    Returns:
        MAD values scaled by 1/1.4826 for Gaussian equivalence
    """
    median = cp.median(arr, axis=axis, keepdims=True)
    mad = cp.median(cp.abs(arr - median), axis=axis, keepdims=True)
    return mad / 1.4826


# =============================================================================
# CPU Transformer Classes (for operations where GPU overhead > benefit)
# =============================================================================


class RobustMAD_CPU:
    """Median Absolute Deviation normalization (CPU version).

    CPU is faster for median/MAD since these operations involve sorting
    which doesn't parallelize well on GPU.

    Formula: (x - median) / (MAD + epsilon)
    """

    def __init__(self, epsilon: float = 1e-18):
        self.epsilon = epsilon
        self.median_: np.ndarray | None = None
        self.mad_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "RobustMAD_CPU":
        """Compute median and MAD from data."""
        self.median_ = np.median(X, axis=0)
        self.mad_ = scipy_mad(X, axis=0).squeeze()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        return (X - self.median_) / (self.mad_ + self.epsilon)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class StandardScaler_CPU:
    """Standard normalization (CPU version).

    Formula: (x - mean) / std
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler_CPU":
        """Compute mean and std from data."""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        return (X - self.mean_) / (self.std_ + 1e-18)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


# =============================================================================
# GPU Transformer Classes
# =============================================================================


class RobustMAD:
    """Median Absolute Deviation normalization (GPU-accelerated).

    Robust to outliers, recommended for Cell Painting data.
    Formula: (x - median) / (MAD + epsilon)

    Example:
        >>> X = cp.random.randn(1000, 100)
        >>> scaler = RobustMAD()
        >>> X_normalized = scaler.fit_transform(X)
    """

    def __init__(self, epsilon: float = 1e-18):
        """Initialize RobustMAD.

        Args:
            epsilon: Small value added to MAD for numerical stability
        """
        self.epsilon = epsilon
        self.median_: cp.ndarray | None = None
        self.mad_: cp.ndarray | None = None

    def fit(self, X: cp.ndarray) -> "RobustMAD":
        """Compute median and MAD from data.

        Args:
            X: Training data (n_samples, n_features)

        Returns:
            self
        """
        self.median_ = cp.median(X, axis=0)
        self.mad_ = median_abs_deviation(X, axis=0).squeeze()
        return self

    def transform(self, X: cp.ndarray) -> cp.ndarray:
        """Apply MAD normalization.

        Args:
            X: Data to transform (n_samples, n_features)

        Returns:
            Normalized data
        """
        return (X - self.median_) / (self.mad_ + self.epsilon)

    def fit_transform(self, X: cp.ndarray) -> cp.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class Spherize:
    """ZCA/PCA whitening transformation (GPU-accelerated).

    Transforms covariance matrix to identity, removing correlations.
    Should be applied AFTER RobustMAD normalization.

    Methods:
        - "ZCA": Zero-phase whitening (maximally similar to original)
        - "PCA": PCA whitening (decorrelated components)
        - "ZCA-cor": ZCA on correlation matrix (standardized first)
        - "PCA-cor": PCA on correlation matrix (standardized first)

    Example:
        >>> X = cp.random.randn(1000, 100)
        >>> spherize = Spherize(method="ZCA-cor")
        >>> X_whitened = spherize.fit_transform(X)
    """

    def __init__(
        self,
        method: str = "ZCA-cor",
        epsilon: float = 1e-6,
        center: bool = True,
        remove_variance_threshold: float | None = None,
        remove_variance_method: str = "threshold",
        n_permutations: int = 10,
    ):
        """Initialize Spherize.

        Args:
            method: Whitening method (ZCA, PCA, ZCA-cor, PCA-cor)
            epsilon: Regularization for numerical stability
            center: Whether to center data (default True)
            remove_variance_threshold: If set, remove top k PCs that explain
                this fraction of variance (e.g. 0.80 = remove 80%), projecting
                onto remaining dims. Reduces dimensionality.
                Ignored when remove_variance_method="mp" or "pa".
            remove_variance_method: How to determine k PCs to remove:
                - "threshold": remove top k PCs covering threshold% of variance
                - "mp": Marchenko-Pastur — remove PCs whose eigenvalues exceed
                  the analytical upper edge of the random noise distribution
                - "pa": Parallel analysis — permute each column independently,
                  compute SVD on shuffled data, find where real singular values
                  drop below the random maximum envelope
            n_permutations: Number of permutations for parallel analysis (default 10)
        """
        self.method = method
        self.epsilon = epsilon
        self.center = center
        self.remove_variance_threshold = remove_variance_threshold
        self.remove_variance_method = remove_variance_method
        self.n_permutations = n_permutations
        self.W_: cp.ndarray | None = None
        self.mean_: cp.ndarray | None = None
        self.std_: cp.ndarray | None = None
        self.ill_conditioned_: bool = False
        self.condition_number_: float | None = None
        self.k_removed_: int | None = None
        self.variance_removed_: float | None = None

    def fit(self, X: cp.ndarray) -> "Spherize":
        """Compute whitening matrix via SVD.

        Args:
            X: Training data (n_samples, n_features)

        Returns:
            self
        """
        X_work = X.copy()

        # For correlation-based variants, standardize (center + scale)
        # For covariance-based variants, just center
        if "cor" in self.method:
            self.mean_ = X_work.mean(axis=0)
            self.std_ = X_work.std(axis=0)
            self.std_ = cp.where(self.std_ < 1e-10, 1.0, self.std_)  # Avoid division by zero
            X_work = (X_work - self.mean_) / self.std_
        elif self.center:
            self.mean_ = X_work.mean(axis=0)
            X_work = X_work - self.mean_

        # SVD decomposition
        n_samples, n_features = X_work.shape

        _, Sigma, Vt = cp.linalg.svd(X_work, full_matrices=False)

        # CRITICAL: When n_samples < n_features, Vt has shape (n_samples, n_features)
        # This causes W to have shape (n_features, n_samples), reducing dimensionality.
        # We need to pad W to (n_features, n_features) to preserve dimensionality.
        rank = len(Sigma)  # Actual rank = min(n_samples, n_features)

        # Truncated projection: remove top k PCs based on chosen method
        do_truncation = (
            self.remove_variance_method in ("mp", "pa")
            or (self.remove_variance_method == "threshold" and self.remove_variance_threshold is not None)
        )
        if do_truncation:
            eigenvalues = Sigma ** 2

            if self.remove_variance_method == "pa":
                # Parallel analysis: shuffle each column independently to destroy
                # correlations, compute SVD on permuted data, track element-wise
                # max singular value across permutations. k = first index where
                # real singular value drops below the random envelope.
                Xr = X_work.copy()
                Sr_max = cp.zeros(rank)
                for perm_i in range(self.n_permutations):
                    # Shuffle each column independently
                    for col in range(n_features):
                        idx = cp.random.permutation(n_samples)
                        Xr[:, col] = X_work[idx, col]
                    Si = cp.linalg.svd(Xr, compute_uv=False)
                    Sr_max = cp.maximum(Sr_max, Si)
                # Find crossover: first index where real S drops below random max
                crossover = cp.argwhere(Sigma <= Sr_max)
                if len(crossover) > 0:
                    k = int(crossover[0, 0])
                else:
                    k = rank  # All singular values exceed random — no truncation
                print(f"  PA: {self.n_permutations} permutations, k={k} signal PCs "
                      f"(of {rank}), top_S={float(Sigma[0]):.2f}, random_max={float(Sr_max[0]):.2f}")
                if k == 0:
                    print(f"  PA: k=0, no signal PCs found, falling through to full spherize")
                elif k >= rank:
                    print(f"  PA: all PCs are signal, falling through to full spherize")
                    k = 0  # Signal "no truncation"
                else:
                    k = min(k, rank - 1)  # Keep at least 1 component

            elif self.remove_variance_method == "mp":
                # Marchenko-Pastur: remove PCs whose eigenvalues exceed the
                # analytical upper edge of the noise distribution.
                # For standardized data (ZCA-cor/PCA-cor), population noise
                # eigenvalue is 1, so lambda_+ = (1 + sqrt(p/n))^2.
                gamma = float(n_features) / float(n_samples)
                # Normalize eigenvalues to per-sample scale (like np.cov)
                eig_scaled = eigenvalues / (n_samples - 1)
                mp_upper = (1.0 + cp.sqrt(cp.array(gamma))) ** 2
                k = int(cp.sum(eig_scaled > mp_upper))
                print(f"  MP: gamma={gamma:.4f}, upper_edge={float(mp_upper):.4f}, "
                      f"k={k} signal PCs, top_eig={float(eig_scaled[0]):.4f}")
                if k == 0:
                    # No eigenvalues exceed MP upper edge — skip truncation
                    print(f"  MP: no eigenvalues exceed upper edge, falling through to full spherize")
                else:
                    k = min(k, rank - 1)  # Keep at least 1 component
            else:
                # Variance threshold method
                cumvar = cp.cumsum(eigenvalues) / cp.sum(eigenvalues)
                k = int(cp.searchsorted(cumvar, cp.array(self.remove_variance_threshold))) + 1
                k = min(k, rank - 1)  # Keep at least 1 component

            if k > 0:
                self.k_removed_ = k
                cumvar = cp.cumsum(eigenvalues) / cp.sum(eigenvalues)
                self.variance_removed_ = float(cumvar[k - 1])

                # Record truncation state for metrics reporting
                update_spherize_truncation_state(
                    method=self.remove_variance_method,
                    k=k,
                    input_dims=int(n_features),
                    variance_removed=self.variance_removed_,
                )

                # Project onto remaining p-k dims (drop top k PCs)
                W = Vt[k:].T  # shape (n_features, rank-k)

                self.W_ = W
                update_spherize_state(False, 0.0)
                return self

        # Check condition number (ratio of largest to smallest singular value)
        sigma_min = float(Sigma[-1])
        sigma_max = float(Sigma[0])
        if sigma_min > 0:
            cond_number = sigma_max / sigma_min
        else:
            cond_number = float("inf")
        self.condition_number_ = cond_number

        if cond_number > 1e10:
            self.ill_conditioned_ = True
            update_spherize_state(True, cond_number)
            warnings.warn(
                f"Spherize: Data is ill-conditioned (cond={cond_number:.2e}, "
                f"sigma_min={sigma_min:.2e}, sigma_max={sigma_max:.2e}). "
                f"Epsilon={self.epsilon} may not be sufficient."
            )
        else:
            update_spherize_state(False, cond_number)

        # Whitening matrix
        # Vt has shape (rank, n_features)
        # W will have shape (n_features, rank)
        W = (Vt / (Sigma + self.epsilon)[:, None]).T * cp.sqrt(n_samples - 1)

        # ZCA rotation to preserve similarity to original data
        if "ZCA" in self.method:
            W = W @ Vt  # (n_features, rank) @ (rank, n_features) = (n_features, n_features)
            # For ZCA, W should already be square

        # If rank < n_features, we need to pad W to maintain dimensionality
        # The null space (unobserved features) are passed through unchanged
        if rank < n_features:
            warnings.warn(
                f"Spherize: n_samples ({n_samples}) < n_features ({n_features}). "
                f"Whitening only {rank} dimensions, passing through {n_features - rank} null space dimensions."
            )
            if "ZCA" in self.method:
                # ZCA already produces (n_features, n_features) matrix
                pass
            else:
                # PCA produces (n_features, rank) matrix
                # Pad with identity for null space: W becomes (n_features, n_features)
                W_padded = cp.eye(n_features, dtype=W.dtype)
                W_padded[:, :rank] = W
                W = W_padded

        self.W_ = W
        return self

    def transform(self, X: cp.ndarray) -> cp.ndarray:
        """Apply whitening transformation.

        Args:
            X: Data to transform (n_samples, n_features)

        Returns:
            Whitened data
        """
        X_work = X.copy()
        if "cor" in self.method:
            X_work = (X_work - self.mean_) / self.std_
        elif self.center and self.mean_ is not None:
            X_work = X_work - self.mean_
        return X_work @ self.W_

    def fit_transform(self, X: cp.ndarray) -> cp.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class TVN:
    """Typical Variation Normalization (GPU-accelerated).

    Removes batch-specific covariance by fractional matrix power transformation.
    This is the legacy/simple TVN implementation.

    Example:
        >>> X = cp.random.randn(1000, 100)
        >>> tvn = TVN(alpha=0.5)
        >>> X_normalized = tvn.fit_transform(X)
    """

    def __init__(self, alpha: float = 0.5, epsilon: float = 1.0):
        """Initialize TVN.

        Args:
            alpha: Fractional power for covariance transformation (default 0.5)
            epsilon: Regularization for ill-conditioned matrices
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.mean_: cp.ndarray | None = None
        self.cov_alpha_: cp.ndarray | None = None
        self.ill_conditioned_: bool = False
        self.condition_number_: float | None = None

    def fit(self, X: cp.ndarray) -> "TVN":
        """Fit TVN on control samples.

        Args:
            X: Training data (n_samples, n_features)

        Returns:
            self
        """
        if len(X) < 2:
            raise ValueError("TVN requires at least 2 samples")

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        cov = covariance(X_centered.T, rowvar=True)
        cond_number = condition_number(cov)
        self.condition_number_ = float(cond_number)
        regularization = self.epsilon

        if cond_number > 1e10:
            self.ill_conditioned_ = True
            regularization = max(self.epsilon, cond_number / 1e8)
            update_tvn_state(True, cond_number)
            warnings.warn(f"TVN: Covariance ill-conditioned (cond={cond_number:.2e}) (Shape = {X.shape})")
        else:
            update_tvn_state(False, cond_number)

        regularized_cov = cov + regularization * cp.eye(cov.shape[0], dtype=cov.dtype)
        self.cov_alpha_ = fractional_matrix_power(regularized_cov, -self.alpha)
        return self

    def transform(self, X: cp.ndarray) -> cp.ndarray:
        """Apply TVN transformation.

        Args:
            X: Data to transform (n_samples, n_features)

        Returns:
            Transformed data
        """
        X_centered = X - self.mean_
        return (X_centered @ self.cov_alpha_).real

    def fit_transform(self, X: cp.ndarray) -> cp.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class TVN_EFAAR:
    """Typical Variation Normalization matching EFAAR benchmarking (GPU-accelerated).

    Implements the proper EFAAR TVN workflow based on CORAL:
    1. Fit target covariance on ALL controls (global reference)
    2. For each batch: whiten using batch covariance, recolor using target covariance

    Based on CORAL (CORrelation ALignment) method.

    Reference:
        https://github.com/recursionpharma/EFAAR_benchmarking/blob/trunk/efaar_benchmarking/efaar.py
    """

    def __init__(self, epsilon: float = 0.5):
        """Initialize TVN_EFAAR.

        Args:
            epsilon: Regularization for covariance matrices (default 0.5 per EFAAR)
        """
        self.epsilon = epsilon
        self.target_cov_: cp.ndarray | None = None
        self.target_cov_sqrt_: cp.ndarray | None = None
        self.n_features_: int | None = None
        self.ill_conditioned_: bool = False
        self.condition_number_: float | None = None

    def fit(self, X_controls: cp.ndarray) -> "TVN_EFAAR":
        """Fit target covariance on ALL control samples.

        Args:
            X_controls: Control embeddings from ALL batches (n_samples, n_features)

        Returns:
            self
        """
        if len(X_controls) < 2:
            raise ValueError("TVN_EFAAR requires at least 2 control samples")

        self.n_features_ = X_controls.shape[1]

        # Compute target covariance from all controls with regularization
        self.target_cov_ = covariance(X_controls.T, rowvar=True) + self.epsilon * cp.eye(self.n_features_)

        # Check condition number
        cond_number = condition_number(self.target_cov_)
        self.condition_number_ = float(cond_number)

        if cond_number > 1e10:
            self.ill_conditioned_ = True
            update_tvn_state(True, cond_number)
            warnings.warn(f"TVN_EFAAR: Target covariance ill-conditioned (cond={cond_number:.2e})")
        else:
            update_tvn_state(False, cond_number)

        # Pre-compute target covariance sqrt for recoloring
        self.target_cov_sqrt_ = fractional_matrix_power(self.target_cov_, 0.5)

        print(f"  TVN_EFAAR fit on {len(X_controls)} controls, {self.n_features_} features")
        print(f"  Target covariance condition number: {cond_number:.2e}")

        return self

    def transform_batch(self, X_batch: cp.ndarray, X_batch_controls: cp.ndarray) -> cp.ndarray:
        """Transform a single batch using CORAL: whiten with batch cov, recolor with target cov.

        Args:
            X_batch: All embeddings in this batch (n_samples, n_features)
            X_batch_controls: Control embeddings in this batch (n_controls, n_features)

        Returns:
            Transformed embeddings for this batch
        """
        if self.target_cov_sqrt_ is None:
            raise ValueError("Must call fit() before transform_batch()")

        if len(X_batch_controls) < 2:
            warnings.warn(f"TVN_EFAAR: Batch has only {len(X_batch_controls)} controls, skipping CORAL")
            return X_batch

        # Compute source (batch-specific) covariance with regularization
        source_cov = covariance(X_batch_controls.T, rowvar=True) + self.epsilon * cp.eye(self.n_features_)

        # Check batch condition number
        batch_cond = condition_number(source_cov)
        if batch_cond > 1e10:
            self.ill_conditioned_ = True
            update_tvn_state(True, batch_cond)
            warnings.warn(f"TVN_EFAAR: Batch covariance ill-conditioned (cond={batch_cond:.2e})")

        # CORAL transformation: whiten with source, recolor with target
        source_cov_inv_sqrt = fractional_matrix_power(source_cov, -0.5)

        X_whitened = X_batch @ source_cov_inv_sqrt
        X_recolored = X_whitened @ self.target_cov_sqrt_

        return X_recolored.real


def tvn_efaar_on_controls(
    embeddings: cp.ndarray,
    control_mask: cp.ndarray,
    batch_labels: cp.ndarray | None = None,
    epsilon: float = 0.5,
) -> cp.ndarray:
    """Apply TVN (Typical Variation Normalization) matching EFAAR implementation.

    This implements the CORAL-based TVN from:
    https://github.com/recursionpharma/EFAAR_benchmarking

    Note: This function assumes embeddings have ALREADY been:
    1. Centered/scaled on controls (globally)
    2. PCA transformed (fit on controls)
    3. Centered/scaled on controls (per-batch)

    Args:
        embeddings: Embeddings to normalize (n_samples, n_features)
        control_mask: Boolean mask indicating control samples
        batch_labels: Batch labels for each sample (None = single batch)
        epsilon: Regularization for covariance matrices (default 0.5 per EFAAR)

    Returns:
        Normalized embeddings with batch effects removed via CORAL
    """
    embeddings = embeddings.copy()
    n_features = embeddings.shape[1]

    # Fit target covariance on ALL controls
    target_cov = covariance(embeddings[control_mask].T, rowvar=True) + epsilon * cp.eye(n_features)
    target_cov_sqrt = fractional_matrix_power(target_cov, 0.5)

    cond_number = condition_number(target_cov)
    if cond_number > 1e10:
        update_tvn_state(True, cond_number)
        warnings.warn(f"TVN_EFAAR: Target covariance ill-conditioned (cond={cond_number:.2e})")
    else:
        update_tvn_state(False, cond_number)

    print(f"  TVN_EFAAR: {int(control_mask.sum())} controls, condition={cond_number:.2e}")

    if batch_labels is not None:
        # Per-batch CORAL transformation
        unique_batches = cp.unique(batch_labels)
        for batch in unique_batches:
            batch_mask = batch_labels == batch
            batch_control_mask = batch_mask & control_mask

            n_batch_controls = int(batch_control_mask.sum())
            if n_batch_controls < 2:
                warnings.warn(f"TVN_EFAAR: Batch '{batch}' has only {n_batch_controls} controls, skipping")
                continue

            # Compute source (batch) covariance
            source_cov = covariance(embeddings[batch_control_mask].T, rowvar=True) + epsilon * cp.eye(n_features)
            source_cov_inv_sqrt = fractional_matrix_power(source_cov, -0.5)

            # CORAL: whiten with source, recolor with target
            embeddings[batch_mask] = embeddings[batch_mask] @ source_cov_inv_sqrt
            embeddings[batch_mask] = embeddings[batch_mask] @ target_cov_sqrt

        embeddings = embeddings.real

    return embeddings


class TVN_Original:
    """Original TVN: Global PCA whitening + per-batch CORAL (no recoloring).

    Unlike TVN_EFAAR, this does NOT recolor to a target covariance.
    Each batch is aligned to identity covariance.

    Args:
        k: Number of PCA whitening components (default 50)
           Rule of thumb: k <= (B * n_neg) / 5
        epsilon: Regularization for small eigenvalues (default 1e-8)
    """

    def __init__(self, k: int = 50, epsilon: float = 1e-8):
        self.k = k
        self.epsilon = epsilon
        self.mu_: cp.ndarray | None = None
        self.W_: cp.ndarray | None = None
        self.coral_: dict = {}
        self.ill_conditioned_: bool = False
        self.condition_number_: float | None = None

    def fit(self, negcons: dict) -> "TVN_Original":
        """Fit TVN on negative control embeddings per batch.

        Args:
            negcons: Dict mapping batch_id -> negcon embeddings (n_neg, n_features)

        Returns:
            self
        """
        # Step 1: Pool all negcons
        all_neg = cp.vstack([negcons[b] for b in negcons])
        print(f"  TVN_Original: pooled {len(all_neg)} negcons, k={self.k}")

        # Step 2: Fit global PCA whitening
        self.mu_, self.W_ = fit_whiten(all_neg, self.k, self.epsilon)

        # Step 3 & 4: Whiten per-batch negcons and fit CORAL
        for b in negcons:
            neg_whitened = (negcons[b] - self.mu_) @ self.W_
            C_b = ledoit_wolf_cov_gpu(neg_whitened)

            # Check condition number
            cond = condition_number(C_b)
            if self.condition_number_ is None or cond > self.condition_number_:
                self.condition_number_ = float(cond)
            if cond > 1e10:
                self.ill_conditioned_ = True
                update_tvn_state(True, cond)

            self.coral_[b] = invsqrt(C_b, self.epsilon)

        return self

    def transform_batch(self, X: cp.ndarray, batch_id) -> cp.ndarray:
        """Transform data for a specific batch.

        Args:
            X: Embeddings to transform (n_samples, n_features)
            batch_id: Batch identifier

        Returns:
            Transformed embeddings (n_samples, k)
        """
        X_whitened = (X - self.mu_) @ self.W_
        return (X_whitened @ self.coral_[batch_id]).real


class TVN_Cascade:
    """Cascade TVN: Two-stage whitening for small n_neg situations.

    Stage 1: Large global whitening (reliable with pooled negcons, no CORAL)
    Stage 2: Smaller whitening + per-batch CORAL (feasible with limited negcons)

    Args:
        k1: Stage 1 components (default 100). Rule: k1 <= (B * n_neg) / 5
        k2: Stage 2 components (default 10). Rule: k2 <= n_neg / 3
        epsilon: Regularization (default 1e-8)
    """

    def __init__(self, k1: int = 100, k2: int = 10, epsilon: float = 1e-8):
        self.k1 = k1
        self.k2 = k2
        self.epsilon = epsilon
        # Stage 1 params
        self.mu1_: cp.ndarray | None = None
        self.W1_: cp.ndarray | None = None
        # Stage 2 params
        self.mu2_: cp.ndarray | None = None
        self.W2_: cp.ndarray | None = None
        self.coral2_: dict = {}
        self.ill_conditioned_: bool = False
        self.condition_number_: float | None = None

    def fit(self, negcons: dict) -> "TVN_Cascade":
        """Fit Cascade TVN on negative control embeddings.

        Args:
            negcons: Dict mapping batch_id -> negcon embeddings (n_neg, n_features)

        Returns:
            self
        """
        # --- Stage 1: Global Whitening (no CORAL) ---
        all_neg = cp.vstack([negcons[b] for b in negcons])
        print(f"  TVN_Cascade Stage 1: pooled {len(all_neg)} negcons, k1={self.k1}")

        self.mu1_, self.W1_ = fit_whiten(all_neg, self.k1, self.epsilon)

        # Project all negcons to Stage 1 space
        neg_s1 = {b: (negcons[b] - self.mu1_) @ self.W1_ for b in negcons}

        # --- Stage 2: Per-Batch Residual Correction ---
        all_neg_s1 = cp.vstack([neg_s1[b] for b in neg_s1])
        print(f"  TVN_Cascade Stage 2: k2={self.k2}")

        self.mu2_, self.W2_ = fit_whiten(all_neg_s1, self.k2, self.epsilon)

        # Fit per-batch CORAL in Stage 2 space
        for b in neg_s1:
            neg_s2 = (neg_s1[b] - self.mu2_) @ self.W2_
            C_b = ledoit_wolf_cov_gpu(neg_s2)

            # Check condition number
            cond = condition_number(C_b)
            if self.condition_number_ is None or cond > self.condition_number_:
                self.condition_number_ = float(cond)
            if cond > 1e10:
                self.ill_conditioned_ = True
                update_tvn_state(True, cond)

            self.coral2_[b] = invsqrt(C_b, self.epsilon)

        return self

    def transform_batch(self, X: cp.ndarray, batch_id) -> cp.ndarray:
        """Transform data for a specific batch.

        Args:
            X: Embeddings to transform (n_samples, n_features)
            batch_id: Batch identifier

        Returns:
            Transformed embeddings (n_samples, k2)
        """
        X_s1 = (X - self.mu1_) @ self.W1_
        X_s2 = (X_s1 - self.mu2_) @ self.W2_
        return (X_s2 @ self.coral2_[batch_id]).real


class PCATransform:
    """PCA dimensionality reduction (GPU-accelerated via cuML).

    Wrapper around cuml.decomposition.PCA with sklearn-compatible API.

    Example:
        >>> X = cp.random.randn(1000, 500)
        >>> pca = PCATransform(n_components=100)
        >>> X_reduced = pca.fit_transform(X)
    """

    def __init__(
        self,
        n_components: int | float = 0.95,
        whiten: bool = False,
    ):
        """Initialize PCA.

        Args:
            n_components: Number of components to keep. If float, select components
                         explaining that fraction of variance.
            whiten: Whether to whiten the output
        """
        self.n_components = n_components
        self.whiten = whiten
        self.pca_: cuPCA | None = None

    def fit(self, X: cp.ndarray) -> "PCATransform":
        """Fit PCA model.

        Args:
            X: Training data (n_samples, n_features)

        Returns:
            self
        """
        # cuML PCA requires integer n_components
        if isinstance(self.n_components, float):
            # For variance ratio, we need to determine n_components empirically
            # First fit with all components to get explained variance
            temp_pca = cuPCA(n_components=min(X.shape), whiten=self.whiten)
            temp_pca.fit(X)
            cumsum = cp.cumsum(temp_pca.explained_variance_ratio_)
            n_components = int(cp.searchsorted(cumsum, self.n_components).get()) + 1
            n_components = min(n_components, X.shape[1])
        else:
            n_components = self.n_components

        self.pca_ = cuPCA(n_components=n_components, whiten=self.whiten)
        self.pca_.fit(X)
        return self

    def transform(self, X: cp.ndarray) -> cp.ndarray:
        """Apply PCA transformation.

        Args:
            X: Data to transform (n_samples, n_features)

        Returns:
            Transformed data with reduced dimensions
        """
        return self.pca_.transform(X)

    def fit_transform(self, X: cp.ndarray) -> cp.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    @property
    def n_components_fitted(self) -> int:
        """Number of components after fitting."""
        return self.pca_.n_components_ if self.pca_ else 0

    @property
    def explained_variance_ratio_(self) -> cp.ndarray | None:
        """Explained variance ratio per component."""
        return self.pca_.explained_variance_ratio_ if self.pca_ else None


class StandardScaler:
    """Standard scaler (GPU-accelerated via cuML).

    Standardizes features by removing the mean and scaling to unit variance.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        """Initialize StandardScaler.

        Args:
            with_mean: Center data to zero mean
            with_std: Scale to unit variance
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self._scaler = cuStandardScaler(with_mean=with_mean, with_std=with_std)

    def fit(self, X: cp.ndarray) -> "StandardScaler":
        """Fit scaler."""
        self._scaler.fit(X)
        return self

    def transform(self, X: cp.ndarray) -> cp.ndarray:
        """Transform data."""
        return self._scaler.transform(X)

    def fit_transform(self, X: cp.ndarray) -> cp.ndarray:
        """Fit and transform."""
        return self._scaler.fit_transform(X)

    @property
    def mean_(self) -> cp.ndarray | None:
        """Per-feature mean."""
        return self._scaler.mean_ if hasattr(self._scaler, "mean_") else None

    @property
    def scale_(self) -> cp.ndarray | None:
        """Per-feature scale (std)."""
        return self._scaler.scale_ if hasattr(self._scaler, "scale_") else None


# =============================================================================
# Feature Selection Functions (GPU-accelerated)
# =============================================================================


def variance_threshold(
    X: cp.ndarray,
    feature_names: list[str],
    freq_cut: float = 0.05,
    unique_cut: float = 0.01,
    var_threshold: float | None = None,
) -> list[str]:
    """Remove low-variance features (GPU-accelerated).

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        freq_cut: Ratio of 2nd to most common value
        unique_cut: Ratio of unique values to samples
        var_threshold: Minimum variance required (None = skip variance check)

    Returns:
        List of features to keep
    """
    if len(feature_names) == 0 or len(X) == 0:
        return feature_names

    n_samples = len(X)
    keep_features = []

    for i, feat in enumerate(feature_names):
        col = X[:, i]

        if var_threshold is not None:
            variance = float(cp.nanvar(col))
            if not cp.isfinite(variance) or variance < var_threshold:
                continue

        # Filter out NaN/inf for counting
        valid_mask = cp.isfinite(col)
        col_clean = col[valid_mask]

        if len(col_clean) == 0:
            continue

        # Count unique values (transfer to CPU for this operation)
        col_clean_cpu = cp.asnumpy(col_clean)
        import numpy as np
        unique_values, counts = np.unique(col_clean_cpu, return_counts=True)
        n_unique = len(unique_values)

        if n_unique / n_samples < unique_cut:
            continue

        if len(counts) >= 2:
            sorted_counts = np.sort(counts)[::-1]
            freq_ratio = sorted_counts[1] / sorted_counts[0] if sorted_counts[0] > 0 else 0
            if freq_ratio < freq_cut:
                continue

        keep_features.append(feat)

    return keep_features


def correlation_threshold(
    X: cp.ndarray,
    feature_names: list[str],
    threshold: float = 0.9,
) -> list[str]:
    """Remove highly correlated features using greedy independent set (GPU-accelerated).

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        threshold: Correlation cutoff

    Returns:
        List of features to keep
    """
    # Compute correlation matrix on GPU
    corr = cp.corrcoef(X, rowvar=False)
    corr = cp.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    # Build adjacency graph for highly correlated features
    adj_graph = cp.abs(corr) > threshold
    adj_graph_cpu = cp.asnumpy(adj_graph)

    # Greedy independent set algorithm (on CPU for graph operations)
    redundant_indices = _greedy_independent_set(adj_graph_cpu)

    return [feat for i, feat in enumerate(feature_names) if i not in redundant_indices]


def _greedy_independent_set(adj_graph) -> list[int]:
    """Find redundant features using greedy independent set algorithm (CPU)."""
    import numpy as np

    adj_graph = adj_graph.copy()
    np.fill_diagonal(adj_graph, 0)

    remaining_nodes = set(range(adj_graph.shape[0]))
    independent_set = []

    while remaining_nodes:
        degrees = np.sum(adj_graph, axis=1)
        min_degree_node = min(remaining_nodes, key=lambda node: degrees[node])

        independent_set.append(min_degree_node)

        neighbors = set(np.where(adj_graph[min_degree_node] == 1)[0])
        remaining_nodes -= neighbors | {min_degree_node}

        adj_graph[min_degree_node, :] = 0
        adj_graph[:, min_degree_node] = 0
        for neighbor in neighbors:
            adj_graph[neighbor, :] = 0
            adj_graph[:, neighbor] = 0

    return list(set(range(len(adj_graph))).difference(independent_set))


def drop_outliers(
    X: cp.ndarray,
    feature_names: list[str],
    outlier_cutoff: float = 500,
) -> list[str]:
    """Remove features with extreme values using z-score (GPU-accelerated).

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        outlier_cutoff: Maximum absolute z-score allowed

    Returns:
        List of features to keep
    """
    if len(feature_names) == 0 or len(X) == 0:
        return feature_names

    # Z-score normalization
    X_z = (X - cp.nanmean(X, axis=0)) / (cp.nanstd(X, axis=0) + 1e-8)

    max_vals = cp.abs(cp.nanmax(X_z, axis=0))
    min_vals = cp.abs(cp.nanmin(X_z, axis=0))

    max_vals = cp.nan_to_num(max_vals, nan=cp.inf, posinf=cp.inf, neginf=cp.inf)
    min_vals = cp.nan_to_num(min_vals, nan=cp.inf, posinf=cp.inf, neginf=cp.inf)

    vals_within_cutoff = (max_vals <= outlier_cutoff) & (min_vals <= outlier_cutoff)
    keep_indices = cp.where(vals_within_cutoff)[0]

    return [feature_names[int(i)] for i in cp.asnumpy(keep_indices)]


def blocklist_filter(
    feature_names: list[str],
    blocklist: list[str] | None = None,
) -> list[str]:
    """Remove blocklisted features by pattern matching.

    Args:
        feature_names: List of feature names
        blocklist: List of patterns to block (any feature containing these strings is removed)
                   Default patterns include Location, ObjectNumber, BoundingBox, ExecutionTime

    Returns:
        List of features to keep
    """
    if blocklist is None:
        blocklist = [
            "Location_Center",
            "Location_Max",
            "Location_Min",
            "Number_Object_Number",
            "ExecutionTime",
            "BoundingBox",
            "ObjectNumber",  # Catches FirstClosestObjectNumber, SecondClosestObjectNumber, etc.
        ]

    return [f for f in feature_names if not any(pattern in f for pattern in blocklist)]


def include_features(
    feature_names: list[str],
    patterns: list[str],
) -> list[str]:
    """Keep only features matching any of the given patterns.

    Args:
        feature_names: List of feature names
        patterns: List of patterns to include (only features containing these strings are kept)

    Returns:
        List of features to keep

    Example:
        >>> include_features(['Cells_Intensity_Mean', 'Nuclei_Shape_Area'], ['Intensity'])
        ['Cells_Intensity_Mean']
    """
    return [f for f in feature_names if any(pattern in f for pattern in patterns)]


def exclude_features(
    feature_names: list[str],
    patterns: list[str],
) -> list[str]:
    """Remove features matching any of the given patterns.

    Args:
        feature_names: List of feature names
        patterns: List of patterns to exclude (any feature containing these strings is removed)

    Returns:
        List of features to keep

    Example:
        >>> exclude_features(['Cells_Intensity_Mean', 'Nuclei_Shape_Area'], ['Intensity'])
        ['Nuclei_Shape_Area']
    """
    return [f for f in feature_names if not any(pattern in f for pattern in patterns)]


def include_from_csv(
    feature_names: list[str],
    csv_path: str,
    column: str = "feature",
) -> list[str]:
    """Keep only features listed in a CSV file.

    Args:
        feature_names: List of feature names
        csv_path: Path to CSV file containing feature names
        column: Name of column containing feature names (default: "feature")

    Returns:
        List of features to keep

    Example CSV format:
        feature
        Cells_Intensity_MeanIntensity_DNA
        Cells_Intensity_MeanIntensity_ER
        Nuclei_Shape_Area
    """
    import polars as pl
    from pathlib import Path

    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pl.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in CSV. Available columns: {df.columns}")

    features_to_keep = set(df[column].to_list())
    return [f for f in feature_names if f in features_to_keep]


def exclude_from_csv(
    feature_names: list[str],
    csv_path: str,
    column: str = "feature",
) -> list[str]:
    """Remove features listed in a CSV file.

    Args:
        feature_names: List of feature names
        csv_path: Path to CSV file containing feature names to exclude
        column: Name of column containing feature names (default: "feature")

    Returns:
        List of features to keep

    Example CSV format:
        feature
        Cells_Location_Center_X
        Cells_Location_Center_Y
        Nuclei_Number_Object_Number
    """
    import polars as pl
    from pathlib import Path

    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pl.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in CSV. Available columns: {df.columns}")

    features_to_exclude = set(df[column].to_list())
    return [f for f in feature_names if f not in features_to_exclude]


# =============================================================================
# Well Position Correction
# =============================================================================


def well_position_correct(
    X: np.ndarray,
    well_positions: np.ndarray,
) -> np.ndarray:
    """
    Correct for well position effects by subtracting the mean of each well position.

    For each well position (e.g., A01, A02, ...), subtracts the mean across all plates.
    This removes systematic spatial biases that are consistent across plates.

    Based on Broad Institute JUMP profiling recipe 'well_correct' step.

    Args:
        X: Feature matrix (n_samples, n_features)
        well_positions: Array of well position labels for each sample

    Returns:
        Corrected feature matrix
    """
    unique_wells = np.unique(well_positions)
    X_corrected = X.copy()

    for well in unique_wells:
        mask = well_positions == well
        if mask.sum() > 0:
            well_mean = np.nanmean(X[mask], axis=0)
            X_corrected[mask] = X[mask] - well_mean

    return X_corrected


# =============================================================================
# Inverse Normal Transform
# =============================================================================


def inverse_normal_transform(X: np.ndarray) -> np.ndarray:
    """
    Apply rank-based inverse normal transformation to features.

    For each feature, ranks the values and maps them to normal distribution quantiles.
    This makes the marginal distribution of each feature approximately normal,
    which can improve downstream analyses that assume normality.

    Based on Broad Institute JUMP profiling recipe 'INT' step.

    Args:
        X: Feature matrix (n_samples, n_features)

    Returns:
        Inverse normal transformed feature matrix
    """
    from scipy.stats import norm

    n, p = X.shape

    # Vectorized ranking using argsort twice: argsort(argsort(x)) gives ranks
    # This gives 0-based ranks, add 1 for 1-based ranks
    ranks = np.argsort(np.argsort(X, axis=0), axis=0) + 1  # Shape: (n, p)

    # Map ranks to normal quantiles using Blom's formula: (r - 3/8) / (n + 1/4)
    # This adjustment avoids -inf/+inf at the tails
    quantiles = (ranks - 0.375) / (n + 0.25)
    X_transformed = norm.ppf(quantiles)

    # Handle any NaN values in original data (keep them as NaN)
    nan_mask = np.isnan(X)
    X_transformed[nan_mask] = np.nan

    return X_transformed


# =============================================================================
# Sample Normalization (L1/L2/Max)
# =============================================================================


def sample_normalize(
    X: np.ndarray,
    norm: Literal["l1", "l2", "max"] = "l2",
) -> np.ndarray:
    """
    Apply L1/L2/max normalization per sample (row-wise).

    Args:
        X: Feature matrix (n_samples, n_features)
        norm: Normalization type ("l1", "l2", or "max")

    Returns:
        Normalized feature matrix
    """
    if norm == "l2":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
    elif norm == "l1":
        norms = np.abs(X).sum(axis=1, keepdims=True)
    else:  # max
        norms = np.abs(X).max(axis=1, keepdims=True)

    X_norm = X / (norms + 1e-10)
    return X_norm
