"""Core transformations for morphological profiles.

This module provides:
- Transformer classes: RobustMAD, Spherize, TVN, PCATransform
- Normalization functions following pycytominer-style APIs
- Feature selection functions
- Aggregation functions
- trommel-style basic_cleanup function

Design follows:
- trommel: basic_cleanup() orchestration pattern
- pycytominer: clean function signatures with method parameter
- EFAAR: batch-aware processing with control fitting
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import polars as pl
from scipy.linalg import fractional_matrix_power
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

from .io import get_numeric_features, infer_columns


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


def update_tvn_state(ill_conditioned: bool, condition_number: float):
    """Update global TVN state if ill-conditioned."""
    global _tvn_ill_conditioned, _tvn_max_condition_number
    if ill_conditioned:
        _tvn_ill_conditioned = True
    if condition_number > _tvn_max_condition_number:
        _tvn_max_condition_number = condition_number


# =============================================================================
# Transformer Classes
# =============================================================================


def median_abs_deviation(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Calculate median absolute deviation with scale factor for Gaussian consistency.

    Args:
        arr: Input array
        axis: Axis along which to compute MAD

    Returns:
        MAD values scaled by 1/1.4826 for Gaussian equivalence
    """
    median = np.median(arr, axis=axis, keepdims=True)
    mad = np.median(np.abs(arr - median), axis=axis, keepdims=True)
    return mad / 1.4826


class RobustMAD:
    """
    Median Absolute Deviation normalization.

    Robust to outliers, recommended for Cell Painting data.
    Formula: (x - median) / (MAD + epsilon)
    """

    def __init__(self, epsilon: float = 1e-18):
        self.epsilon = epsilon
        self.median_: np.ndarray | None = None
        self.mad_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "RobustMAD":
        self.median_ = np.median(X, axis=0)
        self.mad_ = median_abs_deviation(X, axis=0).squeeze()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.median_) / (self.mad_ + self.epsilon)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class Spherize:
    """
    ZCA/PCA whitening transformation for batch correction.

    Transforms covariance matrix to identity, removing correlations.
    Should be applied AFTER RobustMAD normalization.
    """

    def __init__(
        self,
        method: str = "ZCA-cor",
        epsilon: float = 1e-6,
        center: bool = True,
    ):
        self.method = method
        self.epsilon = epsilon
        self.center = center
        self.W_: np.ndarray | None = None
        self.scaler_: StandardScaler | None = None

    def fit(self, X: np.ndarray) -> "Spherize":
        """Compute whitening matrix via SVD."""
        X_work = X.copy()

        # For correlation-based variants, standardize (center + scale)
        # For covariance-based variants, just center
        if "cor" in self.method:
            self.scaler_ = StandardScaler()
            X_work = self.scaler_.fit_transform(X_work)
        elif self.center:
            self.scaler_ = StandardScaler(with_std=False)
            X_work = self.scaler_.fit_transform(X_work)

        # SVD decomposition
        _, Sigma, Vt = np.linalg.svd(X_work, full_matrices=False)

        # Whitening matrix
        W = (Vt / (Sigma + self.epsilon)[:, None]).T * np.sqrt(len(X) - 1)

        # ZCA rotation to preserve similarity to original data
        if "ZCA" in self.method:
            W = W @ Vt

        self.W_ = W
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply whitening transformation."""
        X_work = X.copy()
        if self.scaler_ is not None:
            X_work = self.scaler_.transform(X_work)
        return X_work @ self.W_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class TVN:
    """
    Typical Variation Normalization from EFAAR benchmarking.

    Removes batch-specific covariance by fractional matrix power transformation.
    Particularly effective for batch correction while preserving biological signal.
    """

    def __init__(self, alpha: float = 0.5, epsilon: float = 1.0):
        self.alpha = alpha
        self.epsilon = epsilon
        self.mean_: np.ndarray | None = None
        self.cov_alpha_: np.ndarray | None = None
        self.ill_conditioned_: bool = False  # Track if matrix was ill-conditioned
        self.condition_number_: float | None = None  # Store condition number for reporting

    def fit(self, X: np.ndarray) -> "TVN":
        if len(X) < 2:
            raise ValueError("TVN requires at least 2 samples")

        # Center data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Debug info
        stds = np.sort(np.std(X_centered, axis=0))[:5]
        print("Feature stddev before TVN:", stds)
        print("Feature shape:", X_centered.shape)
        print("Max and Min pre mean feature stddev:", np.max(X), np.min(X))
        print("Max and Min feature stddev:", np.max(X_centered), np.min(X_centered))

        # Compute covariance
        cov = np.cov(X_centered.T)

        # Check condition number and apply adaptive regularization
        cond_number = np.linalg.cond(cov)
        self.condition_number_ = cond_number  # Store for reporting
        regularization = self.epsilon

        if cond_number > 1e10:
            self.ill_conditioned_ = True  # Flag for reporting
            regularization = max(self.epsilon, cond_number / 1e8)

            # Update global state for pipeline-wide tracking
            update_tvn_state(True, cond_number)

            if not hasattr(TVN, '_warned_ill_conditioned'):
                TVN._warned_ill_conditioned = True

                issues = []
                n_samples, n_features = X_centered.shape

                ratio = n_samples / n_features
                if ratio < 3:
                    issues.append(f"Low sample/feature ratio: {ratio:.2f}")

                corr_matrix = np.corrcoef(X_centered.T)
                np.fill_diagonal(corr_matrix, 0)
                high_corr_count = np.sum(np.abs(corr_matrix) > 0.95) // 2
                if high_corr_count > 0:
                    issues.append(f"Highly correlated features: {high_corr_count} pairs")

                variances = np.var(X_centered, axis=0)
                low_var_count = np.sum(variances < 1e-10)
                if low_var_count > 0:
                    issues.append(f"Near-zero variance features: {low_var_count}")

                msg = f"TVN: Covariance ill-conditioned (cond={cond_number:.2e}). Using regularization={regularization:.2e}."
                if issues:
                    msg += "\n  Potential causes: " + ", ".join(issues)
                warnings.warn(msg, UserWarning)
        else:
            # Update global state even if not ill-conditioned (to track max)
            update_tvn_state(False, cond_number)

        # Fractional matrix power with regularization
        self.cov_alpha_ = fractional_matrix_power(
            cov + regularization * np.eye(cov.shape[0]), -self.alpha
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean_
        return (X_centered @ self.cov_alpha_).real

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class PCATransform:
    """PCA dimensionality reduction with optional whitening."""

    def __init__(
        self,
        n_components: int | float = 0.95,
        whiten: bool = False,
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.pca_: PCA | None = None

    def fit(self, X: np.ndarray) -> "PCATransform":
        self.pca_ = PCA(n_components=self.n_components, whiten=self.whiten)
        self.pca_.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca_.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_components_fitted(self) -> int:
        return self.pca_.n_components_ if self.pca_ else 0


# =============================================================================
# Feature Selection Functions
# =============================================================================


def variance_threshold(
    df: pl.DataFrame,
    features: list[str],
    freq_cut: float = 0.05,
    unique_cut: float = 0.01,
    var_threshold: float | None = None,
) -> list[str]:
    """
    Remove low-variance features.

    Args:
        df: Input DataFrame
        features: Feature columns to evaluate
        freq_cut: Ratio of 2nd to most common value
        unique_cut: Ratio of unique values to samples
        var_threshold: Minimum variance required (None = skip variance check)

    Returns:
        List of features to keep
    """
    if not features or len(df) == 0:
        return features

    X = df.select(features).to_numpy()
    n_samples = len(X)

    keep_features = []
    count_variance_drop = 0
    count_unique_drop = 0
    count_freq_drop = 0

    for i, feat in enumerate(features):
        col = X[:, i]

        if var_threshold is not None:
            variance = np.nanvar(col)
            if not np.isfinite(variance) or variance < var_threshold:
                count_variance_drop += 1
                continue

        col_clean = col[~np.isnan(col) & np.isfinite(col)]
        if len(col_clean) == 0:
            count_variance_drop += 1
            continue

        unique_values, counts = np.unique(col_clean, return_counts=True)
        n_unique = len(unique_values)

        if n_unique / n_samples < unique_cut:
            count_unique_drop += 1
            continue

        if len(counts) >= 2:
            sorted_counts = np.sort(counts)[::-1]
            freq_ratio = sorted_counts[1] / sorted_counts[0] if sorted_counts[0] > 0 else 0
            if freq_ratio < freq_cut:
                count_freq_drop += 1
                continue

        keep_features.append(feat)

    print(f"    Variance threshold: dropped {count_variance_drop} features")
    print(f"    Unique value threshold: dropped {count_unique_drop} features")
    print(f"    Frequency ratio threshold: dropped {count_freq_drop} features")

    return keep_features


def correlation_threshold(
    df: pl.DataFrame,
    features: list[str],
    threshold: float = 0.9,
    method: str = "pearson",
) -> list[str]:
    """
    Remove highly correlated features using greedy independent set.

    Args:
        df: Input DataFrame
        features: Feature columns to evaluate
        threshold: Correlation cutoff
        method: Correlation method (pearson, spearman, kendall)

    Returns:
        List of features to keep
    """
    X = df.select(features).to_numpy()

    if method == "pearson":
        corr = np.corrcoef(X, rowvar=False)
    elif method == "spearman":
        from scipy.stats import spearmanr
        corr, _ = spearmanr(X, axis=0)
    elif method == "kendall":
        from scipy.stats import kendalltau
        n_features = len(features)
        corr = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i, n_features):
                tau, _ = kendalltau(X[:, i], X[:, j])
                corr[i, j] = corr[j, i] = tau
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    adj_graph = np.abs(corr) > threshold
    redundant_indices = _greedy_independent_set(adj_graph)

    return [feat for i, feat in enumerate(features) if i not in redundant_indices]


def _greedy_independent_set(adj_graph: np.ndarray) -> list[int]:
    """Find redundant features using greedy independent set algorithm."""
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

    return list(set(np.arange(len(adj_graph))).difference(independent_set))


def drop_na_columns(
    df: pl.DataFrame,
    features: list[str],
    na_cutoff: float = 0.05,
) -> tuple[list[str], list[str]]:
    """
    Remove features with too many missing or infinite values.

    Args:
        df: Input DataFrame
        features: Feature columns to evaluate
        na_cutoff: Maximum fraction of NaN/inf values allowed

    Returns:
        Tuple of (features_to_keep, features_dropped)
    """
    n_samples = len(df)
    keep_features = []
    dropped_features = []

    for feat in features:
        n_null = df[feat].null_count()
        n_nan = df[feat].is_nan().sum()
        n_inf = df[feat].is_infinite().sum()
        n_invalid = n_null + n_nan + n_inf

        if n_invalid / n_samples <= na_cutoff:
            keep_features.append(feat)
        else:
            dropped_features.append(feat)

    return keep_features, dropped_features


def drop_outliers(
    df: pl.DataFrame,
    features: list[str],
    outlier_cutoff: float = 500,
) -> list[str]:
    """
    Remove features with extreme values using z-score normalization.

    Args:
        df: Input DataFrame
        features: Feature columns to evaluate
        outlier_cutoff: Maximum absolute z-score allowed

    Returns:
        List of features to keep
    """
    if not features or len(df) == 0:
        return features

    X = df.select(features).to_numpy()
    X_z_norm = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-8)

    max_vals = np.abs(np.nanmax(X_z_norm, axis=0))
    min_vals = np.abs(np.nanmin(X_z_norm, axis=0))

    max_vals = np.nan_to_num(max_vals, nan=np.inf, posinf=np.inf, neginf=np.inf)
    min_vals = np.nan_to_num(min_vals, nan=np.inf, posinf=np.inf, neginf=np.inf)

    vals_within_cutoff = (max_vals <= outlier_cutoff) & (min_vals <= outlier_cutoff)
    keep_indices = np.where(vals_within_cutoff)[0]

    return [features[i] for i in keep_indices]


def blocklist_filter(
    df: pl.DataFrame,
    features: list[str],
    blocklist: list[str] | None = None,
) -> list[str]:
    """Remove blocklisted features."""
    if blocklist is None:
        blocklist = [
            "Location_Center",
            "Location_Max",
            "Location_Min",
            "Number_Object_Number",
            "ExecutionTime",
        ]

    return [f for f in features if not any(pattern in f for pattern in blocklist)]


# =============================================================================
# High-Level API Functions
# =============================================================================


def normalize(
    df: pl.DataFrame,
    features: list[str] | None = None,
    method: Literal["robustmad", "standardize", "robustize", "tvn", "spherize", "none"] = "robustmad",
    batch_col: str | None = None,
    fit_on_controls: bool = False,
    control_col: str = "Metadata_control_type",
    control_key: str = "negcon",
    epsilon: float = 1e-6,
    tvn_alpha: float = 0.5,
    tvn_epsilon: float = 1.0,
    spherize_method: str = "ZCA-cor",
    spherize_epsilon: float = 1e-6,
) -> pl.DataFrame:
    """
    Normalize features using specified method.

    Args:
        df: Input DataFrame
        features: Feature columns to normalize (None = infer)
        method: Normalization method
        batch_col: Column for batch grouping (None = global)
        fit_on_controls: Fit scaler only on controls
        control_col: Column containing control labels
        control_key: Control identifier
        epsilon: Stability parameter for robustmad
        tvn_alpha: Fractional power for TVN
        tvn_epsilon: Stability for TVN
        spherize_method: Method for spherize
        spherize_epsilon: Epsilon for spherize

    Returns:
        Normalized DataFrame
    """
    if method == "none":
        return df

    if features is None:
        features, _ = infer_columns(df, "Metadata_")
        features = get_numeric_features(df, features)

    # Create scaler factory
    def create_scaler():
        if method == "tvn":
            return TVN(alpha=tvn_alpha, epsilon=tvn_epsilon)
        elif method == "robustmad":
            return RobustMAD(epsilon=epsilon)
        elif method == "standardize":
            return StandardScaler()
        elif method == "robustize":
            return RobustScaler()
        elif method == "spherize":
            return Spherize(method=spherize_method, epsilon=spherize_epsilon)
        else:
            raise ValueError(f"Unknown method: {method}")

    # Normalize by batch or globally
    if batch_col and batch_col in df.columns:
        normalized_dfs = []

        for batch in sorted(df[batch_col].unique().to_list()):
            batch_mask = df[batch_col] == batch
            batch_df = df.filter(batch_mask)
            X = batch_df.select(features).to_numpy()

            if fit_on_controls and control_key and control_col in df.columns:
                control_mask = batch_df[control_col] == control_key
                X_fit = batch_df.filter(control_mask).select(features).to_numpy()
                if len(X_fit) == 0:
                    raise ValueError(f"No controls for batch {batch}")
                scaler = create_scaler()
                scaler.fit(X_fit)
                X_norm = scaler.transform(X)
            else:
                scaler = create_scaler()
                X_norm = scaler.fit_transform(X)

            batch_df = batch_df.with_columns(
                [pl.Series(name=feat, values=X_norm[:, i]) for i, feat in enumerate(features)]
            )
            normalized_dfs.append(batch_df)

        return pl.concat(normalized_dfs)
    else:
        X = df.select(features).to_numpy()

        if fit_on_controls and control_key and control_col in df.columns:
            control_mask = df[control_col] == control_key
            X_fit = df.filter(control_mask).select(features).to_numpy()
            if len(X_fit) == 0:
                raise ValueError(f"No controls found")
            scaler = create_scaler()
            scaler.fit(X_fit)
            X_norm = scaler.transform(X)
        else:
            scaler = create_scaler()
            X_norm = scaler.fit_transform(X)

        return df.with_columns(
            [pl.Series(name=feat, values=X_norm[:, i]) for i, feat in enumerate(features)]
        )


def sample_normalize(
    df: pl.DataFrame,
    features: list[str] | None = None,
    norm: Literal["l1", "l2", "max"] = "l2",
) -> pl.DataFrame:
    """
    Apply L1/L2/max normalization per sample (row-wise).

    Args:
        df: Input DataFrame
        features: Feature columns to normalize (None = infer)
        norm: Normalization type

    Returns:
        DataFrame with normalized features
    """
    if features is None:
        features, _ = infer_columns(df, "Metadata_")
        features = get_numeric_features(df, features)

    X = df.select(features).to_numpy()

    if norm == "l2":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
    elif norm == "l1":
        norms = np.abs(X).sum(axis=1, keepdims=True)
    else:  # max
        norms = np.abs(X).max(axis=1, keepdims=True)

    X_norm = X / (norms + 1e-10)
    df_norm = pl.DataFrame(X_norm, schema=features)

    return pl.concat([df.select(pl.exclude(features)), df_norm], how="horizontal")


def select_features(
    df: pl.DataFrame,
    features: list[str] | None = None,
    operations: list[dict] | None = None,
    na_cutoff: float = 0.30,
    outlier_cutoff: float = 500,
    correlation_threshold_val: float = 0.9,
    variance_threshold_val: float | None = None,
    freq_cut: float = 0.05,
    unique_cut: float = 0.01,
    verbose: bool = True,
) -> tuple[pl.DataFrame, dict]:
    """
    Select features by applying sequential operations.

    Args:
        df: Input DataFrame
        features: Starting feature list (None = infer)
        operations: List of operation configs (if provided, ignores other params)
        na_cutoff: Max fraction of NaN/inf values
        outlier_cutoff: Max z-score for outliers
        correlation_threshold_val: Correlation cutoff
        variance_threshold_val: Min variance (None = skip)
        freq_cut: Frequency ratio cutoff
        unique_cut: Unique value ratio cutoff
        verbose: Print progress

    Returns:
        Tuple of (filtered DataFrame, stats dict)
    """
    if features is None:
        features, metadata = infer_columns(df, "Metadata_")
        features = get_numeric_features(df, features)
    else:
        _, metadata = infer_columns(df, "Metadata_")

    stats = {"initial_features": len(features)}
    current_features = features.copy()

    # If operations provided, use them
    if operations:
        for op in operations:
            name = op["name"]
            params = {k: v for k, v in op.items() if k != "name"}
            n_before = len(current_features)

            if name == "variance_threshold":
                current_features = variance_threshold(df, current_features, **params)
            elif name == "correlation_threshold":
                current_features = correlation_threshold(df, current_features, **params)
            elif name == "drop_na_columns":
                current_features, _ = drop_na_columns(df, current_features, **params)
            elif name == "drop_outliers":
                current_features = drop_outliers(df, current_features, **params)
            elif name == "blocklist":
                current_features = blocklist_filter(df, current_features, **params)

            if verbose:
                n_dropped = n_before - len(current_features)
                print(f"  {name}: dropped {n_dropped} features ({len(current_features)} remaining)")

            if len(current_features) == 0 and n_before > 0:
                print(f"  WARNING: {name} would remove all features, skipping")
                current_features = features.copy()  # Restore
    else:
        # Apply standard operations
        current_features, dropped = drop_na_columns(df, current_features, na_cutoff)
        stats["dropped_na"] = len(dropped)
        if verbose:
            print(f"  drop_na: dropped {len(dropped)} features")

        n_before = len(current_features)
        current_features = drop_outliers(df, current_features, outlier_cutoff)
        stats["dropped_outliers"] = n_before - len(current_features)
        if verbose:
            print(f"  drop_outliers: dropped {stats['dropped_outliers']} features")

        if variance_threshold_val is not None:
            n_before = len(current_features)
            current_features = variance_threshold(
                df, current_features,
                var_threshold=variance_threshold_val,
                freq_cut=freq_cut,
                unique_cut=unique_cut,
            )
            stats["dropped_variance"] = n_before - len(current_features)

        n_before = len(current_features)
        current_features = correlation_threshold(df, current_features, correlation_threshold_val)
        stats["dropped_correlation"] = n_before - len(current_features)
        if verbose:
            print(f"  correlation_threshold: dropped {stats['dropped_correlation']} features")

    stats["final_features"] = len(current_features)
    df_filtered = df.select(metadata + current_features)

    return df_filtered, stats


def aggregate(
    df: pl.DataFrame,
    features: list[str] | None = None,
    group_by: list[str] | None = None,
    method: Literal["median", "mean"] = "median",
) -> pl.DataFrame:
    """
    Aggregate replicates to single profiles.

    Args:
        df: Input DataFrame
        features: Feature columns to aggregate (None = infer)
        group_by: Grouping columns (default: ["Metadata_Plate", "Metadata_Well"])
        method: Aggregation method

    Returns:
        Aggregated profiles
    """
    if group_by is None:
        group_by = ["Metadata_Plate", "Metadata_Well"]

    if features is None:
        features, metadata = infer_columns(df, "Metadata_")
        features = get_numeric_features(df, features)
    else:
        _, metadata = infer_columns(df, "Metadata_")

    missing_cols = [col for col in group_by if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Group-by columns not found: {missing_cols}")

    agg_func = pl.median if method == "median" else pl.mean
    metadata_to_keep = [m for m in metadata if m not in group_by]

    agg_exprs = [agg_func(feat).alias(feat) for feat in features]
    meta_exprs = [pl.first(meta).alias(meta) for meta in metadata_to_keep]

    return df.group_by(group_by).agg(agg_exprs + meta_exprs)


def basic_cleanup(
    df: pl.DataFrame,
    na_cutoff: float = 0.30,
    outlier_cutoff: float = 500,
    correlation_threshold_val: float = 0.9,
) -> tuple[pl.DataFrame, dict]:
    """
    trommel-style basic cleanup: NaN removal, outlier filtering, correlation pruning.

    Args:
        df: Input DataFrame
        na_cutoff: Max fraction of NaN/inf values
        outlier_cutoff: Max z-score for outliers
        correlation_threshold_val: Correlation cutoff

    Returns:
        Tuple of (cleaned DataFrame, stats dict)
    """
    return select_features(
        df,
        na_cutoff=na_cutoff,
        outlier_cutoff=outlier_cutoff,
        correlation_threshold_val=correlation_threshold_val,
        verbose=True,
    )
