"""Normalization transformations for morphological profiles."""

import numpy as np
import polars as pl
from scipy.linalg import fractional_matrix_power
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler


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
    # Scale factor for Gaussian consistency
    return mad / 1.4826


class RobustMAD:
    """
    Median Absolute Deviation normalization.

    Robust to outliers, recommended for Cell Painting data.
    Formula: (x - median) / (MAD + epsilon)
    """

    def __init__(self, epsilon: float = 1e-18):
        self.epsilon = epsilon
        self.median_ = None
        self.mad_ = None

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
        self.W_ = None
        self.scaler_ = None

    def fit(self, X: np.ndarray) -> "Spherize":
        """Compute whitening matrix via SVD."""
        X_work = X.copy()

        # For correlation-based variants, standardize first
        if "cor" in self.method:
            self.scaler_ = StandardScaler()
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

    def __init__(self, alpha: float = 0.5, epsilon: float = 1e-3):
        self.alpha = alpha
        self.epsilon = epsilon
        self.mean_ = None
        self.cov_alpha_ = None

    def fit(self, X: np.ndarray) -> "TVN":
        import warnings

        if len(X) < 2:
            raise ValueError("TVN requires at least 2 samples")
        
        # Center data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Check std
        stds = np.sort(np.std(X_centered, axis=0))[:5]
        print("Feature stddev before TVN:", stds)
        print("Feature shape:", X_centered.shape)
        print("Max and Min pre mean feature stddev:", np.max(X), np.min(X))
        print("Max and Min feature stddev:", np.max(X_centered), np.min(X_centered))
        
        # Compute covariance
        cov = np.cov(X_centered.T)

        # Check condition number and apply adaptive regularization
        cond_number = np.linalg.cond(cov)
        regularization = self.epsilon

        if cond_number > 1e10:
            warnings.warn(
                f"Covariance ill-conditioned (cond={cond_number:.2e}). "
                f"Applying adaptive regularization.",
                UserWarning
            )
            # Adaptive regularization based on condition number
            regularization = max(self.epsilon, cond_number / 1e8)

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
        self.pca_ = None

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


def normalize_profiles_extended(
    df: pl.DataFrame,
    features: list[str],
    method: str = "robustmad",
    batch_col: str | None = None,
    control_key: str | None = None,
    control_col: str = "Metadata_pert_type",
    fit_on_controls: bool = False,
    epsilon: float = 1e-6,
    spherize_method: str = "ZCA-cor",
    spherize_epsilon: float = 1e-6,
    tvn_alpha: float = 0.5,
    tvn_epsilon: float = 1e-3,
    pca_n_components: int | float = 0.95,
    pca_whiten: bool = False,
) -> pl.DataFrame:
    """
    Extended normalization with TVN and PCA support.

    Adds support for:
    - TVN (Typical Variation Normalization)
    - PCA (dimensionality reduction)
    - PCA with control-based fitting

    Args:
        df: Input DataFrame
        features: Feature columns to normalize
        method: Normalization method (robustmad, standardize, robustize, spherize, tvn, pca, pca_control)
        batch_col: Column for batch grouping
        control_key: Control identifier for fit_on_controls
        control_col: Column containing control labels
        fit_on_controls: Fit scaler only on controls
        epsilon: Stability parameter for robustmad
        spherize_method: Method for spherize
        spherize_epsilon: Epsilon for spherize
        tvn_alpha: Fractional power for TVN (default 0.5)
        tvn_epsilon: Stability for TVN (default 1e-3)
        pca_n_components: Components for PCA (int or variance fraction)
        pca_whiten: Whether to whiten PCA components

    Returns:
        Normalized DataFrame (may have different features for PCA methods)
    """
    # For "none" method, return as-is (no normalization)
    if method == "none":
        return df

    # For PCA variants, return transformed with new feature names
    if method in ["pca", "pca_control"]:
        # Get fit data
        if method == "pca_control" and fit_on_controls and control_key and control_col in df.columns:
            control_mask = df[control_col] == control_key
            df_fit = df.filter(control_mask)
            if len(df_fit) == 0:
                raise ValueError(f"No control samples found with {control_col}={control_key}")
        else:
            df_fit = df

        X_fit = df_fit.select(features).to_numpy()
        X_all = df.select(features).to_numpy()

        # Fit and transform
        pca = PCATransform(n_components=pca_n_components, whiten=pca_whiten)
        pca.fit(X_fit)
        X_transformed = pca.transform(X_all)

        # Create new feature names
        n_comp = X_transformed.shape[1]
        new_features = [f"PC{i+1}" for i in range(n_comp)]

        # Build new DataFrame
        metadata_cols = [c for c in df.columns if c.startswith("Metadata")]
        df_meta = df.select(metadata_cols)
        df_pca = pl.DataFrame({feat: X_transformed[:, i] for i, feat in enumerate(new_features)})

        return pl.concat([df_meta, df_pca], how="horizontal")

    # Helper functions for creating scalers with parameters
    def create_tvn():
        return TVN(alpha=tvn_alpha, epsilon=tvn_epsilon)

    def create_robustmad():
        return RobustMAD(epsilon=epsilon)

    def create_spherize():
        return Spherize(method=spherize_method, epsilon=spherize_epsilon)

    # Select scaler factory based on method
    if method == "tvn":
        scaler_class = create_tvn
    elif method == "robustmad":
        scaler_class = create_robustmad
    elif method == "standardize":
        scaler_class = StandardScaler
    elif method == "robustize":
        scaler_class = RobustScaler
    elif method == "spherize":
        scaler_class = create_spherize
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Supported: robustmad, standardize, robustize, spherize, tvn, pca, pca_control"
        )

    # Normalize by batch or whole dataset
    if batch_col and batch_col in df.columns:
        normalized_dfs = []

        for batch in df[batch_col].unique().to_list():
            batch_mask = df[batch_col] == batch
            batch_df = df.filter(batch_mask)

            X = batch_df.select(features).to_numpy()

            # Fit on controls or all data
            if fit_on_controls and control_key and control_col in df.columns:
                control_mask = batch_df[control_col] == control_key
                X_fit = batch_df.filter(control_mask).select(features).to_numpy()
                if len(X_fit) == 0:
                    raise ValueError(
                        f"No control samples found for batch {batch} "
                        f"with {control_col}={control_key}"
                    )
                scaler = scaler_class()
                scaler.fit(X_fit)
                X_norm = scaler.transform(X)
            else:
                scaler = scaler_class()
                X_norm = scaler.fit_transform(X)

            batch_df = batch_df.with_columns(
                [pl.Series(name=feat, values=X_norm[:, i]) for i, feat in enumerate(features)]
            )
            normalized_dfs.append(batch_df)

        df_norm = pl.concat(normalized_dfs)
    else:
        # Normalize entire dataset
        X = df.select(features).to_numpy()

        if fit_on_controls and control_key and control_col in df.columns:
            control_mask = df[control_col] == control_key
            X_fit = df.filter(control_mask).select(features).to_numpy()
            if len(X_fit) == 0:
                raise ValueError(
                    f"No control samples found with {control_col}={control_key}"
                )
            scaler = scaler_class()
            scaler.fit(X_fit)
            X_norm = scaler.transform(X)
        else:
            scaler = scaler_class()
            X_norm = scaler.fit_transform(X)

        df_norm = df.with_columns(
            [pl.Series(name=feat, values=X_norm[:, i]) for i, feat in enumerate(features)]
        )

    return df_norm
