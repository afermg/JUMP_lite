"""Normalization transformations for morphological profiles."""

import numpy as np
import polars as pl
from scipy.linalg import fractional_matrix_power
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler


def sample_normalize(
    df: pl.DataFrame,
    features: list[str],
    norm: str = "l2",
) -> pl.DataFrame:
    """
    Apply L1 or L2 normalization per sample (row-wise).

    Each sample's feature vector is scaled to unit norm.
    Useful for embeddings before other normalization steps.

    Args:
        df: Input DataFrame
        features: Feature columns to normalize
        norm: 'l1', 'l2', or 'max'

    Returns:
        DataFrame with normalized features
    """
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

        # For correlation-based variants, standardize (center + scale)
        # For covariance-based variants, just center
        if "cor" in self.method:
            self.scaler_ = StandardScaler()  # center + scale
            X_work = self.scaler_.fit_transform(X_work)
        elif self.center:
            self.scaler_ = StandardScaler(with_std=False)  # center only
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

    def __init__(self, alpha: float = 0.5, epsilon: float = 1.0):  # epsilon=1.0 per Ando 2017
        self.alpha = alpha
        self.epsilon = epsilon
        self.mean_ = None
        self.cov_alpha_ = None
        self.ill_conditioned_ = False  # Track if matrix was ill-conditioned
        self.condition_number_ = None  # Store condition number for reporting

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
        self.condition_number_ = cond_number  # Store for reporting
        regularization = self.epsilon

        if cond_number > 1e10:
            self.ill_conditioned_ = True  # Flag for reporting
            # Adaptive regularization based on condition number
            regularization = max(self.epsilon, cond_number / 1e8)

            # Diagnose potential causes (only once per session)
            if not hasattr(TVN, '_warned_ill_conditioned'):
                TVN._warned_ill_conditioned = True

                # Check potential causes
                issues = []
                n_samples, n_features = X_centered.shape

                # 1. Check sample to feature ratio
                ratio = n_samples / n_features
                if ratio < 3:
                    issues.append(f"Low sample/feature ratio: {ratio:.2f} (have {n_samples} samples, {n_features} features)")

                # 2. Check for highly correlated features
                corr_matrix = np.corrcoef(X_centered.T)
                np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlation
                max_corr = np.nanmax(np.abs(corr_matrix))
                high_corr_count = np.sum(np.abs(corr_matrix) > 0.95) // 2  # Divide by 2 for symmetry
                if high_corr_count > 0:
                    issues.append(f"Highly correlated features: {high_corr_count} pairs with |r|>0.95 (max={max_corr:.3f})")

                # 3. Check for low variance features
                variances = np.var(X_centered, axis=0)
                low_var_count = np.sum(variances < 1e-10)
                if low_var_count > 0:
                    issues.append(f"Near-zero variance features: {low_var_count} features with var<1e-10")

                # Build warning message
                msg = (
                    f"TVN: Covariance ill-conditioned (cond={cond_number:.2e}). "
                    f"Using adaptive regularization={regularization:.2e}."
                )
                if issues:
                    msg += "\n  Potential causes:\n    - " + "\n    - ".join(issues)
                    msg += "\n  Consider: more aggressive corr pruning, filter_features, or higher tvn_epsilon"

                warnings.warn(msg, UserWarning)

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


class Harmony:
    """
    Harmony batch correction via iterative PCA adjustment (GPU-accelerated).

    From Korsunsky et al. 2019 (Nature Methods).
    Works by adjusting principal components to remove batch effects
    while preserving biological variation.

    Uses rapids_singlecell for GPU-accelerated computation.
    """

    def __init__(
        self,
        n_pcs: int = 50,
        theta: float = 2.0,
        sigma: float = 0.1,
        n_clusters: int | None = None,
        max_iter_harmony: int = 10,
    ):
        """
        Args:
            n_pcs: Number of principal components to use
            theta: Diversity clustering penalty (higher = more mixing)
            sigma: Width of soft kmeans clusters
            n_clusters: Number of clusters (None = auto, min(100, N/30))
            max_iter_harmony: Maximum Harmony iterations
        """
        self.n_pcs = n_pcs
        self.theta = theta
        self.sigma = sigma
        self.n_clusters = n_clusters
        self.max_iter_harmony = max_iter_harmony
        self.pca_ = None

    def fit_transform(self, X: np.ndarray, batch: np.ndarray) -> np.ndarray:
        """
        Apply Harmony batch correction using rapids_singlecell.

        Args:
            X: Feature matrix (samples x features)
            batch: Batch labels array

        Returns:
            Corrected PCA embedding (samples x n_pcs)
        """
        try:
            import anndata as ad
            import rapids_singlecell as rsc
        except ImportError:
            raise ImportError(
                "Harmony requires: rapids-singlecell, cupy, anndata. "
                "Install with: pip install rapids-singlecell cupy-cuda11x anndata"
            )

        # First apply PCA
        n_pcs = min(self.n_pcs, X.shape[1], X.shape[0] - 1)
        self.pca_ = PCA(n_components=n_pcs)
        X_pca = self.pca_.fit_transform(X)

        # rapids_singlecell works with AnnData
        adata = ad.AnnData(X)
        adata.obs["batch"] = batch
        adata.obsm["X_pca"] = X_pca

        kwargs = {
            "theta": self.theta,
            "sigma": self.sigma,
            "max_iter_harmony": self.max_iter_harmony,
        }
        if self.n_clusters is not None:
            kwargs["n_clusters"] = self.n_clusters

        rsc.pp.harmony_integrate(adata, key="batch", **kwargs)
        return adata.obsm["X_pca_harmony"]


class ComBat:
    """
    ComBat batch correction using empirical Bayes.

    From Johnson et al. 2007 (Biostatistics).
    Models batch effects as multiplicative and additive noise
    and removes them using a Bayesian framework.

    Unlike Harmony, ComBat works directly on features (not PCA)
    and returns corrected features with the same dimensions.
    """

    def __init__(self, par_prior: bool = True, precision: float | None = 0.01):
        """
        Args:
            par_prior: Use parametric prior (True) or non-parametric (False)
            precision: Precision level for computation (default 0.01).
                       Helps avoid division by zero warnings with small covariances.
                       Set to None to disable.
        """
        self.par_prior = par_prior
        self.precision = precision

    def fit_transform(self, X: np.ndarray, batch: np.ndarray) -> np.ndarray:
        """
        Apply ComBat batch correction.

        Args:
            X: Feature matrix (samples x features)
            batch: Batch labels array

        Returns:
            Corrected feature matrix (same dimensions as input)
        """
        use_inmoose = False
        try:
            from combat.pycombat import pycombat
        except ImportError:
            try:
                from inmoose.pycombat import pycombat_norm as pycombat
                use_inmoose = True
            except ImportError:
                raise ImportError(
                    "ComBat requires pycombat or inmoose. "
                    "Install with: pip install combat  OR  pip install inmoose"
                )

        import pandas as pd

        # pycombat expects features as rows, samples as columns
        df = pd.DataFrame(X.T)
        batch_list = list(batch)

        # inmoose supports precision parameter, original pycombat does not
        if use_inmoose and self.precision is not None:
            corrected = pycombat(df, batch_list, par_prior=self.par_prior, precision=self.precision)
        else:
            corrected = pycombat(df, batch_list, par_prior=self.par_prior)
        return corrected.values.T


# Module-level state to track TVN ill-conditioning across pipeline run
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
    tvn_epsilon: float = 1.0,  # Per Ando 2017
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
        tvn_epsilon: Stability for TVN (default 1.0 per Ando 2017)
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
        tvn = TVN(alpha=tvn_alpha, epsilon=tvn_epsilon)
        return tvn

    def update_tvn_state(scaler):
        """Update global TVN state if scaler is TVN and was ill-conditioned."""
        global _tvn_ill_conditioned, _tvn_max_condition_number
        if isinstance(scaler, TVN) and scaler.condition_number_ is not None:
            if scaler.ill_conditioned_:
                _tvn_ill_conditioned = True
            if scaler.condition_number_ > _tvn_max_condition_number:
                _tvn_max_condition_number = scaler.condition_number_

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

        for batch in sorted(df[batch_col].unique().to_list()):
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
                update_tvn_state(scaler)  # Track TVN ill-conditioning
                X_norm = scaler.transform(X)
            else:
                scaler = scaler_class()
                X_norm = scaler.fit_transform(X)
                update_tvn_state(scaler)  # Track TVN ill-conditioning

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
            update_tvn_state(scaler)  # Track TVN ill-conditioning
            X_norm = scaler.transform(X)
        else:
            scaler = scaler_class()
            X_norm = scaler.fit_transform(X)
            update_tvn_state(scaler)  # Track TVN ill-conditioning

        df_norm = df.with_columns(
            [pl.Series(name=feat, values=X_norm[:, i]) for i, feat in enumerate(features)]
        )

    return df_norm
