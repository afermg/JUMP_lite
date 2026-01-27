#!/usr/bin/env python3
"""Simple configurable pipeline for profile processing."""

import sys
from pathlib import Path
import yaml
from omegaconf import DictConfig, OmegaConf
import hydra

# Add src to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from norm.data.load import load_profiles, save_profiles, infer_columns
from norm.operations.select import select_features
from norm.operations.normalize import normalize_profiles_extended
from norm.operations.aggregate import aggregate_profiles
import polars as pl
import numpy as np


def validate_pca_variance(
    df: pl.DataFrame,
    max_pc1_variance: float = 0.40,
    abort_on_error: bool = True
) -> tuple[bool, str]:
    """
    Validate that the first PCA component doesn't explain too much variance.

    If PC1 explains > max_pc1_variance of total variance, the data has likely
    collapsed to essentially 1 dimension, indicating a normalization problem.

    Args:
        df: DataFrame to validate
        max_pc1_variance: Maximum allowed variance ratio for PC1 (default 0.95)
        abort_on_error: If True, raise an error; if False, just warn

    Returns:
        Tuple of (is_valid, error_message)
    """
    from sklearn.decomposition import PCA

    features, _ = infer_columns(df, ["Metadata_"])

    # Filter to numeric features only
    numeric_features = [
        f for f in features
        if df[f].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                           pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]

    if len(numeric_features) < 2:
        return True, ""  # Can't do PCA with < 2 features

    # Convert to numpy
    X = df.select(numeric_features).to_numpy()

    # Skip if there are NaN/Inf values (should be caught by validate_features)
    if np.isnan(X).any() or np.isinf(X).any():
        return True, ""  # Skip check, let validate_features handle it

    # Fit PCA with 1 component to get variance ratio
    try:
        pca = PCA(n_components=1)
        pca.fit(X)
        pc1_variance = pca.explained_variance_ratio_[0]
    except Exception as e:
        print(f"  Warning: PCA variance check failed: {e}")
        return True, ""

    if pc1_variance > max_pc1_variance:
        error_msg = (
            f"PCA variance check failed: PC1 explains {pc1_variance*100:.1f}% of variance "
            f"(threshold: {max_pc1_variance*100:.0f}%). "
            f"Data may have collapsed to ~1 dimension."
        )

        if abort_on_error:
            print(f"\nERROR: {error_msg}")
            return False, error_msg
        else:
            print(f"\nWARNING: {error_msg}")
            return True, error_msg

    return True, ""


def validate_features(df: pl.DataFrame, step_name: str, abort_on_error: bool = True) -> tuple[bool, str]:
    """
    Validate that feature columns don't contain NaN or Inf values.

    Args:
        df: DataFrame to validate
        step_name: Name of the step that just ran (for error messages)
        abort_on_error: If True, raise an error; if False, just warn

    Returns:
        Tuple of (is_valid, error_message)
    """
    features, _ = infer_columns(df, ["Metadata_"])

    # Filter to numeric features only
    numeric_features = [
        f for f in features
        if df[f].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                           pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]

    if not numeric_features:
        return True, ""

    # Convert to numpy for efficient NaN/Inf checking
    X = df.select(numeric_features).to_numpy()

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()

    if nan_count > 0 or inf_count > 0:
        total_values = X.size
        error_msg = (
            f"Step '{step_name}' produced invalid values: "
            f"{nan_count} NaN ({nan_count/total_values*100:.2f}%), "
            f"{inf_count} Inf ({inf_count/total_values*100:.2f}%) "
            f"in {len(numeric_features)} features"
        )

        if abort_on_error:
            print(f"\nERROR: {error_msg}")
            return False, error_msg
        else:
            print(f"\nWARNING: {error_msg}")
            return True, error_msg

    return True, ""


def clean_nans(df, config):
    """Remove NaN/inf columns and rows."""
    print("\n=== Step: Clean NaNs/Infs ===")
    features, metadata = infer_columns(df, ["Metadata_"])
    na_cutoff = config.get("na_cutoff", 0.30)

    # # Keep only numeric features
    # numeric_features = [
    #     f for f in features
    #     if df[f].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
    #                        pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    # ]
    # non_numeric = set(features) - set(numeric_features)
    # if non_numeric:
    #     print(f"  Moved {len(non_numeric)} non-numeric columns to metadata")
    #     metadata.extend(non_numeric)
    # features = numeric_features

    # Remove columns with >30% NaNs/infs
    from norm.operations.select import drop_na_columns
    features, dropped_features = drop_na_columns(df, features, na_cutoff=na_cutoff)
    print(f"  Kept {len(features)} features after NaN/inf filter")

    print(f"  Dropped {len(dropped_features)} features due to NaN/inf cutoff:")
    if len(dropped_features) > 0:
        print(f"    {dropped_features[:5]}")

    # Remove rows with NaNs or infs
    df = df.select(metadata + features).filter(
        pl.all_horizontal([
            pl.col(f).is_not_null() & pl.col(f).is_finite()
            for f in features
        ])
    )
    print(f"  Result: {df.shape}")
    return df


def merge_metadata(df, config):
    """Merge with JUMP metadata."""
    print("\n=== Step: Merge Metadata ===")
    from norm.utils import load_metadata

    metadata_dir = Path(config.get("metadata_dir", "analysis/feature_similarity/input"))
    metadata = load_metadata(metadata_dir)

    # Rename non-Metadata columns
    non_metadata_cols = [c for c in metadata.columns if not c.startswith("Metadata_")]
    if non_metadata_cols:
        metadata = metadata.rename({col: f"Metadata_{col}" for col in non_metadata_cols})

    # Merge
    df = df.join(metadata, on=["Metadata_Well"], how="left")

    # Create derived columns
    if "Metadata_target_list" in df.columns:
        df = df.with_columns(pl.col("Metadata_target_list").fill_null("unknown"))
    if "Metadata_control_type" in df.columns:
        df = df.with_columns(pl.col("Metadata_control_type").fill_null("trt"))
    else:
        df = df.with_columns(
            pl.when(pl.col("Metadata_JCP2022") == "JCP2022_085227")
            .then(pl.lit("negcon"))
            .otherwise(pl.lit("trt"))
            .alias("Metadata_control_type")
        )
    df = df.with_columns(
        (pl.col("Metadata_control_type") == "negcon").alias("Metadata_negcon")
    )

    print(f"  Result: {df.shape}")
    return df


def filter_features(df, config):
    """Filter low variance and outlier features."""
    print("\n=== Step: Filter Features ===")
    features, metadata = infer_columns(df, ["Metadata_"])

    # Keep only numeric
    numeric_features = [
        f for f in features
        if df[f].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                           pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]
    features = numeric_features

    ops = config.get("operations", [
        {"name": "variance_threshold", "freq_cut": 0.05, "unique_cut": 0.01},
        {"name": "drop_outliers", "outlier_cutoff": 500},
    ])
    features = select_features(df, features, ops, verbose=True)
    df = df.select(metadata + features)
    print(f"  Result: {df.shape}")
    return df


def aggregate_wells(df, config):
    """Aggregate by well-plate."""
    print("\n=== Step: Aggregate Wells ===")
    features, metadata = infer_columns(df, ["Metadata_"])
    strata = config.get("strata", ["Metadata_Plate", "Metadata_Well"])
    method = config.get("method", "median")

    df = aggregate_profiles(df, features, metadata, strata=strata, method=method)
    print(f"  Result: {df.shape}")
    return df


def normalize_standard(df, config):
    """Normalize using standardization."""
    print("\n=== Step: Normalize (Standard) ===")
    features, metadata = infer_columns(df, ["Metadata_"])

    # Keep only numeric
    numeric_features = [
        f for f in features
        if df[f].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                           pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]

    # Handle empty string as None for batch_col
    batch_col = config.get("batch_col", "Metadata_Plate")
    if batch_col == "":
        batch_col = None

    print(f"  method: {config.get('method', 'standardize')}, batch_col: {batch_col}, fit_on_controls: {config.get('fit_on_controls', False)}")

    df = normalize_profiles_extended(
        df, numeric_features,
        method=config.get("method", "standardize"),
        batch_col=batch_col,
        fit_on_controls=config.get("fit_on_controls", False),
        control_key=config.get("control_key", "negcon"),
        control_col=config.get("control_col", "Metadata_control_type"),
    )
    print(f"  Result: {df.shape}")
    return df


def normalize_tvn(df, config):
    """Normalize using TVN.

    Following the original TVN paper (Ando et al. 2017), fits on negative
    controls by default to model "typical" (unwanted) variation.
    """
    print("\n=== Step: Normalize (TVN) ===")
    features, metadata = infer_columns(df, ["Metadata_"])

    numeric_features = [
        f for f in features
        if df[f].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                           pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]

    # Handle empty string as None for batch_col
    batch_col = config.get("batch_col", "Metadata_Plate")
    if batch_col == "":
        batch_col = None

    # Default to fitting on controls (per original TVN paper)
    fit_on_controls = config.get("fit_on_controls", True)
    control_key = config.get("control_key", "negcon")
    control_col = config.get("control_col", "Metadata_control_type")

    print(f"  batch_col: {batch_col}, fit_on_controls: {fit_on_controls}")

    df = normalize_profiles_extended(
        df, numeric_features,
        method="tvn",
        batch_col=batch_col,
        fit_on_controls=fit_on_controls,
        control_key=control_key,
        control_col=control_col,
        tvn_alpha=config.get("alpha", 0.5),
        tvn_epsilon=config.get("epsilon", 1.0),  # Default per Ando 2017
    )
    print(f"  Result: {df.shape}")
    return df


def normalize_spherize(df, config):
    """Normalize using Spherize."""
    print("\n=== Step: Normalize (Spherize) ===")
    features, metadata = infer_columns(df, ["Metadata_"])

    numeric_features = [
        f for f in features
        if df[f].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                           pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]

    # Handle empty string as None for batch_col (global spherize)
    batch_col = config.get("batch_col", "Metadata_Plate")
    if batch_col == "":
        batch_col = None

    fit_on_controls = config.get("fit_on_controls", False)
    control_key = config.get("control_key", "negcon")
    control_col = config.get("control_col", "Metadata_control_type")
    method = config.get("method", "ZCA-cor")

    print(f"  method: {method}, batch_col: {batch_col}, fit_on_controls: {fit_on_controls}")

    df = normalize_profiles_extended(
        df, numeric_features,
        method="spherize",
        batch_col=batch_col,
        spherize_method=config.get("method", "ZCA-cor"),
        spherize_epsilon=config.get("epsilon", 1e-6),
        fit_on_controls=fit_on_controls,
        control_key=control_key,
        control_col=control_col,
    )
    print(f"  Result: {df.shape}")
    return df


def normalize_pca(df, config):
    """Reduce dimensionality using PCA before whitening operations (EFAAR-style).

    This step applies PCA to reduce feature dimensionality, which can improve
    numerical stability for downstream whitening operations like TVN and Spherize.
    """
    from sklearn.decomposition import PCA
    import numpy as np

    print("\n=== Step: Normalize (PCA) ===")
    features, metadata = infer_columns(df, ["Metadata_"])

    numeric_features = [
        f for f in features
        if df[f].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                           pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]

    n_components = config.get("n_components", 128)
    whiten = config.get("whiten", False)
    fit_on_controls = config.get("fit_on_controls", True)
    control_key = config.get("control_key", "negcon")
    control_col = config.get("control_col", "Metadata_control_type")

    print(f"  n_components: {n_components}, whiten: {whiten}, fit_on_controls: {fit_on_controls}")
    print(f"  Input features: {len(numeric_features)}")

    X = df.select(numeric_features).to_numpy()

    # Fit on controls or all samples
    if fit_on_controls and control_col in df.columns:
        control_mask = (df[control_col] == control_key).to_numpy()
        X_fit = X[control_mask]
        print(f"  Fitting on {X_fit.shape[0]} control samples")
    else:
        X_fit = X
        print(f"  Fitting on all {X_fit.shape[0]} samples")

    # Check if we have enough features/samples for the requested components
    max_components = min(X_fit.shape[0], X_fit.shape[1])
    if n_components >= max_components:
        print(f"  WARNING: n_components={n_components} >= max possible={max_components}")
        if max_components <= 1:
            print(f"  SKIPPING PCA: insufficient features/samples (max_components={max_components})")
            return df
        n_components = max_components - 1  # Leave at least 1 degree of freedom
        print(f"  Reducing to n_components={n_components}")

    # Fit PCA
    pca = PCA(n_components=n_components, whiten=whiten)
    pca.fit(X_fit)

    # Transform all samples
    X_transformed = pca.transform(X)

    # Create new feature names
    n_out = X_transformed.shape[1]
    new_features = [f"PC_{i+1}" for i in range(n_out)]
    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"  Output features: {n_out} (explained variance: {explained_var:.1f}%)")

    # Build output DataFrame: keep metadata, replace features with PCs
    metadata_cols = [c for c in df.columns if c not in numeric_features]
    result = df.select(metadata_cols)

    # Add PC columns
    for i, feat_name in enumerate(new_features):
        result = result.with_columns(pl.Series(name=feat_name, values=X_transformed[:, i]))

    print(f"  Result: {result.shape}")
    return result


def normalize_harmony(df, config):
    """Normalize using Harmony batch correction (GPU-accelerated via rapids_singlecell).

    Harmony works on PCA space and returns a corrected PCA embedding.
    This changes the feature columns from original features to PC1, PC2, etc.
    """
    import numpy as np

    print("\n=== Step: Normalize (Harmony) ===")
    features, metadata = infer_columns(df, ["Metadata_"])

    numeric_features = [
        f for f in features
        if df[f].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                           pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]

    # Get batch labels
    batch_col = config.get("batch_col", "Metadata_Plate")
    if batch_col not in df.columns:
        raise ValueError(f"Batch column '{batch_col}' not found in DataFrame")

    batch_labels = df[batch_col].to_numpy()
    X = df.select(numeric_features).to_numpy()

    # Get parameters
    n_pcs = config.get("n_pcs", 50)
    theta = config.get("theta", 2.0)
    sigma = config.get("sigma", 0.1)
    n_clusters = config.get("n_clusters", None)

    print(f"  n_pcs: {n_pcs}, theta: {theta}, sigma: {sigma}, n_clusters: {n_clusters}")
    print(f"  batch_col: {batch_col}, n_batches: {len(np.unique(batch_labels))}")

    from norm.operations.normalize import Harmony
    harmony = Harmony(
        n_pcs=n_pcs,
        theta=theta,
        sigma=sigma,
        n_clusters=n_clusters,
    )
    X_corrected = harmony.fit_transform(X, batch_labels)

    # Create new feature names (PC1, PC2, ...)
    new_features = [f"PC{i+1}" for i in range(X_corrected.shape[1])]

    # Build new DataFrame with metadata + corrected PCs
    metadata_cols = [c for c in df.columns if c.startswith("Metadata")]
    df_meta = df.select(metadata_cols)
    df_pca = pl.DataFrame({feat: X_corrected[:, i] for i, feat in enumerate(new_features)})

    df = pl.concat([df_meta, df_pca], how="horizontal")
    print(f"  Result: {df.shape} (features changed to {len(new_features)} PCs)")
    return df


def normalize_combat(df, config):
    """Normalize using ComBat batch correction.

    ComBat works directly on features and returns corrected features
    with the same dimensions as input.
    """
    import numpy as np

    print("\n=== Step: Normalize (ComBat) ===")
    features, metadata = infer_columns(df, ["Metadata_"])

    numeric_features = [
        f for f in features
        if df[f].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                           pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]

    # Get batch labels
    batch_col = config.get("batch_col", "Metadata_Plate")
    if batch_col not in df.columns:
        raise ValueError(f"Batch column '{batch_col}' not found in DataFrame")

    batch_labels = df[batch_col].to_numpy()
    X = df.select(numeric_features).to_numpy()

    par_prior = config.get("par_prior", True)
    precision = config.get("precision", 0.01)  # Avoid division by zero warnings
    print(f"  batch_col: {batch_col}, n_batches: {len(np.unique(batch_labels))}")
    print(f"  par_prior: {par_prior}, precision: {precision}")

    from norm.operations.normalize import ComBat
    combat = ComBat(par_prior=par_prior, precision=precision)
    X_corrected = combat.fit_transform(X, batch_labels)

    # Update features in DataFrame
    df = df.with_columns(
        [pl.Series(name=feat, values=X_corrected[:, i]) for i, feat in enumerate(numeric_features)]
    )
    print(f"  Result: {df.shape}")
    return df


def prune_correlated(df, config):
    """Prune correlated features."""
    print("\n=== Step: Prune Correlated Features ===")
    features, metadata = infer_columns(df, ["Metadata_"])

    numeric_features = [
        f for f in features
        if df[f].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                           pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]

    threshold = config.get("threshold", 0.9)
    method = config.get("method", "pearson")
    print(f"  threshold: {threshold}, method: {method}")

    ops = [{"name": "correlation_threshold", "threshold": threshold, "method": method}]
    features = select_features(df, numeric_features, ops, verbose=True)
    df = df.select(metadata + features)
    print(f"  Result: {df.shape}")
    return df


def evaluate_metrics(df, config):
    """Evaluate metrics."""
    print("\n=== Step: Evaluate Metrics ===")
    import json
    from norm.metrics.phenotypic import calculate_phenotypic_activity, calculate_phenotypic_consistency
    from norm.metrics.batch import calculate_batch_metrics
    from norm.visualization import plot_dimensionality_reduction_extended

    # Get output directory from config
    output_dir = Path(config.get("output_dir", "."))
    output_dir.mkdir(exist_ok=True)

    features, metadata = infer_columns(df, ["Metadata_"])
    numeric_features = [
        f for f in features
        if df[f].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                           pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]

    results = {}
    full_results = {}  # Keep full DataFrames for visualization

    try:
        pa = calculate_phenotypic_activity(df, numeric_features)
        results["PA"] = pa["pct_compounds_active"]
        results["n_compounds"] = pa["n_compounds"]
        full_results["phenotypic_activity"] = pa  # Keep full results
        print(f"  PA: {pa['pct_compounds_active']:.2f}%")

        # Save per-compound results
        if pa.get("activity_ap") is not None and len(pa["activity_ap"]) > 0:
            pa_csv = output_dir / "phenotypic_activity_per_compound.csv"
            pa["activity_ap"].to_csv(pa_csv, index=False)
            print(f"  Saved per-compound PA to: {pa_csv}")

        # Save mAP results
        if pa.get("activity_map") is not None and len(pa["activity_map"]) > 0:
            map_csv = output_dir / "phenotypic_activity_map.csv"
            pa["activity_map"].to_csv(map_csv, index=False)
            print(f"  Saved mAP results to: {map_csv}")
    except Exception as e:
        print(f"  PA ERROR: {e}")
        full_results["phenotypic_activity"] = {}

    try:
        pc = calculate_phenotypic_consistency(df, numeric_features)
        results["PC"] = pc["pct_targets_active"]
        results["n_targets_active"] = pc["n_targets_active"]
        results["n_targets_total"] = pc["n_targets_total"]
        print(f"  PC: {pc['pct_targets_active']:.1f}% targets active ({pc['n_targets_active']}/{pc['n_targets_total']})")

        # Save per-target results
        if pc.get("target_consistency") is not None and len(pc["target_consistency"]) > 0:
            pc_csv = output_dir / "phenotypic_consistency_per_target.csv"
            pc["target_consistency"].to_csv(pc_csv, index=False)
            print(f"  Saved per-target PC to: {pc_csv}")
    except Exception as e:
        print(f"  PC ERROR: {e}")

    try:
        batch = calculate_batch_metrics(df, numeric_features)
        results["Silhouette"] = batch["silhouette_batch"]
        results["kBET"] = batch["kbet_score"]
        print(f"  Silhouette: {batch['silhouette_batch']:.4f}")
        print(f"  kBET: {batch['kbet_score']:.4f}")
    except Exception as e:
        print(f"  Batch ERROR: {e}")

    # Add TVN ill-conditioning state to results
    from norm.operations.normalize import get_tvn_state
    tvn_ill_conditioned, tvn_max_condition_number = get_tvn_state()
    results["tvn_ill_conditioned"] = tvn_ill_conditioned
    results["tvn_max_condition_number"] = float(tvn_max_condition_number) if tvn_max_condition_number > 0 else None
    if tvn_ill_conditioned:
        print(f"  WARNING: TVN encountered ill-conditioned matrix (condition number: {tvn_max_condition_number:.2e})")

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create visualization (optional - can be skipped entirely for parallel runs)
    skip_visualization = config.get("skip_visualization", False)
    skip_umap = config.get("skip_umap", False)

    if skip_visualization:
        print("  Skipping visualization (skip_visualization=True)")
    else:
        if skip_umap:
            print("  Creating PCA visualization (UMAP skipped)...")
        else:
            print("  Creating PCA/UMAP visualization...")
        try:
            plot_path = output_dir / "dimreduction.png"
            plot_dimensionality_reduction_extended(
                df, numeric_features, full_results, plot_path,
                n_top_compounds=config.get("n_top_compounds", 20),
                skip_umap=skip_umap
            )
            print(f"  Saved visualization to: {plot_path}")
        except Exception as e:
            print(f"  Visualization ERROR: {e}")

    return df


def sample_norm(df, config):
    """Apply L1/L2 normalization per sample (row-wise)."""
    print("\n=== Step: Sample Normalize ===")
    features, metadata = infer_columns(df, ["Metadata_"])

    numeric_features = [
        f for f in features
        if df[f].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                           pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]

    norm = config.get("norm", "l2")
    print(f"  Applying {norm} normalization to {len(numeric_features)} features")

    from norm.operations.normalize import sample_normalize
    df = sample_normalize(df, numeric_features, norm=norm)
    print(f"  Result: {df.shape}")
    return df


# Step registry
STEPS = {
    "clean_nans": clean_nans,
    "merge_metadata": merge_metadata,
    "filter_features": filter_features,
    "aggregate_wells": aggregate_wells,
    "sample_norm": sample_norm,
    "normalize_standard": normalize_standard,
    "normalize_pca": normalize_pca,
    "normalize_tvn": normalize_tvn,
    "normalize_spherize": normalize_spherize,
    "normalize_harmony": normalize_harmony,
    "normalize_combat": normalize_combat,
    "prune_correlated": prune_correlated,
    "evaluate_metrics": evaluate_metrics,
}

# Batch correction methods that should be mutually exclusive
# These methods all aim to remove batch effects and should not be combined
# Batch correction methods that are mutually exclusive
# Note: Spherize is NOT included here - it can be combined with other methods
# as a post-processing whitening step (TVN + Spherize yields best results)
BATCH_CORRECTION_STEPS = {
    "normalize_tvn",
    "normalize_harmony",
    "normalize_combat",
}


def validate_batch_correction_config(config, mode="error"):
    """Validate that only one batch correction method is enabled.

    Batch correction methods (TVN, Spherize, Harmony, ComBat) are mathematically
    incompatible when combined:
    - TVN + Spherize: Both perform whitening, redundant and numerically unstable
    - Harmony: Changes output to PCA embedding, incompatible with feature-based methods
    - ComBat + others: Over-correction risk, can destroy biological signal

    See docs/batch_correction_guide.md for detailed explanation.

    Args:
        config: Pipeline configuration dict
        mode: How to handle violations:
            - "error": Raise ValueError (default)
            - "warn": Print warning and continue
            - "skip": Return list of steps to skip (keeps first enabled)

    Returns:
        If mode="skip": List of step names to disable
        Otherwise: None

    Raises:
        ValueError: If mode="error" and multiple batch correction methods enabled
    """
    enabled_batch_methods = []

    for step_config in config.get("steps", []):
        if not step_config.get("enabled", True):
            continue
        step_name = step_config.get("name", "")
        if step_name in BATCH_CORRECTION_STEPS:
            enabled_batch_methods.append(step_name)

    if len(enabled_batch_methods) <= 1:
        return [] if mode == "skip" else None

    # Multiple batch correction methods enabled
    method_names = ", ".join(enabled_batch_methods)

    if mode == "error":
        raise ValueError(
            f"Multiple batch correction methods enabled: {method_names}. "
            f"These methods are mutually exclusive and should not be combined. "
            f"See docs/batch_correction_guide.md for details. "
            f"Set validate_batch_correction=false to disable this check."
        )
    elif mode == "warn":
        print(f"\nWARNING: Multiple batch correction methods enabled: {method_names}")
        print("These methods are mutually exclusive. Results may be suboptimal.")
        print("See docs/batch_correction_guide.md for details.\n")
        return None
    elif mode == "skip":
        # Keep first, skip rest
        to_skip = enabled_batch_methods[1:]
        print(f"\nINFO: Multiple batch correction methods detected: {method_names}")
        print(f"Keeping: {enabled_batch_methods[0]}, Skipping: {', '.join(to_skip)}")
        return to_skip
    else:
        raise ValueError(f"Unknown validation mode: {mode}")


# Default values for method-specific parameters
# Used to detect redundant sweep configurations
# NOTE: These must match the FIRST value in each sweep parameter list
# to avoid skipping all configs. If sweep has tvn_alpha: "0.3,0.5",
# set default to 0.3 here.
METHOD_PARAM_DEFAULTS = {
    "tvn": {
        "tvn_alpha": 0.3,  # Match sweep's single/first value
        "tvn_fit_controls": True,
    },
    "spherize": {
        "spherize_method": "ZCA-cor",
        "spherize_fit_negcon": True,
    },
    "harmony": {
        "harmony_n_pcs": 50,
        "harmony_theta": 1.0,  # Match sweep's single/first value
    },
    "combat": {
        "combat_par_prior": True,
    },
}


def is_redundant_config(config):
    """Check if configuration is redundant due to irrelevant method-specific params.

    When a batch correction method is disabled, its method-specific parameters
    don't affect the output. To avoid running redundant combinations, we only
    run configurations where disabled methods have their default parameter values.

    Example: If batch_method=none, we skip configs where tvn_alpha=0.3 (non-default)
    and only run tvn_alpha=0.5 (default). This reduces 40K combinations to ~960.

    Args:
        config: Pipeline configuration dict

    Returns:
        tuple: (is_redundant: bool, reason: str or None)
    """
    # Get batch_method or fall back to individual flags
    # Note: use_spherize is always independent (can combine with other methods)
    batch_method = config.get("batch_method")
    if batch_method is not None:
        use_tvn = (batch_method == "tvn")
        use_harmony = (batch_method == "harmony")
        use_combat = (batch_method == "combat")
    else:
        use_tvn = config.get("use_tvn", False)
        use_harmony = config.get("use_harmony", False)
        use_combat = config.get("use_combat", False)

    # Spherize is independent - get from config directly
    use_spherize = config.get("use_spherize", False)

    # For each disabled method, check if params differ from defaults
    checks = [
        ("tvn", use_tvn, METHOD_PARAM_DEFAULTS["tvn"]),
        ("spherize", use_spherize, METHOD_PARAM_DEFAULTS["spherize"]),
        ("harmony", use_harmony, METHOD_PARAM_DEFAULTS["harmony"]),
        ("combat", use_combat, METHOD_PARAM_DEFAULTS["combat"]),
    ]

    for method_name, is_enabled, defaults in checks:
        if is_enabled:
            continue  # Method is enabled, params matter

        # Method is disabled - check if any param differs from default
        for param, default_val in defaults.items():
            actual_val = config.get(param, default_val)
            # Handle type mismatches (e.g., string "true" vs bool True)
            if isinstance(default_val, bool) and isinstance(actual_val, str):
                actual_val = actual_val.lower() == "true"
            elif isinstance(default_val, (int, float)) and isinstance(actual_val, str):
                try:
                    actual_val = type(default_val)(actual_val)
                except (ValueError, TypeError):
                    pass

            if actual_val != default_val:
                return True, f"{method_name} disabled but {param}={actual_val} (default: {default_val})"

    return False, None


def generate_output_name(config):
    """Generate output directory name based on pipeline configuration.

    Only includes parameters that deviate from defaults.
    """
    parts = []

    for step_config in config.get("steps", []):
        if not step_config.get("enabled", True):
            continue

        step_name = step_config["name"]
        params = step_config.get("params", {})

        # Add step-specific identifiers
        if step_name == "sample_norm":
            norm = params.get("norm", "l2")
            parts.append(f"snorm_{norm}")

        elif step_name == "normalize_standard":
            method = params.get("method", "standardize")
            batch = "platewise" if params.get("batch_col") else "global"
            controls = "ctrl" if params.get("fit_on_controls", True) else "all"

            # Only add method if not default
            name_parts = []
            if method != "standardize":
                name_parts.append(method)
            else:
                name_parts.append("std")
            if batch != "platewise":
                name_parts.append(batch)
            if not params.get("fit_on_controls", True):  # Default is True
                name_parts.append("all")
            parts.append("_".join(name_parts))

        elif step_name == "normalize_pca":
            n_comp = params.get("n_components", 128)
            fit_ctrl = params.get("fit_on_controls", True)
            name_parts = [f"pca{n_comp}"]
            if not fit_ctrl:
                name_parts.append("all")
            parts.append("_".join(name_parts))

        elif step_name == "normalize_tvn":
            alpha = params.get("alpha", 0.5)
            epsilon = params.get("epsilon", 1.0)
            batch = "platewise" if params.get("batch_col") else "global"
            fit_ctrl = params.get("fit_on_controls", True)  # Default is now True

            name_parts = ["tvn"]
            if alpha != 0.5:
                name_parts.append(f"a{alpha}")
            if epsilon != 1.0:
                name_parts.append(f"e{epsilon:.0e}")
            if not fit_ctrl:  # Indicate when NOT fitting on controls (non-default)
                name_parts.append("all")
            if batch != "platewise":
                name_parts.append(batch)
            parts.append("_".join(name_parts))

        elif step_name == "normalize_spherize":
            method = params.get("method", "ZCA-cor")
            epsilon = params.get("epsilon", 1e-6)
            batch = "platewise" if params.get("batch_col") else "global"
            fit_ctrl = params.get("fit_on_controls", False)

            name_parts = [method]
            if fit_ctrl:
                name_parts.append("negcon")
            if epsilon != 1e-6:
                name_parts.append(f"e{epsilon:.0e}")
            if batch != "platewise":
                name_parts.append(batch)
            parts.append("_".join(name_parts))

        elif step_name == "normalize_harmony":
            n_pcs = params.get("n_pcs", 50)
            theta = params.get("theta", 2.0)

            name_parts = ["harmony"]
            if n_pcs != 50:
                name_parts.append(f"pc{n_pcs}")
            if theta != 2.0:
                name_parts.append(f"t{theta}")
            parts.append("_".join(name_parts))

        elif step_name == "normalize_combat":
            par_prior = params.get("par_prior", True)

            name_parts = ["combat"]
            if not par_prior:
                name_parts.append("nonpar")
            parts.append("_".join(name_parts))

        elif step_name == "prune_correlated":
            threshold = params.get("threshold", 0.9)
            method = params.get("method", "pearson")

            name_parts = ["prune"]
            if threshold != 0.9:
                name_parts.append(f"{threshold}")
            if method != "pearson":
                name_parts.append(method)
            parts.append("_".join(name_parts))

        elif step_name == "aggregate_wells":
            method = params.get("method", "median")

            if method != "median":
                parts.append(f"agg_{method}")
            else:
                parts.append("agg")

        elif step_name == "filter_features":
            # Extract key filter operations if they differ from defaults
            ops = params.get("operations", [])
            filter_details = []
            for op in ops:
                if op["name"] == "variance_threshold":
                    # Include variance threshold if specified
                    var_thresh = op.get("var_threshold")
                    if var_thresh is not None:
                        filter_details.append(f"varthresh{var_thresh}")
                    # Include freq/unique cuts if they differ from defaults
                    if op.get("freq_cut", 0.05) != 0.05 or op.get("unique_cut", 0.01) != 0.01:
                        filter_details.append(f"var{op.get('freq_cut', 0.05)}_{op.get('unique_cut', 0.01)}")
                elif op["name"] == "drop_outliers":
                    if op.get("outlier_cutoff", 500) != 500:
                        filter_details.append(f"outlier{op.get('outlier_cutoff')}")

            if filter_details:
                parts.append(f"filter_{'_'.join(filter_details)}")
            else:
                parts.append("filter")

        elif step_name == "clean_nans":
            cutoff = params.get("na_cutoff", 0.30)
            if cutoff != 0.30:
                parts.append(f"clean{cutoff}")
            else:
                parts.append("clean")

        elif step_name == "merge_metadata":
            parts.append("meta")

    # Join parts with double underscores (parameter separator)
    # Single underscores used within parameter values
    if parts:
        return "__".join(parts)
    else:
        return "default"


def run_pipeline(config_path=None, input_override=None, hydra_config=None):
    """Run pipeline from config.

    Args:
        config_path: Path to YAML configuration file (for non-Hydra mode)
        input_override: Optional input file path to override config
        hydra_config: DictConfig from Hydra (takes precedence over config_path)
    """
    import shutil

    # Load config from Hydra or YAML file
    if hydra_config is not None:
        config = OmegaConf.to_container(hydra_config, resolve=True)
        config_path = None  # Not used in Hydra mode
    elif config_path is not None:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Either config_path or hydra_config must be provided")

    # Handle batch_method parameter (converts to use_tvn, use_harmony, use_combat)
    # Note: use_spherize is independent - Spherize can be combined with other methods
    batch_method = config.get("batch_method")
    if batch_method is not None:
        # Set the appropriate use_* flag based on batch_method
        config["use_tvn"] = (batch_method == "tvn")
        config["use_harmony"] = (batch_method == "harmony")
        config["use_combat"] = (batch_method == "combat")
        # use_spherize is NOT set by batch_method - it's an independent toggle

        # Also update the steps list to reflect the enabled method
        for step in config.get("steps", []):
            step_name = step.get("name", "")
            if step_name == "normalize_tvn":
                step["enabled"] = (batch_method == "tvn")
            elif step_name == "normalize_harmony":
                step["enabled"] = (batch_method == "harmony")
            elif step_name == "normalize_combat":
                step["enabled"] = (batch_method == "combat")
            elif step_name == "normalize_spherize":
                # Spherize is independent - get from config's use_spherize
                step["enabled"] = config.get("use_spherize", False)
    else:
        # batch_method not set - only override spherize if use_spherize is explicitly set
        if "use_spherize" in config:
            for step in config.get("steps", []):
                if step.get("name") == "normalize_spherize":
                    step["enabled"] = config.get("use_spherize", False)

    # Check for redundant configuration (skip early to save time)
    skip_redundant = config.get("skip_redundant_configs", False)
    if skip_redundant:
        is_redundant, reason = is_redundant_config(config)
        if is_redundant:
            print(f"SKIPPING redundant config: {reason}")
            return float('inf')  # Return sentinel for Optuna compatibility

    # Reset TVN ill-conditioning state at start of each pipeline run
    from norm.operations.normalize import reset_tvn_state
    reset_tvn_state()

    # Use input override if provided (command line > config file > default)
    if input_override:
        input_path = Path(input_override)
        print(f"Input override (CLI): {input_override}")
    elif config.get("input_override"):
        input_path = Path(config["input_override"])
        print(f"Input override (config): {config['input_override']}")
    else:
        input_path = Path(config["input"]["path"])

    # Generate output directory name from config
    output_name = generate_output_name(config)
    base_output_path = Path(config["output"]["path"])

    # Get input filename (stem without extension)
    input_name = input_path.stem

    # Get sweep name and job number if available (for Hydra sweeps)
    sweep_name = None
    if hydra_config is not None:
        try:
            from hydra.core.hydra_config import HydraConfig
            hydra_cfg = HydraConfig.get()
            job_num = hydra_cfg.job.num
            output_name = f"{job_num:04d}__{output_name}"

            # Extract sweep name from overrides (e.g., "+sweep=tvn_spherize_sweep")
            for override in hydra_cfg.overrides.task:
                if override.startswith("+sweep=") or override.startswith("sweep="):
                    sweep_name = override.split("=")[1]
                    break
        except Exception:
            pass  # Not in Hydra context or single run

    # Create output directory: parent / [sweep_name /] input_name / pipeline_config_name
    if sweep_name:
        output_dir = base_output_path.parent / sweep_name / input_name / output_name
    else:
        output_dir = base_output_path.parent / input_name / output_name
    output_dir.mkdir(exist_ok=True, parents=True)

    # Update output path
    output_path = output_dir / base_output_path.name

    # Save config to output directory for reproducibility
    config_copy_path = output_dir / "pipeline_config.yaml"
    if config_path is not None:
        # Non-Hydra mode: copy the config file
        shutil.copy(config_path, config_copy_path)
    else:
        # Hydra mode: save the resolved config
        with open(config_copy_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    print(f"Pipeline: {output_name}")
    print(f"Output directory: {output_dir}")
    print(f"Config saved to: {config_copy_path}\n")

    # Validate batch correction configuration (optional)
    validate_batch_correction = config.get("validate_batch_correction", False)
    validation_mode = config.get("validation_mode", "error")
    steps_to_skip = set()

    if validate_batch_correction:
        result = validate_batch_correction_config(config, mode=validation_mode)
        if result:  # mode="skip" returns list of steps to skip
            steps_to_skip = set(result)

    # Load input (already set above with override if provided)
    print(f"Loading: {input_path}")
    df = load_profiles(input_path)
    print(f"Initial shape: {df.shape}")

    # Run steps
    for step_config in config.get("steps", []):
        step_name = step_config.get("name", "unknown")

        if not step_config.get("enabled", True):
            print(f"\nSkipping: {step_name}")
            continue

        # Skip steps flagged by batch correction validation
        if step_name in steps_to_skip:
            print(f"\nSkipping (validation): {step_name}")
            continue

        step_func = STEPS.get(step_name)

        if step_func is None:
            print(f"WARNING: Unknown step '{step_name}'")
            continue

        # Update evaluate_metrics output_dir to use the same directory
        if step_name == "evaluate_metrics":
            if "params" not in step_config:
                step_config["params"] = {}
            step_config["params"]["output_dir"] = output_dir / "results"

        # Check PCA variance BEFORE aggregation (data should not have collapsed)
        if step_name == "aggregate_wells":
            max_pc1_var = config.get("max_pc1_variance", 0.50)
            is_valid, error_msg = validate_pca_variance(
                df, max_pc1_var,
                abort_on_error=config.get("abort_on_invalid_features", True)
            )
            if not is_valid:
                print("This indicates the feature space has collapsed, likely due to")
                print("overly aggressive normalization or batch correction.")
                return float('inf')  # Return sentinel for Optuna compatibility

        df = step_func(df, step_config.get("params", {}))

        # Check for empty dataframe
        if len(df) == 0:
            print(f"\nERROR: No rows remaining after {step_name}. Aborting pipeline.")
            print("This usually means metadata merge failed (no matching wells).")
            return float('inf')  # Return sentinel for Optuna compatibility

        # Check for NaN/Inf values in features (skip for evaluate_metrics which doesn't modify data)
        if step_name != "evaluate_metrics":
            is_valid, error_msg = validate_features(
                df, step_name,
                abort_on_error=config.get("abort_on_invalid_features", True)
            )
            if not is_valid:
                print("This usually indicates numerical instability in the normalization method.")
                print("Try: different method combination, larger epsilon, or skip this step.")
                return float('inf')  # Return sentinel for Optuna compatibility

    # Save output
    print(f"\nSaving to: {output_path}")
    save_profiles(df, output_path, compression=config["output"].get("compression", "zstd"))
    print(f"\nPipeline complete! All outputs in: {output_dir}")
    print("Done!")

    # Return metric for Optuna (read from metrics.json if available)
    # Lower is better for Optuna minimization, so return negative PA
    try:
        import json
        metrics_path = output_dir / "results" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            # Return negative PA (higher PA is better, but Optuna minimizes)
            pa = metrics.get("PA", 0.0)
            return -pa  # Negative so Optuna minimization finds highest PA
    except Exception:
        pass
    return 0.0  # Default if metrics unavailable


@hydra.main(version_base=None, config_path="conf", config_name="pipeline")
def main(cfg: DictConfig) -> float:
    """Hydra entry point for running pipeline with sweeps.

    Returns:
        float: Metric value for Optuna optimization (negative PA, lower is better)
    """
    # Get input override from hydra config if provided
    input_override = cfg.get("input_override", None)

    # Run pipeline with Hydra config and return metric for Optuna
    return run_pipeline(hydra_config=cfg, input_override=input_override)


if __name__ == "__main__":
    # Check if using Hydra mode (config files in conf/) or legacy mode
    if len(sys.argv) >= 2 and sys.argv[1].endswith('.yaml') and Path(sys.argv[1]).exists():
        # Legacy mode: python run_pipeline.py <config.yaml> [input_file]
        print("Running in legacy mode (non-Hydra)")
        config_path = sys.argv[1]
        input_override = sys.argv[2] if len(sys.argv) > 2 else None
        run_pipeline(config_path=config_path, input_override=input_override)
    else:
        # Hydra mode: python run_pipeline.py [hydra overrides]
        print("Running in Hydra mode")
        main()
