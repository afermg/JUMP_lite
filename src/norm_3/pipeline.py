#!/usr/bin/env python3
"""GPU-accelerated pipeline orchestration for norm_3.

This module provides:
- Step functions for each pipeline stage (GPU-accelerated)
- Validation functions
- Hydra integration for configuration
- run_pipeline() main entry point

Usage:
    pixi run python -m norm_3.pipeline input.path=data/profiles.parquet
"""

from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np
import polars as pl
import yaml
from omegaconf import DictConfig, OmegaConf

from norm_3.core import (
    PCATransform,
    RobustMAD,
    RobustMAD_CPU,
    Spherize,
    StandardScaler,
    StandardScaler_CPU,
    TVN,
    TVN_EFAAR,
    TVN_Original,
    TVN_Cascade,
    blocklist_filter,
    correlation_threshold,
    drop_outliers,
    exclude_features,
    exclude_from_csv,
    get_spherize_state,
    get_tvn_state,
    include_features,
    include_from_csv,
    reset_spherize_state,
    reset_spherize_truncation_state,
    reset_tvn_state,
    tvn_efaar_on_controls,
    variance_threshold,
)
from norm_3.io import (
    drop_na_columns,
    get_numeric_features,
    infer_columns,
    load_metadata_parquet,
    load_profiles,
    save_profiles,
)
from norm_3.utils import (
    GPUMemoryManager,
    get_gpu_info,
    is_gpu_available,
    print_gpu_info,
    to_cpu,
    to_gpu,
)


# =============================================================================
# Validation Functions
# =============================================================================


def validate_pca_variance(
    df: pl.DataFrame,
    max_pc1_variance: float = 0.40,
    abort_on_error: bool = True,
) -> tuple[bool, str]:
    """Validate that PC1 doesn't explain too much variance."""
    features, _ = infer_columns(df, ["Metadata_"])
    numeric_features = get_numeric_features(df, features)

    if len(numeric_features) < 2:
        return True, ""

    X = df.select(numeric_features).to_numpy()

    if np.isnan(X).any() or np.isinf(X).any():
        return True, ""

    try:
        # Use GPU PCA for validation
        X_gpu = to_gpu(X)
        pca = PCATransform(n_components=1)
        pca.fit(X_gpu)
        pc1_variance = float(pca.explained_variance_ratio_[0])
    except Exception as e:
        print(f"  Warning: PCA variance check failed: {e}")
        return True, ""

    if pc1_variance > max_pc1_variance:
        error_msg = f"PC1 explains {pc1_variance*100:.1f}% (threshold: {max_pc1_variance*100:.0f}%)"
        if abort_on_error:
            print(f"\nERROR: {error_msg}")
            return False, error_msg
        else:
            print(f"\nWARNING: {error_msg}")
            return True, error_msg

    return True, ""


def validate_features(
    df: pl.DataFrame,
    step_name: str,
    abort_on_error: bool = True,
) -> tuple[bool, str]:
    """Validate no NaN/Inf in features after a step."""
    features, _ = infer_columns(df, ["Metadata_"])
    numeric_features = get_numeric_features(df, features)

    if not numeric_features:
        return True, ""

    X = df.select(numeric_features).to_numpy()
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()

    if nan_count > 0 or inf_count > 0:
        error_msg = f"Step '{step_name}': {nan_count} NaN, {inf_count} Inf in {len(numeric_features)} features"

        if abort_on_error:
            print(f"\nERROR: {error_msg}")
            return False, error_msg
        else:
            print(f"\nWARNING: {error_msg}")
            return True, error_msg

    return True, ""


# =============================================================================
# Step Functions (GPU-accelerated)
# =============================================================================


def clean_nans(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Remove NaN/inf columns and rows."""
    print("\n=== Step: Clean NaNs/Infs ===")
    features, metadata = infer_columns(df, ["Metadata_"])
    non_numeric = [f for f in features if f not in get_numeric_features(df, features)]
    if non_numeric:
        print(f"  Moving {len(non_numeric)} non-numeric columns to metadata: {non_numeric}")
        metadata = metadata + non_numeric
        features = [f for f in features if f not in non_numeric]
    na_cutoff = config.get("na_cutoff", 0.30)

    features, dropped_features = drop_na_columns(df, features, na_cutoff=na_cutoff)
    print(f"  Kept {len(features)} features after NaN/inf filter")
    if dropped_features:
        print(f"  Dropped {len(dropped_features)} features")

    df = df.select(metadata + features).filter(
        pl.all_horizontal(
            [pl.col(f).is_not_null() & pl.col(f).is_finite() for f in features]
        )
    )
    print(f"  Result: {df.shape}")
    return df


def merge_metadata(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Merge with metadata."""
    print("\n=== Step: Merge Metadata ===")

    if config.get("skip_merge", False):
        print("  Skipping metadata merge (skip_merge=True)")
        return df

    if "metadata_path" in config:
        metadata = load_metadata_parquet(
            config["metadata_path"],
            merge_how=config.get("merge_how", "left")
        )

        join_cols = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"]

        # Verify join columns exist
        profile_cols = set(df.columns)
        metadata_cols = set(metadata.columns)
        missing_in_profile = [c for c in join_cols if c not in profile_cols]
        missing_in_metadata = [c for c in join_cols if c not in metadata_cols]

        if missing_in_profile or missing_in_metadata:
            print(f"  ERROR: Missing join columns!")
            if missing_in_profile:
                print(f"    Missing in profiles: {missing_in_profile}")
            if missing_in_metadata:
                print(f"    Missing in metadata: {missing_in_metadata}")
            raise ValueError("Required join columns not found")

        # Cast and join
        df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in join_cols])
        metadata = metadata.with_columns([pl.col(c).cast(pl.Utf8) for c in join_cols])

        # Deduplicate metadata on join keys to prevent duplicate rows
        metadata_before = len(metadata)
        metadata = metadata.unique(subset=join_cols, keep="first")
        metadata_after = len(metadata)
        if metadata_before > metadata_after:
            print(f"  WARNING: Removed {metadata_before - metadata_after} duplicate metadata entries")

        common_metadata_cols = [c for c in metadata_cols if c in profile_cols and c not in join_cols]
        if common_metadata_cols:
            df = df.drop(common_metadata_cols)

        join_how = config.get("merge_how", "inner")
        before_join = len(df)
        df = df.join(metadata, on=join_cols, how=join_how)
        after_join = len(df)
        if before_join > after_join:
            print(f"  Dropped {before_join - after_join} rows not in metadata ({join_how} join)")

    # Fill control type
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


def filter_features(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Filter low variance and outlier features (GPU-accelerated)."""
    print("\n=== Step: Filter Features (GPU) ===")
    features, metadata = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    # Transfer to GPU
    X_gpu = to_gpu(df.select(features).to_numpy())

    # Accept both 'operations' and 'filters' keys for backwards compatibility
    ops = config.get("operations") or config.get("filters")
    if ops is None:
        ops = [
            {"name": "variance_threshold", "freq_cut": 0.05, "unique_cut": 0.01},
            {"name": "drop_outliers", "outlier_cutoff": 500},
        ]

    current_features = features.copy()
    for op in ops:
        name = op["name"]
        params = {k: v for k, v in op.items() if k != "name"}
        n_before = len(current_features)

        # Get indices for current features
        feature_indices = [features.index(f) for f in current_features]
        X_subset = X_gpu[:, feature_indices]

        if name == "variance_threshold":
            current_features = variance_threshold(X_subset, current_features, **params)
        elif name == "drop_outliers":
            current_features = drop_outliers(X_subset, current_features, **params)
        elif name == "drop_na_columns":
            # drop_na_columns uses DataFrame, not GPU array
            threshold = params.get("threshold", 0.05)
            current_features, _ = drop_na_columns(df, current_features, na_cutoff=threshold)
        elif name == "blocklist":
            # blocklist doesn't need GPU data - just filters by name patterns
            current_features = blocklist_filter(current_features, **params)
        elif name == "include_features":
            # include_features doesn't need GPU data - just filters by name patterns
            patterns = params.get("patterns", [])
            current_features = include_features(current_features, patterns)
        elif name == "exclude_features":
            # exclude_features doesn't need GPU data - just filters by name patterns
            patterns = params.get("patterns", [])
            current_features = exclude_features(current_features, patterns)
        elif name == "include_from_csv":
            # include_from_csv reads features from a CSV file
            csv_path = params.get("csv_path")
            column = params.get("column", "feature")
            current_features = include_from_csv(current_features, csv_path, column)
        elif name == "exclude_from_csv":
            # exclude_from_csv reads features from a CSV file
            csv_path = params.get("csv_path")
            column = params.get("column", "feature")
            current_features = exclude_from_csv(current_features, csv_path, column)

        n_dropped = n_before - len(current_features)
        print(f"  {name}: dropped {n_dropped} features ({len(current_features)} remaining)")

        if len(current_features) == 0 and n_before > 0:
            print(f"  WARNING: {name} would remove all features, skipping")
            current_features = features.copy()

    df = df.select(metadata + current_features)
    print(f"  Result: {df.shape}")
    return df


def aggregate_wells(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Aggregate by well-plate (CPU - polars is fast enough)."""
    print("\n=== Step: Aggregate Wells ===")
    features, metadata = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)
    strata = config.get("strata", ["Metadata_Plate", "Metadata_Well"])
    method = config.get("method", "median")

    agg_func = pl.median if method == "median" else pl.mean
    metadata_to_keep = [m for m in metadata if m not in strata]

    agg_exprs = [agg_func(feat).alias(feat) for feat in features]
    meta_exprs = [pl.first(meta).alias(meta) for meta in metadata_to_keep]

    df = df.group_by(strata).agg(agg_exprs + meta_exprs)
    print(f"  Result: {df.shape}")
    return df


def normalize_robustmad(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Normalize using RobustMAD (CPU by default, GPU optional)."""
    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    batch_col = config.get("batch_col", "Metadata_Plate")
    if batch_col == "":
        batch_col = None

    fit_on_controls = config.get("fit_on_controls", False)
    control_col = config.get("control_col", "Metadata_control_type")
    control_key = config.get("control_key", "negcon")
    epsilon = config.get("epsilon", 1e-18)
    use_gpu = config.get("use_gpu", False)  # CPU by default

    if use_gpu:
        print("\n=== Step: Normalize (RobustMAD - GPU) ===")
    else:
        print("\n=== Step: Normalize (RobustMAD - CPU) ===")
    print(f"  batch_col: {batch_col}, fit_on_controls: {fit_on_controls}")

    if use_gpu:
        # GPU path
        X_cpu = df.select(features).to_numpy()
        X_gpu = to_gpu(X_cpu)
        X_norm = cp.zeros_like(X_gpu)

        if batch_col and batch_col in df.columns:
            batch_labels = df[batch_col].to_numpy()
            unique_batches = np.unique(batch_labels)

            if fit_on_controls and control_col in df.columns:
                control_mask = (df[control_col] == control_key).to_numpy()
            else:
                control_mask = None

            for batch in unique_batches:
                batch_mask = batch_labels == batch
                batch_indices = np.where(batch_mask)[0]
                X_batch = X_gpu[batch_indices]

                scaler = RobustMAD(epsilon=epsilon)
                if control_mask is not None:
                    batch_control_mask = control_mask[batch_mask]
                    X_fit = X_batch[batch_control_mask]
                    if len(X_fit) > 0:
                        scaler.fit(X_fit)
                        X_norm[batch_indices] = scaler.transform(X_batch)
                    else:
                        warnings.warn(f"No controls for batch {batch}, fitting on all data")
                        X_norm[batch_indices] = scaler.fit_transform(X_batch)
                else:
                    X_norm[batch_indices] = scaler.fit_transform(X_batch)

            print(f"  Processed {len(unique_batches)} batches on GPU")
        else:
            scaler = RobustMAD(epsilon=epsilon)
            if fit_on_controls and control_col in df.columns:
                control_mask = (df[control_col] == control_key).to_numpy()
                X_fit = X_gpu[control_mask]
                if len(X_fit) > 0:
                    scaler.fit(X_fit)
                    X_norm = scaler.transform(X_gpu)
                else:
                    X_norm = scaler.fit_transform(X_gpu)
            else:
                X_norm = scaler.fit_transform(X_gpu)

        X_result = to_cpu(X_norm)
    else:
        # CPU path (default - faster for median/MAD)
        X = df.select(features).to_numpy()
        X_norm = np.zeros_like(X)

        if batch_col and batch_col in df.columns:
            batch_labels = df[batch_col].to_numpy()
            unique_batches = np.unique(batch_labels)

            if fit_on_controls and control_col in df.columns:
                control_mask = (df[control_col] == control_key).to_numpy()
            else:
                control_mask = None

            for batch in unique_batches:
                batch_mask = batch_labels == batch
                batch_indices = np.where(batch_mask)[0]
                X_batch = X[batch_indices]

                scaler = RobustMAD_CPU(epsilon=epsilon)
                if control_mask is not None:
                    batch_control_mask = control_mask[batch_mask]
                    X_fit = X_batch[batch_control_mask]
                    if len(X_fit) > 0:
                        scaler.fit(X_fit)
                        X_norm[batch_indices] = scaler.transform(X_batch)
                    else:
                        warnings.warn(f"No controls for batch {batch}, fitting on all data")
                        X_norm[batch_indices] = scaler.fit_transform(X_batch)
                else:
                    X_norm[batch_indices] = scaler.fit_transform(X_batch)

            print(f"  Processed {len(unique_batches)} batches on CPU")
        else:
            scaler = RobustMAD_CPU(epsilon=epsilon)
            if fit_on_controls and control_col in df.columns:
                control_mask = (df[control_col] == control_key).to_numpy()
                X_fit = X[control_mask]
                if len(X_fit) > 0:
                    scaler.fit(X_fit)
                    X_norm = scaler.transform(X)
                else:
                    X_norm = scaler.fit_transform(X)
            else:
                X_norm = scaler.fit_transform(X)

        X_result = X_norm

    df = df.with_columns(
        [pl.Series(name=feat, values=X_result[:, i]) for i, feat in enumerate(features)]
    )

    print(f"  Result: {df.shape}")
    return df


def normalize_standardize(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Normalize using StandardScaler (CPU by default, GPU optional)."""
    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    batch_col = config.get("batch_col", "Metadata_Plate")
    if batch_col == "":
        batch_col = None

    fit_on_controls = config.get("fit_on_controls", False)
    control_col = config.get("control_col", "Metadata_control_type")
    control_key = config.get("control_key", "negcon")
    use_gpu = config.get("use_gpu", False)  # CPU by default

    if use_gpu:
        print("\n=== Step: Normalize (Standardize - GPU) ===")
    else:
        print("\n=== Step: Normalize (Standardize - CPU) ===")
    print(f"  batch_col: {batch_col}, fit_on_controls: {fit_on_controls}")

    if use_gpu:
        # GPU path
        X_cpu = df.select(features).to_numpy()
        X_gpu = to_gpu(X_cpu)
        X_norm = cp.zeros_like(X_gpu)

        if batch_col and batch_col in df.columns:
            batch_labels = df[batch_col].to_numpy()
            unique_batches = np.unique(batch_labels)

            if fit_on_controls and control_col in df.columns:
                control_mask = (df[control_col] == control_key).to_numpy()
            else:
                control_mask = None

            for batch in unique_batches:
                batch_mask = batch_labels == batch
                batch_indices = np.where(batch_mask)[0]
                X_batch = X_gpu[batch_indices]

                scaler = StandardScaler()
                if control_mask is not None:
                    batch_control_mask = control_mask[batch_mask]
                    X_fit = X_batch[batch_control_mask]
                    if len(X_fit) > 0:
                        scaler.fit(X_fit)
                        X_norm[batch_indices] = scaler.transform(X_batch)
                    else:
                        X_norm[batch_indices] = scaler.fit_transform(X_batch)
                else:
                    X_norm[batch_indices] = scaler.fit_transform(X_batch)

            print(f"  Processed {len(unique_batches)} batches on GPU")
        else:
            scaler = StandardScaler()
            if fit_on_controls and control_col in df.columns:
                control_mask = (df[control_col] == control_key).to_numpy()
                X_fit = X_gpu[control_mask]
                if len(X_fit) > 0:
                    scaler.fit(X_fit)
                    X_norm = scaler.transform(X_gpu)
                else:
                    X_norm = scaler.fit_transform(X_gpu)
            else:
                X_norm = scaler.fit_transform(X_gpu)

        X_result = to_cpu(X_norm)
    else:
        # CPU path (default)
        X = df.select(features).to_numpy()
        X_norm = np.zeros_like(X)

        if batch_col and batch_col in df.columns:
            batch_labels = df[batch_col].to_numpy()
            unique_batches = np.unique(batch_labels)

            if fit_on_controls and control_col in df.columns:
                control_mask = (df[control_col] == control_key).to_numpy()
            else:
                control_mask = None

            for batch in unique_batches:
                batch_mask = batch_labels == batch
                batch_indices = np.where(batch_mask)[0]
                X_batch = X[batch_indices]

                scaler = StandardScaler_CPU()
                if control_mask is not None:
                    batch_control_mask = control_mask[batch_mask]
                    X_fit = X_batch[batch_control_mask]
                    if len(X_fit) > 0:
                        scaler.fit(X_fit)
                        X_norm[batch_indices] = scaler.transform(X_batch)
                    else:
                        X_norm[batch_indices] = scaler.fit_transform(X_batch)
                else:
                    X_norm[batch_indices] = scaler.fit_transform(X_batch)

            print(f"  Processed {len(unique_batches)} batches on CPU")
        else:
            scaler = StandardScaler_CPU()
            if fit_on_controls and control_col in df.columns:
                control_mask = (df[control_col] == control_key).to_numpy()
                X_fit = X[control_mask]
                if len(X_fit) > 0:
                    scaler.fit(X_fit)
                    X_norm = scaler.transform(X)
                else:
                    X_norm = scaler.fit_transform(X)
            else:
                X_norm = scaler.fit_transform(X)

        X_result = X_norm

    df = df.with_columns(
        [pl.Series(name=feat, values=X_result[:, i]) for i, feat in enumerate(features)]
    )

    print(f"  Result: {df.shape}")
    return df


def normalize_tvn(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Normalize using TVN (GPU-accelerated, vectorized batch processing)."""
    print("\n=== Step: Normalize (TVN - GPU) ===")
    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    batch_col = config.get("batch_col", "Metadata_Plate")
    if batch_col == "":
        batch_col = None

    fit_on_controls = config.get("fit_on_controls", True)
    alpha = config.get("alpha", 0.5)
    epsilon = config.get("epsilon", 1.0)
    control_col = config.get("control_col", "Metadata_control_type")
    control_key = config.get("control_key", "negcon")

    print(f"  batch_col: {batch_col}, alpha: {alpha}, fit_on_controls: {fit_on_controls}")

    # Transfer ALL data to GPU once
    X_cpu = df.select(features).to_numpy()
    X_gpu = to_gpu(X_cpu)
    X_norm = X_gpu.copy()  # Start with copy in case some batches are skipped

    if batch_col and batch_col in df.columns:
        batch_labels = df[batch_col].to_numpy()
        unique_batches = np.unique(batch_labels)

        if fit_on_controls and control_col in df.columns:
            control_mask_cpu = (df[control_col] == control_key).to_numpy()
        else:
            control_mask_cpu = None

        for batch in unique_batches:
            batch_mask = batch_labels == batch
            batch_indices = np.where(batch_mask)[0]
            X_batch = X_gpu[batch_indices]

            tvn = TVN(alpha=alpha, epsilon=epsilon)
            if control_mask_cpu is not None:
                batch_control_mask = control_mask_cpu[batch_mask]
                X_fit = X_batch[batch_control_mask]
                if len(X_fit) >= 2:
                    tvn.fit(X_fit)
                    X_norm[batch_indices] = tvn.transform(X_batch)
                else:
                    warnings.warn(f"Batch {batch}: <2 controls, skipping TVN")
            else:
                X_norm[batch_indices] = tvn.fit_transform(X_batch)

        print(f"  Processed {len(unique_batches)} batches on GPU")
    else:
        tvn = TVN(alpha=alpha, epsilon=epsilon)

        if fit_on_controls and control_col in df.columns:
            control_mask = (df[control_col] == control_key).to_numpy()
            X_fit = X_gpu[control_mask]
            if len(X_fit) >= 2:
                tvn.fit(X_fit)
                X_norm = tvn.transform(X_gpu)
            else:
                X_norm = tvn.fit_transform(X_gpu)
        else:
            X_norm = tvn.fit_transform(X_gpu)

    # Single transfer back to CPU
    X_result = to_cpu(X_norm)
    df = df.with_columns(
        [pl.Series(name=feat, values=X_result[:, i]) for i, feat in enumerate(features)]
    )

    print(f"  Result: {df.shape}")
    return df


def _check_dim_control_ratio(
    n_features: int,
    min_controls: int,
    method_name: str,
    config: dict,
) -> int:
    """Check feature-to-control ratio and clamp or abort if too high.

    When batch correction operates in a space where n_features >> n_controls_per_batch,
    the per-batch covariance estimate becomes rank-deficient and whitening amplifies
    noise — creating spurious compound-level similarity (inflated PA).

    Args:
        n_features: requested output dimensionality
        min_controls: minimum number of controls across batches
        method_name: name for logging (e.g. "TVN EFAAR", "TVN Original")
        config: step config dict with dim_ratio_threshold and dim_ratio_action

    Returns:
        Clamped n_features (may be reduced to min_controls - 1)

    Raises:
        ValueError: if action is "abort" and ratio exceeds threshold
    """
    threshold = config.get("dim_ratio_threshold", 2.5)
    action = config.get("dim_ratio_action", "clamp")

    if min_controls <= 0:
        return n_features

    ratio = n_features / min_controls
    if ratio <= threshold:
        return n_features

    clamped = min_controls - 1
    if action == "abort":
        raise ValueError(
            f"[{method_name}] dim/control ratio {ratio:.1f} exceeds threshold {threshold} "
            f"(n_features={n_features}, min_controls_per_batch={min_controls}). "
            f"Set dim_ratio_action=clamp to auto-reduce to {clamped}."
        )

    print(
        f"  WARNING: dim/control ratio {ratio:.1f} > {threshold} "
        f"(n_features={n_features}, min_controls_per_batch={min_controls})"
    )
    print(f"  Clamping output dims: {n_features} -> {clamped}")
    return clamped


def normalize_tvn_efaar(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Normalize using TVN_EFAAR (GPU-accelerated)."""
    print("\n=== Step: Normalize (TVN_EFAAR - GPU) ===")

    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    batch_col = config.get("batch_col", "Metadata_Plate")
    if batch_col == "":
        batch_col = None
    control_col = config.get("control_col", "Metadata_control_type")
    control_key = config.get("control_key", "negcon")
    epsilon = config.get("epsilon", 0.5)
    n_components = config.get("n_components", 128)

    print(f"  batch_col: {batch_col}, epsilon: {epsilon}, n_components: {n_components}")

    with GPUMemoryManager() as gpu:
        X = gpu.transfer(df.select(features).to_numpy())
        control_mask = (df[control_col] == control_key).to_numpy()
        control_mask_gpu = cp.asarray(control_mask)

        # Encode string batch labels to integers for GPU
        if batch_col and batch_col in df.columns:
            batch_labels_str = df[batch_col].to_numpy()
            unique_batches, batch_labels_int = np.unique(batch_labels_str, return_inverse=True)
            batch_labels = cp.asarray(batch_labels_int)
            print(f"  Encoded {len(unique_batches)} unique batches")
        else:
            batch_labels = None

        # Compute min controls per batch for dim/control ratio check
        min_controls_per_batch = int(control_mask.sum())  # fallback: global total
        if batch_labels is not None:
            unique_b = cp.unique(batch_labels)
            batch_ctrl_counts = []
            for b in unique_b:
                bc = int(((batch_labels == b) & control_mask_gpu).sum())
                if bc >= 2:
                    batch_ctrl_counts.append(bc)
            if batch_ctrl_counts:
                min_controls_per_batch = min(batch_ctrl_counts)
                print(f"  Controls per batch: min={min_controls_per_batch}, max={max(batch_ctrl_counts)}")

        print(f"  Input shape: {X.shape}, controls: {control_mask.sum()}")

        # Step 1: Global center/scale on controls
        print("  Step 1: Global center/scale on controls...")
        scaler_global = StandardScaler()
        scaler_global.fit(X[control_mask_gpu])
        X = scaler_global.transform(X)

        # Step 2: PCA fit on controls, transform all
        max_components = min(X.shape[1], int(control_mask.sum()) - 1)
        n_components = min(n_components, max_components)

        # Check dim/control ratio and clamp if needed
        n_components = _check_dim_control_ratio(
            n_components, min_controls_per_batch, "TVN EFAAR", config
        )

        print(f"  Step 2: PCA on controls (n_components={n_components})...")
        pca = PCATransform(n_components=n_components)
        pca.fit(X[control_mask_gpu])
        X = pca.transform(X)
        print(f"    PCA reduced: {X.shape[1]} components")

        # Step 3: Per-batch center/scale on controls
        print("  Step 3: Per-batch center/scale on controls...")
        if batch_labels is not None:
            unique_batches = cp.unique(batch_labels)
            for batch in unique_batches:
                batch_mask = batch_labels == batch
                batch_control_mask = batch_mask & control_mask_gpu
                if int(batch_control_mask.sum()) >= 2:
                    scaler_batch = StandardScaler()
                    scaler_batch.fit(X[batch_control_mask])
                    X[batch_mask] = scaler_batch.transform(X[batch_mask])

        # Step 4: CORAL
        print("  Step 4: CORAL transformation...")
        X = tvn_efaar_on_controls(X, control_mask_gpu, batch_labels, epsilon=epsilon)

        # Create new feature names
        X_cpu = to_cpu(X)
        new_features = [f"PC_{i}" for i in range(X_cpu.shape[1])]

    # Build result dataframe
    metadata_cols = [c for c in df.columns if c not in features]
    result_df = df.select(metadata_cols)

    for i, feat_name in enumerate(new_features):
        result_df = result_df.with_columns(pl.Series(name=feat_name, values=X_cpu[:, i]))

    print(f"  Result: {result_df.shape}")
    return result_df


def normalize_tvn_original(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Normalize using Original TVN: global PCA whitening + per-batch CORAL (no recoloring)."""
    print("\n=== Step: Normalize (TVN Original - GPU) ===")

    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    batch_col = config.get("batch_col", "Metadata_Plate")
    control_col = config.get("control_col", "Metadata_control_type")
    control_key = config.get("control_key", "negcon")
    # Use adaptive k: match input feature count (works with or without PCA)
    k_requested = config.get("k", min(50, len(features)))
    k = min(k_requested, len(features))
    epsilon = config.get("epsilon", 1e-8)

    if k < k_requested:
        print(f"  k clamped: {k_requested} -> {k} (only {len(features)} input features)")
    print(f"  k={k}, n_features={len(features)}, batch_col={batch_col}, epsilon={epsilon}")

    with GPUMemoryManager() as gpu:
        X_all = gpu.transfer(df.select(features).to_numpy())
        batch_labels = df[batch_col].to_numpy()
        control_mask = (df[control_col] == control_key).to_numpy()

        # Build negcons dict and compute min controls per batch
        negcons = {}
        batch_ctrl_counts = []
        unique_batches = np.unique(batch_labels)
        for batch in unique_batches:
            batch_mask = batch_labels == batch
            batch_control_mask = batch_mask & control_mask
            n_ctrl = int(batch_control_mask.sum())
            if n_ctrl >= 2:
                negcons[batch] = X_all[batch_control_mask]
                batch_ctrl_counts.append(n_ctrl)

        min_controls_per_batch = min(batch_ctrl_counts) if batch_ctrl_counts else 0
        print(f"  {len(negcons)} batches with >=2 negcons (min={min_controls_per_batch})")

        # Check dim/control ratio and clamp k if needed
        k = _check_dim_control_ratio(k, min_controls_per_batch, "TVN Original", config)

        # Fit TVN
        tvn = TVN_Original(k=k, epsilon=epsilon)
        tvn.fit(negcons)

        # Transform all data
        X_out = cp.zeros((len(df), k), dtype=cp.float32)
        for batch in unique_batches:
            batch_mask = batch_labels == batch
            batch_indices = np.where(batch_mask)[0]
            if batch in negcons:
                X_out[batch_indices] = tvn.transform_batch(X_all[batch_indices], batch)

        X_cpu = to_cpu(X_out)

    # Create output DataFrame
    new_features = [f"TVN_{i}" for i in range(k)]
    metadata_cols = [c for c in df.columns if c not in features]
    result = df.select(metadata_cols)
    for i, feat in enumerate(new_features):
        result = result.with_columns(pl.Series(name=feat, values=X_cpu[:, i]))

    print(f"  Result: {result.shape}")
    return result


def normalize_tvn_cascade(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Normalize using Cascade TVN: two-stage whitening for small n_neg situations."""
    print("\n=== Step: Normalize (TVN Cascade - GPU) ===")

    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    batch_col = config.get("batch_col", "Metadata_Plate")
    control_col = config.get("control_col", "Metadata_control_type")
    control_key = config.get("control_key", "negcon")
    # Use adaptive k: cap k1 at input features, k2 at min(10, k1)
    k1 = config.get("k1", min(100, len(features)))
    k2 = config.get("k2", min(10, k1))
    epsilon = config.get("epsilon", 1e-8)

    print(f"  k1={k1}, k2={k2} (adaptive from {len(features)} features), batch_col={batch_col}, epsilon={epsilon}")

    with GPUMemoryManager() as gpu:
        X_all = gpu.transfer(df.select(features).to_numpy())
        batch_labels = df[batch_col].to_numpy()
        control_mask = (df[control_col] == control_key).to_numpy()

        # Build negcons dict
        negcons = {}
        unique_batches = np.unique(batch_labels)
        for batch in unique_batches:
            batch_mask = batch_labels == batch
            batch_control_mask = batch_mask & control_mask
            if batch_control_mask.sum() >= 2:
                negcons[batch] = X_all[batch_control_mask]

        print(f"  {len(negcons)} batches with >=2 negcons")

        # Fit Cascade TVN
        tvn = TVN_Cascade(k1=k1, k2=k2, epsilon=epsilon)
        tvn.fit(negcons)

        # Transform all data
        X_out = cp.zeros((len(df), k2), dtype=cp.float32)
        for batch in unique_batches:
            batch_mask = batch_labels == batch
            batch_indices = np.where(batch_mask)[0]
            if batch in negcons:
                X_out[batch_indices] = tvn.transform_batch(X_all[batch_indices], batch)

        X_cpu = to_cpu(X_out)

    # Create output DataFrame
    new_features = [f"TVN_Cascade_{i}" for i in range(k2)]
    metadata_cols = [c for c in df.columns if c not in features]
    result = df.select(metadata_cols)
    for i, feat in enumerate(new_features):
        result = result.with_columns(pl.Series(name=feat, values=X_cpu[:, i]))

    print(f"  Result: {result.shape}")
    return result


def normalize_spherize(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Normalize using Spherize (GPU-accelerated)."""
    print("\n=== Step: Normalize (Spherize - GPU) ===")
    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    batch_col = config.get("batch_col", "Metadata_Plate")
    if batch_col == "":
        batch_col = None

    fit_on_controls = config.get("fit_on_controls", False)
    method = config.get("method", "ZCA-cor")
    epsilon = config.get("epsilon", 1e-6)
    control_col = config.get("control_col", "Metadata_control_type")
    control_key = config.get("control_key", "negcon")

    remove_variance_threshold = config.get("remove_variance_threshold")
    remove_variance_method = config.get("remove_variance_method", "threshold")
    n_permutations = config.get("n_permutations", 10)

    is_global = not batch_col
    scope_str = "GLOBAL (all plates)" if is_global else f"per-batch ({batch_col})"
    print(f"  method: {method}, scope: {scope_str}, fit_on_controls: {fit_on_controls}, epsilon: {epsilon}")
    if remove_variance_method == "pa":
        print(f"  remove_variance_method: pa (parallel analysis, {n_permutations} permutations)")
    elif remove_variance_method == "mp":
        print(f"  remove_variance_method: mp (Marchenko-Pastur truncation)")
    elif remove_variance_threshold is not None:
        print(f"  remove_variance_threshold: {remove_variance_threshold} (truncated projection)")

    def _build_spherize():
        return Spherize(method=method, epsilon=epsilon,
                        remove_variance_threshold=remove_variance_threshold,
                        remove_variance_method=remove_variance_method,
                        n_permutations=n_permutations)

    def _rebuild_df_truncated(df_in, features_in, X_norm, spherize_obj):
        """Rebuild DataFrame with new feature names when dims are reduced."""
        k = spherize_obj.k_removed_
        n_out = X_norm.shape[1]
        print(f"  Removed top {k} PCs ({spherize_obj.variance_removed_*100:.1f}% variance), {n_out} dims remaining")
        new_features = [f"Spherize_{i}" for i in range(n_out)]
        metadata_cols = [c for c in df_in.columns if c not in features_in]
        result = df_in.select(metadata_cols)
        for i, feat in enumerate(new_features):
            result = result.with_columns(pl.Series(name=feat, values=X_norm[:, i]))
        return result

    with GPUMemoryManager() as gpu:
        if batch_col and batch_col in df.columns:
            normalized_dfs = []

            for batch in sorted(df[batch_col].unique().to_list()):
                batch_mask = df[batch_col] == batch
                batch_df = df.filter(batch_mask)
                X = gpu.transfer(batch_df.select(features).to_numpy())

                spherize = _build_spherize()
                if fit_on_controls and control_col in df.columns:
                    control_mask = batch_df[control_col] == control_key
                    X_fit = X[control_mask.to_numpy()]
                    if len(X_fit) > 0:
                        spherize.fit(X_fit)
                        X_norm = to_cpu(spherize.transform(X))
                    else:
                        X_norm = to_cpu(spherize.fit_transform(X))
                else:
                    X_norm = to_cpu(spherize.fit_transform(X))

                if spherize.k_removed_ is not None:
                    batch_df = _rebuild_df_truncated(batch_df, features, X_norm, spherize)
                else:
                    print(f"  DEBUG batch={batch}: len(features)={len(features)}, X.shape={X.shape}, X_norm.shape={X_norm.shape}")
                    if X_norm.shape[1] != len(features):
                        raise ValueError(f"Spherize dimension mismatch: {len(features)} features != {X_norm.shape[1]} columns")
                    batch_df = batch_df.with_columns(
                        [pl.Series(name=feat, values=X_norm[:, i]) for i, feat in enumerate(features)]
                    )
                normalized_dfs.append(batch_df)

            df = pl.concat(normalized_dfs)
        else:
            # Global spherize: process all plates together (JUMP recipe approach)
            X = gpu.transfer(df.select(features).to_numpy())
            spherize = _build_spherize()

            if fit_on_controls and control_col in df.columns:
                control_mask = df[control_col] == control_key
                X_fit = X[control_mask.to_numpy()]
                n_controls = len(X_fit)
                n_features = X.shape[1]
                print(f"  Global spherize: fitting on {n_controls} controls, {n_features} features")
                if n_controls < n_features:
                    print(f"  WARNING: n_controls ({n_controls}) < n_features ({n_features})")
                    print(f"  Covariance will be rank-deficient. Epsilon={epsilon} provides regularization.")
                if len(X_fit) > 0:
                    spherize.fit(X_fit)
                    X_norm = to_cpu(spherize.transform(X))
                else:
                    X_norm = to_cpu(spherize.fit_transform(X))
            else:
                print(f"  Global spherize: fitting on all {X.shape[0]} samples, {X.shape[1]} features")
                X_norm = to_cpu(spherize.fit_transform(X))

            if spherize.k_removed_ is not None:
                df = _rebuild_df_truncated(df, features, X_norm, spherize)
            else:
                print(f"  DEBUG: len(features)={len(features)}, X.shape={X.shape}, X_norm.shape={X_norm.shape}")
                if X_norm.shape[1] != len(features):
                    raise ValueError(f"Spherize dimension mismatch: {len(features)} features != {X_norm.shape[1]} columns")
                df = df.with_columns(
                    [pl.Series(name=feat, values=X_norm[:, i]) for i, feat in enumerate(features)]
                )

    print(f"  Result: {df.shape}")
    return df


def normalize_pca(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Reduce dimensionality using PCA (GPU-accelerated)."""
    print("\n=== Step: Normalize (PCA - GPU) ===")
    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    n_components = config.get("n_components", 128)
    whiten = config.get("whiten", False)
    fit_on_controls = config.get("fit_on_controls", True)
    control_col = config.get("control_col", "Metadata_control_type")
    control_key = config.get("control_key", "negcon")

    print(f"  n_components: {n_components}, whiten: {whiten}, fit_on_controls: {fit_on_controls}")
    print(f"  Input features: {len(features)}")

    with GPUMemoryManager() as gpu:
        X = gpu.transfer(df.select(features).to_numpy())

        if fit_on_controls:
            control_mask = (df[control_col] == control_key).to_numpy()
            X_fit = X[control_mask]
            print(f"  Fitting on {len(X_fit)} control samples")
        else:
            X_fit = X
            print(f"  Fitting on all {len(X_fit)} samples")

        max_components = min(X_fit.shape[0], X_fit.shape[1])
        if n_components >= max_components:
            print(f"  WARNING: n_components={n_components} >= max possible={max_components}")
            if max_components <= 1:
                print(f"  SKIPPING PCA")
                return df
            n_components = max_components - 1

        pca = PCATransform(n_components=n_components, whiten=whiten)
        pca.fit(X_fit)
        X_transformed = to_cpu(pca.transform(X))

    n_out = X_transformed.shape[1]
    new_features = [f"PC_{i+1}" for i in range(n_out)]

    if pca.explained_variance_ratio_ is not None:
        explained_var = float(to_cpu(pca.explained_variance_ratio_).sum()) * 100
        print(f"  Output features: {n_out} (explained variance: {explained_var:.1f}%)")

    metadata_cols = [c for c in df.columns if c not in features]
    result = df.select(metadata_cols)

    for i, feat_name in enumerate(new_features):
        result = result.with_columns(pl.Series(name=feat_name, values=X_transformed[:, i]))

    print(f"  Result: {result.shape}")
    return result


def prune_correlated(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Prune correlated features (GPU-accelerated)."""
    print("\n=== Step: Prune Correlated Features (GPU) ===")
    features, metadata = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    threshold = config.get("threshold", 0.9)
    print(f"  threshold: {threshold}")

    with GPUMemoryManager() as gpu:
        X = gpu.transfer(df.select(features).to_numpy())
        new_features = correlation_threshold(X, features, threshold=threshold)

    n_dropped = len(features) - len(new_features)
    print(f"  Dropped {n_dropped} correlated features ({len(new_features)} remaining)")

    df = df.select(metadata + new_features)
    print(f"  Result: {df.shape}")
    return df


def well_position_correct(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Correct for well position effects.

    Subtracts the mean of each well position across all plates to remove
    systematic spatial biases. Based on Broad JUMP profiling recipe.
    """
    print("\n=== Step: Well Position Correction ===")
    from norm_3.core import well_position_correct as _well_position_correct

    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    well_col = config.get("well_col", "Metadata_Well")
    print(f"  well_col: {well_col}")

    if well_col not in df.columns:
        print(f"  WARNING: Well column '{well_col}' not found, skipping")
        return df

    n_wells = df[well_col].n_unique()
    print(f"  Correcting for {n_wells} unique well positions")

    X = df.select(features).to_numpy()
    well_positions = df[well_col].to_numpy()

    X_corrected = _well_position_correct(X, well_positions)

    df = df.with_columns(
        [pl.Series(name=feat, values=X_corrected[:, i]) for i, feat in enumerate(features)]
    )

    print(f"  Result: {df.shape}")
    return df


def inverse_normal_transform(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Apply rank-based inverse normal transformation.

    Maps each feature to normal distribution quantiles based on ranks.
    Based on Broad JUMP profiling recipe INT step.
    """
    print("\n=== Step: Inverse Normal Transform (INT) ===")
    from norm_3.core import inverse_normal_transform as _inverse_normal_transform

    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    X = df.select(features).to_numpy()
    X_transformed = _inverse_normal_transform(X)

    df = df.with_columns(
        [pl.Series(name=feat, values=X_transformed[:, i]) for i, feat in enumerate(features)]
    )

    print(f"  Result: {df.shape}")
    return df


def sample_norm(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Apply L1/L2/max normalization per sample."""
    print("\n=== Step: Sample Normalize ===")
    from norm_3.core import sample_normalize as _sample_normalize

    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    norm_type = config.get("norm", "l2")
    print(f"  Applying {norm_type} normalization to {len(features)} features")

    X = df.select(features).to_numpy()
    X_normalized = _sample_normalize(X, norm=norm_type)

    df = df.with_columns(
        [pl.Series(name=feat, values=X_normalized[:, i]) for i, feat in enumerate(features)]
    )

    print(f"  Result: {df.shape}")
    return df


def evaluate_metrics(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Evaluate metrics."""
    print("\n=== Step: Evaluate Metrics ===")
    from norm_3.metrics import evaluate_all

    output_dir = Path(config.get("output_dir", "."))
    output_dir.mkdir(exist_ok=True)

    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    evaluate_all(
        df,
        features,
        output_dir=output_dir,
        skip_visualization=config.get("skip_visualization", False),
        skip_umap=config.get("skip_umap", False),
        n_top_compounds=config.get("n_top_compounds", 20),
        min_compounds_per_target=config.get("min_compounds_per_target", 3),
        compound_col=config.get("compound_col", "Metadata_pert_iname"),
        target_col=config.get("target_col", "Metadata_target_list"),
        negcon_col=config.get("negcon_col", "Metadata_negcon"),
        batch_col=config.get("batch_col", "Metadata_Plate"),
        pc_groups=config.get("pc_groups", None),
        skip_batch_effects=config.get("skip_batch_effects", True),
        well_col=config.get("well_col", "Metadata_Well"),
        distance=config.get("distance", None),
    )

    return df


# =============================================================================
# Step Registry
# =============================================================================


STEPS = {
    "clean_nans": clean_nans,
    "merge_metadata": merge_metadata,
    "filter_features": filter_features,
    "aggregate_wells": aggregate_wells,
    "normalize_robustmad": normalize_robustmad,
    "normalize_standardize": normalize_standardize,
    "normalize_tvn": normalize_tvn,
    "normalize_tvn_efaar": normalize_tvn_efaar,
    "normalize_tvn_original": normalize_tvn_original,
    "normalize_tvn_cascade": normalize_tvn_cascade,
    "normalize_spherize": normalize_spherize,
    "normalize_pca": normalize_pca,
    "prune_correlated": prune_correlated,
    "well_position_correct": well_position_correct,
    "inverse_normal_transform": inverse_normal_transform,
    "sample_norm": sample_norm,
    "evaluate_metrics": evaluate_metrics,
}


# =============================================================================
# Output Naming
# =============================================================================


def generate_output_name(config: dict) -> str:
    """Generate output directory name based on pipeline configuration."""
    parts = []

    for step_config in config.get("steps", []):
        if not step_config.get("enabled", True):
            continue

        step_name = step_config["name"]
        params = step_config.get("params", {})

        if step_name == "normalize_robustmad":
            fit_ctrl = params.get("fit_on_controls", False)
            parts.append("robustmad" + ("_ctrl" if fit_ctrl else "_all"))
        elif step_name == "normalize_standardize":
            fit_ctrl = params.get("fit_on_controls", False)
            parts.append("std" + ("_ctrl" if fit_ctrl else "_all"))
        elif step_name == "normalize_pca":
            n_comp = params.get("n_components", 128)
            parts.append(f"pca{n_comp}")
        elif step_name == "normalize_tvn":
            alpha = params.get("alpha", 0.5)
            parts.append(f"tvn_a{alpha}")
        elif step_name == "normalize_tvn_original":
            k = params.get("k", 64)
            parts.append(f"tvn_original_k{k}")
        elif step_name == "normalize_tvn_cascade":
            k1 = params.get("k1", 128)
            k2 = params.get("k2", 32)
            parts.append(f"cascade_tvn_k{k1}_k{k2}")
        elif step_name == "normalize_tvn_efaar":
            eps = params.get("epsilon", 0.5)
            n_comp = params.get("n_components", 128)
            name = f"tvn_efaar_e{eps}"
            if n_comp != 128:
                name += f"_c{n_comp}"
            parts.append(name)
        elif step_name == "normalize_spherize":
            method = params.get("method", "ZCA-cor")
            fit_ctrl = params.get("fit_on_controls", False)
            epsilon = params.get("epsilon", 1.0e-6)
            batch_col = params.get("batch_col", "Metadata_Plate")
            is_global = not batch_col or batch_col == ""
            # Format epsilon: 1e-6 -> "e6", 0.5 -> "e0.5"
            if epsilon < 0.001:
                eps_str = f"e{abs(int(round(np.log10(epsilon))))}"
            else:
                eps_str = f"e{epsilon}"
            scope = "_global" if is_global else ""
            name = method + scope + ("_ctrl" if fit_ctrl else "_all") + f"_{eps_str}"
            rvm = params.get("remove_variance_method", "threshold")
            if rvm == "pa":
                n_perm = params.get("n_permutations", 10)
                name += f"_truncPA{n_perm}"
            elif rvm == "mp":
                name += "_truncMP"
            else:
                rvt = params.get("remove_variance_threshold")
                if rvt is not None:
                    name += f"_trunc{rvt}"
            parts.append(name)
        elif step_name == "filter_features":
            # Add filter config to output name
            filters = params.get("filters", [])
            for f in filters:
                if f.get("name") == "drop_outliers":
                    cutoff = f.get("outlier_cutoff", 50)
                    parts.append(f"outlier{cutoff}")
                    break
        elif step_name == "prune_correlated":
            thresh = params.get("threshold", 0.90)
            parts.append(f"prune{thresh}")
        elif step_name == "well_position_correct":
            parts.append("wellcorr")
        elif step_name == "inverse_normal_transform":
            parts.append("INT")
        elif step_name == "sample_norm":
            norm_type = params.get("norm", "l2")
            parts.append(f"snorm_{norm_type}")
        elif step_name == "aggregate_wells":
            method = params.get("method", "median")
            parts.append(f"agg_{method}" if method != "median" else "agg")

    # When use_prune_correlated is explicitly swept to false, mark it in the name
    # to distinguish from prune=true configs (which get "prune{thresh}" from above)
    if config.get("use_prune_correlated") is False:
        has_prune_part = any(p.startswith("prune") for p in parts)
        if not has_prune_part:
            parts.append("noprune")

    return "__".join(parts) if parts else "default"


# =============================================================================
# Sweep Parameter Handling
# =============================================================================


def apply_sweep_overrides(config: dict) -> dict:
    """Apply top-level sweep parameters to step configurations.

    This allows sweep configs to use simple parameters like:
        norm_method: robustmad
        use_spherize: true

    Instead of modifying nested step configurations directly.
    """
    # Handle batch_method (none, tvn, tvn_efaar)
    batch_method = config.get("batch_method", None)
    norm_method = config.get("norm_method", "robustmad")

    if batch_method is not None:
        for step in config.get("steps", []):
            step_name = step.get("name", "")
            if step_name == "normalize_tvn":
                step["enabled"] = (batch_method == "tvn")
            elif step_name == "normalize_tvn_efaar":
                step["enabled"] = (batch_method == "tvn_efaar")
            elif step_name == "normalize_tvn_original":
                step["enabled"] = (batch_method == "tvn_original")
            elif step_name == "normalize_tvn_cascade":
                step["enabled"] = (batch_method == "cascade_tvn")
            elif step_name == "normalize_spherize":
                # Enable spherize if batch_method=spherize/spherize_global OR use_spherize=true
                step["enabled"] = (batch_method in ("spherize", "spherize_global")) or config.get("use_spherize", False)
            elif step_name == "normalize_robustmad":
                # Disable if norm_method is not robustmad, or if norm_method is none
                step["enabled"] = (norm_method == "robustmad")
            elif step_name == "normalize_standardize":
                # Disable if norm_method is not standardize, or if norm_method is none
                step["enabled"] = (norm_method == "standardize")
            elif step_name == "normalize_pca":
                step["enabled"] = config.get("use_pca", False) or batch_method == "tvn"

    # Propagate top-level sweep parameters to step configurations
    for step in config.get("steps", []):
        step_name = step.get("name", "")
        if "params" not in step:
            step["params"] = {}

        # Apply spherize-specific sweep parameters
        if step_name == "normalize_spherize":
            if "spherize_epsilon" in config:
                step["params"]["epsilon"] = config["spherize_epsilon"]
            if "spherize_fit_controls" in config:
                step["params"]["fit_on_controls"] = config["spherize_fit_controls"]
            if "spherize_remove_variance_threshold" in config:
                step["params"]["remove_variance_threshold"] = config["spherize_remove_variance_threshold"]
            if "spherize_remove_variance_method" in config:
                step["params"]["remove_variance_method"] = config["spherize_remove_variance_method"]
            if "spherize_n_permutations" in config:
                step["params"]["n_permutations"] = config["spherize_n_permutations"]

        # Handle step enable/disable flags
        if step_name == "normalize_spherize" and "use_spherize" in config:
            step["enabled"] = config["use_spherize"]
        elif step_name == "normalize_pca" and "use_pca" in config:
            step["enabled"] = config["use_pca"]
        elif step_name == "filter_features" and "use_filter_features" in config:
            # Don't override if step has always_enabled flag
            if not step.get("params", {}).get("always_enabled", False):
                step["enabled"] = config["use_filter_features"]
        elif step_name == "prune_correlated" and "use_prune_correlated" in config:
            # Don't override if step has always_enabled flag
            if not step.get("params", {}).get("always_enabled", False):
                step["enabled"] = config["use_prune_correlated"]
        elif step_name == "well_position_correct" and "use_wellcorr" in config:
            step["enabled"] = config["use_wellcorr"]
        elif step_name == "inverse_normal_transform" and "use_int" in config:
            step["enabled"] = config["use_int"]
        elif step_name == "sample_norm" and "use_sample_norm" in config:
            step["enabled"] = config["use_sample_norm"]

        # Handle step-specific parameters
        if step_name == "normalize_robustmad":
            if "norm_fit_controls" in config:
                step["params"]["fit_on_controls"] = config["norm_fit_controls"]

        elif step_name == "normalize_standardize":
            if "norm_fit_controls" in config:
                step["params"]["fit_on_controls"] = config["norm_fit_controls"]

        elif step_name == "normalize_tvn":
            if "tvn_alpha" in config:
                step["params"]["alpha"] = config["tvn_alpha"]
            if "tvn_epsilon" in config:
                step["params"]["epsilon"] = config["tvn_epsilon"]
            if "tvn_fit_controls" in config:
                step["params"]["fit_on_controls"] = config["tvn_fit_controls"]

        elif step_name == "normalize_tvn_efaar":
            if "tvn_efaar_epsilon" in config:
                step["params"]["epsilon"] = config["tvn_efaar_epsilon"]
            if "tvn_efaar_n_components" in config:
                step["params"]["n_components"] = config["tvn_efaar_n_components"]
            if "dim_ratio_threshold" in config:
                step["params"]["dim_ratio_threshold"] = config["dim_ratio_threshold"]
            if "dim_ratio_action" in config:
                step["params"]["dim_ratio_action"] = config["dim_ratio_action"]

        elif step_name == "normalize_tvn_original":
            if "tvn_original_k" in config:
                step["params"]["k"] = config["tvn_original_k"]
            if "dim_ratio_threshold" in config:
                step["params"]["dim_ratio_threshold"] = config["dim_ratio_threshold"]
            if "dim_ratio_action" in config:
                step["params"]["dim_ratio_action"] = config["dim_ratio_action"]

        elif step_name == "normalize_tvn_cascade":
            if "tvn_cascade_k1" in config:
                step["params"]["k1"] = config["tvn_cascade_k1"]
            if "tvn_cascade_k2" in config:
                step["params"]["k2"] = config["tvn_cascade_k2"]
            if "dim_ratio_threshold" in config:
                step["params"]["dim_ratio_threshold"] = config["dim_ratio_threshold"]
            if "dim_ratio_action" in config:
                step["params"]["dim_ratio_action"] = config["dim_ratio_action"]

        elif step_name == "normalize_spherize":
            if "spherize_method" in config:
                step["params"]["method"] = config["spherize_method"]
            if "spherize_epsilon" in config:
                step["params"]["epsilon"] = config["spherize_epsilon"]
            if "spherize_remove_variance_threshold" in config:
                step["params"]["remove_variance_threshold"] = config["spherize_remove_variance_threshold"]
            if "spherize_remove_variance_method" in config:
                step["params"]["remove_variance_method"] = config["spherize_remove_variance_method"]
            if "spherize_n_permutations" in config:
                step["params"]["n_permutations"] = config["spherize_n_permutations"]
            if batch_method == "spherize_global":
                # JUMP recipe: global scope (all plates), fit on controls only
                step["params"]["batch_col"] = ""
                step["params"]["fit_on_controls"] = True
            elif batch_method == "spherize":
                # Per-batch spherize: use Metadata_Batch as batch scope
                step["params"]["batch_col"] = "Metadata_Batch"
                if "spherize_fit_controls" in config:
                    step["params"]["fit_on_controls"] = config["spherize_fit_controls"]
            else:
                if "spherize_fit_controls" in config:
                    step["params"]["fit_on_controls"] = config["spherize_fit_controls"]

        elif step_name == "normalize_pca":
            if "pca_n_components" in config:
                step["params"]["n_components"] = config["pca_n_components"]
            if "pca_whiten" in config:
                step["params"]["whiten"] = config["pca_whiten"]

        elif step_name == "aggregate_wells":
            if "agg_method" in config:
                step["params"]["method"] = config["agg_method"]

        elif step_name == "prune_correlated":
            if "corr_thresh" in config:
                step["params"]["threshold"] = config["corr_thresh"]

        elif step_name == "sample_norm":
            if "sample_norm_type" in config:
                step["params"]["norm"] = config["sample_norm_type"]

        elif step_name == "filter_features":
            if "outlier_cutoff" in config:
                # Update the drop_outliers filter with the sweep parameter
                filters = step["params"].get("filters", [])
                for filter_op in filters:
                    if filter_op.get("name") == "drop_outliers":
                        filter_op["outlier_cutoff"] = config["outlier_cutoff"]

        elif step_name == "evaluate_metrics":
            if "skip_batch_effects" in config:
                step["params"]["skip_batch_effects"] = config["skip_batch_effects"]

    return config


def is_redundant_config(config: dict) -> tuple[bool, str | None]:
    """Check if configuration is redundant due to irrelevant method-specific params.

    When a method is disabled, its parameters don't matter. Skip configs where
    disabled methods have non-default parameter values.

    The canonical defaults (first value in sweeps) are:
    - spherize_method: PCA-cor
    - spherize_fit_controls: True
    - pca_n_components: 64 (cellprofiler) or 128 (embeddings)
    - corr_thresh: 0.85 (cellprofiler) or 0.90 (embeddings)
    - tvn_efaar_epsilon: 0.5
    """
    use_spherize = config.get("use_spherize", False)
    use_pca = config.get("use_pca", False)
    batch_method = config.get("batch_method", "none")

    # TVN/TVN_EFAAR + spherize is redundant - TVN already does covariance alignment
    # via CORAL, so adding spherize on top is unnecessary and potentially harmful
    if batch_method in ("tvn", "tvn_efaar") and use_spherize:
        return True, f"batch_method={batch_method} + use_spherize=true is redundant (TVN already aligns covariance)"

    # When use_spherize=false AND batch_method!=spherize, spherize params don't matter
    # Use PCA-cor as canonical default (first value in sweep)
    # BUT: when batch_method=spherize, spherize params DO matter even if use_spherize=false
    # NOTE: spherize_fit_controls is handled by the batch_method check below
    if not use_spherize and batch_method != "spherize":
        spherize_method = config.get("spherize_method", "PCA-cor")
        if spherize_method != "PCA-cor":
            return True, f"use_spherize=false ignores spherize_method={spherize_method}"

    # When batch_method is not spherize or spherize_global, spherize params don't matter
    # Accept 1e-6, 0.1, and 0.5 as canonical defaults (different sweep versions)
    if batch_method not in ("spherize", "spherize_global"):
        spherize_epsilon = config.get("spherize_epsilon", 0.1)
        canonical_eps = {1.0e-6, 0.1, 0.5}
        if not any(abs(spherize_epsilon - c) < 1e-10 for c in canonical_eps):
            return True, f"batch_method={batch_method} ignores spherize_epsilon={spherize_epsilon}"
        spherize_fit_controls = config.get("spherize_fit_controls", False)
        if spherize_fit_controls != False:
            return True, f"batch_method={batch_method} ignores spherize_fit_controls={spherize_fit_controls}"
        spherize_rvt = config.get("spherize_remove_variance_threshold")
        if spherize_rvt is not None:
            return True, f"batch_method={batch_method} ignores spherize_remove_variance_threshold={spherize_rvt}"
        spherize_rvm = config.get("spherize_remove_variance_method", "threshold")
        if spherize_rvm != "threshold":
            return True, f"batch_method={batch_method} ignores spherize_remove_variance_method={spherize_rvm}"

    # When remove_variance_method=mp or pa, threshold values don't matter (k is determined automatically)
    if batch_method in ("spherize", "spherize_global"):
        spherize_rvm = config.get("spherize_remove_variance_method", "threshold")
        if spherize_rvm in ("mp", "pa"):
            spherize_rvt = config.get("spherize_remove_variance_threshold")
            if spherize_rvt is not None:
                return True, f"spherize_remove_variance_method={spherize_rvm} ignores threshold={spherize_rvt}"

    # When use_pca=false, pca params don't matter
    # Only accept 64 (smallest/first sweep value) as canonical default
    if not use_pca and batch_method != "tvn":
        pca_n_components = config.get("pca_n_components", 64)
        if pca_n_components != 64:
            return True, f"use_pca=false ignores pca_n_components={pca_n_components}"

    # When use_prune_correlated=false AND prune step isn't always_enabled, corr_thresh doesn't matter
    # Accept 0.85 and 0.90 as valid defaults (different sweeps)
    use_prune_correlated = config.get("use_prune_correlated", False)
    prune_always_enabled = any(
        s.get("name") == "prune_correlated" and s.get("params", {}).get("always_enabled", False)
        for s in config.get("steps", [])
    )
    if not use_prune_correlated and not prune_always_enabled:
        corr_thresh = config.get("corr_thresh", 0.90)
        if corr_thresh not in (0.85, 0.90):
            return True, f"use_prune_correlated=false ignores corr_thresh={corr_thresh}"

    # When use_filter_features=false, outlier_cutoff doesn't matter
    # Use first sweep value as canonical default
    use_filter_features = config.get("use_filter_features", True)
    if not use_filter_features:
        outlier_cutoff = config.get("outlier_cutoff", 15)
        if outlier_cutoff != 15:
            return True, f"use_filter_features=false ignores outlier_cutoff={outlier_cutoff}"

    # When batch_method != tvn_efaar, tvn_efaar params don't matter
    # Use 0.5 and 128 as canonical defaults
    if batch_method != "tvn_efaar":
        tvn_efaar_epsilon = config.get("tvn_efaar_epsilon", 0.5)
        if tvn_efaar_epsilon != 0.5:
            return True, f"batch_method={batch_method} ignores tvn_efaar_epsilon={tvn_efaar_epsilon}"
        tvn_efaar_n_components = config.get("tvn_efaar_n_components", 128)
        if tvn_efaar_n_components != 128:
            return True, f"batch_method={batch_method} ignores tvn_efaar_n_components={tvn_efaar_n_components}"

    # When batch_method != tvn_original, tvn_original_k doesn't matter
    if batch_method != "tvn_original":
        tvn_original_k = config.get("tvn_original_k", 64)
        if tvn_original_k != 64:
            return True, f"batch_method={batch_method} ignores tvn_original_k={tvn_original_k}"

    # When batch_method != cascade_tvn, tvn_cascade_k1/k2 don't matter
    if batch_method != "cascade_tvn":
        tvn_cascade_k1 = config.get("tvn_cascade_k1", 128)
        if tvn_cascade_k1 != 128:
            return True, f"batch_method={batch_method} ignores tvn_cascade_k1={tvn_cascade_k1}"
        tvn_cascade_k2 = config.get("tvn_cascade_k2", 32)
        if tvn_cascade_k2 != 32:
            return True, f"batch_method={batch_method} ignores tvn_cascade_k2={tvn_cascade_k2}"

    # NOTE: We no longer skip configs based on batch_method=tvn_efaar + norm_method
    # TVN_EFAAR now runs AFTER basic normalization (robustmad/standardize/none) and outlier removal
    # so all combinations are valid

    return False, None


# =============================================================================
# Main Pipeline
# =============================================================================


def run_pipeline(
    config_path: str | Path | None = None,
    input_override: str | None = None,
    hydra_config: DictConfig | None = None,
) -> float:
    """Run GPU-accelerated pipeline from config.

    Args:
        config_path: Path to YAML configuration file
        input_override: Optional input file path to override config
        hydra_config: DictConfig from Hydra (takes precedence)

    Returns:
        Metric value for optimization (negative PA)
    """
    # Check GPU availability
    if not is_gpu_available():
        print("ERROR: GPU not available. norm_3 requires CUDA/cupy.")
        print("Run with: pixi run python -m norm_3.pipeline ...")
        return float("inf")

    if not getattr(run_pipeline, "_gpu_info_printed", False):
        print_gpu_info()
        print()
        run_pipeline._gpu_info_printed = True

    # Load config
    if hydra_config is not None:
        config = OmegaConf.to_container(hydra_config, resolve=True)
        config_path = None
    elif config_path is not None:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Either config_path or hydra_config must be provided")

    # Apply sweep parameter overrides
    config = apply_sweep_overrides(config)

    # Check for redundant configurations (skip to save time)
    if config.get("skip_redundant_configs", True):
        is_redundant, reason = is_redundant_config(config)
        if is_redundant:
            print(f"Skipping redundant config: {reason}")
            return float("inf")

    # Reset TVN and Spherize state
    reset_tvn_state()
    reset_spherize_state()
    reset_spherize_truncation_state()

    # Input path
    if input_override:
        input_path = Path(input_override)
    elif config.get("input_override"):
        input_path = Path(config["input_override"])
    else:
        input_path = Path(config["input"]["path"])

    # Output directory
    output_name = generate_output_name(config)
    base_output_path = Path(config["output"]["path"])
    input_name = input_path.stem
    output_dir = base_output_path.parent / input_name / output_name
    output_path = output_dir / base_output_path.name
    metrics_path = output_dir / "results" / "metrics.json"

    # Skip if already completed
    if config.get("skip_existing", True) and metrics_path.exists():
        print(f"[SKIP] Already completed: {output_name}")
        return 0.0

    output_dir.mkdir(exist_ok=True, parents=True)

    # Save config
    config_copy_path = output_dir / "pipeline_config.yaml"
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Pipeline: {output_name}")
    print(f"Output directory: {output_dir}")
    print()

    # === Intermediate caching for sweep speedup ===
    # When sweeping many configs, early pipeline steps are identical.
    # Cache at two boundaries to avoid redundant computation:
    #   early: after merge_metadata (load + NaN + variance + merge)
    #   mid:   after prune_correlated (+ normalize + outlier + INT + corr prune)
    _use_cache = config.get("use_pipeline_cache", True)
    _cache_resume_after = None
    _early_cache_path = None
    _mid_cache_path = None

    if _use_cache:
        import hashlib

        _cache_dir = base_output_path.parent / input_name / ".pipeline_cache"
        _cache_dir.mkdir(exist_ok=True, parents=True)

        # Input identity: path + modification time (detects regenerated files)
        _input_mtime = int(input_path.stat().st_mtime) if input_path.exists() else 0
        _input_id = hashlib.md5(f"{input_path}|{_input_mtime}".encode()).hexdigest()[:12]

        # Early cache: after merge_metadata (identical for all configs)
        _early_cache_path = _cache_dir / f"after_merge_{_input_id}.parquet"

        # Mid cache: after prune_correlated (varies by normalization variant)
        _nm = config.get("norm_method", "robustmad")
        _nf = config.get("norm_fit_controls", True)
        _ui = config.get("use_int", False)
        _oc = config.get("outlier_cutoff", 500)
        _ct = config.get("corr_thresh", 0.9)
        _mid_tag = f"{_nm}_{'ctrl' if _nf else 'all'}"
        if _oc is not None:
            _mid_tag += f"_out{_oc}"
        if _ui:
            _mid_tag += "_INT"
        if _ct is not None:
            _mid_tag += f"_corr{_ct}"
        _mid_cache_path = _cache_dir / f"after_prune_{_input_id}_{_mid_tag}.parquet"

        # Try to load from cache (prefer mid > early)
        # Validate on load — delete corrupt cache files
        if _mid_cache_path.exists():
            try:
                df = load_profiles(_mid_cache_path)
                print(f"[CACHE HIT] {_mid_cache_path.name} ({df.shape[0]}×{df.shape[1]})")
                _cache_resume_after = "prune_correlated"
            except Exception as e:
                print(f"[CACHE] Corrupt mid cache, deleting: {e}")
                _mid_cache_path.unlink(missing_ok=True)
        if _cache_resume_after is None and _early_cache_path.exists():
            try:
                df = load_profiles(_early_cache_path)
                print(f"[CACHE HIT] {_early_cache_path.name} ({df.shape[0]}×{df.shape[1]})")
                _cache_resume_after = "merge_metadata"
            except Exception as e:
                print(f"[CACHE] Corrupt early cache, deleting: {e}")
                _early_cache_path.unlink(missing_ok=True)

    if _cache_resume_after is None:
        # No cache — load from scratch
        print(f"Loading: {input_path}")
        df = load_profiles(input_path)
        print(f"Initial shape: {df.shape}")

    # Run steps
    _past_resume_point = (_cache_resume_after is None)
    for step_config in config.get("steps", []):
        step_name = step_config.get("name", "unknown")

        # Skip steps already covered by intermediate cache
        if not _past_resume_point:
            if step_name == _cache_resume_after:
                _past_resume_point = True
            if step_config.get("enabled", True):
                print(f"  [CACHE] Skipping: {step_name}")
            continue

        if not step_config.get("enabled", True):
            print(f"\nSkipping: {step_name}")
            continue

        step_func = STEPS.get(step_name)
        if step_func is None:
            print(f"WARNING: Unknown step '{step_name}'")
            continue

        # Update evaluate_metrics output_dir
        if step_name == "evaluate_metrics":
            if "params" not in step_config:
                step_config["params"] = {}
            step_config["params"]["output_dir"] = output_dir / "results"

        # Validate before aggregation
        if step_name == "aggregate_wells":
            max_pc1_var = config.get("max_pc1_variance", 0.50)
            is_valid, _ = validate_pca_variance(
                df, max_pc1_var, abort_on_error=config.get("abort_on_invalid_features", True)
            )
            if not is_valid:
                return float("inf")

        df = step_func(df, step_config.get("params", {}))

        # Check for empty dataframe
        if len(df) == 0:
            print(f"\nERROR: No rows remaining after {step_name}")
            return float("inf")

        # Check for ill-conditioned TVN
        if step_name in ("normalize_tvn", "normalize_tvn_efaar"):
            tvn_ill_conditioned, tvn_condition_number = get_tvn_state()
            if tvn_ill_conditioned and config.get("abort_on_ill_conditioned_tvn", True):
                print(f"\nABORTING: TVN ill-conditioned (condition number: {tvn_condition_number:.2e})")
                return float("inf")

        # Check for ill-conditioned Spherize (warn but don't abort by default)
        if step_name == "normalize_spherize":
            spherize_ill_conditioned, spherize_condition_number = get_spherize_state()
            if spherize_ill_conditioned:
                print(f"\nWARNING: Spherize ill-conditioned (condition number: {spherize_condition_number:.2e})")

        # Validate features
        if step_name != "evaluate_metrics":
            is_valid, _ = validate_features(
                df, step_name, abort_on_error=config.get("abort_on_invalid_features", True)
            )
            if not is_valid:
                return float("inf")

        # Save intermediate cache after key steps (atomic write, per-PID temp file)
        if _use_cache:
            _cache_target = None
            if step_name == "merge_metadata" and _early_cache_path and not _early_cache_path.exists():
                _cache_target = _early_cache_path
            elif step_name == "prune_correlated" and _mid_cache_path and not _mid_cache_path.exists():
                _cache_target = _mid_cache_path
            if _cache_target is not None:
                try:
                    import os
                    _tmp = _cache_target.with_suffix(f".{os.getpid()}.tmp")
                    save_profiles(df, _tmp)
                    _tmp.rename(_cache_target)
                    print(f"  [CACHE] Saved: {_cache_target.name} ({df.shape[0]}×{df.shape[1]})")
                except Exception as e:
                    # Clean up orphaned temp file
                    try:
                        _tmp.unlink(missing_ok=True)
                    except Exception:
                        pass
                    print(f"  [CACHE] Warning: {e}")

    # Save output
    print(f"\nSaving to: {output_path}")
    save_profiles(df, output_path, compression=config["output"].get("compression", "zstd"))
    print(f"\nPipeline complete! All outputs in: {output_dir}")

    # Return metric for optimization
    metric_value = 0.0
    try:
        import json

        metrics_path = output_dir / "results" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            pa = metrics.get("PA", 0.0)
            pc = metrics.get("PC", 0.0)
            balance = pa * pc
            print(f"  Optimization metric: Balance = {pa:.2f} x {pc:.2f} = {balance:.2f}")
            metric_value = -balance
    except Exception:
        pass

    # Clean up GPU memory to prevent accumulation across parallel runs
    try:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass

    return metric_value


# =============================================================================
# Hydra Entry Point
# =============================================================================

try:
    import hydra

    @hydra.main(version_base=None, config_path="conf", config_name="pipeline")
    def main(cfg: DictConfig) -> float:
        """Hydra entry point for running pipeline."""
        import time

        # Check for redundant config BEFORE any heavy operations (no GPU init, no data loading)
        config = OmegaConf.to_container(cfg, resolve=True)
        config = apply_sweep_overrides(config)
        if config.get("skip_redundant_configs", True):
            is_redundant, reason = is_redundant_config(config)
            if is_redundant:
                # Fast skip - no imports, no GPU, no data
                print(f"[SKIP] {reason}")
                return float("inf")

        # Check skip_existing BEFORE heavy imports (GPU, cupy, data loading)
        if config.get("skip_existing", True):
            input_override = cfg.get("input_override", None)
            if input_override:
                input_path = Path(input_override)
            elif config.get("input_override"):
                input_path = Path(config["input_override"])
            else:
                input_path = Path(config["input"]["path"])
            output_name = generate_output_name(config)
            base_output_path = Path(config["output"]["path"])
            metrics_path = base_output_path.parent / input_path.stem / output_name / "results" / "metrics.json"
            if metrics_path.exists():
                print(f"[SKIP] Already completed: {output_name}")
                return 0.0

        print(f"[{time.strftime('%H:%M:%S')}] norm_3 GPU Pipeline starting...")
        input_override = cfg.get("input_override", None)
        return run_pipeline(hydra_config=cfg, input_override=input_override)

except ImportError:
    def main():
        """Fallback when Hydra is not installed."""
        print("Hydra not installed. Use run_pipeline() directly.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".yaml") and Path(sys.argv[1]).exists():
        # Legacy mode
        print("Running in legacy mode (non-Hydra)")
        config_path = sys.argv[1]
        input_override = sys.argv[2] if len(sys.argv) > 2 else None
        run_pipeline(config_path=config_path, input_override=input_override)
    else:
        # Hydra mode
        import time
        print(f"Running norm_3 in Hydra mode (starting at {time.strftime('%H:%M:%S')})")
        main()
