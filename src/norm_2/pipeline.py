#!/usr/bin/env python3
"""Pipeline orchestration for profile processing.

This module provides:
- Step functions for each pipeline stage
- Validation functions
- Hydra integration for configuration
- run_pipeline() main entry point
"""

from __future__ import annotations

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Suppress JAX GPU/TPU warnings

import shutil
import sys
from pathlib import Path
from typing import Any

# Add src to path so imports work when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl
import yaml
from omegaconf import DictConfig, OmegaConf

from norm_2.core import (
    aggregate,
    correlation_threshold,
    drop_na_columns,
    drop_outliers,
    get_tvn_state,
    normalize,
    reset_tvn_state,
    sample_normalize as core_sample_normalize,
    variance_threshold,
)
from norm_2.io import get_numeric_features, infer_columns, load_metadata, load_profiles, save_profiles
from norm_2.metrics import evaluate_all


# =============================================================================
# Validation Functions
# =============================================================================


def validate_pca_variance(
    df: pl.DataFrame,
    max_pc1_variance: float = 0.40,
    abort_on_error: bool = True,
) -> tuple[bool, str]:
    """
    Validate that PC1 doesn't explain too much variance.

    If PC1 explains > max_pc1_variance, data has likely collapsed.
    """
    from sklearn.decomposition import PCA

    features, _ = infer_columns(df, ["Metadata_"])
    numeric_features = get_numeric_features(df, features)

    if len(numeric_features) < 2:
        return True, ""

    X = df.select(numeric_features).to_numpy()

    if np.isnan(X).any() or np.isinf(X).any():
        return True, ""

    try:
        pca = PCA(n_components=1)
        pca.fit(X)
        pc1_variance = pca.explained_variance_ratio_[0]
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
        total_values = X.size
        error_msg = f"Step '{step_name}': {nan_count} NaN, {inf_count} Inf in {len(numeric_features)} features"

        if abort_on_error:
            print(f"\nERROR: {error_msg}")
            return False, error_msg
        else:
            print(f"\nWARNING: {error_msg}")
            return True, error_msg

    return True, ""


# =============================================================================
# Step Functions
# =============================================================================


def clean_nans(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Remove NaN/inf columns and rows."""
    print("\n=== Step: Clean NaNs/Infs ===")
    features, metadata = infer_columns(df, ["Metadata_"])
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
    """Merge with JUMP metadata."""
    print("\n=== Step: Merge Metadata ===")

    metadata_dir = Path(config.get("metadata_dir", "analysis/feature_similarity/input"))
    metadata = load_metadata(metadata_dir)

    non_metadata_cols = [c for c in metadata.columns if not c.startswith("Metadata_")]
    if non_metadata_cols:
        metadata = metadata.rename({col: f"Metadata_{col}" for col in non_metadata_cols})

    df = df.join(metadata, on=["Metadata_Well"], how="left")

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


def filter_features(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Filter low variance and outlier features."""
    print("\n=== Step: Filter Features ===")
    features, metadata = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    ops = config.get(
        "operations",
        [
            {"name": "variance_threshold", "freq_cut": 0.05, "unique_cut": 0.01},
            {"name": "drop_outliers", "outlier_cutoff": 500},
        ],
    )

    current_features = features.copy()
    for op in ops:
        name = op["name"]
        params = {k: v for k, v in op.items() if k != "name"}
        n_before = len(current_features)

        if name == "variance_threshold":
            current_features = variance_threshold(df, current_features, **params)
        elif name == "drop_outliers":
            current_features = drop_outliers(df, current_features, **params)

        n_dropped = n_before - len(current_features)
        print(f"  {name}: dropped {n_dropped} features ({len(current_features)} remaining)")

        if len(current_features) == 0 and n_before > 0:
            print(f"  WARNING: {name} would remove all features, skipping")
            current_features = features.copy()

    df = df.select(metadata + current_features)
    print(f"  Result: {df.shape}")
    return df


def aggregate_wells(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Aggregate by well-plate."""
    print("\n=== Step: Aggregate Wells ===")
    features, metadata = infer_columns(df, ["Metadata_"])
    strata = config.get("strata", ["Metadata_Plate", "Metadata_Well"])
    method = config.get("method", "median")

    df = aggregate(df, features=features, group_by=strata, method=method)
    print(f"  Result: {df.shape}")
    return df


def normalize_standard(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Normalize using standardization."""
    print("\n=== Step: Normalize (Standard) ===")
    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    batch_col = config.get("batch_col", "Metadata_Plate")
    if batch_col == "":
        batch_col = None

    method = config.get("method", "standardize")
    print(f"  method: {method}, batch_col: {batch_col}, fit_on_controls: {config.get('fit_on_controls', False)}")

    df = normalize(
        df,
        features=features,
        method=method,
        batch_col=batch_col,
        fit_on_controls=config.get("fit_on_controls", False),
        control_key=config.get("control_key", "negcon"),
        control_col=config.get("control_col", "Metadata_control_type"),
    )
    print(f"  Result: {df.shape}")
    return df


def normalize_tvn(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Normalize using TVN."""
    print("\n=== Step: Normalize (TVN) ===")
    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    batch_col = config.get("batch_col", "Metadata_Plate")
    if batch_col == "":
        batch_col = None

    fit_on_controls = config.get("fit_on_controls", True)
    print(f"  batch_col: {batch_col}, fit_on_controls: {fit_on_controls}")

    df = normalize(
        df,
        features=features,
        method="tvn",
        batch_col=batch_col,
        fit_on_controls=fit_on_controls,
        control_key=config.get("control_key", "negcon"),
        control_col=config.get("control_col", "Metadata_control_type"),
        tvn_alpha=config.get("alpha", 0.5),
        tvn_epsilon=config.get("epsilon", 1.0),
    )
    print(f"  Result: {df.shape}")
    return df


def normalize_spherize(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Normalize using Spherize."""
    print("\n=== Step: Normalize (Spherize) ===")
    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    batch_col = config.get("batch_col", "Metadata_Plate")
    if batch_col == "":
        batch_col = None

    fit_on_controls = config.get("fit_on_controls", False)
    method = config.get("method", "ZCA-cor")
    print(f"  method: {method}, batch_col: {batch_col}, fit_on_controls: {fit_on_controls}")

    df = normalize(
        df,
        features=features,
        method="spherize",
        batch_col=batch_col,
        fit_on_controls=fit_on_controls,
        control_key=config.get("control_key", "negcon"),
        control_col=config.get("control_col", "Metadata_control_type"),
        spherize_method=method,
        spherize_epsilon=config.get("epsilon", 1e-6),
    )
    print(f"  Result: {df.shape}")
    return df


def normalize_pca(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Reduce dimensionality using PCA before whitening operations (EFAAR-style)."""
    print("\n=== Step: Normalize (PCA) ===")
    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    n_components = config.get("n_components", 128)
    whiten = config.get("whiten", False)
    fit_on_controls = config.get("fit_on_controls", True)
    control_col = config.get("control_col", "Metadata_control_type")
    control_key = config.get("control_key", "negcon")

    print(f"  n_components: {n_components}, whiten: {whiten}, fit_on_controls: {fit_on_controls}")
    print(f"  Input features: {len(features)}")

    X = df.select(features).to_numpy()

    # Fit on controls or all samples
    if fit_on_controls:
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
    from norm_2.core import PCATransform

    pca = PCATransform(n_components=n_components, whiten=whiten)
    pca.fit(X_fit)

    # Transform all samples
    X_transformed = pca.transform(X)

    # Create new feature names
    n_out = X_transformed.shape[1]
    new_features = [f"PC_{i+1}" for i in range(n_out)]
    explained_var = sum(pca.pca_.explained_variance_ratio_) * 100
    print(f"  Output features: {n_out} (explained variance: {explained_var:.1f}%)")

    # Build output DataFrame: keep metadata, replace features with PCs
    metadata_cols = [c for c in df.columns if c not in features]
    result = df.select(metadata_cols)

    # Add PC columns
    for i, feat_name in enumerate(new_features):
        result = result.with_columns(pl.Series(name=feat_name, values=X_transformed[:, i]))

    print(f"  Result: {result.shape}")
    return result


def _skip_harmony(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Stub for Harmony - skipped in norm_2."""
    print("\n=== Step: Normalize (Harmony) - SKIPPED ===")
    print("  Harmony has been removed from norm_2. Use norm/ if needed.")
    return df


def _skip_combat(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Stub for ComBat - skipped in norm_2."""
    print("\n=== Step: Normalize (ComBat) - SKIPPED ===")
    print("  ComBat has been removed from norm_2. Use norm/ if needed.")
    return df


def prune_correlated(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Prune correlated features."""
    print("\n=== Step: Prune Correlated Features ===")
    features, metadata = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    threshold = config.get("threshold", 0.9)
    method = config.get("method", "pearson")
    print(f"  threshold: {threshold}, method: {method}")

    new_features = correlation_threshold(df, features, threshold=threshold, method=method)
    n_dropped = len(features) - len(new_features)
    print(f"  Dropped {n_dropped} correlated features ({len(new_features)} remaining)")

    df = df.select(metadata + new_features)
    print(f"  Result: {df.shape}")
    return df


def sample_norm(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Apply L1/L2 normalization per sample."""
    print("\n=== Step: Sample Normalize ===")
    features, _ = infer_columns(df, ["Metadata_"])
    features = get_numeric_features(df, features)

    norm = config.get("norm", "l2")
    print(f"  Applying {norm} normalization to {len(features)} features")

    df = core_sample_normalize(df, features=features, norm=norm)
    print(f"  Result: {df.shape}")
    return df


def evaluate_metrics(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Evaluate metrics."""
    print("\n=== Step: Evaluate Metrics ===")

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
    "sample_norm": sample_norm,
    "normalize_standard": normalize_standard,
    "normalize_pca": normalize_pca,
    "normalize_tvn": normalize_tvn,
    "normalize_spherize": normalize_spherize,
    "normalize_harmony": _skip_harmony,
    "normalize_combat": _skip_combat,
    "prune_correlated": prune_correlated,
    "evaluate_metrics": evaluate_metrics,
}


# =============================================================================
# Redundant Configuration Detection
# =============================================================================

# Default values for method-specific parameters
# Used to detect redundant sweep configurations
# NOTE: These must match the FIRST value in each sweep parameter list
# to avoid skipping all configs. If sweep has tvn_alpha: "0.3,0.5",
# set default to 0.3 here.
METHOD_PARAM_DEFAULTS = {
    "tvn": {
        "tvn_fit_controls": True,  # Match sweep's first value
        "tvn_epsilon": 0.5,  # Match sweep's first value
    },
    "spherize": {
        "spherize_method": "ZCA-cor",
        "spherize_epsilon": 1e-6,  # Match sweep's first value
    },
    "pca": {
        # pca_n_components and pca_fit_controls not swept, no redundancy check needed
    },
}


def is_redundant_config(config: dict) -> tuple[bool, str | None]:
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
    else:
        use_tvn = config.get("use_tvn", False)

    # Spherize and PCA are independent - get from config directly
    use_spherize = config.get("use_spherize", False)
    use_pca = config.get("use_pca", False)

    # For each disabled method, check if params differ from defaults
    checks = [
        ("tvn", use_tvn, METHOD_PARAM_DEFAULTS["tvn"]),
        ("spherize", use_spherize, METHOD_PARAM_DEFAULTS["spherize"]),
        ("pca", use_pca, METHOD_PARAM_DEFAULTS["pca"]),
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

        if step_name == "sample_norm":
            norm = params.get("norm", "l2")
            parts.append(f"snorm_{norm}")

        elif step_name == "normalize_standard":
            method = params.get("method", "standardize")
            batch = "platewise" if params.get("batch_col") else "global"
            name_parts = [method if method != "standardize" else "std"]
            if batch != "platewise":
                name_parts.append(batch)
            if not params.get("fit_on_controls", True):
                name_parts.append("all")
            parts.append("_".join(name_parts))

        elif step_name == "normalize_pca":
            n_comp = params.get("n_components", 128)
            parts.append(f"pca{n_comp}")

        elif step_name == "normalize_tvn":
            alpha = params.get("alpha", 0.5)
            name_parts = ["tvn"]
            if alpha != 0.5:
                name_parts.append(f"a{alpha}")
            if not params.get("fit_on_controls", True):
                name_parts.append("all")
            parts.append("_".join(name_parts))

        elif step_name == "normalize_spherize":
            method = params.get("method", "ZCA-cor")
            fit_ctrl = params.get("fit_on_controls", False)
            name_parts = [method]
            if fit_ctrl:
                name_parts.append("negcon")
            parts.append("_".join(name_parts))

        elif step_name == "prune_correlated":
            threshold = params.get("threshold", 0.9)
            name_parts = ["prune"]
            if threshold != 0.9:
                name_parts.append(f"{threshold}")
            parts.append("_".join(name_parts))

        elif step_name == "aggregate_wells":
            method = params.get("method", "median")
            parts.append(f"agg_{method}" if method != "median" else "agg")

    return "__".join(parts) if parts else "default"


# =============================================================================
# Main Pipeline
# =============================================================================


def run_pipeline(
    config_path: str | Path | None = None,
    input_override: str | None = None,
    hydra_config: DictConfig | None = None,
) -> float:
    """
    Run pipeline from config.

    Args:
        config_path: Path to YAML configuration file
        input_override: Optional input file path to override config
        hydra_config: DictConfig from Hydra (takes precedence)

    Returns:
        Metric value for optimization (negative PA)
    """
    # Load config
    if hydra_config is not None:
        config = OmegaConf.to_container(hydra_config, resolve=True)
        config_path = None
    elif config_path is not None:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Either config_path or hydra_config must be provided")

    # Reset TVN ill-conditioning state at start of each pipeline run
    from norm_2.core import reset_tvn_state
    reset_tvn_state()

    # Handle batch_method parameter
    batch_method = config.get("batch_method")
    if batch_method is not None:
        config["use_tvn"] = batch_method == "tvn"
        config["use_harmony"] = batch_method == "harmony"
        config["use_combat"] = batch_method == "combat"

        for step in config.get("steps", []):
            step_name = step.get("name", "")
            if step_name == "normalize_tvn":
                step["enabled"] = batch_method == "tvn"
            elif step_name == "normalize_harmony":
                step["enabled"] = batch_method == "harmony"
            elif step_name == "normalize_combat":
                step["enabled"] = batch_method == "combat"
            elif step_name == "normalize_spherize":
                step["enabled"] = config.get("use_spherize", False)
            elif step_name == "normalize_pca":
                step["enabled"] = config.get("use_pca", False)
    else:
        # batch_method not set - override steps if flags are explicitly set
        for step in config.get("steps", []):
            step_name = step.get("name", "")
            if step_name == "normalize_spherize" and "use_spherize" in config:
                step["enabled"] = config.get("use_spherize", False)
            elif step_name == "normalize_pca" and "use_pca" in config:
                step["enabled"] = config.get("use_pca", False)
            elif step_name == "normalize_tvn" and "use_tvn" in config:
                step["enabled"] = config.get("use_tvn", False)

    # Check for redundant configuration (skip early to save time)
    skip_redundant = config.get("skip_redundant_configs", False)
    if skip_redundant:
        is_redundant, reason = is_redundant_config(config)
        if is_redundant:
            print(f"SKIPPING redundant config: {reason}")
            return float("inf")  # Return sentinel for Optuna compatibility

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

    # Handle Hydra sweep
    sweep_name = None
    if hydra_config is not None:
        try:
            from hydra.core.hydra_config import HydraConfig

            hydra_cfg = HydraConfig.get()
            job_num = hydra_cfg.job.num
            output_name = f"{job_num:04d}__{output_name}"

            for override in hydra_cfg.overrides.task:
                if override.startswith("+sweep=") or override.startswith("sweep="):
                    sweep_name = override.split("=")[1]
                    break
        except Exception:
            pass

    if sweep_name:
        output_dir = base_output_path.parent / sweep_name / input_name / output_name
    else:
        output_dir = base_output_path.parent / input_name / output_name
    output_dir.mkdir(exist_ok=True, parents=True)

    output_path = output_dir / base_output_path.name

    # Save config
    config_copy_path = output_dir / "pipeline_config.yaml"
    if config_path is not None:
        shutil.copy(config_path, config_copy_path)
    else:
        with open(config_copy_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    print(f"Pipeline: {output_name}")
    print(f"Output directory: {output_dir}")
    print(f"Config saved to: {config_copy_path}\n")

    # Load input
    print(f"Loading: {input_path}")
    df = load_profiles(input_path)
    print(f"Initial shape: {df.shape}")

    # Run steps
    for step_config in config.get("steps", []):
        step_name = step_config.get("name", "unknown")

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
        if step_name == "normalize_tvn":
            tvn_ill_conditioned, tvn_condition_number = get_tvn_state()
            if tvn_ill_conditioned and config.get("abort_on_ill_conditioned_tvn", True):
                print(f"\nABORTING: TVN ill-conditioned (condition number: {tvn_condition_number:.2e})")
                return float("inf")

        # Validate features
        if step_name != "evaluate_metrics":
            is_valid, _ = validate_features(
                df, step_name, abort_on_error=config.get("abort_on_invalid_features", True)
            )
            if not is_valid:
                return float("inf")

    # Save output
    print(f"\nSaving to: {output_path}")
    save_profiles(df, output_path, compression=config["output"].get("compression", "zstd"))
    print(f"\nPipeline complete! All outputs in: {output_dir}")

    # Return metric for optimization
    try:
        import json

        metrics_path = output_dir / "results" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            return -metrics.get("PA", 0.0)
    except Exception:
        pass

    return 0.0


# =============================================================================
# Hydra Entry Point
# =============================================================================

try:
    import hydra

    @hydra.main(version_base=None, config_path="conf", config_name="pipeline")
    def main(cfg: DictConfig) -> float:
        """Hydra entry point for running pipeline with sweeps."""
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
        print("Running in Hydra mode")
        main()
