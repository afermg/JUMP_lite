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


def clean_nans(df, config):
    """Remove NaN/inf columns and rows."""
    print("\n=== Step: Clean NaNs/Infs ===")
    features, metadata = infer_columns(df, ["Metadata_"])
    na_cutoff = config.get("na_cutoff", 0.30)

    # Remove columns with >30% NaNs/infs
    from norm.operations.select import drop_na_columns
    features = drop_na_columns(df, features, na_cutoff=na_cutoff)
    print(f"  Kept {len(features)} features after NaN/inf filter")

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

    df = normalize_profiles_extended(
        df, numeric_features,
        method=config.get("method", "standardize"),
        batch_col=config.get("batch_col", "Metadata_Plate"),
        fit_on_controls=config.get("fit_on_controls", True),
        control_key=config.get("control_key", "negcon"),
        control_col=config.get("control_col", "Metadata_control_type"),
    )
    print(f"  Result: {df.shape}")
    return df


def normalize_tvn(df, config):
    """Normalize using TVN."""
    print("\n=== Step: Normalize (TVN) ===")
    features, metadata = infer_columns(df, ["Metadata_"])

    numeric_features = [
        f for f in features
        if df[f].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                           pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]

    df = normalize_profiles_extended(
        df, numeric_features,
        method="tvn",
        batch_col=config.get("batch_col", "Metadata_Plate"),
        tvn_alpha=config.get("alpha", 0.5),
        tvn_epsilon=config.get("epsilon", 1e-3),
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

    df = normalize_profiles_extended(
        df, numeric_features,
        method="spherize",
        batch_col=config.get("batch_col", "Metadata_Plate"),
        spherize_method=config.get("method", "ZCA-cor"),
        spherize_epsilon=config.get("epsilon", 1e-6),
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

    ops = [{"name": "correlation_threshold", "threshold": config.get("threshold", 0.9), "method": "pearson"}]
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
        results["PC"] = pc["n_targets_active"]
        results["n_targets_total"] = pc["n_targets_total"]
        print(f"  PC: {pc['n_targets_active']} active targets")

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

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create visualization
    print("  Creating PCA/UMAP visualization...")
    try:
        plot_path = output_dir / "dimreduction.png"
        plot_dimensionality_reduction_extended(
            df, numeric_features, full_results, plot_path,
            n_top_compounds=config.get("n_top_compounds", 20)
        )
        print(f"  Saved visualization to: {plot_path}")
    except Exception as e:
        print(f"  Visualization ERROR: {e}")

    return df


# Step registry
STEPS = {
    "clean_nans": clean_nans,
    "merge_metadata": merge_metadata,
    "filter_features": filter_features,
    "aggregate_wells": aggregate_wells,
    "normalize_standard": normalize_standard,
    "normalize_tvn": normalize_tvn,
    "normalize_spherize": normalize_spherize,
    "prune_correlated": prune_correlated,
    "evaluate_metrics": evaluate_metrics,
}


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
        if step_name == "normalize_standard":
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

        elif step_name == "normalize_tvn":
            alpha = params.get("alpha", 0.5)
            epsilon = params.get("epsilon", 1e-3)
            batch = "platewise" if params.get("batch_col") else "global"

            name_parts = ["tvn"]
            if alpha != 0.5:
                name_parts.append(f"a{alpha}")
            if epsilon != 1e-3:
                name_parts.append(f"e{epsilon:.0e}")
            if batch != "platewise":
                name_parts.append(batch)
            parts.append("_".join(name_parts))

        elif step_name == "normalize_spherize":
            method = params.get("method", "ZCA-cor")
            epsilon = params.get("epsilon", 1e-6)
            batch = "platewise" if params.get("batch_col") else "global"

            name_parts = [method]
            if epsilon != 1e-6:
                name_parts.append(f"e{epsilon:.0e}")
            if batch != "platewise":
                name_parts.append(batch)
            parts.append("_".join(name_parts))

        elif step_name == "prune_correlated":
            threshold = params.get("threshold", 0.9)

            if threshold != 0.9:
                parts.append(f"prune{threshold}")
            else:
                parts.append("prune")

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

    # Join parts with underscores
    if parts:
        return "_".join(parts)
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

    # Use input override if provided, otherwise use config
    if input_override:
        input_path = Path(input_override)
        print(f"Input override: {input_override}")
    else:
        input_path = Path(config["input"]["path"])

    # Generate output directory name from config
    output_name = generate_output_name(config)
    base_output_path = Path(config["output"]["path"])

    # Get input filename (stem without extension)
    input_name = input_path.stem

    # Create output directory: parent / input_name / pipeline_config_name
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

    # Load input (already set above with override if provided)
    print(f"Loading: {input_path}")
    df = load_profiles(input_path)
    print(f"Initial shape: {df.shape}")

    # Run steps
    for step_config in config.get("steps", []):
        if not step_config.get("enabled", True):
            print(f"\nSkipping: {step_config['name']}")
            continue

        step_name = step_config["name"]
        step_func = STEPS.get(step_name)

        if step_func is None:
            print(f"WARNING: Unknown step '{step_name}'")
            continue

        # Update evaluate_metrics output_dir to use the same directory
        if step_name == "evaluate_metrics":
            if "params" not in step_config:
                step_config["params"] = {}
            step_config["params"]["output_dir"] = output_dir / "results"

        df = step_func(df, step_config.get("params", {}))

    # Save output
    print(f"\nSaving to: {output_path}")
    save_profiles(df, output_path, compression=config["output"].get("compression", "zstd"))
    print(f"\nPipeline complete! All outputs in: {output_dir}")
    print("Done!")


@hydra.main(version_base=None, config_path="conf", config_name="pipeline")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for running pipeline with sweeps."""
    # Get input override from hydra config if provided
    input_override = cfg.get("input_override", None)

    # Run pipeline with Hydra config
    run_pipeline(hydra_config=cfg, input_override=input_override)


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
