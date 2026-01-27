#!/usr/bin/env python3
"""Debug morphem pipeline to find divergence point."""
import sys
sys.path.insert(0, 'src')
import yaml
import numpy as np
import polars as pl
from pathlib import Path

def hash_df(df, features):
    """Hash feature values for quick comparison."""
    X = df.select(features).to_numpy()
    return np.mean(X), np.std(X), np.max(np.abs(X))

print("=" * 60)
print("DEBUGGING MORPHEM PIPELINE")
print("=" * 60)

# Load config
config_path = Path("src/norm/conf/preset/morphem.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

input_path = "output/morphem_jump_target2_4plate_zstd_raw_features.parquet"

# Initialize both pipelines
from norm.run_pipeline import STEPS as OLD_STEPS
from norm.data.load import load_profiles as old_load, infer_columns as old_infer
from norm_2.pipeline import STEPS as NEW_STEPS
from norm_2.io import load_profiles as new_load, infer_columns as new_infer, get_numeric_features

# Load data
df_old = old_load(input_path)
df_new = new_load(input_path)

print(f"\nInitial shape: old={df_old.shape}, new={df_new.shape}")

# Track both dataframes through each step
for step_config in config.get("steps", []):
    step_name = step_config.get("name", "unknown")
    if not step_config.get("enabled", True):
        continue

    old_func = OLD_STEPS.get(step_name)
    new_func = NEW_STEPS.get(step_name)

    if old_func is None or new_func is None:
        print(f"\n{step_name}: SKIP (not in both)")
        continue

    # Skip evaluate_metrics
    if step_name == "evaluate_metrics":
        continue

    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")

    # Run old
    df_old = old_func(df_old, step_config.get("params", {}))
    features_old, _ = old_infer(df_old, ["Metadata_"])
    features_old = [f for f in features_old if df_old[f].dtype in (pl.Float32, pl.Float64)]

    # Run new
    df_new = new_func(df_new, step_config.get("params", {}))
    features_new, _ = new_infer(df_new, ["Metadata_"])
    features_new = get_numeric_features(df_new, features_new)

    print(f"  Shapes: old={df_old.shape}, new={df_new.shape}")
    print(f"  Features: old={len(features_old)}, new={len(features_new)}")

    # Check feature overlap
    common = set(features_old) & set(features_new)
    print(f"  Common features: {len(common)}")

    if common:
        # Sort both by Plate+Well
        df_old_sorted = df_old.sort(["Metadata_Plate", "Metadata_Well"])
        df_new_sorted = df_new.sort(["Metadata_Plate", "Metadata_Well"])

        common_list = sorted(list(common))[:5]  # First 5

        # Compare values
        max_diff = 0
        for feat in common_list:
            old_vals = df_old_sorted[feat].to_numpy()
            new_vals = df_new_sorted[feat].to_numpy()
            diff = np.max(np.abs(old_vals - new_vals))
            if diff > max_diff:
                max_diff = diff

        print(f"  Max diff (first 5 features): {max_diff:.2e}")

        if max_diff > 1e-6:
            print("  *** DIVERGENCE DETECTED ***")
            # Show more details
            feat = common_list[0]
            old_vals = df_old_sorted[feat].to_numpy()
            new_vals = df_new_sorted[feat].to_numpy()
            print(f"    Feature: {feat}")
            print(f"    Old: mean={np.mean(old_vals):.6f}, std={np.std(old_vals):.6f}")
            print(f"    New: mean={np.mean(new_vals):.6f}, std={np.std(new_vals):.6f}")
            print(f"    First 5 old: {old_vals[:5]}")
            print(f"    First 5 new: {new_vals[:5]}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
