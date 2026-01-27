#!/usr/bin/env python3
"""Trace morphem pipeline step by step to find divergence."""
import sys
sys.path.insert(0, 'src')
import yaml
import numpy as np
import polars as pl
from pathlib import Path

def compare_dfs(df1, df2, name=""):
    """Compare two dataframes after sorting."""
    # Sort both
    df1 = df1.sort(["Metadata_Plate", "Metadata_Well"])
    df2 = df2.sort(["Metadata_Plate", "Metadata_Well"])

    features = [c for c in df1.columns if not c.startswith('Metadata_')]
    features2 = [c for c in df2.columns if not c.startswith('Metadata_')]

    if features != features2:
        print(f"  {name}: Feature columns differ! {len(features)} vs {len(features2)}")
        return

    max_diff = 0
    worst_feat = None
    for feat in features[:50]:  # Check first 50
        v1 = df1[feat].to_numpy()
        v2 = df2[feat].to_numpy()
        diff = np.max(np.abs(v1 - v2))
        if diff > max_diff:
            max_diff = diff
            worst_feat = feat

    print(f"  {name}: shapes={df1.shape}/{df2.shape}, max_diff={max_diff:.2e}")
    if max_diff > 1e-6:
        print(f"    *** DIVERGENCE at {worst_feat} ***")
        return True
    return False

print("=" * 60)
print("TRACING MORPHEM PIPELINE")
print("=" * 60)

# Load config
config_path = Path("src/norm/conf/preset/morphem.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

input_path = "output/morphem_jump_target2_4plate_zstd_raw_features.parquet"

# Initialize both pipelines
from norm.run_pipeline import STEPS as OLD_STEPS
from norm.data.load import load_profiles as old_load
from norm_2.pipeline import STEPS as NEW_STEPS
from norm_2.io import load_profiles as new_load

# Load data
df_old = old_load(input_path)
df_new = new_load(input_path)

print(f"\nInitial: old={df_old.shape}, new={df_new.shape}")
compare_dfs(df_old, df_new, "Initial")

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

    # Run old
    df_old = old_func(df_old, step_config.get("params", {}))

    # Run new
    df_new = new_func(df_new, step_config.get("params", {}))

    diverged = compare_dfs(df_old, df_new, f"After {step_name}")

    if diverged:
        print(f"\n  STOPPING: Divergence detected at {step_name}")
        # Show more details
        df_old_s = df_old.sort(["Metadata_Plate", "Metadata_Well"])
        df_new_s = df_new.sort(["Metadata_Plate", "Metadata_Well"])
        features = [c for c in df_old.columns if not c.startswith('Metadata_')][:3]
        for feat in features:
            old_vals = df_old_s[feat].to_numpy()[:10]
            new_vals = df_new_s[feat].to_numpy()[:10]
            print(f"\n  {feat}:")
            print(f"    Old[:10]: {old_vals}")
            print(f"    New[:10]: {new_vals}")
        break

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
