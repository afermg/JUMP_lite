#!/usr/bin/env python3
"""Test morphem preset with both pipelines after the fix."""
import sys
sys.path.insert(0, 'src')
import yaml
import json
import numpy as np
import polars as pl
from pathlib import Path

print("=" * 60)
print("Testing morphem preset (after batch ordering fix)")
print("=" * 60)

preset = "morphem"
input_path = Path("output/morphem_jump_target2_4plate_zstd_raw_features.parquet")
output_dir_old = Path("test_comparison/old/morphem")
output_dir_new = Path("test_comparison/new/morphem")

# Load config
config_path = Path(f"src/norm/conf/preset/{preset}.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Test OLD pipeline
print("\n[OLD PIPELINE]")
from norm.run_pipeline import STEPS as OLD_STEPS
from norm.data.load import load_profiles as old_load, save_profiles as old_save

output_dir_old.mkdir(parents=True, exist_ok=True)
results_dir_old = output_dir_old / "results"
results_dir_old.mkdir(exist_ok=True)

df = old_load(str(input_path))
print(f"  Loaded: {df.shape}")

for step_config in config.get("steps", []):
    step_name = step_config.get("name", "unknown")
    if not step_config.get("enabled", True):
        continue
    func = OLD_STEPS.get(step_name)
    if func is None:
        continue
    if step_name == "evaluate_metrics":
        step_config["params"] = step_config.get("params", {})
        step_config["params"]["skip_visualization"] = True
        step_config["params"]["output_dir"] = str(results_dir_old)
    df = func(df, step_config.get("params", {}))

old_save(df, str(output_dir_old / "processed.parquet"), compression="zstd")
print(f"  Final shape: {df.shape}")

# Test NEW pipeline
print("\n[NEW PIPELINE]")
from norm_2.pipeline import STEPS as NEW_STEPS
from norm_2.io import load_profiles as new_load, save_profiles as new_save

# Reload config (in case it was mutated)
with open(config_path) as f:
    config = yaml.safe_load(f)

output_dir_new.mkdir(parents=True, exist_ok=True)
results_dir_new = output_dir_new / "results"
results_dir_new.mkdir(exist_ok=True)

df = new_load(str(input_path))
print(f"  Loaded: {df.shape}")

for step_config in config.get("steps", []):
    step_name = step_config.get("name", "unknown")
    if not step_config.get("enabled", True):
        continue
    func = NEW_STEPS.get(step_name)
    if func is None:
        continue
    if step_name == "evaluate_metrics":
        step_config["params"] = step_config.get("params", {})
        step_config["params"]["skip_visualization"] = True
        step_config["params"]["output_dir"] = str(results_dir_new)
    df = func(df, step_config.get("params", {}))

new_save(df, str(output_dir_new / "processed.parquet"), compression="zstd")
print(f"  Final shape: {df.shape}")

# Compare outputs
print("\n[COMPARISON]")
old = pl.read_parquet(output_dir_old / "processed.parquet")
new = pl.read_parquet(output_dir_new / "processed.parquet")

# Sort both
old = old.sort(["Metadata_Plate", "Metadata_Well"])
new = new.sort(["Metadata_Plate", "Metadata_Well"])

features = [c for c in old.columns if not c.startswith("Metadata_")]
max_diff = 0
for feat in features:
    old_vals = old[feat].to_numpy()
    new_vals = new[feat].to_numpy()
    diff = np.max(np.abs(old_vals - new_vals))
    if diff > max_diff:
        max_diff = diff

if max_diff < 1e-6:
    print(f"  PASS: Max diff = {max_diff:.2e}")
else:
    print(f"  FAIL: Max diff = {max_diff:.2e}")
    # Show first failing feature
    for feat in features[:5]:
        old_vals = old[feat].to_numpy()
        new_vals = new[feat].to_numpy()
        diff = np.max(np.abs(old_vals - new_vals))
        if diff > 1e-6:
            print(f"    {feat}: diff={diff:.2e}")

print("\nDONE!")
