#!/usr/bin/env python3
"""Test cp_measure preset with both pipelines."""
import sys
sys.path.insert(0, 'src')
import yaml
import json
from pathlib import Path

print("=" * 60)
print("Testing cp_measure preset")
print("=" * 60)

# Test OLD pipeline
print("\n[OLD PIPELINE]")
from norm.run_pipeline import STEPS as OLD_STEPS
from norm.data.load import load_profiles, save_profiles

config_path = Path("src/norm/conf/preset/cp_measure.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

input_path = "output/cp_measure_jump_target2_4plate_zstd_raw_features.parquet"
output_dir = Path("test_comparison/old/cp_measure")
output_dir.mkdir(parents=True, exist_ok=True)

df = load_profiles(input_path)
print(f"  Loaded: {df.shape}")

for step_config in config.get("steps", []):
    step_name = step_config.get("name", "unknown")
    if not step_config.get("enabled", True):
        continue

    step_func = OLD_STEPS.get(step_name)
    if step_func is None:
        continue

    if step_name == "evaluate_metrics":
        step_config["params"] = step_config.get("params", {})
        step_config["params"]["skip_visualization"] = True
        step_config["params"]["output_dir"] = str(output_dir / "results")
        (output_dir / "results").mkdir(exist_ok=True)

    print(f"  Running: {step_name}")
    df = step_func(df, step_config.get("params", {}))

save_profiles(df, str(output_dir / "processed.parquet"), compression="zstd")
old_shape = df.shape
print(f"  Final shape: {old_shape}")

# Load metrics
metrics_path = output_dir / "results" / "metrics.json"
old_pa = None
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
    old_pa = metrics.get('PA')
    print(f"  PA: {old_pa}")

print("\n[NEW PIPELINE]")
from norm_2.pipeline import STEPS as NEW_STEPS
from norm_2.io import load_profiles as load_profiles_new, save_profiles as save_profiles_new

config_path = Path("src/norm_2/conf/preset/cp_measure.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

output_dir = Path("test_comparison/new/cp_measure")
output_dir.mkdir(parents=True, exist_ok=True)

df = load_profiles_new(input_path)
print(f"  Loaded: {df.shape}")

for step_config in config.get("steps", []):
    step_name = step_config.get("name", "unknown")
    if not step_config.get("enabled", True):
        continue

    step_func = NEW_STEPS.get(step_name)
    if step_func is None:
        continue

    if step_name == "evaluate_metrics":
        step_config["params"] = step_config.get("params", {})
        step_config["params"]["skip_visualization"] = True
        step_config["params"]["output_dir"] = str(output_dir / "results")
        (output_dir / "results").mkdir(exist_ok=True)

    print(f"  Running: {step_name}")
    df = step_func(df, step_config.get("params", {}))

save_profiles_new(df, str(output_dir / "processed.parquet"), compression="zstd")
new_shape = df.shape
print(f"  Final shape: {new_shape}")

# Load metrics
metrics_path = output_dir / "results" / "metrics.json"
new_pa = None
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
    new_pa = metrics.get('PA')
    print(f"  PA: {new_pa}")

# Compare
print("\n[COMPARISON]")
print(f"  Shapes match: {old_shape == new_shape}")
if old_pa and new_pa:
    pa_diff = abs(old_pa - new_pa)
    print(f"  PA difference: {pa_diff:.4f}")
    print(f"  PA match: {pa_diff < 0.01}")

print("\nDONE!")
