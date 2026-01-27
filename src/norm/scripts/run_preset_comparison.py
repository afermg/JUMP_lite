#!/usr/bin/env python3
"""Run all presets through both norm and norm_2 pipelines and compare outputs.

Usage:
    nix develop . --command python scripts/run_preset_comparison.py
"""

import sys
import json
import shutil
from pathlib import Path

sys.path.insert(0, "src")

import yaml
import polars as pl

# Presets to test
PRESETS = [
    "cp_measure",
    "dinov2_490",
    "dinov2_random",
    "dinov2_tilesize_224",
    "morphem",
    "openphenom_8bit",
    "subcell",
]

# Base paths
BASE_DIR = Path("/home/jfredinh/projects/JUMP_core")
OUTPUT_DIR = BASE_DIR / "output"
TEST_DIR = BASE_DIR / "test_comparison"


def get_input_path(preset: str) -> Path:
    """Get input parquet path for a preset."""
    # Map preset name to input file pattern
    if preset == "dinov2_490":
        name = "dinov2_490"
    else:
        name = preset
    return OUTPUT_DIR / f"{name}_jump_target2_4plate_zstd_raw_features.parquet"


def run_old_pipeline(preset: str, input_path: Path, output_dir: Path) -> dict:
    """Run the old norm/ pipeline."""
    from norm.run_pipeline import STEPS
    from norm.data.load import load_profiles, save_profiles, infer_columns

    # Load config
    config_path = BASE_DIR / "src" / "norm" / "conf" / "preset" / f"{preset}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "processed.parquet"
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"  Loading: {input_path}")
    df = load_profiles(str(input_path))
    print(f"  Initial shape: {df.shape}")

    # Run steps
    for step_config in config.get("steps", []):
        step_name = step_config.get("name", "unknown")

        if not step_config.get("enabled", True):
            continue

        step_func = STEPS.get(step_name)
        if step_func is None:
            print(f"    WARNING: Unknown step '{step_name}'")
            continue

        # Skip visualization
        if step_name == "evaluate_metrics":
            if "params" not in step_config:
                step_config["params"] = {}
            step_config["params"]["skip_visualization"] = True
            step_config["params"]["output_dir"] = str(results_dir)

        print(f"    Running: {step_name}")
        df = step_func(df, step_config.get("params", {}))

        if len(df) == 0:
            print(f"    ERROR: No rows remaining after {step_name}")
            return {"error": f"No rows after {step_name}"}

    # Save
    print(f"  Saving to: {output_path}")
    save_profiles(df, str(output_path), compression="zstd")

    # Load metrics
    metrics = {}
    metrics_path = results_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return {"shape": df.shape, "metrics": metrics, "output_path": str(output_path)}


def run_new_pipeline(preset: str, input_path: Path, output_dir: Path) -> dict:
    """Run the new norm_2/ pipeline."""
    from norm_2.pipeline import STEPS
    from norm_2.io import load_profiles, save_profiles

    # Load config
    config_path = BASE_DIR / "src" / "norm_2" / "conf" / "preset" / f"{preset}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "processed.parquet"
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"  Loading: {input_path}")
    df = load_profiles(str(input_path))
    print(f"  Initial shape: {df.shape}")

    # Run steps
    for step_config in config.get("steps", []):
        step_name = step_config.get("name", "unknown")

        if not step_config.get("enabled", True):
            continue

        step_func = STEPS.get(step_name)
        if step_func is None:
            print(f"    WARNING: Unknown step '{step_name}'")
            continue

        # Skip visualization
        if step_name == "evaluate_metrics":
            if "params" not in step_config:
                step_config["params"] = {}
            step_config["params"]["skip_visualization"] = True
            step_config["params"]["output_dir"] = str(results_dir)

        print(f"    Running: {step_name}")
        df = step_func(df, step_config.get("params", {}))

        if len(df) == 0:
            print(f"    ERROR: No rows remaining after {step_name}")
            return {"error": f"No rows after {step_name}"}

    # Save
    print(f"  Saving to: {output_path}")
    save_profiles(df, str(output_path), compression="zstd")

    # Load metrics
    metrics = {}
    metrics_path = results_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return {"shape": df.shape, "metrics": metrics, "output_path": str(output_path)}


def compare_outputs(old_path: Path, new_path: Path, tolerance: float = 1e-6) -> dict:
    """Compare two parquet outputs."""
    import numpy as np

    old = pl.read_parquet(old_path)
    new = pl.read_parquet(new_path)

    result = {"match": True, "errors": []}

    # Check shape
    if old.shape != new.shape:
        result["match"] = False
        result["errors"].append(f"Shape mismatch: {old.shape} vs {new.shape}")
        return result

    # Check columns
    if old.columns != new.columns:
        result["match"] = False
        result["errors"].append(f"Column mismatch")
        return result

    # Sort both dataframes by Plate+Well to ensure consistent row order
    sort_cols = ["Metadata_Plate", "Metadata_Well"]
    if all(c in old.columns for c in sort_cols):
        old = old.sort(sort_cols)
        new = new.sort(sort_cols)

    # Check values
    feature_cols = [c for c in old.columns if not c.startswith("Metadata_")]
    max_diff = 0.0
    worst_col = None

    for col in feature_cols:
        if old[col].dtype in (pl.Float32, pl.Float64):
            old_vals = old[col].to_numpy()
            new_vals = new[col].to_numpy()

            # Handle NaN
            old_nan = np.isnan(old_vals)
            new_nan = np.isnan(new_vals)

            if not np.array_equal(old_nan, new_nan):
                result["errors"].append(f"NaN pattern differs in {col}")
                continue

            mask = ~old_nan
            if mask.any():
                diff = np.max(np.abs(old_vals[mask] - new_vals[mask]))
                if diff > max_diff:
                    max_diff = diff
                    worst_col = col

                if diff > tolerance:
                    result["match"] = False
                    result["errors"].append(f"{col}: diff={diff:.2e}")

    result["max_diff"] = max_diff
    result["worst_col"] = worst_col

    return result


def main():
    print("=" * 80)
    print("PRESET COMPARISON: norm/ vs norm_2/")
    print("=" * 80)

    # Clean up old test results
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True)

    results = {}

    for preset in PRESETS:
        print(f"\n{'=' * 60}")
        print(f"PRESET: {preset}")
        print("=" * 60)

        input_path = get_input_path(preset)
        if not input_path.exists():
            print(f"  SKIP: Input not found: {input_path}")
            results[preset] = {"status": "skipped", "reason": "input not found"}
            continue

        old_output_dir = TEST_DIR / "old" / preset
        new_output_dir = TEST_DIR / "new" / preset

        # Run old pipeline
        print(f"\n[OLD PIPELINE]")
        try:
            old_result = run_old_pipeline(preset, input_path, old_output_dir)
            print(f"  Shape: {old_result.get('shape')}")
            print(f"  Metrics: PA={old_result.get('metrics', {}).get('PA', 'N/A')}")
        except Exception as e:
            print(f"  ERROR: {e}")
            old_result = {"error": str(e)}

        # Run new pipeline
        print(f"\n[NEW PIPELINE]")
        try:
            new_result = run_new_pipeline(preset, input_path, new_output_dir)
            print(f"  Shape: {new_result.get('shape')}")
            print(f"  Metrics: PA={new_result.get('metrics', {}).get('PA', 'N/A')}")
        except Exception as e:
            print(f"  ERROR: {e}")
            new_result = {"error": str(e)}

        # Compare
        print(f"\n[COMPARISON]")
        if "error" in old_result or "error" in new_result:
            print(f"  SKIP: One pipeline failed")
            results[preset] = {"status": "error", "old": old_result, "new": new_result}
        else:
            old_path = Path(old_result["output_path"])
            new_path = Path(new_result["output_path"])
            comparison = compare_outputs(old_path, new_path)

            if comparison["match"]:
                print(f"  PASS: Outputs match (max diff: {comparison['max_diff']:.2e})")
                results[preset] = {"status": "pass", "max_diff": comparison["max_diff"]}
            else:
                print(f"  FAIL: Outputs differ")
                for err in comparison["errors"][:5]:
                    print(f"    - {err}")
                results[preset] = {"status": "fail", "errors": comparison["errors"]}

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results.values() if r.get("status") == "pass")
    failed = sum(1 for r in results.values() if r.get("status") == "fail")
    skipped = sum(1 for r in results.values() if r.get("status") in ("skipped", "error"))

    for preset, result in results.items():
        status = result.get("status", "unknown")
        if status == "pass":
            print(f"  {preset}: PASS (max diff: {result.get('max_diff', 0):.2e})")
        elif status == "fail":
            print(f"  {preset}: FAIL")
        else:
            print(f"  {preset}: {status.upper()}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    # Save results
    with open(TEST_DIR / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {TEST_DIR / 'comparison_results.json'}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
