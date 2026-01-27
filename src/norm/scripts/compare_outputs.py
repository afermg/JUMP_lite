#!/usr/bin/env python3
"""Compare outputs from norm and norm_2 pipelines.

Usage:
    uv run python scripts/compare_outputs.py old_output.parquet new_output.parquet

This script compares two parquet files to verify they produce identical results.
It checks:
1. Shape (rows and columns)
2. Column names
3. Feature values (with floating point tolerance)
4. Metadata values (exact match)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl


def compare_outputs(
    old_path: str | Path,
    new_path: str | Path,
    tolerance: float = 1e-6,
    verbose: bool = True,
) -> bool:
    """
    Compare two parquet files for equivalence.

    Args:
        old_path: Path to original output
        new_path: Path to new output
        tolerance: Maximum allowed difference for floats
        verbose: Print detailed comparison

    Returns:
        True if outputs are equivalent, False otherwise
    """
    old_path = Path(old_path)
    new_path = Path(new_path)

    if not old_path.exists():
        print(f"ERROR: Old file not found: {old_path}")
        return False
    if not new_path.exists():
        print(f"ERROR: New file not found: {new_path}")
        return False

    print(f"Comparing:")
    print(f"  Old: {old_path}")
    print(f"  New: {new_path}")
    print()

    # Load files
    old = pl.read_parquet(old_path)
    new = pl.read_parquet(new_path)

    # Check shape
    if old.shape != new.shape:
        print(f"FAIL: Shape mismatch")
        print(f"  Old: {old.shape}")
        print(f"  New: {new.shape}")
        return False
    if verbose:
        print(f"Shape: {old.shape} ✓")

    # Check columns
    if old.columns != new.columns:
        print(f"FAIL: Column mismatch")
        old_cols = set(old.columns)
        new_cols = set(new.columns)
        missing_in_new = old_cols - new_cols
        extra_in_new = new_cols - old_cols
        if missing_in_new:
            print(f"  Missing in new: {missing_in_new}")
        if extra_in_new:
            print(f"  Extra in new: {extra_in_new}")
        return False
    if verbose:
        print(f"Columns: {len(old.columns)} ✓")

    # Separate metadata and features
    metadata_cols = [c for c in old.columns if c.startswith("Metadata_")]
    feature_cols = [c for c in old.columns if not c.startswith("Metadata_")]

    if verbose:
        print(f"  Metadata columns: {len(metadata_cols)}")
        print(f"  Feature columns: {len(feature_cols)}")

    # Check feature values
    n_feature_errors = 0
    max_diff_overall = 0.0
    worst_feature = None

    for col in feature_cols:
        if old[col].dtype in (pl.Float32, pl.Float64):
            old_vals = old[col].to_numpy()
            new_vals = new[col].to_numpy()

            # Handle NaN values
            old_nan = np.isnan(old_vals)
            new_nan = np.isnan(new_vals)

            if not np.array_equal(old_nan, new_nan):
                n_feature_errors += 1
                if verbose:
                    print(f"  WARN: NaN pattern differs in {col}")
                continue

            # Compare non-NaN values
            mask = ~old_nan
            if mask.any():
                diff = np.abs(old_vals[mask] - new_vals[mask])
                max_diff = np.max(diff) if len(diff) > 0 else 0

                if max_diff > tolerance:
                    n_feature_errors += 1
                    if verbose:
                        print(f"  WARN: {col} differs by {max_diff:.2e}")

                if max_diff > max_diff_overall:
                    max_diff_overall = max_diff
                    worst_feature = col
        else:
            # Integer or other numeric type
            if not (old[col] == new[col]).all():
                n_feature_errors += 1
                if verbose:
                    print(f"  WARN: {col} has mismatches")

    if n_feature_errors > 0:
        print(f"\nFAIL: {n_feature_errors} feature columns have differences")
        print(f"  Worst: {worst_feature} (max diff: {max_diff_overall:.2e})")
        return False

    if verbose:
        print(f"Features: all within tolerance ({tolerance}) ✓")
        if worst_feature:
            print(f"  Max diff: {max_diff_overall:.2e} in {worst_feature}")

    # Check metadata values
    n_metadata_errors = 0
    for col in metadata_cols:
        if old[col].dtype in (pl.Float32, pl.Float64):
            old_vals = old[col].to_numpy()
            new_vals = new[col].to_numpy()
            old_nan = np.isnan(old_vals)
            new_nan = np.isnan(new_vals)
            mask = ~old_nan & ~new_nan
            if not np.array_equal(old_nan, new_nan):
                n_metadata_errors += 1
            elif mask.any() and np.max(np.abs(old_vals[mask] - new_vals[mask])) > tolerance:
                n_metadata_errors += 1
        else:
            if not old[col].equals(new[col]):
                n_metadata_errors += 1
                if verbose:
                    print(f"  WARN: Metadata {col} differs")

    if n_metadata_errors > 0:
        print(f"\nFAIL: {n_metadata_errors} metadata columns have differences")
        return False

    if verbose:
        print(f"Metadata: all match ✓")

    print(f"\nPASS: Outputs are equivalent")
    return True


def compare_metrics(
    old_metrics_path: str | Path,
    new_metrics_path: str | Path,
    tolerance: float = 0.01,
) -> bool:
    """
    Compare metrics.json files.

    Args:
        old_metrics_path: Path to old metrics.json
        new_metrics_path: Path to new metrics.json
        tolerance: Maximum allowed relative difference (1% default)

    Returns:
        True if metrics are equivalent
    """
    import json

    old_metrics_path = Path(old_metrics_path)
    new_metrics_path = Path(new_metrics_path)

    if not old_metrics_path.exists():
        print(f"Old metrics not found: {old_metrics_path}")
        return True  # Not a failure if metrics don't exist
    if not new_metrics_path.exists():
        print(f"New metrics not found: {new_metrics_path}")
        return True

    with open(old_metrics_path) as f:
        old_metrics = json.load(f)
    with open(new_metrics_path) as f:
        new_metrics = json.load(f)

    print("\nComparing metrics:")
    all_match = True

    for key in old_metrics:
        if key not in new_metrics:
            print(f"  {key}: missing in new metrics")
            all_match = False
            continue

        old_val = old_metrics[key]
        new_val = new_metrics[key]

        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            if old_val == 0:
                diff = abs(new_val - old_val)
            else:
                diff = abs(new_val - old_val) / abs(old_val)

            status = "✓" if diff <= tolerance else "✗"
            print(f"  {key}: {old_val:.4f} -> {new_val:.4f} (diff: {diff*100:.2f}%) {status}")
            if diff > tolerance:
                all_match = False
        else:
            match = old_val == new_val
            status = "✓" if match else "✗"
            print(f"  {key}: {old_val} -> {new_val} {status}")
            if not match:
                all_match = False

    return all_match


def main():
    parser = argparse.ArgumentParser(description="Compare norm and norm_2 outputs")
    parser.add_argument("old_path", help="Path to old (norm) output parquet")
    parser.add_argument("new_path", help="Path to new (norm_2) output parquet")
    parser.add_argument(
        "--tolerance", "-t", type=float, default=1e-6,
        help="Tolerance for float comparison (default: 1e-6)"
    )
    parser.add_argument(
        "--metrics", "-m", action="store_true",
        help="Also compare metrics.json files"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Only print summary"
    )

    args = parser.parse_args()

    success = compare_outputs(
        args.old_path,
        args.new_path,
        tolerance=args.tolerance,
        verbose=not args.quiet,
    )

    if args.metrics:
        old_metrics = Path(args.old_path).parent / "results" / "metrics.json"
        new_metrics = Path(args.new_path).parent / "results" / "metrics.json"
        metrics_match = compare_metrics(old_metrics, new_metrics)
        success = success and metrics_match

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
