#!/usr/bin/env python3
"""Analyze morphem outputs to understand differences."""
import sys
sys.path.insert(0, 'src')
import polars as pl
import numpy as np

# Load both outputs
old = pl.read_parquet('test_comparison/old/morphem/processed.parquet')
new = pl.read_parquet('test_comparison/new/morphem/processed.parquet')

print("Morphem Output Analysis")
print("=" * 60)

# Sort both
old_sorted = old.sort(["Metadata_Plate", "Metadata_Well"])
new_sorted = new.sort(["Metadata_Plate", "Metadata_Well"])

# Get features
features = [c for c in old.columns if not c.startswith('Metadata_')]
print(f"Features: {len(features)}")

# Get metadata
plates_old = old_sorted['Metadata_Plate'].unique().to_list()
plates_new = new_sorted['Metadata_Plate'].unique().to_list()
print(f"Plates old: {plates_old}")
print(f"Plates new: {plates_new}")

# Check per-plate statistics
print("\nPer-plate analysis for first feature:")
feat = features[0]

for plate in sorted(plates_old):
    old_plate = old_sorted.filter(pl.col('Metadata_Plate') == plate)
    new_plate = new_sorted.filter(pl.col('Metadata_Plate') == plate)

    old_vals = old_plate[feat].to_numpy()
    new_vals = new_plate[feat].to_numpy()

    old_mean = np.mean(old_vals)
    new_mean = np.mean(new_vals)
    old_std = np.std(old_vals)
    new_std = np.std(new_vals)

    max_diff = np.max(np.abs(old_vals - new_vals))
    corr = np.corrcoef(old_vals, new_vals)[0, 1]

    print(f"\n  Plate {plate}:")
    print(f"    Old: mean={old_mean:.4f}, std={old_std:.4f}")
    print(f"    New: mean={new_mean:.4f}, std={new_std:.4f}")
    print(f"    Max diff: {max_diff:.2e}, Corr: {corr:.4f}")

# Check if the transformation is linearly related (same up to scale/shift)
print("\n" + "=" * 60)
print("Checking linear relationship:")
print("=" * 60)

for feat in features[:3]:
    old_vals = old_sorted[feat].to_numpy()
    new_vals = new_sorted[feat].to_numpy()

    # Fit linear regression: new = a*old + b
    A = np.vstack([old_vals, np.ones(len(old_vals))]).T
    a, b = np.linalg.lstsq(A, new_vals, rcond=None)[0]

    predicted = a * old_vals + b
    residual = np.max(np.abs(new_vals - predicted))

    print(f"{feat[:35]}:")
    print(f"  Linear fit: new = {a:.4f}*old + {b:.4f}")
    print(f"  Max residual after linear fit: {residual:.2e}")
