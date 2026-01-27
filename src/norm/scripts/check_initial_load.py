#!/usr/bin/env python3
"""Check if initial data loading produces same row order."""
import sys
sys.path.insert(0, 'src')
import polars as pl
import numpy as np

input_path = "output/morphem_jump_target2_4plate_zstd_raw_features.parquet"

# Load with both methods
from norm.data.load import load_profiles as old_load
from norm_2.io import load_profiles as new_load

df_old = old_load(input_path)
df_new = new_load(input_path)

print("Initial Load Comparison")
print("=" * 60)
print(f"Old shape: {df_old.shape}")
print(f"New shape: {df_new.shape}")

# Check row order
old_ids = df_old['Metadata_Plate'].to_list()[:20]
new_ids = df_new['Metadata_Plate'].to_list()[:20]
print(f"\nFirst 20 Metadata_Plate values:")
print(f"  Old: {old_ids}")
print(f"  New: {new_ids}")
print(f"  Match: {old_ids == new_ids}")

# Check unique order
old_unique = df_old['Metadata_Plate'].unique().to_list()
new_unique = df_new['Metadata_Plate'].unique().to_list()
print(f"\nUnique Metadata_Plate values:")
print(f"  Old: {old_unique}")
print(f"  New: {new_unique}")

# Check columns
old_cols = df_old.columns
new_cols = df_new.columns
print(f"\nColumns match: {old_cols == new_cols}")

# Check first feature values
features = [c for c in old_cols if not c.startswith('Metadata_')]
feat = features[0]
old_vals = df_old[feat].to_numpy()[:10]
new_vals = df_new[feat].to_numpy()[:10]
print(f"\nFirst 10 values of {feat}:")
print(f"  Old: {old_vals}")
print(f"  New: {new_vals}")
print(f"  Max diff: {np.max(np.abs(old_vals - new_vals)):.2e}")
