#!/usr/bin/env python3
"""Check batch order in both pipelines."""
import sys
sys.path.insert(0, 'src')
import polars as pl

# Load morphem data
input_path = "output/morphem_jump_target2_4plate_zstd_raw_features.parquet"
df = pl.read_parquet(input_path)

print("Checking batch order in Polars")
print("=" * 60)

# Check unique batch values
batch_col = "Metadata_Plate"
batches = df[batch_col].unique().to_list()
print(f"Batches from unique().to_list(): {batches}")

# Check if sorted
batches_sorted = sorted(batches)
print(f"Batches sorted: {batches_sorted}")

# The order from unique() is NOT guaranteed to be consistent
# Both implementations use the same call, so order should be the same
# But let's verify by checking the actual order in the dataframe

# Check first occurrence of each batch
first_occurrences = {}
for i, val in enumerate(df[batch_col].to_list()):
    if val not in first_occurrences:
        first_occurrences[val] = i

print(f"\nFirst occurrence indices: {first_occurrences}")

# Now check how normalize function handles this
print("\n" + "=" * 60)
print("Testing normalize function batch order")
print("=" * 60)

from norm.operations.normalize import normalize_profiles_extended as old_normalize
from norm_2.core import normalize as new_normalize
from norm_2.io import infer_columns, get_numeric_features

features, _ = infer_columns(df, "Metadata_")
features = get_numeric_features(df, features)
print(f"Number of features: {len(features)}")

# Check a simple normalization (robustmad) to verify batch order
# We'll compare the output row order
df_old = old_normalize(df, features[:10], method="robustmad", batch_col=batch_col)
df_new = new_normalize(df, features=features[:10], method="robustmad", batch_col=batch_col)

# Check if rows are in same order by comparing a feature
old_vals = df_old[features[0]].to_list()[:20]
new_vals = df_new[features[0]].to_list()[:20]

print(f"\nFirst 20 values of first feature after robustmad normalization:")
print(f"Old: {old_vals[:5]}")
print(f"New: {new_vals[:5]}")
print(f"Match: {old_vals == new_vals}")

# Sort both and compare
df_old_sorted = df_old.sort(["Metadata_Plate", "Metadata_Well"])
df_new_sorted = df_new.sort(["Metadata_Plate", "Metadata_Well"])

old_sorted_vals = df_old_sorted[features[0]].to_list()[:20]
new_sorted_vals = df_new_sorted[features[0]].to_list()[:20]

import numpy as np
diff = np.max(np.abs(np.array(old_sorted_vals) - np.array(new_sorted_vals)))
print(f"\nAfter sorting - max diff: {diff:.2e}")
