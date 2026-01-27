#!/usr/bin/env python3
"""Test if Polars unique() is deterministic."""
import polars as pl
import numpy as np

print("Testing Polars unique() determinism")
print("=" * 40)

# Create test data with repeated values
np.random.seed(42)
data = np.random.choice(['A', 'B', 'C', 'D'], size=100)
df = pl.DataFrame({"col": data})

# Call unique multiple times
results = []
for i in range(5):
    u = df['col'].unique().to_list()
    results.append(tuple(u))
    print(f"Run {i+1}: {u}")

# Check if all results are identical
if len(set(results)) == 1:
    print("\nPASS: unique() is deterministic")
else:
    print("\nFAIL: unique() returned different orders!")

# Also test with string column from real parquet
print("\n" + "=" * 40)
print("Testing with real parquet data")
df_real = pl.read_parquet("output/morphem_jump_target2_4plate_zstd_raw_features.parquet")

results_real = []
for i in range(5):
    u = df_real['Metadata_Plate'].unique().to_list()
    results_real.append(tuple(u))
    print(f"Run {i+1}: {u}")

if len(set(results_real)) == 1:
    print("\nPASS: unique() is deterministic on real data")
else:
    print("\nFAIL: unique() returned different orders!")
