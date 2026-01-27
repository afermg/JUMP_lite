#!/usr/bin/env python3
"""Check if TVN produces identical values for same batch."""
import sys
sys.path.insert(0, 'src')
import polars as pl
import numpy as np
import yaml
from pathlib import Path

print("TVN Batch Comparison")
print("=" * 60)

# Load config
config_path = Path("src/norm/conf/preset/morphem.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

input_path = "output/morphem_jump_target2_4plate_zstd_raw_features.parquet"

# Load data
from norm_2.io import load_profiles, infer_columns, get_numeric_features

df = load_profiles(input_path)
print(f"Initial shape: {df.shape}")

# Run steps up to TVN
from norm_2.pipeline import STEPS

for step_config in config.get("steps", []):
    step_name = step_config.get("name", "unknown")
    if not step_config.get("enabled", True):
        continue

    if step_name == "normalize_tvn":
        break

    func = STEPS.get(step_name)
    if func is None:
        continue

    df = func(df, step_config.get("params", {}))

print(f"Before TVN shape: {df.shape}")

# Get features
features, _ = infer_columns(df, ["Metadata_"])
features = get_numeric_features(df, features)
print(f"Features: {len(features)}")

# Now manually apply TVN to ONE batch using both implementations
from norm.operations.normalize import TVN as OldTVN
from norm_2.core import TVN as NewTVN

batch_col = "Metadata_Plate"
batches = sorted(df[batch_col].unique().to_list())
print(f"Batches: {batches}")

# Test on first batch
batch = batches[0]
print(f"\nTesting batch: {batch}")

batch_df = df.filter(pl.col(batch_col) == batch)
X = batch_df.select(features).to_numpy()
print(f"Batch data shape: {X.shape}")

# Apply old TVN
old_tvn = OldTVN(alpha=0.3, epsilon=1.0)
X_old = old_tvn.fit_transform(X)
print(f"\nOld TVN output: mean={np.mean(X_old):.6f}, std={np.std(X_old):.6f}")

# Apply new TVN
new_tvn = NewTVN(alpha=0.3, epsilon=1.0)
X_new = new_tvn.fit_transform(X)
print(f"New TVN output: mean={np.mean(X_new):.6f}, std={np.std(X_new):.6f}")

# Compare
max_diff = np.max(np.abs(X_old - X_new))
print(f"\nMax difference: {max_diff:.2e}")

if max_diff < 1e-10:
    print("TVN outputs are IDENTICAL")
else:
    print("*** TVN outputs DIFFER ***")
    # Show first few values
    print(f"  First values old: {X_old[0, :5]}")
    print(f"  First values new: {X_new[0, :5]}")
