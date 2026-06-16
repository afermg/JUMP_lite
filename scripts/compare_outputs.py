"""Compare new feature extraction output against reference file."""
import polars as pl
import numpy as np

ref = pl.read_parquet("data/features/jump_lite_old/morphem_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet")
new = pl.read_parquet("output/morphem_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet")

print("=== SHAPE ===")
print(f"Reference: {ref.shape}")
print(f"New:       {new.shape}")

print("\n=== COLUMNS ===")
ref_cols = set(ref.columns)
new_cols = set(new.columns)
print(f"Reference columns: {len(ref_cols)}")
print(f"New columns:       {len(new_cols)}")
print(f"In reference only: {sorted(ref_cols - new_cols)}")
print(f"In new only:       {sorted(new_cols - ref_cols)}")

print("\n=== ROWS ===")
ref_ids = set(ref["Metadata_id"].to_list())
new_ids = set(new["Metadata_id"].to_list())
print(f"Reference well_ids: {len(ref_ids)}")
print(f"New well_ids:       {len(new_ids)}")
print(f"In reference only:  {len(ref_ids - new_ids)}")
print(f"In new only:        {len(new_ids - ref_ids)}")
print(f"Common:             {len(ref_ids & new_ids)}")

print("\n=== METADATA VALUES ===")
for col in ["Metadata_model", "Metadata_dataset", "Metadata_compression", "Metadata_Site"]:
    if col in ref.columns and col in new.columns:
        print(f"  {col}: ref={ref[col].unique().to_list()}, new={new[col].unique().to_list()}")

print("\n=== FEATURE VALUE COMPARISON ===")
common_cols = sorted(ref_cols & new_cols)
feature_cols = [c for c in common_cols if not c.startswith("Metadata")]
print(f"Common feature columns: {len(feature_cols)}")

# Join on Metadata_id and compare
ref_sorted = ref.select(["Metadata_id"] + feature_cols).sort("Metadata_id")
new_sorted = new.select(["Metadata_id"] + feature_cols).sort("Metadata_id")
merged = ref_sorted.join(new_sorted, on="Metadata_id", suffix="_new")

# Sample first 20 columns
print(f"\nFirst 20 feature columns:")
header = f"{'Column':<35} {'Max Diff':<15} {'Mean Diff':<15} {'Correlation':<12}"
print(header)
print("-" * len(header))
for col in feature_cols[:20]:
    ref_vals = merged[col].to_numpy()
    new_vals = merged[f"{col}_new"].to_numpy()
    mask = ~(np.isnan(ref_vals) | np.isnan(new_vals))
    if mask.sum() > 1:
        max_diff = np.max(np.abs(ref_vals[mask] - new_vals[mask]))
        mean_diff = np.mean(np.abs(ref_vals[mask] - new_vals[mask]))
        corr = np.corrcoef(ref_vals[mask], new_vals[mask])[0, 1]
        print(f"{col:<35} {max_diff:<15.6e} {mean_diff:<15.6e} {corr:<12.8f}")

# Overall stats across ALL feature columns
print("\nOverall stats across ALL feature columns...")
all_max_diffs = []
all_mean_diffs = []
all_corrs = []
for col in feature_cols:
    ref_vals = merged[col].to_numpy()
    new_vals = merged[f"{col}_new"].to_numpy()
    mask = ~(np.isnan(ref_vals) | np.isnan(new_vals))
    if mask.sum() > 1:
        max_diff = np.max(np.abs(ref_vals[mask] - new_vals[mask]))
        mean_diff = np.mean(np.abs(ref_vals[mask] - new_vals[mask]))
        corr = np.corrcoef(ref_vals[mask], new_vals[mask])[0, 1]
        all_max_diffs.append(max_diff)
        all_mean_diffs.append(mean_diff)
        all_corrs.append(corr)

print(f"Features compared: {len(all_corrs)}")
print(f"Max diff across all features:  {max(all_max_diffs):.6e}")
print(f"Mean of mean diffs:            {np.mean(all_mean_diffs):.6e}")
print(f"Min correlation:               {min(all_corrs):.8f}")
print(f"Mean correlation:              {np.mean(all_corrs):.8f}")
print(f"Features with correlation < 0.99: {sum(1 for c in all_corrs if c < 0.99)}")

# Check if old used median (values should differ) or mean (values should match)
if max(all_max_diffs) > 0.01:
    print("\nNOTE: Large differences detected — likely because reference used median and new uses mean.")
else:
    print("\nValues match closely — same aggregation method.")
