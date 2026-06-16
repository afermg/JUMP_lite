"""Compare openphenom vs openphenom_confusing features for 10 random shared sites."""

import numpy as np
import polars as pl
from pathlib import Path

DIR_A = Path("/work/datasets/aliby_output/jump_lite_raw/jump_lite/imgs/openphenom/raw/profiles")
DIR_B = Path("/work/datasets/aliby_output/jump_lite_raw/jump_lite/imgs/openphenom_confusing/raw/profiles")
N_SAMPLES = 10
SEED = 42


def load_wide(path: Path) -> np.ndarray:
    """Load parquet and pivot to (tiles, features) array."""
    df = pl.read_parquet(path)
    wide = df.pivot(on="metric", index=["tile", "label"], values="value").sort("tile")
    feat_cols = [c for c in wide.columns if c.startswith("X_")]
    feat_cols_sorted = sorted(feat_cols, key=lambda x: int(x.split("_")[1]))
    return wide.select(feat_cols_sorted).to_numpy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Row-wise cosine similarity, averaged."""
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return float(np.mean(dot / (norm_a * norm_b + 1e-12)))


def main():
    files_a = {p.name for p in DIR_A.glob("*.parquet")}
    files_b = {p.name for p in DIR_B.glob("*.parquet")}
    common = sorted(files_a & files_b)
    print(f"Total common sites: {len(common)}")

    rng = np.random.default_rng(SEED)
    sample = rng.choice(common, size=min(N_SAMPLES, len(common)), replace=False)

    results = []
    for fname in sample:
        arr_a = load_wide(DIR_A / fname)
        arr_b = load_wide(DIR_B / fname)

        # Per-tile cosine similarity
        cos = cosine_sim(arr_a, arr_b)

        # Pearson correlation of flattened vectors
        flat_a, flat_b = arr_a.flatten(), arr_b.flatten()
        pearson = float(np.corrcoef(flat_a, flat_b)[0, 1])

        # Mean absolute difference
        mad = float(np.mean(np.abs(arr_a - arr_b)))

        # Max absolute difference
        max_diff = float(np.max(np.abs(arr_a - arr_b)))

        # Mean and std of values in each
        mean_a, mean_b = float(arr_a.mean()), float(arr_b.mean())
        std_a, std_b = float(arr_a.std()), float(arr_b.std())

        site_name = fname.replace(".parquet", "")
        results.append({
            "site": site_name,
            "cosine_sim": cos,
            "pearson_r": pearson,
            "mean_abs_diff": mad,
            "max_abs_diff": max_diff,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "std_a": std_a,
            "std_b": std_b,
        })
        print(f"  {site_name}")
        print(f"    cosine={cos:.4f}  pearson={pearson:.4f}  MAD={mad:.4f}  max_diff={max_diff:.4f}")
        print(f"    mean_a={mean_a:.4f} std_a={std_a:.4f}  mean_b={mean_b:.4f} std_b={std_b:.4f}")

    df = pl.DataFrame(results)
    print("\n=== Summary across 10 sampled sites ===")
    print(f"  Cosine similarity:  mean={df['cosine_sim'].mean():.4f}  std={df['cosine_sim'].std():.4f}  min={df['cosine_sim'].min():.4f}  max={df['cosine_sim'].max():.4f}")
    print(f"  Pearson correlation: mean={df['pearson_r'].mean():.4f}  std={df['pearson_r'].std():.4f}  min={df['pearson_r'].min():.4f}  max={df['pearson_r'].max():.4f}")
    print(f"  Mean abs diff:      mean={df['mean_abs_diff'].mean():.4f}  std={df['mean_abs_diff'].std():.4f}")
    print(f"  Max abs diff:       mean={df['max_abs_diff'].mean():.4f}  max={df['max_abs_diff'].max():.4f}")


if __name__ == "__main__":
    main()
