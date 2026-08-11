#!/usr/bin/env python3
"""Quantify rank-stability of model rankings across compression levels.

Computes Spearman rank correlation of model family rankings between all pairs
of compression codecs, both for the best normalization config per model and
for every individual normalization strategy.

Outputs:
  - Heatmap of pairwise Spearman rho (best config per model)
  - Heatmap per normalization config
  - Summary CSV with all pairwise correlations
  - Headline statistics
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Model family extraction from model column
FAMILY_PREFIXES = [
    "cellprofiler",
    "cell_count",
    "dinov2_random",
    "dinov2",
    "morphem",
    "openphenom",
    "subcell__clip01",
]


def get_family(model: str) -> str:
    for prefix in FAMILY_PREFIXES:
        if model.startswith(prefix):
            return prefix
    return model


def get_codec(model: str) -> str:
    """Extract codec from model name like 'dinov2_lite_jpegxl_lossy_mq_raw'."""
    # Models without codec (cell_count_lite_raw, cellprofiler_lite_raw)
    if "_lite_raw" in model and "jpegxl" not in model:
        return "raw"
    # Extract codec: everything between '_lite_' and '_raw' at end
    parts = model.split("_lite_")
    if len(parts) < 2:
        return "raw"
    codec_part = parts[1]
    if codec_part.endswith("_raw"):
        codec_part = codec_part[:-4]
    return codec_part


CODEC_ORDER = ["jpegxl_lossy_raw", "jpegxl_lossy_hq", "jpegxl_lossy_mq", "jpegxl_lossy_d20"]
CODEC_LABELS = {
    "jpegxl_lossy_raw": "Raw",
    "jpegxl_lossy_hq": "HQ",
    "jpegxl_lossy_mq": "MQ",
    "jpegxl_lossy_d20": "D20",
    "raw": "Raw",
}


def compute_nap_balanced(pa: float, pc: float) -> float:
    if pa <= 0 or pc <= 0:
        return np.nan
    return np.sqrt(pa * pc)


def rank_models_best_config(df: pd.DataFrame) -> pd.DataFrame:
    """For each (family, codec), find the best nap_balanced across configs."""
    df = df.copy()
    df["family"] = df["model"].apply(get_family)
    df["codec"] = df["model"].apply(get_codec)
    df["nap_balanced"] = df.apply(
        lambda r: compute_nap_balanced(r["PA_mean_nap"], r["PC_mean_nap"]), axis=1
    )

    # Best config per (family, codec)
    idx = df.groupby(["family", "codec"])["nap_balanced"].idxmax()
    best = df.loc[idx.dropna()].copy()
    return best


def rank_models_per_config(df: pd.DataFrame) -> pd.DataFrame:
    """Keep all configs, add family/codec/nap_balanced columns."""
    df = df.copy()
    df["family"] = df["model"].apply(get_family)
    df["codec"] = df["model"].apply(get_codec)
    df["nap_balanced"] = df.apply(
        lambda r: compute_nap_balanced(r["PA_mean_nap"], r["PC_mean_nap"]), axis=1
    )
    return df


def compute_pairwise_rho(ranked: pd.DataFrame, codecs: list[str]) -> pd.DataFrame:
    """Compute Spearman rho between model rankings at each pair of codecs."""
    results = []
    for c1, c2 in combinations(codecs, 2):
        r1 = ranked[ranked["codec"] == c1].set_index("family")["nap_balanced"]
        r2 = ranked[ranked["codec"] == c2].set_index("family")["nap_balanced"]
        common = r1.index.intersection(r2.index)
        if len(common) < 3:
            continue
        rho, pval = spearmanr(r1.loc[common], r2.loc[common])
        results.append({
            "codec_1": c1, "codec_2": c2,
            "rho": rho, "p_value": pval, "n_models": len(common),
        })
    return pd.DataFrame(results)


def plot_heatmap(pairwise: pd.DataFrame, codecs: list[str], output_path: Path, title: str):
    """Plot a symmetric heatmap of Spearman rho values."""
    labels = [CODEC_LABELS.get(c, c) for c in codecs]
    n = len(codecs)
    matrix = np.ones((n, n))

    for _, row in pairwise.iterrows():
        i = codecs.index(row["codec_1"])
        j = codecs.index(row["codec_2"])
        matrix[i, j] = row["rho"]
        matrix[j, i] = row["rho"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdYlGn", aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=11)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if abs(matrix[i, j]) > 0.7 else "black")

    plt.colorbar(im, ax=ax, label="Spearman rho", shrink=0.8)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


FAMILY_LABELS = {
    "cellprofiler": "CellProfiler",
    "cell_count": "Cell Count",
    "dinov2": "DINOv2",
    "dinov2_random": "DINOv2-rand",
    "morphem": "MorphEm",
    "openphenom": "OpenPhenom",
    "subcell__clip01": "SubCell",
}

FAMILY_MARKERS = {
    "cellprofiler": "s",
    "cell_count": "^",
    "dinov2": "o",
    "dinov2_random": "D",
    "morphem": "P",
    "openphenom": "X",
    "subcell__clip01": "v",
}


def plot_correlation_scatter(ranked: pd.DataFrame, codecs: list[str], output_dir: Path):
    """Scatter plot of NAP scores at one codec vs another, one panel per pair."""
    pairs = list(combinations(codecs, 2))
    n_pairs = len(pairs)
    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, (c1, c2) in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        r1 = ranked[ranked["codec"] == c1].set_index("family")["nap_balanced"]
        r2 = ranked[ranked["codec"] == c2].set_index("family")["nap_balanced"]
        common = r1.index.intersection(r2.index)
        if len(common) < 2:
            ax.set_visible(False)
            continue

        # Plot identity line
        all_vals = pd.concat([r1.loc[common], r2.loc[common]])
        lo, hi = all_vals.min() * 0.9, all_vals.max() * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)

        # Scatter each model family
        for fam in common:
            ax.scatter(
                r1.loc[fam], r2.loc[fam],
                marker=FAMILY_MARKERS.get(fam, "o"),
                s=100, zorder=3,
                label=FAMILY_LABELS.get(fam, fam),
            )

        rho, pval = spearmanr(r1.loc[common], r2.loc[common])
        ax.set_xlabel(f"NAP ({CODEC_LABELS.get(c1, c1)})")
        ax.set_ylabel(f"NAP ({CODEC_LABELS.get(c2, c2)})")
        ax.set_title(f"rho = {rho:.2f} (p = {pval:.3f})", fontsize=11)
        ax.grid(True, alpha=0.3)

    # Single legend for all panels
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    # Hide unused axes
    for idx in range(n_pairs, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Model NAP Correlation Across Compression Levels\n(Best Config per Model)", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = output_dir / "rank_stability_correlation_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_correlation_scatter_all_configs(data: pd.DataFrame, codecs: list[str], output_dir: Path):
    """Scatter plot with one point per (model_family, normalization_config) at each codec pair."""
    pairs = list(combinations(codecs, 2))
    n_pairs = len(pairs)
    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, (c1, c2) in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        d1 = data[data["codec"] == c1].set_index(["family", "config"])["nap_balanced"]
        d2 = data[data["codec"] == c2].set_index(["family", "config"])["nap_balanced"]
        common = d1.index.intersection(d2.index)
        if len(common) < 2:
            ax.set_visible(False)
            continue

        vals1 = d1.loc[common].dropna()
        vals2 = d2.loc[common].dropna()
        shared = vals1.index.intersection(vals2.index)
        vals1 = vals1.loc[shared]
        vals2 = vals2.loc[shared]

        # Identity line
        all_vals = pd.concat([vals1, vals2])
        lo, hi = all_vals.min() * 0.9, all_vals.max() * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)

        # Scatter per family
        for fam in sorted(set(f for f, _ in shared)):
            mask = [f == fam for f, _ in shared]
            ax.scatter(
                vals1.values[mask], vals2.values[mask],
                marker=FAMILY_MARKERS.get(fam, "o"),
                s=30, alpha=0.5, zorder=3,
                label=FAMILY_LABELS.get(fam, fam),
            )

        rho, pval = spearmanr(vals1, vals2)
        ax.set_xlabel(f"NAP ({CODEC_LABELS.get(c1, c1)})")
        ax.set_ylabel(f"NAP ({CODEC_LABELS.get(c2, c2)})")
        ax.set_title(f"rho = {rho:.2f} (p = {pval:.1e})", fontsize=11)
        ax.grid(True, alpha=0.3)

    # Deduplicated legend
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    for idx in range(n_pairs, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Model NAP Correlation Across Compression Levels\n(All Normalization Configs)", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = output_dir / "rank_stability_correlation_scatter_all_configs.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("src/norm_3/data/features/sweep_results_v11_lite_full.csv"),
        help="Path to sweep_results CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/output/rank_stability"),
        help="Output directory",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows, {df['model'].nunique()} models, {df['config'].nunique()} configs")

    # Determine available codecs
    df_tmp = df.copy()
    df_tmp["codec"] = df_tmp["model"].apply(get_codec)
    codecs = [c for c in CODEC_ORDER if c in df_tmp["codec"].unique()]
    if not codecs:
        codecs = sorted(df_tmp["codec"].unique())
    print(f"Codecs: {codecs}")

    # === 1. Best config per model family ===
    best = rank_models_best_config(df)
    pairwise_best = compute_pairwise_rho(best, codecs)

    plot_heatmap(
        pairwise_best, codecs,
        args.output_dir / "rank_stability_best_config.png",
        "Model Rank Stability Across Compression\n(Best Normalization Config per Model)",
    )

    # Print headline
    if not pairwise_best.empty:
        mean_rho = pairwise_best["rho"].mean()
        min_rho = pairwise_best["rho"].min()
        print(f"\n=== Best Config: Mean rho = {mean_rho:.3f}, Min rho = {min_rho:.3f} ===")
        print(pairwise_best.to_string(index=False))

    # === 2. Per normalization config ===
    all_data = rank_models_per_config(df)

    # Find configs that appear across multiple codecs and families
    config_counts = (
        all_data.dropna(subset=["nap_balanced"])
        .groupby("config")
        .apply(lambda g: g["codec"].nunique() >= 2 and g["family"].nunique() >= 3)
    )
    valid_configs = config_counts[config_counts].index.tolist()
    print(f"\n{len(valid_configs)} configs with >= 2 codecs and >= 3 families")

    per_config_results = []
    for config in valid_configs:
        sub = all_data[(all_data["config"] == config) & all_data["nap_balanced"].notna()]
        pairwise = compute_pairwise_rho(sub, codecs)
        if pairwise.empty:
            continue
        for _, row in pairwise.iterrows():
            per_config_results.append({**row.to_dict(), "config": config})

    per_config_df = pd.DataFrame(per_config_results)

    if not per_config_df.empty:
        # Summary stats across all configs
        summary = per_config_df.groupby(["codec_1", "codec_2"])["rho"].agg(
            ["mean", "std", "min", "max", "count"]
        ).reset_index()
        summary.to_csv(args.output_dir / "rank_stability_per_config_summary.csv", index=False)
        print(f"\n=== Per-Config Summary ===")
        print(summary.to_string(index=False))

        overall_mean = per_config_df["rho"].mean()
        overall_min = per_config_df["rho"].min()
        print(f"\nAcross all configs: Mean rho = {overall_mean:.3f}, Min rho = {overall_min:.3f}")

        # Heatmap of mean rho across configs
        mean_pairwise = summary[["codec_1", "codec_2", "mean"]].rename(columns={"mean": "rho"})
        plot_heatmap(
            mean_pairwise, codecs,
            args.output_dir / "rank_stability_mean_across_configs.png",
            "Model Rank Stability Across Compression\n(Mean Across All Normalization Configs)",
        )

        # Distribution plot of rho values
        fig, ax = plt.subplots(figsize=(8, 4))
        for (c1, c2), grp in per_config_df.groupby(["codec_1", "codec_2"]):
            label = f"{CODEC_LABELS.get(c1, c1)} vs {CODEC_LABELS.get(c2, c2)}"
            ax.hist(grp["rho"], bins=20, alpha=0.5, label=label)
        ax.set_xlabel("Spearman rho")
        ax.set_ylabel("Count (normalization configs)")
        ax.set_title("Distribution of Rank Stability Across Normalization Configs")
        ax.legend(fontsize=8)
        ax.axvline(x=0.9, color="green", linestyle="--", alpha=0.5, label="rho=0.9")
        fig.tight_layout()
        fig.savefig(args.output_dir / "rank_stability_distribution.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {args.output_dir / 'rank_stability_distribution.png'}")
        plt.close(fig)

    # === 3. Correlation scatter plots (best config) ===
    plot_correlation_scatter(best, codecs, args.output_dir)

    # === 4. Correlation scatter plots (all configs) ===
    plot_correlation_scatter_all_configs(all_data, codecs, args.output_dir)

    # Save all results
    pairwise_best.to_csv(args.output_dir / "rank_stability_best_config.csv", index=False)
    if not per_config_df.empty:
        per_config_df.to_csv(args.output_dir / "rank_stability_all_configs.csv", index=False)

    # Save model rankings at each codec (best config)
    rankings = best.pivot_table(index="family", columns="codec", values="nap_balanced")
    rankings = rankings.rank(ascending=False)
    rankings.to_csv(args.output_dir / "model_rankings_by_codec.csv")
    print(f"\n=== Model Rankings (best config) ===")
    print(rankings.to_string())


if __name__ == "__main__":
    main()
