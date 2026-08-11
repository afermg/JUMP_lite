#!/usr/bin/env python3
"""
Per-feature cross-well consistency analysis.

For each CellProfiler feature, compares cell-level distributions between
replicate wells (same treatment, same plate) using multiple metrics.
Runs for GT (zstd) and each lossy codec to measure whether compression
degrades cross-well consistency.

Metrics per feature:
  - KS statistic (max CDF distance)
  - Cohen's d (standardized mean difference)
  - Normalized Wasserstein (earth mover's distance / pooled std)
  - Variance ratio (max(var_a, var_b) / min(var_a, var_b))

Plus one whole-profile metric per group:
  - Well-profile similarity (cosine similarity of z-scored median profiles)

Optionally correlates cross-well metrics with codec feature correlations
from compare_codec_features.py to test whether features that are inherently
inconsistent across wells are also poorly preserved by compression.

Usage:
    python compare_cross_well_features.py \
        --features-base /path/to/cp_measure/jump_target2_4plate \
        --metadata metadata/metadata.parquet
"""

import argparse
import numpy as np
import polars as pl
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from scipy import stats

from compare_codec_features import (
    load_cell_features,
    sort_codecs_by_quality,
    CODEC_QUALITY_ORDER,
    get_feature_group,
)


def build_well_groups(
    profiles_dir: Path, metadata_path: Path
) -> list[dict]:
    """Build (plate, treatment) groups with exactly 2 replicate wells.

    Parses profile filenames, joins to metadata, and groups by
    (plate, treatment) keeping only groups with 2+ distinct wells.

    Returns list of dicts:
        {"plate", "treatment", "wells": [{"well", "source_ids": [str]}]}
    """
    stems = [p.stem for p in profiles_dir.glob("*.parquet")]
    records = []
    for stem in stems:
        parts = stem.split("__")
        if len(parts) != 5:
            continue
        source, batch, plate, well, site = parts
        records.append({
            "Metadata_Source": source,
            "Metadata_Batch": batch,
            "Metadata_Plate": plate,
            "Metadata_Well": well,
            "site": site,
            "source_id": stem,
        })

    profiles_df = pl.DataFrame(records)
    metadata = pl.read_parquet(metadata_path)

    joined = profiles_df.join(
        metadata,
        on=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"],
        how="inner",
    )

    groups = []
    for (plate, treatment), group_df in joined.group_by(
        ["Metadata_Plate", "Metadata_JCP2022"]
    ):
        wells = group_df["Metadata_Well"].unique().to_list()
        if len(wells) < 2:
            continue
        well_entries = []
        for well in sorted(wells)[:2]:
            well_rows = group_df.filter(pl.col("Metadata_Well") == well)
            source_ids = well_rows["source_id"].to_list()
            well_entries.append({"well": well, "source_ids": source_ids})
        groups.append({
            "plate": plate,
            "treatment": treatment,
            "wells": well_entries,
        })

    return groups


def load_well_features(
    profiles_dir: Path, source_ids: list[str], object_type: str
) -> tuple[np.ndarray, list[str]] | None:
    """Load and concatenate cell features for all sites in a well.

    Returns (feature_matrix, feature_names) or None if no data.
    """
    frames = []
    for sid in source_ids:
        path = profiles_dir / f"{sid}.parquet"
        if not path.exists():
            continue
        df = load_cell_features(path, object_type)
        if len(df) > 0:
            frames.append(df)

    if not frames:
        return None

    combined = pl.concat(frames)
    feature_names = [c for c in combined.columns if c != "label"]
    matrix = combined.select(feature_names).to_numpy().astype(np.float64)
    return matrix, feature_names


def compute_per_feature_metrics(
    values_a: np.ndarray, values_b: np.ndarray
) -> dict:
    """Compute distributional metrics between two 1-D arrays of cell values.

    Returns dict with ks_stat, cohens_d, wasserstein_norm, variance_ratio.
    """
    valid_a = values_a[~np.isnan(values_a)]
    valid_b = values_b[~np.isnan(values_b)]

    if len(valid_a) < 5 or len(valid_b) < 5:
        return {
            "ks_stat": np.nan,
            "cohens_d": np.nan,
            "wasserstein_norm": np.nan,
            "variance_ratio": np.nan,
        }

    # KS statistic
    ks_stat = stats.ks_2samp(valid_a, valid_b).statistic

    # Cohen's d
    mean_diff = abs(np.mean(valid_a) - np.mean(valid_b))
    pooled_var = (np.var(valid_a, ddof=1) + np.var(valid_b, ddof=1)) / 2
    cohens_d = mean_diff / np.sqrt(pooled_var) if pooled_var > 0 else 0.0

    # Normalized Wasserstein: divide by pooled std to make scale-invariant
    pooled_std = np.sqrt(pooled_var) if pooled_var > 1e-30 else np.nan
    if pooled_std is not np.nan and np.isfinite(pooled_std):
        raw_wass = stats.wasserstein_distance(valid_a, valid_b)
        wass = raw_wass / pooled_std if np.isfinite(raw_wass) else np.nan
    else:
        wass = np.nan

    # Variance ratio: max/min of the two variances (>=1, 1=identical spread)
    # Capped at 100 to avoid inf from near-zero-variance features
    var_a = np.var(valid_a, ddof=1)
    var_b = np.var(valid_b, ddof=1)
    if min(var_a, var_b) > 1e-30:
        variance_ratio = min(max(var_a, var_b) / min(var_a, var_b), 100.0)
    else:
        variance_ratio = np.nan

    return {
        "ks_stat": float(ks_stat),
        "cohens_d": float(cohens_d),
        "wasserstein_norm": float(wass),
        "variance_ratio": float(variance_ratio),
    }


def compute_well_profile_correlation(
    features_a: np.ndarray, features_b: np.ndarray
) -> float:
    """Spearman rank correlation of median feature profiles between two wells.

    Computes the median of each feature per well, then ranks the features
    and correlates the ranks. Spearman is scale-invariant, avoiding the
    saturation-at-1.0 problem that Pearson on raw profiles has.
    """
    median_a = np.nanmedian(features_a, axis=0)
    median_b = np.nanmedian(features_b, axis=0)

    valid = ~(np.isnan(median_a) | np.isnan(median_b))
    if valid.sum() < 3:
        return np.nan

    rho, _ = stats.spearmanr(median_a[valid], median_b[valid])
    return float(rho)



def compute_replicate_correlations_from_raw(
    well_groups: list[dict],
    codec: str,
    profiles_base: Path,
    object_type: str,
    min_cells: int,
) -> pd.DataFrame:
    """Compute per-feature replicate correlation from raw data.

    For each feature, collects the well-level median across all groups,
    then Spearman-correlates well_A medians vs well_B medians.

    Returns DataFrame: feature, feature_group, replicate_corr, n_groups.
    """
    profiles_dir = profiles_base / codec / "profiles"

    # Collect per-group, per-feature medians for well_A and well_B
    medians_a = []  # list of dicts: {feature: median_value}
    medians_b = []

    for group in well_groups:
        well_a_info = group["wells"][0]
        well_b_info = group["wells"][1]

        result_a = load_well_features(profiles_dir, well_a_info["source_ids"], object_type)
        result_b = load_well_features(profiles_dir, well_b_info["source_ids"], object_type)

        if result_a is None or result_b is None:
            continue

        matrix_a, names_a = result_a
        matrix_b, names_b = result_b

        if matrix_a.shape[0] < min_cells or matrix_b.shape[0] < min_cells:
            continue

        common = sorted(set(names_a) & set(names_b))
        idx_a = [names_a.index(f) for f in common]
        idx_b = [names_b.index(f) for f in common]

        med_a = np.nanmedian(matrix_a[:, idx_a], axis=0)
        med_b = np.nanmedian(matrix_b[:, idx_b], axis=0)

        medians_a.append(dict(zip(common, med_a)))
        medians_b.append(dict(zip(common, med_b)))

    if not medians_a:
        return pd.DataFrame()

    # Convert to DataFrames (groups × features)
    df_a = pd.DataFrame(medians_a)
    df_b = pd.DataFrame(medians_b)

    # For each feature, Spearman correlate the column across groups
    results = []
    for feature in df_a.columns:
        if feature not in df_b.columns:
            continue
        vals_a = df_a[feature].values
        vals_b = df_b[feature].values
        valid = ~(np.isnan(vals_a) | np.isnan(vals_b))
        n_valid = valid.sum()
        if n_valid < 5:
            continue
        rho, _ = stats.spearmanr(vals_a[valid], vals_b[valid])
        results.append({
            "feature": feature,
            "feature_group": get_feature_group(feature),
            "replicate_corr": float(rho),
            "n_groups": int(n_valid),
        })

    return pd.DataFrame(results)


def process_group(
    group: dict,
    codecs: list[str],
    profiles_base: Path,
    object_type: str,
    min_cells: int,
) -> tuple[list[dict], list[dict]]:
    """Process one (plate, treatment) group across all codecs.

    Returns (per_feature_rows, profile_corr_rows).
    """
    plate = group["plate"]
    treatment = group["treatment"]
    well_a_info = group["wells"][0]
    well_b_info = group["wells"][1]

    per_feature_rows = []
    profile_corr_rows = []

    for codec in codecs:
        profiles_dir = profiles_base / codec / "profiles"

        result_a = load_well_features(profiles_dir, well_a_info["source_ids"], object_type)
        result_b = load_well_features(profiles_dir, well_b_info["source_ids"], object_type)

        if result_a is None or result_b is None:
            continue

        matrix_a, names_a = result_a
        matrix_b, names_b = result_b

        if matrix_a.shape[0] < min_cells or matrix_b.shape[0] < min_cells:
            continue

        # Align features (should match, but be safe)
        common = sorted(set(names_a) & set(names_b))
        idx_a = [names_a.index(f) for f in common]
        idx_b = [names_b.index(f) for f in common]
        mat_a = matrix_a[:, idx_a]
        mat_b = matrix_b[:, idx_b]

        # Well-profile correlation
        prof_corr = compute_well_profile_correlation(mat_a, mat_b)
        profile_corr_rows.append({
            "plate": plate,
            "treatment": treatment,
            "codec": codec,
            "well_a": well_a_info["well"],
            "well_b": well_b_info["well"],
            "n_cells_a": matrix_a.shape[0],
            "n_cells_b": matrix_b.shape[0],
            "well_profile_correlation": prof_corr,
        })

        # Per-feature metrics
        for i, feature in enumerate(common):
            metrics = compute_per_feature_metrics(mat_a[:, i], mat_b[:, i])
            per_feature_rows.append({
                "plate": plate,
                "treatment": treatment,
                "codec": codec,
                "feature": feature,
                "feature_group": get_feature_group(feature),
                "well_a": well_a_info["well"],
                "well_b": well_b_info["well"],
                "n_cells_a": matrix_a.shape[0],
                "n_cells_b": matrix_b.shape[0],
                **metrics,
            })

    return per_feature_rows, profile_corr_rows


# ── Plotting ──────────────────────────────────────────────────────────


def plot_metric_violin(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    codecs_ordered: list[str],
    output_path: Path,
    title_extra: str = "",
):
    """Violin plot of one metric across codecs."""
    codec_labels = {
        c: c.replace(".zarr", "").replace("jpegxl_lossy_", "jxl_")
        for c in codecs_ordered
    }
    plot_df = df.copy()
    plot_df["codec_label"] = plot_df["codec"].map(codec_labels)
    label_order = [codec_labels[c] for c in codecs_ordered if c in codec_labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=plot_df,
        x="codec_label",
        y=metric,
        order=label_order,
        palette="viridis",
        inner="box",
        cut=0,
        ax=ax,
    )
    ax.set_xlabel("Codec", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f"Cross-Well {ylabel} per Feature{title_extra}", fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_gt_feature_ranking(
    df: pd.DataFrame,
    gt_codec: str,
    metric: str,
    ylabel: str,
    output_path: Path,
    top_n: int = 50,
):
    """Bar chart ranking features by their GT consistency (mean metric across groups)."""
    gt_df = df[df["codec"] == gt_codec]
    if gt_df.empty:
        return

    ranking = gt_df.groupby("feature")[metric].mean().sort_values()

    # Show top and bottom
    n_show = min(top_n, len(ranking))
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, n_show * 0.22)))

    # Most consistent (lowest KS / Cohen's d)
    best = ranking.head(n_show)
    colors_best = [plt.cm.viridis(0.2 + 0.6 * i / len(best)) for i in range(len(best))]
    axes[0].barh(range(len(best)), best.values, color=colors_best)
    axes[0].set_yticks(range(len(best)))
    axes[0].set_yticklabels(best.index, fontsize=7)
    axes[0].set_xlabel(ylabel, fontsize=12)
    axes[0].set_title(f"Most Consistent (GT, lowest {metric})", fontsize=14, fontweight="bold")
    axes[0].invert_yaxis()

    # Least consistent (highest KS / Cohen's d)
    worst = ranking.tail(n_show).sort_values(ascending=False)
    colors_worst = [plt.cm.magma(0.2 + 0.6 * i / len(worst)) for i in range(len(worst))]
    axes[1].barh(range(len(worst)), worst.values, color=colors_worst)
    axes[1].set_yticks(range(len(worst)))
    axes[1].set_yticklabels(worst.index, fontsize=7)
    axes[1].set_xlabel(ylabel, fontsize=12)
    axes[1].set_title(f"Least Consistent (GT, highest {metric})", fontsize=14, fontweight="bold")
    axes[1].invert_yaxis()

    plt.suptitle(f"GT Feature Ranking by Cross-Well {ylabel}", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_compression_delta(
    df: pd.DataFrame,
    gt_codec: str,
    metric: str,
    ylabel: str,
    codecs_ordered: list[str],
    output_path: Path,
):
    """Violin of metric(codec) - metric(GT) per feature, faceted by codec."""
    gt_df = df[df["codec"] == gt_codec][["plate", "treatment", "feature", metric]].rename(
        columns={metric: f"{metric}_gt"}
    )
    lossy_codecs = [c for c in codecs_ordered if c != gt_codec]
    if not lossy_codecs:
        return

    codec_labels = {
        c: c.replace(".zarr", "").replace("jpegxl_lossy_", "jxl_")
        for c in lossy_codecs
    }

    merged = df[df["codec"].isin(lossy_codecs)].merge(
        gt_df, on=["plate", "treatment", "feature"], how="inner"
    )
    merged["delta"] = merged[metric] - merged[f"{metric}_gt"]
    merged["codec_label"] = merged["codec"].map(codec_labels)
    label_order = [codec_labels[c] for c in lossy_codecs if c in codec_labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=merged,
        x="codec_label",
        y="delta",
        order=label_order,
        palette="coolwarm",
        inner="box",
        cut=0,
        ax=ax,
    )
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Codec", fontsize=14)
    ax.set_ylabel(f"Delta {ylabel} (codec - GT)", fontsize=14)
    ax.set_title(
        f"Compression Impact on Cross-Well {ylabel}",
        fontsize=16,
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=30, labelsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_by_feature_group(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    codecs_ordered: list[str],
    output_path: Path,
):
    """Box/violin per feature group, faceted by codec."""
    codec_labels = {
        c: c.replace(".zarr", "").replace("jpegxl_lossy_", "jxl_")
        for c in codecs_ordered
    }
    plot_df = df.copy()
    plot_df["codec_label"] = plot_df["codec"].map(codec_labels)
    label_order = [codec_labels[c] for c in codecs_ordered if c in codec_labels]
    groups = sorted(plot_df["feature_group"].unique())

    n_codecs = len(label_order)
    fig, axes = plt.subplots(
        1, n_codecs, figsize=(4 * n_codecs, 6), sharey=True, squeeze=False
    )
    axes = axes.flatten()

    for i, label in enumerate(label_order):
        sub = plot_df[plot_df["codec_label"] == label]
        ax = axes[i]
        sns.boxplot(
            data=sub,
            x="feature_group",
            y=metric,
            order=groups,
            palette="Set2",
            ax=ax,
        )
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        else:
            ax.set_ylabel("")

    plt.suptitle(
        f"Cross-Well {ylabel} by Feature Group",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_profile_correlation_bar(
    prof_df: pd.DataFrame,
    codecs_ordered: list[str],
    output_path: Path,
):
    """Bar chart of median well-profile correlation per codec."""
    codec_labels = {
        c: c.replace(".zarr", "").replace("jpegxl_lossy_", "jxl_")
        for c in codecs_ordered
    }
    medians = (
        prof_df.groupby("codec")["well_profile_correlation"]
        .median()
        .reindex(codecs_ordered)
    )
    labels = [codec_labels.get(c, c) for c in codecs_ordered]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
    ax.bar(labels, medians.values, color=colors)
    ax.set_ylabel("Median Well-Profile Correlation", fontsize=13)
    ax.set_xlabel("Codec", fontsize=13)
    ax.set_title(
        "Well-Profile Correlation (median across 52 groups)",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_ylim(max(0, medians.min() - 0.05), min(1.02, medians.max() + 0.02))
    ax.tick_params(axis="x", rotation=30, labelsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_by_plate(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    codecs_ordered: list[str],
    output_path: Path,
):
    """Violin of metric per codec, faceted by plate."""
    codec_labels = {
        c: c.replace(".zarr", "").replace("jpegxl_lossy_", "jxl_")
        for c in codecs_ordered
    }
    plot_df = df.copy()
    plot_df["codec_label"] = plot_df["codec"].map(codec_labels)
    label_order = [codec_labels[c] for c in codecs_ordered if c in codec_labels]

    plates = sorted(plot_df["plate"].unique())
    n_plates = len(plates)
    fig, axes = plt.subplots(1, n_plates, figsize=(5 * n_plates, 6), sharey=True, squeeze=False)
    axes = axes.flatten()

    for i, plate in enumerate(plates):
        sub = plot_df[plot_df["plate"] == plate]
        ax = axes[i]
        sns.violinplot(
            data=sub,
            x="codec_label",
            y=metric,
            order=label_order,
            palette="viridis",
            inner="box",
            cut=0,
            ax=ax,
        )
        ax.set_title(plate, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        else:
            ax.set_ylabel("")

    plt.suptitle(
        f"Cross-Well {ylabel} per Plate",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_crosswell_vs_codec_correlation(
    cross_well_summary: pd.DataFrame,
    codec_corr_df: pd.DataFrame,
    gt_codec: str,
    output_dir: Path,
):
    """Scatter: cross-well consistency (GT) vs codec feature correlation (zstd→HQ).

    For each feature, plots its inherent cross-well variability (KS, Cohen's d)
    against how well compression preserves it (cell-level Pearson r from the
    codec comparison script).
    """
    # Get GT cross-well metrics (averaged across groups)
    gt_summary = cross_well_summary[cross_well_summary["codec"] == gt_codec].copy()
    if gt_summary.empty:
        print("  WARNING: No GT data in cross-well summary, skipping correlation plot")
        return

    # Codec corr has columns: feature, d10, lq, mq, effort_3, d2_e8, hq
    # We want the HQ column as the best lossy codec
    hq_col = "hq" if "hq" in codec_corr_df.columns else None
    if hq_col is None:
        for col in codec_corr_df.columns:
            if "hq" in col.lower():
                hq_col = col
                break
    if hq_col is None:
        print("  WARNING: No HQ column found in codec correlation CSV")
        return

    # Join on feature name
    merged = gt_summary.merge(
        codec_corr_df[["feature", hq_col]].rename(columns={hq_col: "codec_corr_hq"}),
        on="feature",
        how="inner",
    )
    print(f"  Joined {len(merged)} features between cross-well and codec correlation")

    if len(merged) < 10:
        print("  WARNING: Too few features to plot correlation")
        return

    # Also join all codec columns for the CSV
    all_codec_cols = [c for c in codec_corr_df.columns if c != "feature"]
    merged_full = gt_summary.merge(codec_corr_df, on="feature", how="inner")

    # Save joined CSV
    joined_csv = output_dir / "cross_well_vs_codec_correlation.csv"
    merged_full.to_csv(joined_csv, index=False)
    print(f"  Saved joined data to: {joined_csv}")

    # Color by feature group
    group_colors = {
        "SizeShape": "#1f77b4",
        "Intensity": "#ff7f0e",
        "Texture": "#2ca02c",
        "Zernike": "#d62728",
        "RadialDistribution": "#9467bd",
        "Granularity": "#8c564b",
        "Other": "#7f7f7f",
    }

    for metric, ylabel in [
        ("ks_stat_mean", "Mean KS Statistic (GT, cross-well)"),
        ("cohens_d_mean", "Mean Cohen's d (GT, cross-well)"),
    ]:
        if metric not in merged.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        for group, color in group_colors.items():
            sub = merged[merged["feature_group"] == group]
            if sub.empty:
                continue
            ax.scatter(
                sub["codec_corr_hq"],
                sub[metric],
                c=color,
                label=f"{group} ({len(sub)})",
                alpha=0.6,
                s=20,
            )

        # Compute Spearman correlation
        valid = merged[["codec_corr_hq", metric]].dropna()
        if len(valid) > 5:
            rho, pval = stats.spearmanr(valid["codec_corr_hq"], valid[metric])
            ax.text(
                0.05, 0.95,
                f"Spearman rho={rho:.3f} (p={pval:.2e})",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        ax.set_xlabel("Codec Feature Correlation (Raw vs JXL-HQ, cell-level)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(
            "Cross-Well Consistency vs Compression Fidelity",
            fontsize=15,
            fontweight="bold",
        )
        ax.legend(loc="lower left", fontsize=9, title="Feature Group")
        plt.tight_layout()

        metric_short = metric.replace("_mean", "")
        out = output_dir / f"cross_well_vs_codec_{metric_short}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")

    # Also do per-codec analysis: for each lossy codec, correlate its
    # cross-well KS with its codec feature correlation
    lossy_codecs_in_summary = [
        c for c in cross_well_summary["codec"].unique() if c != gt_codec
    ]
    if lossy_codecs_in_summary and len(all_codec_cols) > 0:
        # Map codec names to short names used in codec_corr_df columns
        codec_to_col = {}
        for codec in lossy_codecs_in_summary:
            short = codec.replace(".zarr", "").replace("jpegxl_lossy_", "")
            if short in codec_corr_df.columns:
                codec_to_col[codec] = short

        if codec_to_col:
            n_codecs = len(codec_to_col)
            fig, axes = plt.subplots(
                1, n_codecs, figsize=(5 * n_codecs, 5), squeeze=False
            )
            axes = axes.flatten()

            for i, (codec, col) in enumerate(codec_to_col.items()):
                codec_summary = cross_well_summary[
                    cross_well_summary["codec"] == codec
                ]
                m = codec_summary.merge(
                    codec_corr_df[["feature", col]].rename(
                        columns={col: "codec_corr"}
                    ),
                    on="feature",
                    how="inner",
                )
                ax = axes[i]
                for group, color in group_colors.items():
                    sub = m[m["feature_group"] == group]
                    if sub.empty:
                        continue
                    ax.scatter(
                        sub["codec_corr"],
                        sub["ks_stat_mean"],
                        c=color,
                        alpha=0.5,
                        s=15,
                    )

                valid = m[["codec_corr", "ks_stat_mean"]].dropna()
                if len(valid) > 5:
                    rho, _ = stats.spearmanr(
                        valid["codec_corr"], valid["ks_stat_mean"]
                    )
                    ax.set_title(f"{col} (rho={rho:.3f})", fontsize=11, fontweight="bold")
                else:
                    ax.set_title(col, fontsize=11, fontweight="bold")

                ax.set_xlabel("Codec Corr" if i == n_codecs // 2 else "")
                if i == 0:
                    ax.set_ylabel("Cross-Well KS (this codec)")

            handles = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=g)
                for g, c in group_colors.items()
            ]
            fig.legend(handles=handles, loc="upper right", fontsize=8)
            plt.suptitle(
                "Per-Codec: Cross-Well KS vs Codec Feature Correlation",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()
            out = output_dir / "cross_well_vs_codec_per_codec.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {out.name}")


def plot_replicate_vs_codec_correlation(
    replicate_df: pd.DataFrame,
    codec_corr_df: pd.DataFrame,
    output_dir: Path,
):
    """Scatter: per-feature replicate correlation vs codec feature correlation.

    X = codec correlation (zstd vs HQ): how well compression preserves the feature
    Y = replicate correlation: how reproducibly the feature measures biology

    High X, High Y = signal, well-preserved (best features)
    Low X, Low Y = noise, poorly preserved (discard candidates)
    High X, Low Y = preserved by compression but not biologically reproducible
    Low X, High Y = real signal destroyed by compression (worst case for compression)
    """
    hq_col = "hq" if "hq" in codec_corr_df.columns else None
    if hq_col is None:
        for col in codec_corr_df.columns:
            if "hq" in col.lower():
                hq_col = col
                break
    if hq_col is None:
        print("  WARNING: No HQ column found in codec correlation CSV")
        return

    merged = replicate_df.merge(
        codec_corr_df[["feature", hq_col]].rename(columns={hq_col: "codec_corr_hq"}),
        on="feature",
        how="inner",
    )
    print(f"  Joined {len(merged)} features for replicate vs codec plot")

    if len(merged) < 10:
        return

    # Save CSV
    csv_path = output_dir / "replicate_vs_codec_correlation.csv"
    # Also join all codec columns
    merged_full = replicate_df.merge(codec_corr_df, on="feature", how="inner")
    merged_full.to_csv(csv_path, index=False)
    print(f"  Saved joined data to: {csv_path}")

    group_colors = {
        "SizeShape": "#1f77b4",
        "Intensity": "#ff7f0e",
        "Texture": "#2ca02c",
        "Zernike": "#d62728",
        "RadialDistribution": "#9467bd",
        "Granularity": "#8c564b",
        "Other": "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    for group, color in group_colors.items():
        sub = merged[merged["feature_group"] == group]
        if sub.empty:
            continue
        ax.scatter(
            sub["codec_corr_hq"],
            sub["replicate_corr"],
            c=color,
            label=f"{group} ({len(sub)})",
            alpha=0.6,
            s=20,
        )

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)

    # Spearman rho (placed outside figure, below axes)
    valid = merged[["codec_corr_hq", "replicate_corr"]].dropna()
    if len(valid) > 5:
        import math
        rho, pval = stats.spearmanr(valid["codec_corr_hq"], valid["replicate_corr"])
        if pval < 1e-12:
            pval_str = r"$p < 10^{-12}$"
        else:
            exp = -int(math.floor(math.log10(pval)))
            pval_str = rf"$p < 10^{{-{exp}}}$"
        fig.text(
            0.5, -0.02,
            f"Spearman $\\rho$={rho:.3f} ({pval_str})",
            fontsize=12,
            fontweight="bold",
            ha="center",
        )

    ax.set_xlabel("Codec Feature Correlation (Raw vs JXL-HQ, cell-level)", fontsize=13)
    ax.set_ylabel("Replicate Correlation (well-pair median reproducibility)", fontsize=13)
    ax.set_title(
        "Feature Signal vs Compression Fidelity",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=9, title="Feature Group")
    plt.tight_layout()

    out = output_dir / "replicate_vs_codec_correlation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # Also plot replicate correlation distribution by feature group
    fig, ax = plt.subplots(figsize=(10, 6))
    groups_present = sorted(merged["feature_group"].unique())
    group_data = [merged[merged["feature_group"] == g]["replicate_corr"].dropna().values
                  for g in groups_present]
    bp = ax.boxplot(group_data, labels=groups_present, patch_artist=True)
    colors = [group_colors.get(g, "#7f7f7f") for g in groups_present]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Replicate Correlation", fontsize=13)
    ax.set_xlabel("Feature Group", fontsize=13)
    ax.set_title("Feature Reproducibility by Group (GT)", fontsize=15, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=11)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out = output_dir / "replicate_correlation_by_group.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Per-feature cross-well consistency analysis"
    )
    parser.add_argument(
        "--features-base",
        type=str,
        default="data/aliby_output/cp_measure/jump_target2_4plate",
        help="Base path for codec profiles",
    )
    parser.add_argument(
        "--gt-codec",
        type=str,
        default="zstd.zarr",
        help="Ground truth codec",
    )
    parser.add_argument(
        "--codecs",
        nargs="+",
        default=None,
        help="Codecs to compare (default: auto-discover)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="metadata/metadata.parquet",
        help="Metadata parquet path",
    )
    parser.add_argument(
        "--object-type",
        type=str,
        nargs="+",
        default=["cell", "nuclei"],
        choices=["cell", "nuclei"],
        help="Object types to analyze (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/feature_similarity/output",
        help="Output directory",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=50,
        help="Minimum cells per well to include",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max parallel workers",
    )
    parser.add_argument(
        "--codec-correlation-csv",
        type=str,
        default="analysis/output/codec_feature_correlation.csv",
        help="CSV with per-feature codec correlations (from compare_codec_features.py)",
    )

    args = parser.parse_args()

    features_base = Path(args.features_base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-discover codecs
    if args.codecs is None:
        all_codec_dirs = [
            d.name
            for d in features_base.iterdir()
            if d.is_dir() and d.name.endswith(".zarr") and d.name != args.gt_codec
        ]
        codecs = [args.gt_codec] + sort_codecs_by_quality(all_codec_dirs)
    else:
        codecs = [args.gt_codec] + [c for c in args.codecs if c != args.gt_codec]

    print(f"Codecs ({len(codecs)}): {codecs}")

    # Build well groups from GT profiles
    gt_profiles_dir = features_base / args.gt_codec / "profiles"
    print(f"\nBuilding well groups from {gt_profiles_dir} ...")
    well_groups = build_well_groups(gt_profiles_dir, args.metadata)
    print(f"Found {len(well_groups)} (plate, treatment) groups with 2+ wells")

    # Summary by plate
    plates = {}
    for g in well_groups:
        plates.setdefault(g["plate"], []).append(g["treatment"])
    for plate, treatments in sorted(plates.items()):
        print(f"  {plate}: {len(treatments)} treatments")

    object_types = args.object_type

    for object_type in object_types:
        obj_dir = output_dir / object_type
        obj_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"  Object type: {object_type}")
        print(f"{'#'*70}")

        # Process all groups
        print(f"\nProcessing {len(well_groups)} groups x {len(codecs)} codecs ...")
        all_feature_rows = []
        all_profile_rows = []

        def _worker(group, _ot=object_type):
            return process_group(group, codecs, features_base, _ot, args.min_cells)

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(_worker, well_groups),
                    total=len(well_groups),
                    desc=f"Processing groups ({object_type})",
                    unit="group",
                )
            )

        for feat_rows, prof_rows in results:
            all_feature_rows.extend(feat_rows)
            all_profile_rows.extend(prof_rows)

        if not all_feature_rows:
            print(f"WARNING: No results for {object_type}. Skipping.")
            continue

        detail_df = pd.DataFrame(all_feature_rows)
        prof_df = pd.DataFrame(all_profile_rows)

        print(f"\nResults: {len(detail_df)} per-feature rows, {len(prof_df)} profile-correlation rows")
        print(f"  Unique features: {detail_df['feature'].nunique()}")
        print(f"  Unique codecs: {detail_df['codec'].nunique()}")
        print(f"  Unique (plate, treatment) groups: {detail_df.groupby(['plate', 'treatment']).ngroups}")

        # ── Save CSVs ─────────────────────────────────────────────
        detail_csv = obj_dir / "cross_well_metrics_detailed.csv"
        detail_df.to_csv(detail_csv, index=False)
        print(f"\nSaved detailed metrics to: {detail_csv}")

        summary_df = (
            detail_df.groupby(["codec", "feature", "feature_group"])
            .agg(
                ks_stat_mean=("ks_stat", "mean"),
                ks_stat_median=("ks_stat", "median"),
                cohens_d_mean=("cohens_d", "mean"),
                cohens_d_median=("cohens_d", "median"),
                wasserstein_norm_mean=("wasserstein_norm", "mean"),
                wasserstein_norm_median=("wasserstein_norm", "median"),
                variance_ratio_mean=("variance_ratio", "mean"),
                variance_ratio_median=("variance_ratio", "median"),
                n_groups=("ks_stat", "count"),
            )
            .reset_index()
        )
        summary_csv = obj_dir / "cross_well_metrics_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved summary metrics to: {summary_csv}")

        prof_csv = obj_dir / "cross_well_profile_correlation.csv"
        prof_df.to_csv(prof_csv, index=False)
        print(f"Saved profile correlations to: {prof_csv}")

        # ── Codec ordering for plots ──────────────────────────────
        codecs_in_data = [c for c in codecs if c in detail_df["codec"].unique()]

        # ── Quick stats ───────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Summary statistics per codec ({object_type})")
        print(f"{'='*60}")
        for codec in codecs_in_data:
            sub = detail_df[detail_df["codec"] == codec]
            print(f"\n  {codec}:")
            for metric in ["ks_stat", "cohens_d", "wasserstein_norm", "variance_ratio"]:
                vals = sub[metric].dropna()
                if len(vals) > 0:
                    print(f"    {metric:20s}: mean={vals.mean():.4f}  median={vals.median():.4f}  std={vals.std():.4f}")
        if not prof_df.empty:
            print(f"\n  Well-profile correlation:")
            for codec in codecs_in_data:
                sub = prof_df[prof_df["codec"] == codec]
                vals = sub["well_profile_correlation"].dropna()
                if len(vals) > 0:
                    print(f"    {codec:40s}: median={vals.median():.4f}  mean={vals.mean():.4f}")

        # ── Plots ─────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Generating plots ({object_type}) ...")
        print(f"{'='*60}")

        metrics_info = [
            ("ks_stat", "KS Statistic"),
            ("cohens_d", "Cohen's d"),
            ("wasserstein_norm", "Normalized Wasserstein"),
            ("variance_ratio", "Variance Ratio"),
        ]

        for metric, ylabel in metrics_info:
            out = obj_dir / f"cross_well_violin_{metric}.png"
            plot_metric_violin(detail_df, metric, ylabel, codecs_in_data, out)
            print(f"  Saved: {out.name}")

        for metric, ylabel in [("ks_stat", "KS Statistic"), ("cohens_d", "Cohen's d")]:
            out = obj_dir / f"cross_well_gt_feature_ranking_{metric}.png"
            plot_gt_feature_ranking(detail_df, args.gt_codec, metric, ylabel, out)
            print(f"  Saved: {out.name}")

        for metric, ylabel in metrics_info:
            out = obj_dir / f"cross_well_compression_delta_{metric}.png"
            plot_compression_delta(detail_df, args.gt_codec, metric, ylabel, codecs_in_data, out)
            print(f"  Saved: {out.name}")

        for metric, ylabel in metrics_info:
            out = obj_dir / f"cross_well_by_feature_group_{metric}.png"
            plot_by_feature_group(detail_df, metric, ylabel, codecs_in_data, out)
            print(f"  Saved: {out.name}")

        if not prof_df.empty:
            out = obj_dir / "cross_well_profile_correlation.png"
            plot_profile_correlation_bar(prof_df, codecs_in_data, out)
            print(f"  Saved: {out.name}")

        out = obj_dir / "cross_well_by_plate.png"
        plot_by_plate(detail_df, "ks_stat", "KS Statistic", codecs_in_data, out)
        print(f"  Saved: {out.name}")

        # ── Replicate correlation ─────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Computing per-feature replicate correlation ({object_type}, GT) ...")
        print(f"{'='*60}")
        replicate_df = compute_replicate_correlations_from_raw(
            well_groups, args.gt_codec, features_base, object_type, args.min_cells
        )
        if replicate_df.empty:
            print(f"  WARNING: No replicate data for {object_type}")
            continue

        rep_csv = obj_dir / "cross_well_replicate_correlation.csv"
        replicate_df.to_csv(rep_csv, index=False)
        print(f"  Saved {len(replicate_df)} features to: {rep_csv}")
        print(f"  Replicate corr: mean={replicate_df['replicate_corr'].mean():.4f}  "
              f"median={replicate_df['replicate_corr'].median():.4f}")
        print(f"  By group:")
        for group, gdf in replicate_df.groupby("feature_group"):
            print(f"    {group:25s}: median={gdf['replicate_corr'].median():.4f}  "
                  f"n={len(gdf)}")

        # Greenlist: features with replicate correlation >= 0.9
        greenlist = replicate_df[replicate_df["replicate_corr"] >= 0.9].sort_values(
            "replicate_corr", ascending=False
        )
        greenlist_csv = obj_dir / f"greenlist_features_{object_type}.csv"
        greenlist.to_csv(greenlist_csv, index=False)
        print(f"\n  GREENLIST (replicate_corr >= 0.9): {len(greenlist)}/{len(replicate_df)} features")
        print(f"  Saved to: {greenlist_csv}")
        print(f"  By group:")
        for group, gdf in sorted(
            greenlist.groupby("feature_group"),
            key=lambda x: -len(x[1]),
        ):
            total_in_group = len(replicate_df[replicate_df["feature_group"] == group])
            print(f"    {group:25s}: {len(gdf):3d}/{total_in_group:3d} "
                  f"({100*len(gdf)/total_in_group:.0f}%)")

        # Redlist: features with replicate correlation < 0.5
        redlist = replicate_df[replicate_df["replicate_corr"] < 0.5].sort_values(
            "replicate_corr", ascending=True
        )
        redlist_csv = obj_dir / f"redlist_features_{object_type}.csv"
        redlist.to_csv(redlist_csv, index=False)
        print(f"\n  REDLIST (replicate_corr < 0.5): {len(redlist)}/{len(replicate_df)} features")
        print(f"  Saved to: {redlist_csv}")
        print(f"  By group:")
        for group, gdf in sorted(
            redlist.groupby("feature_group"),
            key=lambda x: -len(x[1]),
        ):
            total_in_group = len(replicate_df[replicate_df["feature_group"] == group])
            print(f"    {group:25s}: {len(gdf):3d}/{total_in_group:3d} "
                  f"({100*len(gdf)/total_in_group:.0f}%)")

        # ── Codec correlation analysis ────────────────────────────
        codec_corr_path = Path(args.codec_correlation_csv)
        if codec_corr_path.exists():
            print(f"\n{'='*60}")
            print(f"Cross-well vs codec feature correlation ({object_type}) ...")
            print(f"{'='*60}")
            codec_corr_df = pd.read_csv(codec_corr_path)
            print(f"  Loaded {len(codec_corr_df)} features from {codec_corr_path}")
            plot_crosswell_vs_codec_correlation(
                summary_df, codec_corr_df, args.gt_codec, obj_dir
            )

            print(f"\n{'='*60}")
            print(f"Replicate correlation vs codec feature correlation ({object_type}) ...")
            print(f"{'='*60}")
            plot_replicate_vs_codec_correlation(
                replicate_df, codec_corr_df, obj_dir
            )
        else:
            print(f"\n  Codec correlation CSV not found at {codec_corr_path}, skipping.")

        print(f"\nDone with {object_type}. Outputs in: {obj_dir}")

    print(f"\nAll done. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
