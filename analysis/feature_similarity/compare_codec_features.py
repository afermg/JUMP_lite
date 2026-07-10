#!/usr/bin/env python3
"""
Compare per-cell features between ground truth and compressed codecs.

Uses instance mappings to align cells between GT and each codec,
then computes feature correlations.

Usage:
    python compare_codec_features.py --mappings-dir /path/to/instance_mappings/
    python compare_codec_features.py --object-type nuclei --n-samples 10
"""

import argparse
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from tqdm import tqdm


# Known quality order for JPEG XL codecs (best to worst quality, left to right)
CODEC_QUALITY_ORDER = {
    'jpegxl_lossy_hq.zarr': 0,
    'jpegxl_lossy_effort_3.zarr': 1,
    'jpegxl_lossy_d2_e8.zarr': 2,
    'jpegxl_lossy_mq.zarr': 3,
    'jpegxl_lossy_lq.zarr': 4,
    'jpegxl_lossy_d10.zarr': 5,
    'jpegxl_lossy_d15.zarr': 6,
    'jpegxl_lossy_d20_e2.zarr': 7,
    'jpegxl_lossy_d30.zarr': 8,
}


def sort_codecs_by_quality(codecs: list[str]) -> list[str]:
    """Sort codecs by known quality order (best to worst). Unknown codecs go at the end."""
    max_order = max(CODEC_QUALITY_ORDER.values()) + 1
    return sorted(codecs, key=lambda c: CODEC_QUALITY_ORDER.get(c, max_order))


@lru_cache(maxsize=1024)  # Increased cache for better performance with many sources
def load_cell_features_cached(path_str: str, object_type: str = "cell") -> pl.DataFrame:
    """Cached version of load_cell_features."""
    return _load_cell_features_impl(Path(path_str), object_type)


def load_cell_features(path: Path, object_type: str = "cell") -> pl.DataFrame:
    """Load per-cell features (cached wrapper)."""
    return load_cell_features_cached(str(path), object_type)


def _load_cell_features_impl(path: Path, object_type: str = "cell") -> pl.DataFrame:
    """Load per-cell features, pivot to one row per cell.

    Includes all feature types:
    - sizeshape: from channel 0 only (same across channels)
    - intensity, texture, zernike, etc.: from all channels (prefixed with channel)
    """
    df = pl.read_parquet(path)
    df = df.filter(pl.col("object") == object_type)

    # Parse branch into channel and feature_type
    df = df.with_columns([
        pl.col("branch").str.extract(r"^(\d+)/").alias("channel"),
        pl.col("branch").str.extract(r"/max/(.+)$").alias("feature_type"),
    ])

    # For sizeshape/ferret, use channel 0 only (mask-based, same across channels)
    # For others (intensity-based), use all channels with channel prefix
    mask_based = df.filter(
        (pl.col("feature_type").is_in(["sizeshape", "ferret"])) &
        (pl.col("channel") == "0")
    ).select(["label", "metric", "value"])

    intensity_based = df.filter(
        ~pl.col("feature_type").is_in(["sizeshape", "ferret"])
    ).with_columns(
        (pl.col("channel") + "_" + pl.col("metric")).alias("metric")
    ).select(["label", "metric", "value"])

    # Combine and pivot - use pivot_wider which is faster in recent polars versions
    combined = pl.concat([mask_based, intensity_based])
    # Optimize pivot by sorting first (significantly speeds up pivot)
    combined = combined.sort(["label", "metric"])
    return combined.pivot(on="metric", index="label", values="value")


def load_instance_mapping(path: Path, thresh: float = 0.5) -> pl.DataFrame:
    """Load instance mapping parquet, filter to threshold."""
    df = pl.read_parquet(path)
    if "thresh" in df.columns:
        closest = min(df["thresh"].unique().to_list(), key=lambda x: abs(x - thresh))
        df = df.filter(pl.col("thresh") == closest)
    # Keep only TP and BELOW_THRESH (matched cells)
    df = df.filter(pl.col("match_type").is_in(["TP", "BELOW_THRESH"]))
    return df.select(["source_id", "file", "gt_id", "pred_id", "iou_score"])


def get_matched_features(
    gt_features: pl.DataFrame,
    codec_features: pl.DataFrame,
    mapping: pl.DataFrame,
) -> pl.DataFrame | None:
    """Join GT and codec features using instance mapping."""
    gt_with_map = gt_features.join(
        mapping.select(["gt_id", "pred_id"]).cast({"gt_id": pl.Int64, "pred_id": pl.Int64}),
        left_on="label",
        right_on="gt_id",
    )
    merged = gt_with_map.join(
        codec_features,
        left_on="pred_id",
        right_on="label",
        suffix="_codec",
    )
    return merged if len(merged) > 0 else None


def compute_feature_correlations(
    gt_features: pl.DataFrame,
    codec_features: pl.DataFrame,
    mapping: pl.DataFrame,
) -> dict:
    """Compute per-feature correlations between GT and codec."""
    n_gt_cells = len(gt_features)
    n_codec_cells = len(codec_features)
    n_mapped = len(mapping)

    # Join GT features with mapping
    gt_with_map = gt_features.join(
        mapping.select(["gt_id", "pred_id"]).cast({"gt_id": pl.Int64, "pred_id": pl.Int64}),
        left_on="label",
        right_on="gt_id",
    )

    # Join with codec features
    merged = gt_with_map.join(
        codec_features,
        left_on="pred_id",
        right_on="label",
        suffix="_codec",
    )

    n_matched = len(merged)

    if n_matched == 0:
        return {
            "n_gt_cells": n_gt_cells,
            "n_codec_cells": n_codec_cells,
            "n_mapped": n_mapped,
            "n_matched": 0,
            "match_rate": 0.0,
            "correlations": {},
        }

    # Get feature columns (exclude label, gt_id, pred_id)
    feature_cols = [c for c in gt_features.columns if c != "label"]

    # GPU-accelerated correlation using PyTorch (fully vectorized)
    correlations = {}

    # Get pairs of columns (gt, codec) that exist
    valid_cols = [(c, f"{c}_codec") for c in feature_cols if f"{c}_codec" in merged.columns]

    if valid_cols:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gt_cols = [c[0] for c in valid_cols]
        codec_cols = [c[1] for c in valid_cols]

        # Convert to numpy, then torch tensors on GPU
        gt_matrix = merged.select(gt_cols).to_numpy()
        codec_matrix = merged.select(codec_cols).to_numpy()

        # Replace NaN with 0 and create mask
        nan_mask = np.isnan(gt_matrix) | np.isnan(codec_matrix)
        gt_matrix = np.nan_to_num(gt_matrix, nan=0.0)
        codec_matrix = np.nan_to_num(codec_matrix, nan=0.0)

        gt_tensor = torch.from_numpy(gt_matrix).float().to(device)
        codec_tensor = torch.from_numpy(codec_matrix).float().to(device)
        mask_tensor = torch.from_numpy(~nan_mask).float().to(device)

        # Fully vectorized: compute all correlations at once
        # Shapes: (n_cells, n_features)
        n_valid = mask_tensor.sum(dim=0)  # (n_features,)

        # Masked mean per feature
        gt_masked = gt_tensor * mask_tensor
        codec_masked = codec_tensor * mask_tensor
        gt_mean = gt_masked.sum(dim=0) / n_valid.clamp(min=1)
        codec_mean = codec_masked.sum(dim=0) / n_valid.clamp(min=1)

        # Center values and apply mask
        gt_centered = (gt_tensor - gt_mean) * mask_tensor
        codec_centered = (codec_tensor - codec_mean) * mask_tensor

        # Correlation = cov / (std_x * std_y)
        numer = (gt_centered * codec_centered).sum(dim=0)
        denom = torch.sqrt((gt_centered ** 2).sum(dim=0) * (codec_centered ** 2).sum(dim=0))

        # Compute correlations where valid
        corr_values = torch.where(denom > 0, numer / denom, torch.zeros_like(numer))
        valid_mask = n_valid >= 2

        # Transfer to CPU and build dict
        corr_cpu = corr_values.cpu().numpy()
        valid_cpu = valid_mask.cpu().numpy()
        correlations = {col: float(corr_cpu[i]) for i, col in enumerate(gt_cols) if valid_cpu[i]}

    return {
        "n_gt_cells": n_gt_cells,
        "n_codec_cells": n_codec_cells,
        "n_mapped": n_mapped,
        "n_matched": n_matched,
        "match_rate": n_matched / n_gt_cells if n_gt_cells > 0 else 0.0,
        "correlations": correlations,
    }


def get_feature_group(feature_name: str) -> str:
    """Categorize feature into a group based on its name.

    Uses pattern matching based on CellProfiler measurement naming conventions.
    """
    # Strip channel prefix if present (e.g., "0_IntegratedIntensity" -> "IntegratedIntensity")
    if len(feature_name) > 2 and feature_name[0].isdigit() and feature_name[1] == "_":
        base_name = feature_name[2:]
    else:
        base_name = feature_name

    # Texture features (Haralick texture features)
    texture_patterns = [
        "AngularSecondMoment", "Contrast", "Correlation", "DifferenceEntropy",
        "DifferenceVariance", "Entropy", "InfoMeas1", "InfoMeas2",
        "InverseDifferenceMoment", "SumAverage", "SumEntropy", "SumVariance", "Variance"
    ]
    if any(base_name.startswith(p) for p in texture_patterns):
        return "Texture"

    # Intensity features
    intensity_patterns = [
        "IntegratedIntensity", "MeanIntensity", "StdIntensity", "MaxIntensity",
        "MinIntensity", "MedianIntensity", "MADIntensity", "MassDisplacement",
        "LowerQuartileIntensity", "UpperQuartileIntensity", "Location_Center",
        "Location_Max"
    ]
    if any(p in base_name for p in intensity_patterns):
        return "Intensity"

    # Radial distribution features
    if "RadialDistribution" in base_name or base_name.startswith("FracAtD") or \
       base_name.startswith("MeanFrac") or base_name.startswith("RadialCV"):
        return "RadialDistribution"

    # Zernike features
    if "Zernike" in base_name:
        return "Zernike"

    # Granularity features
    if base_name.startswith("Granularity"):
        return "Granularity"

    # SizeShape features (default for non-prefixed features from sizeshape branch)
    sizeshape_patterns = [
        "Area", "Perimeter", "FormFactor", "Solidity", "Extent", "Eccentricity",
        "MajorAxisLength", "MinorAxisLength", "Orientation", "Compactness",
        "EulerNumber", "Center_X", "Center_Y", "BoundingBox", "EquivalentDiameter",
        "ConvexArea", "MaxFeretDiameter", "MinFeretDiameter", "MaximumRadius",
        "MeanRadius", "MedianRadius", "MinimumRadius", "Moment", "InertiaTensor"
    ]
    if any(base_name.startswith(p) for p in sizeshape_patterns):
        return "SizeShape"

    return "Other"


def gpu_correlations(gt_matrix: np.ndarray, codec_matrix: np.ndarray, col_names: list[str],
                      return_diagnostics: bool = False) -> dict[str, float] | tuple[dict[str, float], dict]:
    """Compute per-feature correlations using GPU (vectorized)."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nan_mask = np.isnan(gt_matrix) | np.isnan(codec_matrix)
    gt_clean = np.nan_to_num(gt_matrix, nan=0.0)
    codec_clean = np.nan_to_num(codec_matrix, nan=0.0)

    # Use double precision (float64) to handle large moment features
    gt_tensor = torch.from_numpy(gt_clean).double().to(device)
    codec_tensor = torch.from_numpy(codec_clean).double().to(device)
    mask_tensor = torch.from_numpy(~nan_mask).double().to(device)

    n_valid = mask_tensor.sum(dim=0)
    gt_masked = gt_tensor * mask_tensor
    codec_masked = codec_tensor * mask_tensor
    gt_mean = gt_masked.sum(dim=0) / n_valid.clamp(min=1)
    codec_mean = codec_masked.sum(dim=0) / n_valid.clamp(min=1)

    gt_centered = (gt_tensor - gt_mean) * mask_tensor
    codec_centered = (codec_tensor - codec_mean) * mask_tensor

    numer = (gt_centered * codec_centered).sum(dim=0)
    denom = torch.sqrt((gt_centered ** 2).sum(dim=0) * (codec_centered ** 2).sum(dim=0))

    # Compute variance for diagnostics
    gt_var = (gt_centered ** 2).sum(dim=0) / n_valid.clamp(min=1)
    codec_var = (codec_centered ** 2).sum(dim=0) / n_valid.clamp(min=1)

    # Compute std for diagnostics
    gt_std = torch.sqrt(gt_var)
    codec_std = torch.sqrt(codec_var)

    corr_values = torch.where(denom > 0, numer / denom, torch.full_like(numer, float('nan')))
    valid_mask = n_valid >= 2

    corr_cpu = corr_values.cpu().numpy()
    valid_cpu = valid_mask.cpu().numpy()
    n_valid_cpu = n_valid.cpu().numpy()
    gt_var_cpu = gt_var.cpu().numpy()
    codec_var_cpu = codec_var.cpu().numpy()
    gt_std_cpu = gt_std.cpu().numpy()
    codec_std_cpu = codec_std.cpu().numpy()
    gt_mean_cpu = gt_mean.cpu().numpy()
    codec_mean_cpu = codec_mean.cpu().numpy()
    numer_cpu = numer.cpu().numpy()
    denom_cpu = denom.cpu().numpy()

    correlations = {
        col: float(corr_cpu[i]) for i, col in enumerate(col_names)
        if valid_cpu[i] and not np.isnan(corr_cpu[i])
    }

    if return_diagnostics:
        # Compute median for each feature (CPU operation since we need numpy)
        gt_medians = []
        codec_medians = []
        for i in range(gt_matrix.shape[1]):
            # Get valid (non-NaN) values for this feature
            valid_mask_col = ~nan_mask[:, i]
            if valid_mask_col.sum() > 0:
                gt_medians.append(np.median(gt_matrix[valid_mask_col, i]))
                codec_medians.append(np.median(codec_matrix[valid_mask_col, i]))
            else:
                gt_medians.append(np.nan)
                codec_medians.append(np.nan)

        diagnostics = {
            col: {
                'correlation': float(corr_cpu[i]),
                'n_valid': int(n_valid_cpu[i]),
                'nan_rate': float(1 - n_valid_cpu[i] / len(gt_matrix)),
                'gt_mean': float(gt_mean_cpu[i]),
                'gt_median': float(gt_medians[i]),
                'gt_std': float(gt_std_cpu[i]),
                'gt_var': float(gt_var_cpu[i]),
                'codec_mean': float(codec_mean_cpu[i]),
                'codec_median': float(codec_medians[i]),
                'codec_std': float(codec_std_cpu[i]),
                'codec_var': float(codec_var_cpu[i]),
                'numerator': float(numer_cpu[i]),
                'denominator': float(denom_cpu[i]),
            }
            for i, col in enumerate(col_names)
            if valid_cpu[i]
        }
        return correlations, diagnostics

    return correlations


def compute_correlations_from_merged(
    merged_dfs: dict[str, list[pl.DataFrame]],
    codecs: list[str],
    level: str = "cell",
    return_diagnostics: bool = False,
) -> tuple[dict, dict, dict, dict, dict] | tuple[dict, dict, dict, dict, dict, dict]:
    """Compute correlations from matched cell data.

    Args:
        merged_dfs: Dict of codec -> list of merged DataFrames (one per site)
        codecs: List of codec names
        level: "cell" for cell-level or "site" for site-level (median per site)
        return_diagnostics: If True, return NaN diagnostics

    Returns:
        mean_correlations, total_matched, total_gt_cells, match_rates, features_missing_in_parquet, [diagnostics]
    """
    all_correlations = {codec: {} for codec in codecs}
    all_diagnostics = {codec: {} for codec in codecs}
    total_matched = {codec: 0 for codec in codecs}
    total_gt_cells = {codec: 0 for codec in codecs}
    match_rates = {codec: [] for codec in codecs}

    if level == "site":
        # Site-level: compute median per site, then correlate across sites
        def process_codec_site(codec):
            if codec not in merged_dfs or not merged_dfs[codec]:
                return codec, {}, {}, 0, 0, [], 0

            # Add site_id and concatenate all at once
            all_merged = []
            matched = 0
            gt_cells = 0
            rates = []
            for site_idx, (merged, n_gt) in enumerate(merged_dfs[codec]):
                matched += len(merged)
                gt_cells += n_gt
                rates.append(len(merged) / n_gt if n_gt > 0 else 0)
                all_merged.append(merged.with_columns(pl.lit(site_idx).alias("_site_id")))

            if not all_merged:
                return codec, {}, {}, matched, gt_cells, rates, 0

            combined = pl.concat(all_merged)

            # Get feature columns
            gt_cols = [c for c in combined.columns if not c.endswith("_codec") and c not in ["label", "pred_id", "_site_id"]]
            codec_cols = [f"{c}_codec" for c in gt_cols if f"{c}_codec" in combined.columns]
            gt_cols_before = len(gt_cols)
            gt_cols = [c for c in gt_cols if f"{c}_codec" in combined.columns]
            missing_in_codec = gt_cols_before - len(gt_cols)

            # Single group_by + median
            gt_medians = combined.group_by("_site_id").agg([pl.col(c).median() for c in gt_cols]).sort("_site_id")
            codec_medians = combined.group_by("_site_id").agg([pl.col(c).median() for c in codec_cols]).sort("_site_id")

            gt_matrix = gt_medians.select(gt_cols).to_numpy()
            codec_matrix = codec_medians.select(codec_cols).to_numpy()

            if return_diagnostics:
                corrs, diags = gpu_correlations(gt_matrix, codec_matrix, gt_cols, return_diagnostics=True)
                return codec, corrs, diags, matched, gt_cells, rates, missing_in_codec
            else:
                corrs = gpu_correlations(gt_matrix, codec_matrix, gt_cols)
                return codec, corrs, {}, matched, gt_cells, rates, missing_in_codec

        with ThreadPoolExecutor(max_workers=len(codecs)) as executor:
            results = list(executor.map(process_codec_site, codecs))

        # Track reasons for feature loss
        features_missing_in_parquet = {}
        for codec, corrs, diags, matched, gt_cells, rates, missing_count in results:
            all_correlations[codec] = corrs
            all_diagnostics[codec] = diags
            total_matched[codec] = matched
            total_gt_cells[codec] = gt_cells
            match_rates[codec] = rates
            features_missing_in_parquet[codec] = missing_count

        mean_correlations = {codec: all_correlations[codec] for codec in codecs}

    else:
        # Cell-level: correlate all matched cells across all sites
        def process_codec_cell(codec):
            if codec not in merged_dfs or not merged_dfs[codec]:
                return codec, {}, {}, 0, 0, [], 0

            all_merged = []
            matched = 0
            gt_cells = 0
            rates = []
            for merged, n_gt in merged_dfs[codec]:
                matched += len(merged)
                gt_cells += n_gt
                rates.append(len(merged) / n_gt if n_gt > 0 else 0)
                all_merged.append(merged)

            if not all_merged:
                return codec, {}, {}, matched, gt_cells, rates, 0

            combined = pl.concat(all_merged)

            # Diagnostic: Report columns in merged dataframe (only for first codec)
            if 'lq' in codec or codec == codecs[0]:
                all_cols = list(combined.columns)
                gt_only_cols = [c for c in all_cols if not c.endswith("_codec") and c not in ["label", "pred_id", "pred_id_right"]]
                codec_only_cols = [c for c in all_cols if c.endswith("_codec")]
                print(f"\n  DEBUG {codec}: After merging {len(all_merged)} sites:")
                print(f"    Total columns in merged dataframe: {len(all_cols)}")
                print(f"    GT feature columns: {len(gt_only_cols)}")
                print(f"    Codec feature columns: {len(codec_only_cols)}")

            gt_cols = [c for c in combined.columns if not c.endswith("_codec") and c not in ["label", "pred_id"]]
            codec_cols = [f"{c}_codec" for c in gt_cols if f"{c}_codec" in combined.columns]
            gt_cols_before = len(gt_cols)
            gt_cols = [c for c in gt_cols if f"{c}_codec" in combined.columns]
            missing_in_codec = gt_cols_before - len(gt_cols)

            gt_matrix = combined.select(gt_cols).to_numpy()
            codec_matrix = combined.select(codec_cols).to_numpy()

            if return_diagnostics:
                corrs, diags = gpu_correlations(gt_matrix, codec_matrix, gt_cols, return_diagnostics=True)
                return codec, corrs, diags, matched, gt_cells, rates, missing_in_codec
            else:
                corrs = gpu_correlations(gt_matrix, codec_matrix, gt_cols)
                return codec, corrs, {}, matched, gt_cells, rates, missing_in_codec

        with ThreadPoolExecutor(max_workers=len(codecs)) as executor:
            results = list(executor.map(process_codec_cell, codecs))

        # Track reasons for feature loss
        features_missing_in_parquet = {}
        for codec, corrs, diags, matched, gt_cells, rates, missing_count in results:
            all_correlations[codec] = corrs
            all_diagnostics[codec] = diags
            total_matched[codec] = matched
            total_gt_cells[codec] = gt_cells
            match_rates[codec] = rates
            features_missing_in_parquet[codec] = missing_count

        mean_correlations = {codec: all_correlations[codec] for codec in codecs}

    if return_diagnostics:
        return mean_correlations, total_matched, total_gt_cells, match_rates, all_diagnostics, features_missing_in_parquet
    else:
        return mean_correlations, total_matched, total_gt_cells, match_rates, features_missing_in_parquet


def generate_plot_and_csv(
    mean_correlations: dict,
    total_matched: dict,
    total_gt_cells: dict,
    common_features: list[str],
    codecs: list[str],
    object_type: str,
    level_str: str,
    output_path: Path,
):
    """Generate heatmap, boxplot, and CSV for correlation results."""
    codec_names = [c.replace(".zarr", "").replace("jpegxl_lossy_", "") for c in codecs]
    corr_matrix = np.array([[mean_correlations[c].get(f, np.nan) for c in codecs] for f in common_features])

    # Dynamic range based on data
    valid_corrs = corr_matrix[~np.isnan(corr_matrix)]
    if len(valid_corrs) > 0:
        vmin = max(0, np.percentile(valid_corrs, 1) - 0.05)
        vmax = min(1, np.percentile(valid_corrs, 99) + 0.02)
    else:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(common_features) * 0.15)))

    # Heatmap
    ax1 = axes[0]
    im = ax1.imshow(corr_matrix, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax1.set_xticks(range(len(codec_names)))
    ax1.set_xticklabels(codec_names, rotation=45, ha="right")
    ax1.set_yticks(range(len(common_features)))
    ax1.set_yticklabels(common_features, fontsize=6)
    ax1.set_xlabel("Codec")
    ax1.set_ylabel("Feature")
    title_suffix = f" ({object_type}, {level_str}-level)"
    ax1.set_title(f"Feature Correlation with GT{title_suffix}")
    plt.colorbar(im, ax=ax1, label="Pearson r")

    # Box plot
    ax2 = axes[1]
    box_data = [corr_matrix[:, i][~np.isnan(corr_matrix[:, i])] for i in range(len(codecs))]
    bp = ax2.boxplot(box_data, labels=codec_names, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax2.set_ylabel("Correlation with GT")
    ax2.set_xlabel("Codec")
    ax2.set_title(f"Correlation Distribution{title_suffix}")

    all_box_data = np.concatenate([d for d in box_data if len(d) > 0]) if box_data else np.array([0, 1])
    y_min = max(0, np.min(all_box_data) - 0.05)
    y_max = min(1.05, np.max(all_box_data) + 0.02)
    ax2.set_ylim(y_min, y_max)
    ax2.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)

    for i, codec in enumerate(codecs):
        rate = total_matched[codec] / max(1, total_gt_cells[codec]) * 100
        ax2.text(i + 1, y_min - 0.02, f"{rate:.0f}%\nmatch", ha="center", va="top", fontsize=8, color="gray")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Save CSV
    summary_data = {"feature": common_features}
    for codec in codecs:
        codec_short = codec.replace(".zarr", "").replace("jpegxl_lossy_", "")
        summary_data[codec_short] = [mean_correlations[codec].get(f, np.nan) for f in common_features]
    summary_df = pl.DataFrame(summary_data)
    csv_path = output_path.with_suffix(".csv")
    summary_df.write_csv(csv_path)

    return output_path, csv_path


def plot_correlation_violinplots(
    correlations: dict,
    codecs: list,
    features: list,
    level: str,
    output_prefix: str,
    large_diff_features: set = None,
    hq_codec: str = "jpegxl_lossy_hq.zarr",
    hq_threshold: float = 0.75
):
    """
    Create violin plots showing distribution of feature correlations.

    Generates 4 versions:
    1. All features
    2. Excluding features with large cell-site differences (if provided)
    3. Only high-quality features (correlation > threshold in HQ codec)
    4. Excluding RadialDistribution and Zernike features

    Args:
        correlations: Dict of codec -> feature -> correlation
        codecs: List of codec names
        features: List of feature names
        level: "cell" or "site"
        output_prefix: Output file prefix
        large_diff_features: Set of features to exclude (optional)
        hq_codec: Reference codec for quality filtering
        hq_threshold: Correlation threshold for quality filter
    """
    # Dynamic codec labeling
    codec_short_map = {c: c.replace('.zarr', '').replace('jpegxl_lossy_', 'jxl_') for c in codecs}
    _known_labels = {
        'jxl_lq': 'Low', 'jxl_mq': 'Medium', 'jxl_effort_3': 'Mid-High',
        'jxl_hq': 'High', 'jxl_d2_e8': 'D2 E8', 'jxl_d10': 'D10',
        'jxl_d15': 'D15', 'jxl_d20_e2': 'D20 E2', 'jxl_d30': 'D30',
    }
    codec_labels = {c: _known_labels.get(codec_short_map[c], codec_short_map[c]) for c in codecs}

    # Build DataFrame with all codecs present in input
    data = []
    for codec in codecs:
        for feature in features:
            corr = correlations[codec].get(feature, np.nan)
            if not np.isnan(corr):
                data.append({
                    'codec': codec_labels[codec],
                    'feature': feature,
                    'correlation': corr
                })

    if len(data) == 0:
        print(f"Warning: No data for {level}-level violin plots")
        return

    df = pd.DataFrame(data)

    # Order codecs by ascending mean correlation (worst -> best left to right)
    codec_mean_corr = df.groupby('codec')['correlation'].mean().sort_values(ascending=False)
    label_order = list(codec_mean_corr.index)

    # Version 1: All features
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.violinplot(
        data=df,
        x='codec',
        y='correlation',
        order=label_order,
        palette='viridis',
        inner='box',
        cut=0,
        ax=ax
    )
    ax.set_xlabel('Compression Quality', fontsize=16, fontweight='bold')
    ax.set_ylabel('Feature Correlation', fontsize=16, fontweight='bold')
    ax.set_title(f'{level.capitalize()}-level Feature Correlations - All Features (n={len(features)})',
                 fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(-0.1, 1.05)
    plt.tight_layout()
    output_path = f"{output_prefix}_{level}_violin_all.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {level}-level violin plot (all features) to: {output_path}")
    plt.close()

    # Version 2: Exclude large-diff features
    if large_diff_features is not None and len(large_diff_features) > 0:
        df_filtered = df[~df['feature'].isin(large_diff_features)]
        n_excluded = len(features) - df_filtered['feature'].nunique()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.violinplot(
            data=df_filtered,
            x='codec',
            y='correlation',
            order=label_order,
            palette='viridis',
            inner='box',
            cut=0,
            ax=ax
        )
        ax.set_xlabel('Compression Quality', fontsize=16, fontweight='bold')
        ax.set_ylabel('Feature Correlation', fontsize=16, fontweight='bold')
        ax.set_title(f'{level.capitalize()}-level Feature Correlations - Stable Features\n'
                     f'(n={df_filtered["feature"].nunique()}, excluded {n_excluded} with |cell-site diff|>0.1)',
                     fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_ylim(-0.1, 1.05)
        plt.tight_layout()
        output_path = f"{output_prefix}_{level}_violin_stable.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved {level}-level violin plot (stable features) to: {output_path}")
        plt.close()

    # Version 3: High-quality features only (HQ codec > threshold)
    if hq_codec in correlations:
        hq_features = [f for f in features if correlations[hq_codec].get(f, 0) > hq_threshold]
        df_hq = df[df['feature'].isin(hq_features)]

        if len(hq_features) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.violinplot(
                data=df_hq,
                x='codec',
                y='correlation',
                order=label_order,
                palette='viridis',
                inner='box',
                cut=0,
                ax=ax
            )
            ax.set_xlabel('Compression Quality', fontsize=16, fontweight='bold')
            ax.set_ylabel('Feature Correlation', fontsize=16, fontweight='bold')
            ax.set_title(f'{level.capitalize()}-level Feature Correlations - High Quality Features\n'
                         f'(n={len(hq_features)}, HQ correlation >{hq_threshold})',
                         fontsize=18, fontweight='bold')
            ax.tick_params(axis='both', labelsize=14)
            ax.set_ylim(-0.1, 1.05)
            plt.tight_layout()
            output_path = f"{output_prefix}_{level}_violin_highquality.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved {level}-level violin plot (high-quality features) to: {output_path}")
            plt.close()
        else:
            print(f"Warning: No {level}-level features with HQ correlation > {hq_threshold}")

    # Version 4: Exclude RadialDistribution and Zernike features
    radial_zernike_features = [f for f in features
                               if 'RadialDistribution' in f or 'Zernike' in f]
    if len(radial_zernike_features) > 0:
        df_no_radial = df[~df['feature'].isin(radial_zernike_features)]
        n_excluded = len(features) - df_no_radial['feature'].nunique()

        if df_no_radial['feature'].nunique() > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.violinplot(
                data=df_no_radial,
                x='codec',
                y='correlation',
                order=label_order,
                palette='viridis',
                inner='box',
                cut=0,
                ax=ax
            )
            ax.set_xlabel('Compression Quality', fontsize=16, fontweight='bold')
            ax.set_ylabel('Feature Correlation', fontsize=16, fontweight='bold')
            ax.set_title(f'{level.capitalize()}-level Feature Correlations - Excluding Radial/Zernike\n'
                         f'(n={df_no_radial["feature"].nunique()}, excluded {n_excluded} Radial/Zernike features)',
                         fontsize=18, fontweight='bold')
            ax.tick_params(axis='both', labelsize=14)
            ax.set_ylim(-0.1, 1.05)
            plt.tight_layout()
            output_path = f"{output_prefix}_{level}_violin_no_radial.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved {level}-level violin plot (no radial/zernike) to: {output_path}")
            plt.close()
        else:
            print(f"Warning: All features are RadialDistribution or Zernike at {level}-level")


def main():
    parser = argparse.ArgumentParser(description="Compare features across codecs")
    parser.add_argument("--mappings-dir", type=str, required=True,
                        help="Directory with instance mapping parquet files")
    parser.add_argument("--features-base", type=str,
                        default="/work/datasets/aliby_output/cp_measure/jump_target2_4plate",
                        help="Base path for feature profiles")
    parser.add_argument("--gt-codec", type=str, default="zstd.zarr",
                        help="Ground truth codec")
    parser.add_argument("--codecs", nargs="+",
                        default=None,
                        help="Comparison codecs (default: auto-discover from features-base)")
    parser.add_argument("--object-type", type=str, default="cell",
                        choices=["cell", "nuclei"], help="Object type")
    parser.add_argument("--thresh", type=float, default=0.5,
                        help="IoU threshold for instance matching")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of random source_ids to sample")
    parser.add_argument("--output", type=str, default="analysis/output/codec_feature_correlation.png",
                        help="Output path for plot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--features", nargs="+", default=None,
                        help="Specific features to analyze (e.g., Area Perimeter)")
    parser.add_argument("--feature-pattern", type=str, default=None,
                        help="Regex pattern to filter features (e.g., 'Area|Perimeter|Extent')")
    parser.add_argument("--list-features", action="store_true",
                        help="List available features and exit")
    parser.add_argument("--site-level", action="store_true",
                        help="Also run site-level analysis (median of matched cells per site)")
    parser.add_argument("--min-cells", type=int, default=5,
                        help="Minimum GT cell count per site (sites below this are skipped)")
    parser.add_argument("--filter-percentile", type=float, default=None,
                        help="Filter out sites in bottom and top N percentile of cell count (e.g., 5 filters bottom 5%% and top 5%%)")

    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    mappings_dir = Path(args.mappings_dir)
    features_base = Path(args.features_base)
    segment_step = "segment_cell" if args.object_type == "cell" else "segment_nuclei"

    # Auto-discover codecs if not specified
    if args.codecs is None:
        gt_codec = args.gt_codec
        all_codec_dirs = [d.name for d in features_base.iterdir()
                         if d.is_dir() and d.name.endswith(".zarr") and d.name != gt_codec]
        args.codecs = sort_codecs_by_quality(all_codec_dirs)
        print(f"Auto-discovered {len(args.codecs)} codecs: {args.codecs}")

    # Load mapping files for each codec
    codec_mappings = {}
    for codec in args.codecs:
        codec_base = codec.replace(".zarr", "")
        pq_file = mappings_dir / f"{segment_step}_{codec_base}.parquet"
        if pq_file.exists():
            codec_mappings[codec] = load_instance_mapping(pq_file, args.thresh)
            print(f"Loaded mapping for {codec}: {len(codec_mappings[codec])} matched cells")
        else:
            print(f"Warning: No mapping file for {codec}")

    if not codec_mappings:
        print("Error: No mapping files found")
        return

    # Get available source_ids from GT profiles
    gt_profiles_dir = features_base / args.gt_codec / "profiles"
    all_source_ids = [p.stem for p in gt_profiles_dir.glob("*.parquet")]

    # List features and exit if requested
    if args.list_features:
        sample_path = gt_profiles_dir / f"{all_source_ids[0]}.parquet"
        sample_features = load_cell_features(sample_path, args.object_type)
        feature_names = [c for c in sample_features.columns if c != "label"]
        print(f"Available features ({len(feature_names)}):")
        for f in sorted(feature_names):
            print(f"  {f}")
        return

    sample_source_ids = random.sample(all_source_ids, min(args.n_samples, len(all_source_ids)))

    # Check original feature count from GT
    sample_path = gt_profiles_dir / f"{all_source_ids[0]}.parquet"
    sample_features = load_cell_features(sample_path, args.object_type)
    original_feature_count = len([c for c in sample_features.columns if c != "label"])
    print(f"\nOriginal feature count in GT profiles: {original_feature_count}")

    # Check feature count in codec parquet files
    print(f"\nFeature count in codec parquet files:")
    for codec in args.codecs:
        codec_profiles_dir = features_base / codec / "profiles"
        codec_sample_path = codec_profiles_dir / f"{all_source_ids[0]}.parquet"
        if codec_sample_path.exists():
            codec_features = load_cell_features(codec_sample_path, args.object_type)
            codec_feature_count = len([c for c in codec_features.columns if c != "label"])
            codec_short = codec.replace('.zarr', '').replace('jpegxl_lossy_', '')
            print(f"  {codec_short}: {codec_feature_count} features")
            if codec_feature_count != original_feature_count:
                print(f"    WARNING: {abs(codec_feature_count - original_feature_count)} features different from GT!")

    # Apply percentile filtering if requested (similar to segmentation analysis)
    if args.filter_percentile is not None:
        print(f"\nApplying {args.filter_percentile}th percentile filtering on cell counts...")

        # Try to load cell counts from segmentation detailed CSV (much faster!)
        seg_step = "segment_cell" if args.object_type == "cell" else "segment_nuclei"
        seg_csv_pattern = mappings_dir.parent / f"*_{seg_step}_detailed.csv"
        seg_csv_files = list(mappings_dir.parent.glob(f"*_{seg_step}_detailed.csv"))

        print(f"  Looking for CSV files in: {mappings_dir.parent}")
        print(f"  Pattern: *_{seg_step}_detailed.csv")
        print(f"  Found {len(seg_csv_files)} CSV files")
        if len(seg_csv_files) > 0:
            for f in seg_csv_files:
                print(f"    - {f.name}")

        source_cell_counts = {}
        if len(seg_csv_files) > 0:
            print(f"  Loading cell counts from segmentation CSV: {seg_csv_files[0].name}")
            seg_df = pd.read_csv(seg_csv_files[0])
            print(f"  CSV has {len(seg_df)} rows, {len(seg_df['method'].unique())} methods")
            print(f"  Methods: {seg_df['method'].unique()}")
            # Filter to GT method (typically zstd.zarr)
            gt_methods = [m for m in seg_df['method'].unique() if 'zstd' in m.lower()]
            print(f"  GT methods found: {gt_methods}")
            if len(gt_methods) > 0:
                gt_seg = seg_df[seg_df['method'] == gt_methods[0]]
                print(f"  GT method has {len(gt_seg)} rows")
                unique_sources_in_csv = gt_seg['source_id'].unique()
                print(f"  Unique sources in CSV: {len(unique_sources_in_csv)}")
                print(f"  Sample sources to match: {len(sample_source_ids)}")
                overlap = set(unique_sources_in_csv) & set(sample_source_ids)
                print(f"  Overlap: {len(overlap)} sources")
                for _, row in gt_seg.iterrows():
                    if row['source_id'] in sample_source_ids:
                        source_cell_counts[row['source_id']] = row['n_true']
                print(f"  Loaded cell counts for {len(source_cell_counts)} sources from CSV")
            else:
                print(f"  Warning: No GT method with 'zstd' found in CSV")

        # Fallback: load from feature files if CSV not available
        if len(source_cell_counts) == 0:
            print(f"  Segmentation CSV not found, loading from feature files (slower)...")
            for source_id in tqdm(sample_source_ids, desc="Counting cells", unit="source"):
                gt_path = gt_profiles_dir / f"{source_id}.parquet"
                if gt_path.exists():
                    try:
                        gt_features = load_cell_features(gt_path, args.object_type)
                        source_cell_counts[source_id] = len(gt_features)
                    except Exception:
                        pass

        if len(source_cell_counts) > 0:
            counts = list(source_cell_counts.values())
            lower_bound = np.percentile(counts, args.filter_percentile)
            upper_bound = np.percentile(counts, 100 - args.filter_percentile)

            # Filter sources
            filtered_sources = [
                source_id for source_id, count in source_cell_counts.items()
                if lower_bound <= count <= upper_bound
            ]

            print(f"  Cell count range: {min(counts):.0f} - {max(counts):.0f}")
            print(f"  Percentile bounds: {lower_bound:.0f} - {upper_bound:.0f}")
            print(f"  Kept {len(filtered_sources)}/{len(sample_source_ids)} sources")
            print(f"  Filtered out: {len(sample_source_ids) - len(filtered_sources)} sources")

            sample_source_ids = filtered_sources
        else:
            print(f"  Warning: Could not load cell counts, skipping percentile filtering")

    # Check GPU availability once
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"\nProcessing {len(sample_source_ids)} samples for {args.object_type}...")

    # Collect matched cell data for all sites (only matched cells via instance mapping)
    merged_dfs: dict[str, list[tuple[pl.DataFrame, int]]] = {codec: [] for codec in args.codecs}

    def process_source(source_id: str) -> dict[str, tuple[pl.DataFrame, int] | None]:
        """Load and match features for one source_id across all codecs."""
        results = {}
        gt_path = gt_profiles_dir / f"{source_id}.parquet"
        if not gt_path.exists():
            return results
        gt_features = load_cell_features(gt_path, args.object_type)
        n_gt = len(gt_features)

        # Skip sites with too few GT cells
        if n_gt < args.min_cells:
            return results

        for codec in args.codecs:
            if codec not in codec_mappings:
                continue
            mapping = codec_mappings[codec].filter(pl.col("source_id") == source_id)
            if len(mapping) == 0:
                continue
            codec_path = features_base / codec / "profiles" / f"{source_id}.parquet"
            if not codec_path.exists():
                continue
            codec_features = load_cell_features(codec_path, args.object_type)
            merged = get_matched_features(gt_features, codec_features, mapping)
            if merged is not None:
                results[codec] = (merged, n_gt)
        return results

    t_start = time.time()
    print(f"  Loading data in parallel (this may take several minutes for large datasets)...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Use tqdm to show progress
        all_results = list(tqdm(
            executor.map(process_source, sample_source_ids),
            total=len(sample_source_ids),
            desc="Loading sources",
            unit="source"
        ))

    n_valid_sites = 0
    for results in all_results:
        if results:
            n_valid_sites += 1
        for codec, data in results.items():
            merged_dfs[codec].append(data)

    elapsed = time.time() - t_start
    print(f"\n✓ Data loading completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"  Sites with >= {args.min_cells} GT cells: {n_valid_sites}/{len(sample_source_ids)}")
    print(f"  Average time per source: {elapsed/len(sample_source_ids):.2f}s")

    # Determine which levels to run
    levels_to_run = ["cell"]
    if args.site_level:
        levels_to_run.append("site")

    output_base = Path(args.output)

    # Store results for comparison plot
    level_results = {}

    for level in levels_to_run:
        print(f"\n{'='*60}")
        print(f"Computing {level}-level correlations...")
        print(f"{'='*60}")

        t_corr = time.time()
        mean_correlations, total_matched, total_gt_cells, match_rates, diagnostics, features_missing_in_parquet = compute_correlations_from_merged(
            merged_dfs, args.codecs, level=level, return_diagnostics=True
        )
        print(f"Correlation computation took {time.time() - t_corr:.1f}s")

        # Get common features across all codecs
        feature_sets = [set(mean_correlations[c].keys()) for c in args.codecs if mean_correlations[c]]
        if not feature_sets:
            print(f"No features found for {level}-level analysis")
            continue

        # Diagnostic: Show feature counts per codec
        print(f"\nFeature availability per codec ({level}-level):")
        for i, codec in enumerate(args.codecs):
            codec_short = codec.replace('.zarr', '').replace('jpegxl_lossy_', '')
            if mean_correlations[codec]:
                n_features = len(mean_correlations[codec])
                missing_in_parquet = features_missing_in_parquet.get(codec, 0)
                lost_in_correlation = original_feature_count - missing_in_parquet - n_features
                print(f"  {codec_short}: {n_features} features")
                if missing_in_parquet > 0:
                    print(f"    → {missing_in_parquet} features missing from codec parquet files")
                if lost_in_correlation > 0:
                    print(f"    → {lost_in_correlation} features filtered during correlation (NaN/low variance)")

        common_features = sorted(set.intersection(*feature_sets))
        print(f"\nIntersection: {len(common_features)} features common to ALL codecs")
        print(f"  Loss from original: {original_feature_count} → {len(common_features)} "
              f"({100*(1-len(common_features)/original_feature_count):.1f}% lost)")

        # Show which features are missing in which codecs
        all_features_set = set.union(*feature_sets)
        all_features = sorted(all_features_set)
        missing_per_codec = {}
        for codec in args.codecs:
            if mean_correlations[codec]:
                codec_features = set(mean_correlations[codec].keys())
                missing = all_features_set - codec_features
                if len(missing) > 0:
                    missing_per_codec[codec] = missing

        if len(missing_per_codec) > 0:
            print(f"\nFeatures missing in some codecs:")
            for codec, missing in missing_per_codec.items():
                codec_short = codec.replace('.zarr', '').replace('jpegxl_lossy_', '')
                print(f"  {codec_short}: {len(missing)} features missing")
                if len(missing) <= 10:
                    print(f"    {', '.join(list(missing)[:10])}")
                else:
                    print(f"    (showing 10/{len(missing)}): {', '.join(list(missing)[:10])}")

        # Apply redlist: exclude problematic features
        REDLIST_PATTERNS = [
            'CentralMoment_0_1',  # Always 0 (Y-coordinate at origin)
            'CentralMoment_1_0',  # Always 0 (X-coordinate at origin)
            'ZernikePhase',       # Circular data, inappropriate for Pearson correlation
        ]

        original_count = len(common_features)
        excluded_features = []
        for pattern in REDLIST_PATTERNS:
            excluded = [f for f in common_features if pattern in f]
            excluded_features.extend(excluded)
            common_features = [f for f in common_features if pattern not in f]

        if len(excluded_features) > 0:
            print(f"\nRedlist: Excluded {len(excluded_features)} problematic features:")
            for pattern in REDLIST_PATTERNS:
                pattern_excluded = [f for f in excluded_features if pattern in f]
                if len(pattern_excluded) > 0:
                    print(f"  - {pattern}: {len(pattern_excluded)} features")
                    if len(pattern_excluded) <= 5:
                        for feat in pattern_excluded:
                            print(f"      {feat}")
            print(f"Remaining: {len(common_features)} features")

        # Filter features if specified
        if args.features:
            common_features = [f for f in common_features if f in args.features]
            print(f"Filtered to {len(common_features)} specified features")
        elif args.feature_pattern:
            import re
            pattern = re.compile(args.feature_pattern, re.IGNORECASE)
            common_features = [f for f in common_features if pattern.search(f)]
            print(f"Filtered to {len(common_features)} features matching '{args.feature_pattern}'")

        print(f"\nFound {len(common_features)} common features across all codecs")
        print("\nMatch statistics per codec:")
        for codec in args.codecs:
            avg_match_rate = np.mean(match_rates[codec]) if match_rates[codec] else 0
            print(f"  {codec}:")
            print(f"    Matched: {total_matched[codec]}/{total_gt_cells[codec]} cells ({100*total_matched[codec]/max(1,total_gt_cells[codec]):.1f}%)")
            print(f"    Avg match rate per image: {100*avg_match_rate:.1f}%")

        print(f"\nMean correlation per codec ({level}-level):")
        for codec in args.codecs:
            codec_short = codec.replace(".zarr", "").replace("jpegxl_lossy_", "")
            vals = [mean_correlations[codec].get(f, np.nan) for f in common_features]
            if vals:
                print(f"  {codec_short}: {np.nanmean(vals):.4f} (min={np.nanmin(vals):.4f}, max={np.nanmax(vals):.4f})")

        # Diagnostic: Check for large-value features that need double precision
        print(f"\nDiagnostic: Feature value ranges ({level}-level):")
        for codec in args.codecs:
            codec_short = codec.replace(".zarr", "").replace("jpegxl_lossy_", "")
            if codec in diagnostics and len(diagnostics[codec]) > 0:
                variances = [diag.get('gt_var', 0) for diag in diagnostics[codec].values()]
                max_var = max(variances)
                if max_var > 1e15:
                    large_var_features = [feat for feat, diag in diagnostics[codec].items()
                                         if diag.get('gt_var', 0) > 1e15]
                    print(f"  {codec_short}: max variance = {max_var:.2e} ({len(large_var_features)} features >1e15)")
                    print(f"    Note: Using double precision (float64) for these large moment features")
                    break

        # Diagnostic: Overall NaN statistics
        print(f"\nDiagnostic: Overall NaN statistics ({level}-level):")
        for codec in args.codecs:
            codec_short = codec.replace(".zarr", "").replace("jpegxl_lossy_", "")
            if codec in diagnostics and len(diagnostics[codec]) > 0:
                nan_rates = [diag.get('nan_rate', 0) * 100 for diag in diagnostics[codec].values()]
                print(f"  {codec_short}: mean NaN rate = {np.mean(nan_rates):.2f}% (min={np.min(nan_rates):.2f}%, max={np.max(nan_rates):.2f}%)")
                high_nan = [feat for feat, diag in diagnostics[codec].items()
                           if diag.get('nan_rate', 0) > 0.1 and feat in common_features]
                if len(high_nan) > 0:
                    print(f"    {len(high_nan)} features with >10% NaN rate")

        # Diagnostic: Check for features with suspiciously low correlations
        print(f"\nDiagnostic: Features with correlation near 0 ({level}-level):")
        zero_corr_features = {}
        for codec in args.codecs:
            codec_short = codec.replace(".zarr", "").replace("jpegxl_lossy_", "")
            near_zero = [f for f in common_features
                        if abs(mean_correlations[codec].get(f, np.nan)) < 0.01]
            if len(near_zero) > 0:
                zero_corr_features[codec_short] = near_zero
                print(f"  {codec_short}: {len(near_zero)} features with |r| < 0.01")
                if len(near_zero) <= 10:
                    for feat in near_zero:
                        corr = mean_correlations[codec].get(feat, np.nan)
                        diag = diagnostics[codec].get(feat, {})
                        nan_rate = diag.get('nan_rate', 0) * 100
                        gt_mean = diag.get('gt_mean', np.nan)
                        gt_median = diag.get('gt_median', np.nan)
                        gt_std = diag.get('gt_std', np.nan)
                        codec_mean = diag.get('codec_mean', np.nan)
                        codec_median = diag.get('codec_median', np.nan)
                        codec_std = diag.get('codec_std', np.nan)
                        # Diagnose cause
                        numer = diag.get('numerator', 0)
                        denom = diag.get('denominator', 1)

                        if gt_std < 1e-10 or codec_std < 1e-10:
                            cause = "CONSTANT (zero variance)"
                        elif gt_std < 1e-6 or codec_std < 1e-6:
                            cause = "NEAR-CONSTANT (very low variance)"
                        elif abs(gt_mean - codec_mean) / max(gt_std, 1e-10) > 3:
                            cause = "MEAN SHIFT (compression changed mean)"
                        elif nan_rate > 5:
                            cause = "HIGH NaN rate"
                        elif abs(numer) < denom * 0.01 and gt_std > 1e-6 and codec_std > 1e-6:
                            # Distributions similar but covariance near zero
                            # Since most features correlate well, matching is correct
                            # This is feature-specific: quantization, angles, or noise
                            if 'Phase' in feat:
                                cause = "CIRCULAR (phase angle, need circular correlation not Pearson)"
                            elif 'Frac' in feat or 'RadialDistribution' in feat:
                                cause = "QUANTIZED (limited range + compression noise breaks pairing)"
                            else:
                                cause = "NOISY (high measurement noise breaks cell-level pairing)"
                        else:
                            cause = "ORTHOGONAL (genuinely uncorrelated)"

                        print(f"    - {feat}: [{cause}]")
                        print(f"        r={corr:.6f}, NaN={nan_rate:.1f}%, covar={numer:.3e}, denom={denom:.3e}")
                        print(f"        GT:    mean={gt_mean:.6f}, median={gt_median:.6f}, std={gt_std:.6f}")
                        print(f"        Codec: mean={codec_mean:.6f}, median={codec_median:.6f}, std={codec_std:.6f}")
                else:
                    print(f"    (showing first 10)")
                    for feat in near_zero[:10]:
                        corr = mean_correlations[codec].get(feat, np.nan)
                        diag = diagnostics[codec].get(feat, {})
                        nan_rate = diag.get('nan_rate', 0) * 100
                        gt_mean = diag.get('gt_mean', np.nan)
                        gt_median = diag.get('gt_median', np.nan)
                        gt_std = diag.get('gt_std', np.nan)
                        codec_mean = diag.get('codec_mean', np.nan)
                        codec_median = diag.get('codec_median', np.nan)
                        codec_std = diag.get('codec_std', np.nan)
                        # Diagnose cause
                        numer = diag.get('numerator', 0)
                        denom = diag.get('denominator', 1)

                        if gt_std < 1e-10 or codec_std < 1e-10:
                            cause = "CONSTANT (zero variance)"
                        elif gt_std < 1e-6 or codec_std < 1e-6:
                            cause = "NEAR-CONSTANT (very low variance)"
                        elif abs(gt_mean - codec_mean) / max(gt_std, 1e-10) > 3:
                            cause = "MEAN SHIFT (compression changed mean)"
                        elif nan_rate > 5:
                            cause = "HIGH NaN rate"
                        elif abs(numer) < denom * 0.01 and gt_std > 1e-6 and codec_std > 1e-6:
                            # Distributions similar but covariance near zero
                            # Since most features correlate well, matching is correct
                            # This is feature-specific: quantization, angles, or noise
                            if 'Phase' in feat:
                                cause = "CIRCULAR (phase angle, need circular correlation not Pearson)"
                            elif 'Frac' in feat or 'RadialDistribution' in feat:
                                cause = "QUANTIZED (limited range + compression noise breaks pairing)"
                            else:
                                cause = "NOISY (high measurement noise breaks cell-level pairing)"
                        else:
                            cause = "ORTHOGONAL (genuinely uncorrelated)"

                        print(f"    - {feat}: [{cause}]")
                        print(f"        r={corr:.6f}, NaN={nan_rate:.1f}%, covar={numer:.3e}, denom={denom:.3e}")
                        print(f"        GT:    mean={gt_mean:.6f}, median={gt_median:.6f}, std={gt_std:.6f}")
                        print(f"        Codec: mean={codec_mean:.6f}, median={codec_median:.6f}, std={codec_std:.6f}")

        # Check if same features appear across codecs
        if len(zero_corr_features) > 1:
            all_zero_sets = [set(feats) for feats in zero_corr_features.values()]
            common_zero = set.intersection(*all_zero_sets)
            if len(common_zero) > 0:
                print(f"\n  WARNING: {len(common_zero)} features have near-zero correlation in ALL codecs:")
                for feat in list(common_zero)[:10]:
                    print(f"    - {feat}")
                if len(common_zero) > 10:
                    print(f"    ... and {len(common_zero) - 10} more")

        # Store for comparison
        level_results[level] = {
            "correlations": mean_correlations,
            "features": common_features,
            "total_matched": total_matched,
            "total_gt_cells": total_gt_cells,
        }

        # Generate output path with level suffix
        if level == "site":
            out_path = output_base.parent / f"{output_base.stem}_site{output_base.suffix}"
        else:
            out_path = output_base

        plot_path, csv_path = generate_plot_and_csv(
            mean_correlations, total_matched, total_gt_cells, common_features,
            args.codecs, args.object_type, level, out_path
        )
        print(f"\nSaved {level}-level plot to: {plot_path}")
        print(f"Saved {level}-level summary to: {csv_path}")

    # Generate cell vs site comparison plot if both levels were computed
    if args.site_level and "cell" in level_results and "site" in level_results:
        print(f"\n{'='*60}")
        print("Generating cell-level vs site-level comparison plot...")
        print(f"{'='*60}")

        cell_corrs = level_results["cell"]["correlations"]
        site_corrs = level_results["site"]["correlations"]

        # Get features common to both levels and all codecs
        common_features_both = sorted(
            set(level_results["cell"]["features"]) & set(level_results["site"]["features"])
        )

        # Define colors for feature groups
        group_colors = {
            "SizeShape": "#1f77b4",
            "Intensity": "#ff7f0e",
            "Texture": "#2ca02c",
            "Zernike": "#d62728",
            "RadialDistribution": "#9467bd",
            "Granularity": "#8c564b",
            "Other": "#7f7f7f",
        }

        # Create subplot for each codec
        n_codecs = len(args.codecs)
        n_cols = min(2, n_codecs)
        n_rows = (n_codecs + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)
        axes = axes.flatten()

        for idx, codec in enumerate(args.codecs):
            ax = axes[idx]
            codec_short = codec.replace(".zarr", "").replace("jpegxl_lossy_", "")

            # Collect data points
            for feature in common_features_both:
                cell_r = cell_corrs[codec].get(feature, np.nan)
                site_r = site_corrs[codec].get(feature, np.nan)
                if np.isnan(cell_r) or np.isnan(site_r):
                    continue

                group = get_feature_group(feature)
                color = group_colors.get(group, "#7f7f7f")
                ax.scatter(cell_r, site_r, c=color, alpha=0.5, s=15, label=group)

            # Plot diagonal line
            ax.plot([-1, 1], [-1, 1], "k--", alpha=0.3, lw=1)

            # Compute correlation between cell and site level
            cell_vals = [cell_corrs[codec].get(f, np.nan) for f in common_features_both]
            site_vals = [site_corrs[codec].get(f, np.nan) for f in common_features_both]
            valid = ~(np.isnan(cell_vals) | np.isnan(site_vals))
            if np.sum(valid) > 2:
                r = np.corrcoef(np.array(cell_vals)[valid], np.array(site_vals)[valid])[0, 1]
                ax.text(0.05, 0.95, f"r={r:.3f}", transform=ax.transAxes, fontsize=10,
                        verticalalignment="top", fontweight="bold")

            ax.set_xlabel("Cell-level correlation")
            ax.set_ylabel("Site-level correlation")
            ax.set_title(f"{codec_short}")
            ax.set_xlim(-1.03, 1.03)
            ax.set_ylim(-1.03, 1.03)
            ax.set_aspect("equal")

        # Hide unused subplots
        for idx in range(n_codecs, len(axes)):
            axes[idx].axis("off")

        # Create legend with unique groups
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=g)
                   for g, c in group_colors.items()]
        fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.12, 0.5), title="Feature Group")

        plt.suptitle(f"Cell-level vs Site-level Correlation ({args.object_type})", fontsize=14, fontweight="bold")
        plt.tight_layout()

        comparison_path = output_base.parent / f"{output_base.stem}_cell_vs_site{output_base.suffix}"
        plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved comparison plot to: {comparison_path}")

        # Compute feature ranking based on average correlation across cell/site and all codecs
        print(f"\n{'='*60}")
        print("Computing feature ranking (average across cell/site levels and all codecs)...")
        print(f"{'='*60}")

        feature_avg_corrs = {}
        for feature in common_features_both:
            corrs = []
            for codec in args.codecs:
                cell_r = cell_corrs[codec].get(feature, np.nan)
                site_r = site_corrs[codec].get(feature, np.nan)
                if not np.isnan(cell_r):
                    corrs.append(cell_r)
                if not np.isnan(site_r):
                    corrs.append(site_r)

            if len(corrs) > 0:
                feature_avg_corrs[feature] = np.mean(corrs)

        # Sort by average correlation (descending)
        ranked_features = sorted(feature_avg_corrs.items(), key=lambda x: x[1], reverse=True)

        print(f"\nTop 50 features ranked by average correlation (cell+site, all codecs):")
        print(f"{'Rank':<6} {'Feature':<60} {'Avg Corr':<10} {'Group':<20}")
        print("-" * 96)
        for rank, (feature, avg_corr) in enumerate(ranked_features[:50], 1):
            group = get_feature_group(feature)
            print(f"{rank:<6} {feature:<60} {avg_corr:.4f}     {group:<20}")

        # Save full ranking to CSV
        ranking_df = pd.DataFrame([
            {
                'rank': rank,
                'feature': feature,
                'avg_correlation': avg_corr,
                'feature_group': get_feature_group(feature),
                **{f'{codec.replace(".zarr", "")}_cell': cell_corrs[codec].get(feature, np.nan)
                   for codec in args.codecs},
                **{f'{codec.replace(".zarr", "")}_site': site_corrs[codec].get(feature, np.nan)
                   for codec in args.codecs}
            }
            for rank, (feature, avg_corr) in enumerate(ranked_features, 1)
        ])

        ranking_csv = output_base.parent / f"{output_base.stem}_feature_ranking.csv"
        ranking_df.to_csv(ranking_csv, index=False)
        print(f"\nSaved full feature ranking to: {ranking_csv}")

        # Identify features with large cell-site differences (>0.1) per codec
        print(f"\n{'='*60}")
        print("Features with large cell-site correlation differences (|diff| > 0.1)...")
        print(f"{'='*60}")

        # Per-codec analysis
        features_by_codec = {}
        for codec in args.codecs:
            codec_short = codec.replace(".zarr", "")
            codec_features = []

            for feature in common_features_both:
                cell_r = cell_corrs[codec].get(feature, np.nan)
                site_r = site_corrs[codec].get(feature, np.nan)

                if not np.isnan(cell_r) and not np.isnan(site_r):
                    diff = abs(cell_r - site_r)
                    if diff > 0.1:
                        codec_features.append({
                            'feature': feature,
                            'diff': diff,
                            'cell_corr': cell_r,
                            'site_corr': site_r,
                            'feature_group': get_feature_group(feature)
                        })

            # Sort by difference (descending)
            codec_features.sort(key=lambda x: x['diff'], reverse=True)
            features_by_codec[codec] = codec_features

            # Print per-codec summary
            print(f"\n{codec_short}: {len(codec_features)} features with |diff| > 0.1")
            if len(codec_features) > 0:
                print(f"  Top 10:")
                print(f"  {'Feature':<50} {'Diff':<8} {'Cell':<8} {'Site':<8} {'Group':<15}")
                print("  " + "-" * 99)
                for feat in codec_features[:10]:
                    print(f"  {feat['feature']:<50} {feat['diff']:.4f}   {feat['cell_corr']:.4f}   "
                          f"{feat['site_corr']:.4f}   {feat['feature_group']:<15}")

                # Save per-codec CSV
                codec_df = pd.DataFrame(codec_features)
                codec_csv = output_base.parent / f"{output_base.stem}_cell_site_large_diff_{codec_short}.csv"
                codec_df.to_csv(codec_csv, index=False)
                print(f"  Saved to: {codec_csv}")

        # Find features present in ALL codecs (intersection)
        print(f"\n{'='*60}")
        print("Features with |diff| > 0.1 in ALL codecs (intersection)...")
        print(f"{'='*60}")

        # Get sets of features for each codec
        feature_sets = {codec: set(f['feature'] for f in features_by_codec[codec])
                       for codec in args.codecs}

        # Find intersection
        if len(feature_sets) > 0:
            common_diff_features = set.intersection(*feature_sets.values())

            if len(common_diff_features) > 0:
                # Compile info for common features
                common_features_info = []
                for feature in common_diff_features:
                    feat_info = {
                        'feature': feature,
                        'feature_group': get_feature_group(feature),
                        'avg_diff': np.mean([abs(cell_corrs[codec].get(feature, np.nan) -
                                                site_corrs[codec].get(feature, np.nan))
                                            for codec in args.codecs]),
                        'max_diff': max([abs(cell_corrs[codec].get(feature, np.nan) -
                                            site_corrs[codec].get(feature, np.nan))
                                        for codec in args.codecs]),
                        'cell_avg': np.mean([cell_corrs[codec].get(feature, np.nan) for codec in args.codecs]),
                        'site_avg': np.mean([site_corrs[codec].get(feature, np.nan) for codec in args.codecs]),
                    }

                    # Add per-codec values
                    for codec in args.codecs:
                        codec_short = codec.replace(".zarr", "")
                        feat_info[f'{codec_short}_diff'] = abs(cell_corrs[codec].get(feature, np.nan) -
                                                               site_corrs[codec].get(feature, np.nan))
                        feat_info[f'{codec_short}_cell'] = cell_corrs[codec].get(feature, np.nan)
                        feat_info[f'{codec_short}_site'] = site_corrs[codec].get(feature, np.nan)

                    common_features_info.append(feat_info)

                # Sort by average difference
                common_features_info.sort(key=lambda x: x['avg_diff'], reverse=True)

                print(f"\nFound {len(common_features_info)} features with |diff| > 0.1 in ALL codecs:")
                print(f"{'Rank':<6} {'Feature':<55} {'Avg Diff':<10} {'Max Diff':<10} {'Cell Avg':<10} {'Site Avg':<10} {'Group':<15}")
                print("-" * 125)
                for rank, feat in enumerate(common_features_info, 1):
                    print(f"{rank:<6} {feat['feature']:<55} {feat['avg_diff']:.4f}     {feat['max_diff']:.4f}     "
                          f"{feat['cell_avg']:.4f}     {feat['site_avg']:.4f}     {feat['feature_group']:<15}")

                # Save intersection CSV
                common_df = pd.DataFrame(common_features_info)
                common_csv = output_base.parent / f"{output_base.stem}_cell_site_large_diff_all_codecs.csv"
                common_df.to_csv(common_csv, index=False)
                print(f"\nSaved features common to all codecs to: {common_csv}")
            else:
                print("\nNo features with |diff| > 0.1 found in ALL codecs.")
                common_diff_features = set()
        else:
            print("\nNo codecs with large differences to analyze.")
            common_diff_features = set()

        # Generate violin plots for each level now that we have large_diff_features
        print(f"\n{'='*60}")
        print("Generating violin plots...")
        print(f"{'='*60}")

        # Use stem to remove extension from output path
        violin_prefix = str(output_base.parent / output_base.stem)

        for level in ["cell", "site"]:
            if level in level_results:
                print(f"\nGenerating {level}-level violin plots...")
                plot_correlation_violinplots(
                    correlations=level_results[level]["correlations"],
                    codecs=args.codecs,
                    features=level_results[level]["features"],
                    level=level,
                    output_prefix=violin_prefix,
                    large_diff_features=common_diff_features,
                    hq_codec="jpegxl_lossy_hq.zarr",
                    hq_threshold=0.95
                )
    else:
        # Only one level computed, generate violin plots without large_diff_features
        # Use stem to remove extension from output path
        violin_prefix = str(output_base.parent / output_base.stem)

        for level in levels_to_run:
            if level in level_results:
                print(f"\nGenerating {level}-level violin plots...")
                plot_correlation_violinplots(
                    correlations=level_results[level]["correlations"],
                    codecs=args.codecs,
                    features=level_results[level]["features"],
                    level=level,
                    output_prefix=violin_prefix,
                    large_diff_features=None,
                    hq_codec="jpegxl_lossy_hq.zarr",
                    hq_threshold=0.95
                )


if __name__ == "__main__":
    main()
