"""
Phenotypic Activity Analysis Script
Converted from Marimo notebook to standalone Python script with functions
"""

import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import islice
from math import pow
from pathlib import Path, PosixPath

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from polars import selectors as cs
from trommel.core import basic_cleanup
from umap import UMAP
import pandas as pd

import os
import glob

from copairs import map
from copairs.matching import assign_reference_index
from copairs.map.average_precision import p_values
from broad_babel.data import get_table
import seaborn as sns


def load_metadata(target_plates=None):
    """Load well metadata for Target-2 plates using broad_babel + local compound metadata."""
    # Get well metadata from broad_babel (has plate/well mapping to JCP2022)
    well_metadata = get_table('well').to_pandas()

    # Filter wells to target plates if specified
    if target_plates:
        well_metadata = well_metadata[well_metadata['Metadata_Plate'].isin(target_plates)]

    # Load local compound metadata which has pert names and control types
    # This file has the compound details (names, targets, control types, etc.)
    local_compound_meta = pd.read_csv("../input/JUMP-Target-2_compound_metadata.tsv", sep="\t").drop_duplicates()

    # The local file uses broad_sample, but we need to map it to JCP2022
    # Get the platemap which has the broad_sample mapping
    plate_map = pd.read_csv("../input/JUMP-Target-2_compound_platemap.tsv", sep="\t")

    # Merge platemap with compound metadata to get well_position -> compound info
    plate_map_annotated = plate_map.merge(local_compound_meta, on="broad_sample", how="left")

    # Now merge with well_metadata on well position
    full_metadata = well_metadata.merge(
        plate_map_annotated,
        left_on='Metadata_Well',
        right_on='well_position',
        how='left'
    )

    # Add Metadata_ prefix to new columns
    rename_cols = {col: f'Metadata_{col}' for col in plate_map_annotated.columns
                   if not col.startswith('Metadata_') and col not in ['well_position', 'broad_sample']}
    full_metadata = full_metadata.rename(columns=rename_cols)

    print(f"Loaded metadata for {len(full_metadata)} wells across {full_metadata['Metadata_Plate'].nunique()} plates")
    if 'Metadata_pert_iname' in full_metadata.columns:
        print(f"  - {full_metadata['Metadata_pert_iname'].notna().sum()} wells with compound annotations")
    else:
        print(f"  - WARNING: No pert_iname column found in merged metadata")
        print(f"  - Available columns: {[c for c in full_metadata.columns if c.startswith('Metadata')]}")

    return full_metadata


def find_available_codecs(output_dir="../output"):
    """Find all available codec parquet files in consistent order."""
    parquet_files = glob.glob(f"{output_dir}/*_clean.parquet")
    available_codecs = [Path(f).stem.replace("_clean", "") for f in parquet_files]

    # Define standard codec order (matching feature_correlation_cp_measure_script.py)
    standard_order = [
        "jpegxl_lossy_lq",
        "jpegxl_lossy_mq",
        "jpegxl_lossy_effort_3",
        "jpegxl_lossy_hq",
        "zstd"
    ]

    # Return codecs in standard order, only including those that exist
    codecs = [c for c in standard_order if c in available_codecs]

    # Add any codecs not in standard order at the end (for future extensibility)
    remaining = [c for c in available_codecs if c not in standard_order]
    codecs.extend(sorted(remaining))

    return codecs


def load_features(codec="zstd"):
    """Load features from parquet file."""
    # Try _clean.parquet first, fallback to .parquet
    try:
        clean_zarr = pd.read_parquet(f"../output/{codec}_clean.parquet")
    except FileNotFoundError:
        clean_zarr = pd.read_parquet(f"../output/{codec}.parquet")

    # Drop 'nr' column if it exists
    if "nr" in clean_zarr.columns:
        clean_zarr = clean_zarr.drop(columns="nr")

    return clean_zarr


def merge_features_metadata(clean_zarr, well_metadata):
    """Merge features with well metadata."""
    df = clean_zarr.merge(
        well_metadata,
        on=["Metadata_Plate", "Metadata_Well"],
        how="left"
    )

    print(f"Merged {len(df)} rows, {df['Metadata_pert_iname'].notna().sum()} with compound info")

    return df


def fill_missing_metadata(df):
    """Fill missing values in metadata columns."""
    df["Metadata_target_list"] = df["Metadata_target_list"].fillna("unknown")
    df["Metadata_control_type"] = df["Metadata_control_type"].fillna("trt")

    return df


def aggregate_to_median(df):
    """Compute median aggregation by well."""
    # Define grouping columns - only use those that exist
    grouping_cols = [
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_pert_iname",
        "Metadata_pert_type",
        "Metadata_control_type",
    ]

    # Add optional columns if they exist
    optional_cols = ["Metadata_Source", "Metadata_Batch", "Metadata_Site", "Metadata_target_list"]
    for col in optional_cols:
        if col in df.columns:
            grouping_cols.append(col)

    # Get only numeric feature columns (exclude ALL metadata columns)
    feature_cols = [col for col in df.columns if not col.startswith('Metadata') and col not in ['well_position', 'broad_sample', 'solvent']]

    df_median = df.groupby(grouping_cols, as_index=False)[feature_cols].median()

    return df_median


def add_negcon_indicator(df_median):
    """Add negative control indicator column."""
    df_median["Metadata_negcon"] = df_median["Metadata_control_type"] == "negcon"

    return df_median


def calculate_average_precision(df_median, pos_sameby, pos_diffby, neg_sameby, neg_diffby):
    """Calculate average precision for phenotypic activity."""
    metadata = df_median.filter(regex="^Metadata")
    profiles = df_median.filter(regex="^(?!Metadata)").values

    activity_ap = map.average_precision(
        metadata, profiles, pos_sameby, pos_diffby, neg_sameby, neg_diffby
    )
    activity_ap = activity_ap.query("Metadata_pert_iname != 'DMSO'")  # remove DMSO

    return activity_ap


def calculate_pvalues(activity_ap, null_size=10_000, seed=0):
    """Calculate p-values and add derived columns."""
    activity_ap["p_value"] = p_values(activity_ap, null_size=null_size, seed=seed)
    activity_ap["-log10(p-value)"] = -activity_ap["p_value"].apply(np.log10)
    activity_ap["below_p"] = activity_ap["p_value"] < 0.05

    active_ratio_ap = activity_ap["below_p"].mean()

    return activity_ap, active_ratio_ap


def plot_replicate_retrieval(activity_ap, active_ratio_ap, codec_name, output_path="../output/replicate_retrieval_{codec}.png"):
    """Plot average precision vs -log10 p-values."""
    fig, axes = plt.subplots(1, 1, figsize=(14, 14))

    axes.scatter(
        data=activity_ap,
        x="average_precision",
        y="-log10(p-value)",
        c="below_p",
        cmap="tab10",
        s=10,
    )
    axes.axhline(-np.log10(0.05), color="black", linestyle="--")
    axes.set_xlabel("AP")
    axes.set_ylabel("-log10(p-value)")
    axes.set_title(f"Replicate retrieval - {codec_name}")
    axes.text(
        0.65,
        1.5,
        f"Retrieved = {100 * active_ratio_ap:.2f}%",
        va="center",
        ha="left",
    )

    plt.tight_layout()
    output_path = output_path.format(codec=codec_name)
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")
    plt.close()


def calculate_mean_average_precision(activity_ap, pos_sameby, null_size=10_000, threshold=0.05, seed=0):
    """Calculate mean average precision."""
    activity_map = map.mean_average_precision(
        activity_ap, pos_sameby, null_size=null_size, threshold=threshold, seed=seed
    )
    activity_map["-log10(p-value)"] = -activity_map["corrected_p_value"].apply(np.log10)

    return activity_map



def calculate_target_consistency(df_median, neg_sameby, neg_diffby, min_compounds_per_target=2):
    """
    Calculate phenotypic consistency for compounds grouped by target.
    Uses copairs to measure how well compounds with the same target retrieve each other.

    Args:
        df_median: DataFrame with aggregated features and metadata
        neg_sameby: List of columns to match negative controls
        neg_diffby: List of columns to differentiate negatives
        min_compounds_per_target: Minimum number of compounds per target to analyze

    Returns:
        Dictionary with target consistency metrics and detailed results
    """
    print("\n" + "="*80)
    print("TARGET-BASED PHENOTYPIC CONSISTENCY ANALYSIS")
    print("="*80)
    
    # Step 1: Get consensus profiles
    
    feature_cols = [c for c in df_median.columns if not c.startswith("Metadata")]
    df_consensus = df_median.groupby(
        ["Metadata_pert_iname", "Metadata_target_list", "Metadata_negcon"], as_index=False
    )[feature_cols].median()
    df_consensus["Metadata_target"] = df_consensus["Metadata_target_list"].str.split("|")
    df_consensus.head()
    
    n_negcons = df_consensus['Metadata_negcon'].sum()
    print(f"Included {n_negcons} negative controls for matching")

    # Step 3: Calculate mAP using targets as the grouping variable
    # We use target as pos_sameby - compounds with same target should retrieve each other
    print("\nCalculating mean average precision per target...")

    metadata = df_consensus.filter(regex="^Metadata")
    profiles = df_consensus.filter(regex="^(?!Metadata)").values

    pos_sameby_target = ["Metadata_target"]  # Group by target
    pos_diffby_target = []  # No additional differentiation needed
    neg_sameby_target = []
    neg_diffby_target = ["Metadata_target"]  # Use same negative differentiation as before

    # try:
    # Calculate AP for target-based retrieval
    target_ap = map.multilabel.average_precision(
        metadata,
        profiles,
        pos_sameby=pos_sameby_target,
        pos_diffby=pos_diffby_target,
        neg_sameby=neg_sameby_target,
        neg_diffby=neg_diffby_target,
        multilabel_col="Metadata_target",
    )

    # Calculate mAP per target
    target_map = map.mean_average_precision(
        target_ap,
        pos_sameby_target,
        null_size=10_000,
        threshold=0.05,
        seed=0
    )

    # Add derived columns
    target_map["-log10(p-value)"] = -target_map["corrected_p_value"].apply(np.log10)
    target_map["below_corrected_p"] = target_map["corrected_p_value"] < 0.05
    valid_targets = target_map.shape[0]
    
    # Calculate summary statistics
    mean_target_map = target_map['mean_average_precision'].mean()
    median_target_map = target_map['mean_average_precision'].median()
    pct_significant = (target_map['below_corrected_p'].sum() / len(target_map)) * 100

    print("\nTarget Consistency Results:")
    print(f"  - Mean target mAP: {mean_target_map:.4f}")
    print(f"  - Median target mAP: {median_target_map:.4f}")
    print(f"  - % significant targets (p<0.05): {pct_significant:.2f}%")
    print(f"  - Targets below corrected p-value: {target_map['below_corrected_p'].sum()}/{len(target_map)}")

    return {
        "n_targets": valid_targets,
        "n_compound_target_pairs": len(df_consensus),
        "target_map": target_map,
        "mean_target_map": mean_target_map,
        "median_target_map": median_target_map,
        "pct_significant_targets": pct_significant,
        "n_significant_targets": target_map['below_corrected_p'].sum(),
    }

    # except Exception as e:
    #     print(f"\nERROR: Failed to calculate target consistency: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return {
    #         "n_targets": len(valid_targets),
    #         "n_compound_target_pairs": len(df_targets),
    #         "target_map": None,
    #         "mean_target_map": None,
    #         "median_target_map": None,
    #         "pct_significant_targets": None,
    #     }


def plot_summary_heatmap(summary_df, all_activity_map, output_path="../output/phenotypic_activity_heatmap.png"):
    """
    Create a color-coded heatmap of phenotypic activity metrics across codecs.
    Each metric category gets its own subplot (row).

    Args:
        summary_df: DataFrame with summary metrics
        all_activity_map: Dict of activity_map DataFrames per codec
        output_path: Path to save the heatmap
    """
    # Include ALL codecs, not just successful ones
    df_plot = summary_df.copy()

    if len(df_plot) == 0:
        print("No codecs to plot")
        return

    # Sort by standard codec order (matching feature_correlation_cp_measure_script.py)
    standard_order = [
        "jpegxl_lossy_lq",
        "jpegxl_lossy_mq",
        "jpegxl_lossy_effort_3",
        "jpegxl_lossy_hq",
        "zstd"
    ]

    # Create a categorical type with the standard order
    df_plot['codec'] = pd.Categorical(df_plot['codec'], categories=standard_order, ordered=True)
    df_plot = df_plot.sort_values('codec').reset_index(drop=True)

    # Calculate additional MAP metrics from activity_map
    map_metrics = []
    for codec in df_plot['codec']:
        if codec in all_activity_map and all_activity_map[codec] is not None:
            activity_map = all_activity_map[codec]
            map_metrics.append({
                'codec': codec,
                'mean_map': activity_map['mean_average_precision'].mean(),
                'median_map': activity_map['mean_average_precision'].median(),
                'pct_below_corrected_p_compound': (activity_map['below_corrected_p'].sum() / len(activity_map)) * 100,
                'n_perturbations': len(activity_map),  # Number of unique perturbations
            })

    map_df = pd.DataFrame(map_metrics)
    df_with_map = df_plot.merge(map_df, on='codec', how='left')

    # Prepare data for subplots - organize by category
    codecs = df_with_map['codec'].tolist()

    # Category 1: AP-based metrics (from compound-level average precision)
    ap_metrics = df_with_map[['codec', 'mean_ap', 'median_ap', 'active_ratio_ap']].set_index('codec').T
    ap_metrics.index = ['Mean AP', 'Median AP', 'Active Ratio\n(p<0.05)']

    # Category 2: MAP-based metrics (from mean average precision)
    map_metrics_df = df_with_map[['codec', 'mean_map', 'median_map', 'below_corrected_p']].set_index('codec').T
    map_metrics_df.index = ['Mean mAP', 'Median mAP', 'Below Corrected\np-value (mAP)']

    # Category 3: Target consistency metrics
    target_metrics = df_with_map[['codec', 'mean_target_map', 'median_target_map', 'pct_significant_targets']].set_index('codec').T
    target_metrics.index = ['Mean Target mAP', 'Median Target mAP', '% Significant\nTargets']

    # Category 4: Compound-level metrics
    # Note: This section doesn't have mean/median, just percentage and normalized count
    compound_metrics = df_with_map[['codec', 'pct_below_corrected_p_compound', 'n_compounds']].set_index('codec').T
    # Normalize n_compounds for color scale
    compound_metrics.loc['n_compounds_norm'] = compound_metrics.loc['n_compounds'] / compound_metrics.loc['n_compounds'].max()
    compound_metrics = compound_metrics.drop('n_compounds')
    compound_metrics.index = ['% Significant\nCompounds', 'N Compounds\n(normalized)']

    # Create figure with subplots (one row per category) - added one more for target consistency
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), gridspec_kw={'height_ratios': [3, 3, 3, 2, 1.5]})

    # Subplot 1: AP-based metrics (continuous color scale per row)
    # Create a custom heatmap by plotting each row separately
    for idx, (metric_name, row_data) in enumerate(ap_metrics.iterrows()):
        # Use continuous scale from 0 to row max (not normalized 0-1)
        row_max = row_data.max()

        # Plot this row
        for col_idx, (codec, value) in enumerate(row_data.items()):
            # Use gray for missing values (NaN)
            if pd.isna(value):
                color = (0.9, 0.9, 0.9)  # light gray
                text = 'N/A'
            else:
                # Map value to 0-1 range for colormap using actual values
                color_val = value / row_max if row_max > 0 else 0
                color = plt.cm.RdYlGn(color_val)
                text = f'{value:.4f}'

            rect = plt.Rectangle((col_idx, len(ap_metrics) - idx - 1), 1, 1,
                                  facecolor=color, edgecolor='white', linewidth=1)
            axes[0].add_patch(rect)
            # Add annotation
            axes[0].text(col_idx + 0.5, len(ap_metrics) - idx - 0.5, text,
                        ha='center', va='center', fontsize=10, weight='bold',
                        color='gray' if pd.isna(value) else 'black')

    axes[0].set_xlim(0, len(ap_metrics.columns))
    axes[0].set_ylim(0, len(ap_metrics))
    axes[0].set_xticks(np.arange(len(ap_metrics.columns)) + 0.5)
    axes[0].set_xticklabels(ap_metrics.columns, rotation=45, ha='right')
    axes[0].set_yticks(np.arange(len(ap_metrics)) + 0.5)
    axes[0].set_yticklabels(ap_metrics.index[::-1])
    axes[0].set_title('Average Precision (AP) Metrics - Compound Level\n(Continuous color scale: 0 to max per row)',
                      fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')

    # Subplot 2: MAP-based metrics (continuous color scale per row)
    for idx, (metric_name, row_data) in enumerate(map_metrics_df.iterrows()):
        # Use continuous scale from 0 to row max (not normalized 0-1)
        row_max = row_data.max()

        for col_idx, (codec, value) in enumerate(row_data.items()):
            # Use gray for missing values (NaN)
            if pd.isna(value):
                color = (0.9, 0.9, 0.9)  # light gray
                text = 'N/A'
            else:
                # Map value to 0-1 range for colormap using actual values
                color_val = value / row_max if row_max > 0 else 0
                color = plt.cm.RdYlGn(color_val)
                text = f'{value:.4f}'

            rect = plt.Rectangle((col_idx, len(map_metrics_df) - idx - 1), 1, 1,
                                  facecolor=color, edgecolor='white', linewidth=1)
            axes[1].add_patch(rect)
            axes[1].text(col_idx + 0.5, len(map_metrics_df) - idx - 0.5, text,
                        ha='center', va='center', fontsize=10, weight='bold',
                        color='gray' if pd.isna(value) else 'black')

    axes[1].set_xlim(0, len(map_metrics_df.columns))
    axes[1].set_ylim(0, len(map_metrics_df))
    axes[1].set_xticks(np.arange(len(map_metrics_df.columns)) + 0.5)
    axes[1].set_xticklabels(map_metrics_df.columns, rotation=45, ha='right')
    axes[1].set_yticks(np.arange(len(map_metrics_df)) + 0.5)
    axes[1].set_yticklabels(map_metrics_df.index[::-1])
    axes[1].set_title('Mean Average Precision (mAP) Metrics - Perturbation Level\n(Continuous color scale: 0 to max per row)',
                      fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')

    # Subplot 3: Target consistency metrics (continuous color scale per row)
    for idx, (metric_name, row_data) in enumerate(target_metrics.iterrows()):
        # Use continuous scale from 0 to row max (not normalized 0-1)
        row_max = row_data.max()

        for col_idx, (codec, value) in enumerate(row_data.items()):
            # Use gray for missing values (NaN)
            if pd.isna(value):
                color = (0.9, 0.9, 0.9)  # light gray
                text = 'N/A'
            else:
                # Map value to 0-1 range for colormap using actual values
                color_val = value / row_max if row_max > 0 else 0
                color = plt.cm.RdYlGn(color_val)
                text = f'{value:.4f}' if 'mAP' in metric_name else f'{value:.2f}'

            rect = plt.Rectangle((col_idx, len(target_metrics) - idx - 1), 1, 1,
                                  facecolor=color, edgecolor='white', linewidth=1)
            axes[2].add_patch(rect)
            axes[2].text(col_idx + 0.5, len(target_metrics) - idx - 0.5, text,
                        ha='center', va='center', fontsize=10, weight='bold',
                        color='gray' if pd.isna(value) else 'black')

    axes[2].set_xlim(0, len(target_metrics.columns))
    axes[2].set_ylim(0, len(target_metrics))
    axes[2].set_xticks(np.arange(len(target_metrics.columns)) + 0.5)
    axes[2].set_xticklabels(target_metrics.columns, rotation=45, ha='right')
    axes[2].set_yticks(np.arange(len(target_metrics)) + 0.5)
    axes[2].set_yticklabels(target_metrics.index[::-1])
    axes[2].set_title('Target-Based Phenotypic Consistency\n(Continuous color scale: 0 to max per row)',
                      fontsize=12, fontweight='bold', pad=10)
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')

    # Subplot 4: Compound-level metrics (continuous color scale per row)
    for idx, (metric_name, row_data) in enumerate(compound_metrics.iterrows()):
        # Use continuous scale from 0 to row max (not normalized 0-1)
        row_max = row_data.max()

        for col_idx, (codec, value) in enumerate(row_data.items()):
            # Use gray for missing values (NaN)
            if pd.isna(value):
                color = (0.9, 0.9, 0.9)  # light gray
                text = 'N/A'
            else:
                # Map value to 0-1 range for colormap using actual values
                color_val = value / row_max if row_max > 0 else 0
                color = plt.cm.RdYlGn(color_val)
                text = f'{value:.2f}'

            rect = plt.Rectangle((col_idx, len(compound_metrics) - idx - 1), 1, 1,
                                  facecolor=color, edgecolor='white', linewidth=1)
            axes[3].add_patch(rect)
            axes[3].text(col_idx + 0.5, len(compound_metrics) - idx - 0.5, text,
                        ha='center', va='center', fontsize=10, weight='bold',
                        color='gray' if pd.isna(value) else 'black')

    axes[3].set_xlim(0, len(compound_metrics.columns))
    axes[3].set_ylim(0, len(compound_metrics))
    axes[3].set_xticks(np.arange(len(compound_metrics.columns)) + 0.5)
    axes[3].set_xticklabels(compound_metrics.columns, rotation=45, ha='right')
    axes[3].set_yticks(np.arange(len(compound_metrics)) + 0.5)
    axes[3].set_yticklabels(compound_metrics.index[::-1])
    axes[3].set_title('Compound-Level Statistics\n(Continuous color scale: 0 to max per row)',
                      fontsize=12, fontweight='bold', pad=10)
    axes[3].set_xlabel('')
    axes[3].set_ylabel('')

    # Subplot 5: Sample size table (N Compounds, N Perturbations, N Targets, Status)
    axes[4].axis('off')

    # Create horizontal table with four rows
    table_data = []

    # Row 1: N Compounds
    n_compounds_row = ['N Compounds'] + [f'{int(row["n_compounds"]):,}' if pd.notna(row["n_compounds"]) else 'N/A'
                                          for _, row in df_with_map.iterrows()]
    table_data.append(n_compounds_row)

    # Row 2: N Perturbations
    n_perturbations_row = ['N Perturbations'] + [f'{int(row["n_perturbations"]):,}' if pd.notna(row.get("n_perturbations")) else 'N/A'
                                                   for _, row in df_with_map.iterrows()]
    table_data.append(n_perturbations_row)

    # Row 3: N Targets
    n_targets_row = ['N Targets'] + [f'{int(row["n_targets"]):,}' if pd.notna(row.get("n_targets")) else 'N/A'
                                      for _, row in df_with_map.iterrows()]
    table_data.append(n_targets_row)

    # Row 4: Status
    status_row = ['Status'] + [row['status'] for _, row in df_with_map.iterrows()]
    table_data.append(status_row)

    col_labels = ['Metric'] + codecs

    table = axes[4].table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.15] + [0.15] * len(codecs)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for row_idx in [1, 2, 3, 4]:
        table[(row_idx, 0)].set_facecolor('#e0e0e0')
        table[(row_idx, 0)].set_text_props(weight='bold')
        # Alternate row colors
        if row_idx % 2 == 0:
            for col_idx in range(1, len(col_labels)):
                table[(row_idx, col_idx)].set_facecolor('#f9f9f9')
        # Color code status row
        if row_idx == 4:  # Status row
            for col_idx in range(1, len(col_labels)):
                status = df_with_map.iloc[col_idx-1]['status']
                if status == 'success':
                    table[(row_idx, col_idx)].set_facecolor('#d4edda')  # light green
                elif 'skipped' in status:
                    table[(row_idx, col_idx)].set_facecolor('#fff3cd')  # light yellow
                else:
                    table[(row_idx, col_idx)].set_facecolor('#f8d7da')  # light red

    # Overall title
    fig.suptitle('Phenotypic Activity Summary Across Compression Codecs',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved heatmap to: {output_path}")
    plt.close()


def process_single_codec(codec, well_metadata, pos_sameby, pos_diffby, neg_sameby, neg_diffby):
    """Process phenotypic activity analysis for a single codec."""
    print(f"\n{'='*80}")
    print(f"Processing codec: {codec}")
    print(f"{'='*80}")

    print("Loading features...")
    clean_zarr = load_features(codec=codec)

    print("Merging features with metadata...")
    df = merge_features_metadata(clean_zarr, well_metadata)

    print(f"DataFrame shape: {df.shape}")
    print("\nControl type counts:")
    print(df.Metadata_control_type.value_counts())

    print("Filling missing metadata...")
    df = fill_missing_metadata(df)

    print("Aggregating to median...")
    df_median = aggregate_to_median(df)

    print("Adding negative control indicator...")
    df_median = add_negcon_indicator(df_median)

    # Check if we have negative controls
    if 'Metadata_negcon' not in df_median.columns or df_median['Metadata_negcon'].sum() == 0:
        print(f"\nWARNING: No negative controls found for codec {codec}")
        print("Skipping phenotypic activity analysis for this codec")
        return {
            "codec": codec,
            "active_ratio_ap": None,
            "below_corrected_p": None,
            "n_compounds": len(df_median),
            "mean_ap": None,
            "median_ap": None,
            "activity_ap": None,
            "activity_map": None,
            "status": "skipped_no_negcon",
        }

    print("\nCalculating average precision...")
    try:
        activity_ap = calculate_average_precision(
            df_median, pos_sameby, pos_diffby, neg_sameby, neg_diffby
        )
    except Exception as e:
        print(f"\nERROR: Failed to calculate average precision for codec {codec}: {e}")
        print("Skipping this codec")
        return {
            "codec": codec,
            "active_ratio_ap": None,
            "below_corrected_p": None,
            "n_compounds": len(df_median),
            "mean_ap": None,
            "median_ap": None,
            "activity_ap": None,
            "activity_map": None,
            "status": f"error: {str(e)}",
        }

    print("Calculating p-values...")
    activity_ap, active_ratio_ap = calculate_pvalues(activity_ap)

    print(f"\nActive ratio (AP): {active_ratio_ap:.4f}")

    print("\nPlotting replicate retrieval...")
    plot_replicate_retrieval(activity_ap, active_ratio_ap, codec)

    print("\nCalculating mean average precision...")
    activity_map = calculate_mean_average_precision(activity_ap, pos_sameby)

    print(f"Below corrected p-value ratio: {activity_map.below_corrected_p.mean():.4f}")

    # Calculate target-based phenotypic consistency
    print("\nCalculating target-based phenotypic consistency...")
    target_consistency = calculate_target_consistency(
        df_median, neg_sameby, neg_diffby, min_compounds_per_target=2
    )

    # Save individual results
    activity_ap.to_csv(f"../output/activity_ap_{codec}.csv", index=False)
    activity_map.to_csv(f"../output/activity_map_{codec}.csv", index=False)
    print(f"Saved results to ../output/activity_ap_{codec}.csv and activity_map_{codec}.csv")

    # Save target consistency results if available
    if target_consistency['target_map'] is not None:
        target_consistency['target_map'].to_csv(f"../output/target_consistency_{codec}.csv", index=False)
        print(f"Saved target consistency to ../output/target_consistency_{codec}.csv")

    return {
        "codec": codec,
        "active_ratio_ap": active_ratio_ap,
        "below_corrected_p": activity_map.below_corrected_p.mean(),
        "n_compounds": len(activity_ap),
        "mean_ap": activity_ap["average_precision"].mean(),
        "median_ap": activity_ap["average_precision"].median(),
        "activity_ap": activity_ap,
        "activity_map": activity_map,
        "target_consistency": target_consistency,
        "status": "success",
    }


def main():
    """Main pipeline for phenotypic activity analysis across all codecs."""
    # Define positive and negative pairs for copairs
    pos_sameby = ["Metadata_pert_iname"]
    pos_diffby = []
    neg_sameby = ["Metadata_Plate"]
    neg_diffby = ["Metadata_pert_iname", "Metadata_negcon"]

    # Find all available codecs first to determine which plates we have
    print("Finding available codecs...")
    codecs = find_available_codecs()

    if not codecs:
        print("ERROR: No codec parquet files found in ../output/*_clean.parquet or ../output/*.parquet")
        print("Please run the feature extraction pipeline first.")
        return None, None, None

    print(f"Found {len(codecs)} codecs: {', '.join(codecs)}")

    # Load a sample to find which plates we have
    print("\nDetecting target plates...")
    sample_features = load_features(codec=codecs[0])
    target_plates = sample_features['Metadata_Plate'].unique().tolist()
    print(f"Found plates: {target_plates}")

    # Load metadata for those plates (only once)
    print("\nLoading well metadata...")
    well_metadata = load_metadata(target_plates=target_plates)

    # Process each codec
    results = []
    all_activity_ap = {}
    all_activity_map = {}
    all_target_consistency = {}

    for codec in codecs:
        result = process_single_codec(
            codec, well_metadata, pos_sameby, pos_diffby, neg_sameby, neg_diffby
        )

        # Collect target consistency metrics
        target_consistency = result.get("target_consistency", {})

        results.append({
            "codec": result["codec"],
            "active_ratio_ap": result["active_ratio_ap"],
            "below_corrected_p": result["below_corrected_p"],
            "n_compounds": result["n_compounds"],
            "mean_ap": result["mean_ap"],
            "median_ap": result["median_ap"],
            "n_targets": target_consistency.get("n_targets"),
            "mean_target_map": target_consistency.get("mean_target_map"),
            "median_target_map": target_consistency.get("median_target_map"),
            "pct_significant_targets": target_consistency.get("pct_significant_targets"),
            "status": result.get("status", "success"),
        })
        if result["activity_ap"] is not None:
            all_activity_ap[codec] = result["activity_ap"]
            all_activity_map[codec] = result["activity_map"]
        if target_consistency.get("target_map") is not None:
            all_target_consistency[codec] = target_consistency["target_map"]

    # Create combined summary table
    print("\n" + "="*80)
    print("COMBINED SUMMARY ACROSS ALL CODECS")
    print("="*80)

    summary_df = pd.DataFrame(results)
    print("\n", summary_df.to_string(index=False))

    # Save combined results
    summary_df.to_csv("../output/phenotypic_activity_summary.csv", index=False)
    summary_df.to_json("../output/phenotypic_activity_summary.json", orient="records", indent=2)
    print("\nSaved combined summary to:")
    print("  - ../output/phenotypic_activity_summary.csv")
    print("  - ../output/phenotypic_activity_summary.json")

    # Create target consistency summary if we have data
    if all_target_consistency:
        print("\n" + "="*80)
        print("TARGET CONSISTENCY SUMMARY")
        print("="*80)
        target_summary = summary_df[['codec', 'n_targets', 'mean_target_map',
                                       'median_target_map', 'pct_significant_targets']].copy()
        print("\n", target_summary.to_string(index=False))
        target_summary.to_csv("../output/target_consistency_summary.csv", index=False)
        print("\nSaved target consistency summary to ../output/target_consistency_summary.csv")

    # Create summary heatmap
    print("\nGenerating summary heatmap...")
    plot_summary_heatmap(summary_df, all_activity_map)

    return summary_df, all_activity_ap, all_activity_map


if __name__ == "__main__":
    summary_df, all_activity_ap, all_activity_map = main()
