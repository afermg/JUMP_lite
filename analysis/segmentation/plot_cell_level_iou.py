#!/usr/bin/env python3
"""
Plot cell-level (instance-level) IoU distributions across compression methods.

This plots the IoU for individual cell instances (all matched cells including
TP and BELOW_THRESH), as opposed to well-level IoU which treats all cells in
a well as a single binary mask.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List


def load_instance_mappings(mappings_dir: Path, segment_step: str = "segment_cell") -> pd.DataFrame:
    """Load all instance mapping parquet files for a given segment step."""
    parquet_files = list(mappings_dir.glob(f"{segment_step}_*.parquet"))

    if len(parquet_files) == 0:
        raise ValueError(f"No parquet files found in {mappings_dir} for {segment_step}")

    dfs = []
    for pq_file in parquet_files:
        # Extract method name from filename: segment_cell_jpegxl_lossy_lq.parquet -> jpegxl_lossy_lq
        method = pq_file.stem.replace(f"{segment_step}_", "")

        df = pd.read_parquet(pq_file)
        df['method'] = f"{method}.zarr"
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} instance mappings from {len(parquet_files)} files")
    return combined


def plot_cell_level_iou_combined(df_cell: pd.DataFrame, df_nuclei: pd.DataFrame,
                                  output_prefix: str, thresh: float = 0.5):
    """
    Create violin plots showing cell-level IoU distributions for both cell and nuclei segmentation.
    Uses all matched cells (TP + BELOW_THRESH) - excludes unmatched cells (FN, FP).

    Args:
        df_cell: DataFrame with cell instance mappings
        df_nuclei: DataFrame with nuclei instance mappings
        output_prefix: Output file prefix
        thresh: IoU threshold to use for filtering (default: 0.5)
    """
    # Filter to specific threshold and ALL matched instances (TP + BELOW_THRESH)
    # Exclude FN (no predicted match) and FP (no GT match) as they have iou_score=0
    df_cell_filt = df_cell[(df_cell['thresh'] == thresh) & (df_cell['match_type'].isin(['TP', 'BELOW_THRESH']))].copy()
    df_nuclei_filt = df_nuclei[(df_nuclei['thresh'] == thresh) & (df_nuclei['match_type'].isin(['TP', 'BELOW_THRESH']))].copy()

    # Add segmentation type column
    df_cell_filt['segmentation'] = 'Cell'
    df_nuclei_filt['segmentation'] = 'Nuclei'
    df_combined = pd.concat([df_cell_filt, df_nuclei_filt], ignore_index=True)

    # Clean up method names
    df_combined['codec'] = df_combined['method'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')

    # Order codecs by mean IoU (lowest to highest quality)
    codec_mean_iou = df_combined.groupby('codec')['iou_score'].mean().sort_values(ascending=False)
    label_order = list(codec_mean_iou.index)
    df_plot = df_combined

    # Nice display names
    _known_labels = {
        'jxl_lq': 'Low', 'jxl_mq': 'Medium', 'jxl_effort_3': 'Mid-High',
        'jxl_hq': 'High', 'jxl_d2_e8': 'D2 E8', 'jxl_d10': 'D10',
        'jxl_d15': 'D15', 'jxl_d20_e2': 'D20 E2', 'jxl_d30': 'D30',
    }
    codec_labels = {c: _known_labels.get(c, c.replace('jxl_', '').upper()) for c in label_order}

    print(f"\nCell-level IoU statistics (threshold={thresh}, all matched cells):")
    for seg in ['Cell', 'Nuclei']:
        print(f"\n{seg} Segmentation:")
        for codec in label_order:
            subset = df_plot[(df_plot['codec'] == codec) & (df_plot['segmentation'] == seg)]
            if len(subset) > 0:
                print(f"  {codec_labels[codec]}: n={len(subset):,}, "
                      f"mean={subset['iou_score'].mean():.4f}, "
                      f"median={subset['iou_score'].median():.4f}, "
                      f"std={subset['iou_score'].std():.4f}, "
                      f"p5={np.percentile(subset['iou_score'], 5):.4f}, "
                      f"p95={np.percentile(subset['iou_score'], 95):.4f}")

    # --- Cell-level IoU Violin Plot ---
    fig, ax = plt.subplots(figsize=(7, 7))

    sns.violinplot(
        data=df_plot,
        x='codec',
        y='iou_score',
        hue='segmentation',
        order=label_order,
        hue_order=['Cell', 'Nuclei'],
        palette=['tab:green', 'tab:blue'],
        inner='box',
        cut=0,
        ax=ax
    )

    ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
    ax.set_ylabel('Cell-level IoU', fontsize=24, fontweight='bold')
    ax.set_title(f'Cell-level IoU - All Matched Cells', fontsize=26, fontweight='bold')

    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20)
    ax.tick_params(axis='both', labelsize=20)

    ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')

    plt.tight_layout()
    output_path = f"{output_prefix}_cell_level_iou_violinplot_combined.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved cell-level IoU violinplot to: {output_path}")
    plt.close()

    # --- Cell-level IoU Violin Plot with Percentiles ---
    fig, ax = plt.subplots(figsize=(7, 7))

    sns.violinplot(
        data=df_plot,
        x='codec',
        y='iou_score',
        hue='segmentation',
        order=label_order,
        hue_order=['Cell', 'Nuclei'],
        palette=['tab:green', 'tab:blue'],
        inner='box',
        cut=0,
        ax=ax
    )

    ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
    ax.set_ylabel('Cell-level IoU', fontsize=24, fontweight='bold')
    ax.set_title(f'Cell-level IoU - All Matched Cells (5th & 95th percentile)', fontsize=26, fontweight='bold')

    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20)
    ax.tick_params(axis='both', labelsize=20)

    ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')

    # Add 5th and 95th percentile markers
    hue_order = ['Cell', 'Nuclei']
    colors = {'Cell': 'darkgreen', 'Nuclei': 'darkblue'}
    offset = 0.2

    for i, codec in enumerate(label_order):
        for j, seg in enumerate(hue_order):
            subset = df_plot[(df_plot['codec'] == codec) & (df_plot['segmentation'] == seg)]
            if len(subset) > 0:
                p5 = np.percentile(subset['iou_score'], 5)
                p95 = np.percentile(subset['iou_score'], 95)
                x_pos = i + (j - 0.5) * offset * 2
                # 95th percentile marker
                ax.scatter([x_pos], [p95], marker='_', s=300, linewidths=3,
                          color=colors[seg], zorder=10)
                ax.annotate(f'{p95:.3f}', (x_pos, p95), textcoords='offset points',
                           xytext=(0, 8), ha='center', fontsize=10, fontweight='bold',
                           color=colors[seg])
                # 5th percentile marker
                ax.scatter([x_pos], [p5], marker='_', s=300, linewidths=3,
                          color=colors[seg], zorder=10)
                ax.annotate(f'{p5:.3f}', (x_pos, p5), textcoords='offset points',
                           xytext=(0, -14), ha='center', fontsize=10, fontweight='bold',
                           color=colors[seg])

    plt.tight_layout()
    output_path = f"{output_prefix}_cell_level_iou_violinplot_combined_p5_p95.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved cell-level IoU violinplot with percentiles to: {output_path}")
    plt.close()

    # --- Cell-level IoU Boxen Plot ---
    fig, ax = plt.subplots(figsize=(7, 7))

    sns.boxenplot(
        data=df_plot,
        x='codec',
        y='iou_score',
        hue='segmentation',
        order=label_order,
        hue_order=['Cell', 'Nuclei'],
        palette=['tab:green', 'tab:blue'],
        ax=ax
    )

    ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
    ax.set_ylabel('Cell-level IoU', fontsize=24, fontweight='bold')
    ax.set_title(f'Cell-level IoU - All Matched Cells', fontsize=26, fontweight='bold')

    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20)
    ax.tick_params(axis='both', labelsize=20)

    ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')

    plt.tight_layout()
    output_path = f"{output_prefix}_cell_level_iou_boxenplot_combined.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved cell-level IoU boxenplot to: {output_path}")
    plt.close()


def plot_match_type_distribution(df_cell: pd.DataFrame, df_nuclei: pd.DataFrame,
                                  output_prefix: str, thresh: float = 0.5):
    """Plot distribution of match types across compression methods."""
    df_cell_filt = df_cell[df_cell['thresh'] == thresh].copy()
    df_nuclei_filt = df_nuclei[df_nuclei['thresh'] == thresh].copy()

    df_cell_filt['segmentation'] = 'Cell'
    df_nuclei_filt['segmentation'] = 'Nuclei'
    df_combined = pd.concat([df_cell_filt, df_nuclei_filt], ignore_index=True)

    df_combined['codec'] = df_combined['method'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')

    # Count match types per codec
    match_counts = df_combined.groupby(['codec', 'segmentation', 'match_type']).size().reset_index(name='count')

    # Calculate percentages
    totals = df_combined.groupby(['codec', 'segmentation']).size().reset_index(name='total')
    match_counts = match_counts.merge(totals, on=['codec', 'segmentation'])
    match_counts['percentage'] = (match_counts['count'] / match_counts['total']) * 100

    # Order codecs by mean IoU (lowest to highest)
    codec_mean_iou = df_combined.groupby('codec')['iou_score'].mean().sort_values(ascending=False)
    codec_order = list(codec_mean_iou.index)
    _known_labels = {
        'jxl_lq': 'Low', 'jxl_mq': 'Medium', 'jxl_effort_3': 'Mid-High',
        'jxl_hq': 'High', 'jxl_d2_e8': 'D2 E8', 'jxl_d10': 'D10',
        'jxl_d15': 'D15', 'jxl_d20_e2': 'D20 E2', 'jxl_d30': 'D30',
    }
    codec_labels = {c: _known_labels.get(c, c.replace('jxl_', '').upper()) for c in codec_order}

    n_codecs = len(codec_order)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, n_codecs * 2.5), 7))

    for ax, seg in zip([ax1, ax2], ['Cell', 'Nuclei']):
        subset = match_counts[match_counts['segmentation'] == seg]

        # Pivot for stacked bar plot
        pivot = subset.pivot(index='codec', columns='match_type', values='percentage')
        pivot = pivot.reindex(codec_order)

        pivot.plot(kind='bar', stacked=True, ax=ax,
                  color=['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6'])

        ax.set_xlabel('Compression Quality', fontsize=16, fontweight='bold')
        ax.set_ylabel('Percentage', fontsize=16, fontweight='bold')
        ax.set_title(f'{seg} Segmentation - Match Types', fontsize=18, fontweight='bold')
        ax.set_xticklabels([codec_labels[c] for c in codec_order], rotation=45, ha='right', fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(title='Match Type', fontsize=12, title_fontsize=12)
        ax.set_ylim(0, 100)

    plt.tight_layout()
    output_path = f"{output_prefix}_match_type_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved match type distribution to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot cell-level IoU distributions for all matched cells")
    parser.add_argument("--mappings-dir", type=str, required=True,
                       help="Directory containing instance mapping parquet files")
    parser.add_argument("--output", type=str, default="segmentation_cell_level_iou",
                       help="Output file prefix")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: analysis/segmentation/output/)")
    parser.add_argument("--thresh", type=float, default=0.5,
                       help="IoU threshold to use for filtering (default: 0.5)")

    args = parser.parse_args()

    mappings_dir = Path(args.mappings_dir)
    if not mappings_dir.exists():
        raise ValueError(f"Mappings directory does not exist: {mappings_dir}")

    # Load cell and nuclei instance mappings
    print("Loading cell instance mappings...")
    df_cell = load_instance_mappings(mappings_dir, segment_step="segment_cell")

    print("\nLoading nuclei instance mappings...")
    df_nuclei = load_instance_mappings(mappings_dir, segment_step="segment_nuclei")

    # Create output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_dir / args.output)

    # Generate plots
    print(f"\nGenerating plots with threshold={args.thresh}...")
    print("Using all matched cells (TP + BELOW_THRESH)")
    plot_cell_level_iou_combined(df_cell, df_nuclei, output_prefix, args.thresh)
    plot_match_type_distribution(df_cell, df_nuclei, output_prefix, args.thresh)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
