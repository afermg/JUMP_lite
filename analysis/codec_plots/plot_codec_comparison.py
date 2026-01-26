"""Create codec comparison plots from normalized feature results.

Automatically reads metrics.json files from a specified directory
and generates PA/PC comparison plots.

Usage:
    python plot_codec_comparison.py /path/to/normalized/data
    python plot_codec_comparison.py /path/to/data --output /path/to/output
"""
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# File size ratios (compressed/raw) - same as codec_comparison
FILESIZE_RATIOS = {
    'zstd': 0.5949,
    'jpegxl_lossy_hq': 0.0305,
    'jpegxl_lossy_mq': 0.0118,
    'jpegxl_lossy_lq': 0.0071,
    'jpegxl_lossy_effort_3': 0.0261,
}

# Convert to percentage of ZSTD size
ZSTD_SIZE = FILESIZE_RATIOS['zstd']
SIZE_PERCENTAGES = {k: (v / ZSTD_SIZE) * 100 for k, v in FILESIZE_RATIOS.items()}

# Display labels for codecs
CODEC_LABELS = {
    'zstd': 'ZSTD',
    'jpegxl_lossy_hq': 'JPEG-XL HQ',
    'jpegxl_lossy_mq': 'JPEG-XL MQ',
    'jpegxl_lossy_lq': 'JPEG-XL LQ',
    'jpegxl_lossy_effort_3': 'JPEG-XL Effort3',
}


def extract_codec_from_path(path: Path) -> str | None:
    """Extract codec type from directory name."""
    name = str(path)
    if 'zstd' in name:
        return 'zstd'
    elif 'lossy_hq' in name:
        return 'jpegxl_lossy_hq'
    elif 'lossy_mq' in name:
        return 'jpegxl_lossy_mq'
    elif 'lossy_lq' in name:
        return 'jpegxl_lossy_lq'
    elif 'effort_3' in name:
        return 'jpegxl_lossy_effort_3'
    return None


def load_metrics(data_dir: Path) -> dict:
    """Load metrics from all codec results.

    Returns:
        Dictionary mapping codec names to their metrics (pa, pc, balance, etc.)
    """
    results = {}

    for metrics_file in data_dir.glob('**/results/metrics.json'):
        codec = extract_codec_from_path(metrics_file)
        if codec:
            with open(metrics_file) as f:
                data = json.load(f)

            results_dir = metrics_file.parent

            # Load per-compound activity data
            active_compounds = set()
            activity_file = results_dir / 'phenotypic_activity_map.csv'
            if activity_file.exists():
                df_activity = pd.read_csv(activity_file)
                active_compounds = set(
                    df_activity[df_activity['below_corrected_p'] == True]['Metadata_pert_iname']
                )

            # Load per-target consistency data
            active_targets = set()
            consistency_file = results_dir / 'phenotypic_consistency_per_target.csv'
            if consistency_file.exists():
                df_consistency = pd.read_csv(consistency_file)
                active_targets = set(
                    df_consistency[df_consistency['below_corrected_p'] == True]['Metadata_target']
                )

            results[codec] = {
                'pa': data['PA'],
                'pc': data['PC'],
                'n_compounds': data.get('n_compounds', 0),
                'n_targets_total': data.get('n_targets_total', 0),
                'silhouette': data.get('Silhouette', 0),
                'kbet': data.get('kBET', 0),
                'balance': data['PA'] * data['PC'],
                'active_compounds': active_compounds,
                'active_targets': active_targets,
                'n_active_compounds': len(active_compounds),
                'n_active_targets': len(active_targets),
            }

    return results


def create_pa_pc_plot(data: dict, output_path: Path, title: str = None):
    """Create dual-axis plot with file size on x-axis, PA and PC on y-axes."""

    if not data:
        print("No data to plot!")
        return

    # Prepare data
    codecs = list(data.keys())
    sizes = [SIZE_PERCENTAGES.get(codec, 100) for codec in codecs]
    pa_values = [data[codec]['pa'] for codec in codecs]
    pc_values = [data[codec]['pc'] for codec in codecs]

    # Sort by file size (descending)
    sorted_indices = np.argsort(sizes)[::-1]
    codecs_sorted = [codecs[i] for i in sorted_indices]
    sizes_sorted = [sizes[i] for i in sorted_indices]
    pa_sorted = [pa_values[i] for i in sorted_indices]
    pc_sorted = [pc_values[i] for i in sorted_indices]

    # Get display labels
    labels_sorted = [CODEC_LABELS.get(c, c) for c in codecs_sorted]

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # PA axis (left)
    color_pa = 'tab:blue'
    ax1.set_xlabel('File Size (% of ZSTD)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Phenotypic Activity (%)', color=color_pa, fontsize=13, fontweight='bold')
    line1 = ax1.plot(sizes_sorted, pa_sorted, 'o-', color=color_pa, linewidth=2.5,
                     markersize=12, label='PA (%)', markeredgewidth=2, markeredgecolor='white')
    ax1.tick_params(axis='y', labelcolor=color_pa, labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # PC axis (right)
    ax2 = ax1.twinx()
    color_pc = 'tab:orange'
    ax2.set_ylabel('Profile Consistency (%)', color=color_pc, fontsize=13, fontweight='bold')
    line2 = ax2.plot(sizes_sorted, pc_sorted, 's-', color=color_pc, linewidth=2.5,
                     markersize=12, label='PC', markeredgewidth=2, markeredgecolor='white')
    ax2.tick_params(axis='y', labelcolor=color_pc, labelsize=11)

    # Set x-axis to log scale
    ax1.set_xscale('log')

    # Set x-axis ticks at data points
    ax1.set_xticks(sizes_sorted)
    ax1.set_xticklabels([f'{s:.1f}%\n{labels_sorted[i]}'
                         for i, s in enumerate(sizes_sorted)], fontsize=10,
                        rotation=45, ha='right')

    # Set x-axis range with padding
    min_size = min(sizes_sorted)
    max_size = max(sizes_sorted)
    ax1.set_xlim(min_size * 0.7, max_size * 1.3)

    # Set y-axis ranges with 5% padding
    pa_max = max(pa_sorted) if pa_sorted else 1
    ax1.set_ylim(0, pa_max * 1.15)

    pc_max = max(pc_sorted) if pc_sorted else 1
    ax2.set_ylim(0, pc_max * 1.15)

    # Add value labels for PA
    for i, (x, y) in enumerate(zip(sizes_sorted, pa_sorted)):
        ax1.annotate(f'{y:.2f}%', (x, y), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9, color=color_pa,
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                    facecolor='white', edgecolor=color_pa, alpha=0.9, linewidth=1.5))

    # Add value labels for PC
    for i, (x, y) in enumerate(zip(sizes_sorted, pc_sorted)):
        ax2.annotate(f'{y}', (x, y), textcoords="offset points",
                    xytext=(0, -20), ha='center', fontsize=9, color=color_pc,
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                    facecolor='white', edgecolor=color_pc, alpha=0.9, linewidth=1.5))

    # Add title
    if title is None:
        title = 'Codec Comparison\n(Phenotypic Activity & Profile Consistency)'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.9)

    # Adjust layout
    fig.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_balance_plot(data: dict, output_path: Path, title: str = None):
    """Create plot showing Balance score (PA × PC) vs file size."""

    if not data:
        print("No data to plot!")
        return

    # Prepare data
    codecs = list(data.keys())
    sizes = [SIZE_PERCENTAGES.get(codec, 100) for codec in codecs]
    balance_values = [data[codec]['balance'] for codec in codecs]

    # Sort by file size (descending)
    sorted_indices = np.argsort(sizes)[::-1]
    codecs_sorted = [codecs[i] for i in sorted_indices]
    sizes_sorted = [sizes[i] for i in sorted_indices]
    balance_sorted = [balance_values[i] for i in sorted_indices]

    # Get display labels
    labels_sorted = [CODEC_LABELS.get(c, c) for c in codecs_sorted]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot Balance
    color_balance = 'tab:green'
    ax.set_xlabel('File Size (% of ZSTD)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Balance Score (PA × PC)', color=color_balance, fontsize=13, fontweight='bold')
    ax.plot(sizes_sorted, balance_sorted, 'D-', color=color_balance, linewidth=2.5,
            markersize=12, label='Balance Score', markeredgewidth=2, markeredgecolor='white')
    ax.tick_params(axis='y', labelcolor=color_balance, labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set x-axis to log scale
    ax.set_xscale('log')

    # Set x-axis ticks at data points
    ax.set_xticks(sizes_sorted)
    ax.set_xticklabels([f'{s:.1f}%\n{labels_sorted[i]}'
                        for i, s in enumerate(sizes_sorted)], fontsize=10,
                       rotation=45, ha='right')

    # Set x-axis range with padding
    min_size = min(sizes_sorted)
    max_size = max(sizes_sorted)
    ax.set_xlim(min_size * 0.7, max_size * 1.3)

    # Set y-axis range with padding
    balance_max = max(balance_sorted) if balance_sorted else 1
    ax.set_ylim(0, balance_max * 1.15)

    # Add value labels
    for i, (x, y) in enumerate(zip(sizes_sorted, balance_sorted)):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=10, color=color_balance,
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.4',
                   facecolor='white', edgecolor=color_balance, alpha=0.9, linewidth=1.5))

    # Add title
    if title is None:
        title = 'Codec Comparison: Balance Score vs File Size\n(Balance = PA × PC)'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)

    # Add legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # Adjust layout
    fig.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def calculate_iou(set1: set, set2: set) -> float:
    """Calculate Intersection over Union between two sets."""
    if not set1 and not set2:
        return 1.0  # Both empty = perfect match
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def create_iou_heatmaps(data: dict, output_path: Path, title_prefix: str = "Codec"):
    """Create IoU heatmaps comparing active compounds and targets across codecs.

    Args:
        data: Dictionary with codec metrics including active_compounds and active_targets sets
        output_path: Path to save the heatmap image
        title_prefix: Prefix for the plot title
    """
    if not data:
        print("No data to plot!")
        return

    # Sort codecs by file size (descending)
    codecs = sorted(data.keys(),
                   key=lambda x: SIZE_PERCENTAGES.get(x, 100),
                   reverse=True)

    n_codecs = len(codecs)

    # Calculate IoU matrices
    compound_iou = np.zeros((n_codecs, n_codecs))
    target_iou = np.zeros((n_codecs, n_codecs))
    compound_intersect = np.zeros((n_codecs, n_codecs), dtype=int)
    target_intersect = np.zeros((n_codecs, n_codecs), dtype=int)

    for i, codec1 in enumerate(codecs):
        for j, codec2 in enumerate(codecs):
            compounds1 = data[codec1].get('active_compounds', set())
            compounds2 = data[codec2].get('active_compounds', set())
            targets1 = data[codec1].get('active_targets', set())
            targets2 = data[codec2].get('active_targets', set())

            compound_iou[i, j] = calculate_iou(compounds1, compounds2) * 100
            target_iou[i, j] = calculate_iou(targets1, targets2) * 100
            compound_intersect[i, j] = len(compounds1 & compounds2)
            target_intersect[i, j] = len(targets1 & targets2)

    # Create labels with counts and percentages
    def make_label(codec, n_active, pct):
        label = CODEC_LABELS.get(codec, codec)
        return f"{label}\n({n_active}, {pct:.1f}%)"

    compound_labels = [
        make_label(c, data[c].get('n_active_compounds', 0), data[c].get('pa', 0))
        for c in codecs
    ]
    target_labels = [
        make_label(c, data[c].get('n_active_targets', 0), data[c].get('pc', 0))
        for c in codecs
    ]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Custom annotation function to show both IoU % and intersection count
    def annotate_heatmap_iou(ax, iou_matrix, intersect_matrix):
        for i in range(iou_matrix.shape[0]):
            for j in range(iou_matrix.shape[1]):
                iou_val = iou_matrix[i, j]
                intersect_val = intersect_matrix[i, j]
                text_color = 'white' if iou_val < 50 else 'black'
                # Heatmap cells are centered at (j + 0.5, i + 0.5)
                ax.text(j + 0.5, i + 0.5, f'{iou_val:.1f}%\n({intersect_val})',
                       ha='center', va='center', fontsize=9,
                       color=text_color, fontweight='bold')

    # Custom annotation function for intersection count heatmaps
    def annotate_heatmap_count(ax, intersect_matrix, max_val):
        for i in range(intersect_matrix.shape[0]):
            for j in range(intersect_matrix.shape[1]):
                intersect_val = intersect_matrix[i, j]
                # Use white text on dark (high value) cells
                text_color = 'white' if intersect_val > max_val * 0.5 else 'black'
                ax.text(j + 0.5, i + 0.5, f'{intersect_val}',
                       ha='center', va='center', fontsize=10,
                       color=text_color, fontweight='bold')

    # Top row: IoU heatmaps
    # Compound IoU heatmap
    ax1 = axes[0, 0]
    sns.heatmap(compound_iou, ax=ax1, cmap='RdBu_r', vmin=0, vmax=100,
                xticklabels=compound_labels, yticklabels=compound_labels,
                cbar_kws={'label': 'IoU (%)'}, annot=False)
    annotate_heatmap_iou(ax1, compound_iou, compound_intersect)
    ax1.set_title(f'{title_prefix}: Active Compounds IoU\n(count, PA%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=9)

    # Target IoU heatmap
    ax2 = axes[0, 1]
    sns.heatmap(target_iou, ax=ax2, cmap='RdBu_r', vmin=0, vmax=100,
                xticklabels=target_labels, yticklabels=target_labels,
                cbar_kws={'label': 'IoU (%)'}, annot=False)
    annotate_heatmap_iou(ax2, target_iou, target_intersect)
    ax2.set_title(f'{title_prefix}: Active Targets IoU\n(count, PC%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=9)

    # Bottom row: Intersection count heatmaps
    # Compound intersection count heatmap
    ax3 = axes[1, 0]
    compound_max = compound_intersect.max()
    sns.heatmap(compound_intersect, ax=ax3, cmap='RdBu_r', vmin=0, vmax=compound_max,
                xticklabels=compound_labels, yticklabels=compound_labels,
                cbar_kws={'label': '# Overlapping'}, annot=False)
    annotate_heatmap_count(ax3, compound_intersect, compound_max)
    ax3.set_title(f'{title_prefix}: Overlapping Active Compounds\n(count, PA%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('')
    ax3.set_ylabel('')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=9)

    # Target intersection count heatmap
    ax4 = axes[1, 1]
    target_max = target_intersect.max()
    sns.heatmap(target_intersect, ax=ax4, cmap='RdBu_r', vmin=0, vmax=target_max,
                xticklabels=target_labels, yticklabels=target_labels,
                cbar_kws={'label': '# Overlapping'}, annot=False)
    annotate_heatmap_count(ax4, target_intersect, target_max)
    ax4.set_title(f'{title_prefix}: Overlapping Active Targets\n(count, PC%)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('')
    ax4.set_ylabel('')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax4.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Also save the IoU data to CSV
    csv_path = output_path.with_suffix('.csv')
    rows = []
    for i, codec1 in enumerate(codecs):
        for j, codec2 in enumerate(codecs):
            rows.append({
                'codec1': codec1,
                'codec2': codec2,
                'codec1_label': CODEC_LABELS.get(codec1, codec1),
                'codec2_label': CODEC_LABELS.get(codec2, codec2),
                'compound_iou_pct': compound_iou[i, j],
                'compound_intersection': compound_intersect[i, j],
                'target_iou_pct': target_iou[i, j],
                'target_intersection': target_intersect[i, j],
                'codec1_n_active_compounds': data[codec1].get('n_active_compounds', 0),
                'codec2_n_active_compounds': data[codec2].get('n_active_compounds', 0),
                'codec1_n_active_targets': data[codec1].get('n_active_targets', 0),
                'codec2_n_active_targets': data[codec2].get('n_active_targets', 0),
            })
    df_iou = pd.DataFrame(rows)
    df_iou.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def print_summary(data: dict):
    """Print summary table of all metrics."""
    print("\nCodec Comparison Results")
    print("=" * 110)
    print(f"{'Codec':<20} {'PA (%)':<10} {'# Active':<10} {'PC (%)':<10} {'# Active':<10} {'Balance':<10} {'File Size':<10}")
    print(f"{'':20} {'':10} {'Compounds':<10} {'':10} {'Targets':<10} {'':10} {'':10}")
    print("-" * 110)

    # Sort by file size (descending)
    sorted_codecs = sorted(data.keys(),
                          key=lambda x: SIZE_PERCENTAGES.get(x, 100),
                          reverse=True)

    for codec in sorted_codecs:
        metrics = data[codec]
        size_pct = SIZE_PERCENTAGES.get(codec, 100)
        label = CODEC_LABELS.get(codec, codec)
        n_active_compounds = metrics.get('n_active_compounds', 0)
        n_active_targets = metrics.get('n_active_targets', 0)
        print(f"{label:<20} {metrics['pa']:>7.2f}%   {n_active_compounds:>6}     "
              f"{metrics['pc']:>7.2f}%   {n_active_targets:>6}     "
              f"{metrics['balance']:>7.2f}    {size_pct:>6.1f}%")

    print("-" * 110)


def save_results_csv(data: dict, output_path: Path):
    """Save results to CSV file."""
    rows = []
    for codec, metrics in data.items():
        rows.append({
            'codec': codec,
            'codec_label': CODEC_LABELS.get(codec, codec),
            'pa': metrics['pa'],
            'pc': metrics['pc'],
            'balance': metrics['balance'],
            'n_compounds': metrics.get('n_compounds', 0),
            'n_targets_total': metrics.get('n_targets_total', 0),
            'silhouette': metrics.get('silhouette', 0),
            'kbet': metrics.get('kbet', 0),
            'file_size_pct': SIZE_PERCENTAGES.get(codec, 100),
            'file_size_ratio': FILESIZE_RATIOS.get(codec, 1.0),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('file_size_pct', ascending=False)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate codec comparison plots from normalized feature results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python plot_codec_comparison.py /path/to/normalized/data
    python plot_codec_comparison.py ./data/normalized_dino --output ./plots
    python plot_codec_comparison.py /data/results -o /data/results/plots
        """
    )
    parser.add_argument(
        'data_dir',
        type=Path,
        help='Path to directory containing normalized results with metrics.json files'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output directory for plots (default: <data_dir>/plots)'
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Custom title prefix for plots'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='codec',
        help='Prefix for output filenames (default: codec)'
    )
    return parser.parse_args()


def main():
    """Main function to generate all plots."""
    args = parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        return 1

    # Setup output directory
    output_dir = args.output if args.output else data_dir / 'plots'
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics from all codec results
    print(f"Loading metrics from: {data_dir}")
    data = load_metrics(data_dir)

    if not data:
        print(f"No metrics.json files found in {data_dir}")
        print("Make sure the directory contains results with metrics.json files.")
        return 1

    print(f"Found {len(data)} codec results: {list(data.keys())}")

    # Print summary
    print_summary(data)

    # Save results to CSV
    save_results_csv(data, output_dir / f'{args.prefix}_results.csv')

    # Create plots
    print("\nGenerating plots...")

    title_prefix = args.title if args.title else "Codec Comparison"

    create_pa_pc_plot(
        data,
        output_dir / f'{args.prefix}_pa_pc.png',
        title=f'{title_prefix}\n(Phenotypic Activity & Profile Consistency)'
    )

    create_balance_plot(
        data,
        output_dir / f'{args.prefix}_balance.png',
        title=f'{title_prefix}\n(Balance = PA × PC)'
    )

    # Create IoU heatmaps comparing active compounds and targets
    create_iou_heatmaps(
        data,
        output_dir / f'{args.prefix}_iou_heatmap.png',
        title_prefix=title_prefix
    )

    print(f"\nAll plots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
