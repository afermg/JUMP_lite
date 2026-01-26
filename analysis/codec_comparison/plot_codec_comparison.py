"""Create codec comparison plots with file size vs PA/PC metrics."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# File size ratios from README.md (compressed/raw)
filesize_ratios = {
    'zstd': 0.5949,
    'jpegxl_hq': 0.0305,
    'jpegxl_mq': 0.0118,
    'jpegxl_lq': 0.0071,
    'jpegxl_effort3': 0.0261,
}

# Convert to percentage of ZSTD size
zstd_size = filesize_ratios['zstd']
size_percentages = {k: (v / zstd_size) * 100 for k, v in filesize_ratios.items()}

# Graph 1: Highest PA configs (no TVN/Spherize)
highest_pa_data = {
    'zstd': {'pa': 33.77, 'pc': 46, 'config': 'order1_mad'},
    'jpegxl_mq': {'pa': 32.12, 'pc': 42, 'config': 'order1_mad'},
    'jpegxl_hq': {'pa': 32.12, 'pc': 41, 'config': 'order1_mad'},
    'jpegxl_effort3': {'pa': 31.13, 'pc': 40, 'config': 'order1_mad'},
    'jpegxl_lq': {'pa': 29.80, 'pc': 39, 'config': 'order1_std'},
}

# Graph 2: Best Balance configs (no TVN/Spherize)
best_balance_data = {
    'jpegxl_hq': {'pa': 29.80, 'pc': 63, 'config': 'order4_std', 'balance': 1877},
    'jpegxl_effort3': {'pa': 29.47, 'pc': 63, 'config': 'order2_std', 'balance': 1857},
    'zstd': {'pa': 31.46, 'pc': 51, 'config': 'order1_mad', 'balance': 1604},
    'jpegxl_mq': {'pa': 32.12, 'pc': 42, 'config': 'order1_mad', 'balance': 1349},
    'jpegxl_lq': {'pa': 28.48, 'pc': 45, 'config': 'order1_std', 'balance': 1281},
}

def create_pa_pc_plot(data, title, output_path):
    """Create a plot with file size % on x-axis and PA/PC on dual y-axes."""

    # Prepare data for plotting
    codecs = list(data.keys())
    sizes = [size_percentages[codec] for codec in codecs]
    pas = [data[codec]['pa'] for codec in codecs]
    pcs = [data[codec]['pc'] for codec in codecs]

    # Sort by file size (descending)
    sorted_indices = np.argsort(sizes)[::-1]
    codecs_sorted = [codecs[i] for i in sorted_indices]
    sizes_sorted = [sizes[i] for i in sorted_indices]
    pas_sorted = [pas[i] for i in sorted_indices]
    pcs_sorted = [pcs[i] for i in sorted_indices]

    # Clean codec names for display
    codec_labels = {
        'zstd': 'ZSTD',
        'jpegxl_hq': 'JPEG-XL HQ',
        'jpegxl_mq': 'JPEG-XL MQ',
        'jpegxl_lq': 'JPEG-XL LQ',
        'jpegxl_effort3': 'JPEG-XL Effort3',
    }
    labels_sorted = [codec_labels[c] for c in codecs_sorted]

    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot PA on left y-axis
    color_pa = 'tab:blue'
    ax1.set_xlabel('File Size (% of ZSTD)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Phenotypic Activity - PA (%)', color=color_pa, fontsize=13, fontweight='bold')
    line1 = ax1.plot(sizes_sorted, pas_sorted, 'o-', color=color_pa, linewidth=2.5,
                     markersize=10, label='PA', markeredgewidth=2, markeredgecolor='white')
    ax1.tick_params(axis='y', labelcolor=color_pa, labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Add PA value labels
    for i, (x, y) in enumerate(zip(sizes_sorted, pas_sorted)):
        ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9, color=color_pa,
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                    facecolor='white', edgecolor=color_pa, alpha=0.8))

    # Create second y-axis for PC
    ax2 = ax1.twinx()
    color_pc = 'tab:orange'
    ax2.set_ylabel('Profile Consistency - PC (%)', color=color_pc,
                   fontsize=13, fontweight='bold')
    line2 = ax2.plot(sizes_sorted, pcs_sorted, 's-', color=color_pc, linewidth=2.5,
                     markersize=10, label='PC', markeredgewidth=2, markeredgecolor='white')
    ax2.tick_params(axis='y', labelcolor=color_pc, labelsize=11)

    # Add PC value labels
    for i, (x, y) in enumerate(zip(sizes_sorted, pcs_sorted)):
        ax2.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                    xytext=(0, -20), ha='center', fontsize=9, color=color_pc,
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                    facecolor='white', edgecolor=color_pc, alpha=0.8))

    # Set x-axis to log scale
    ax1.set_xscale('log')

    # Set x-axis ticks at data points only (can't include 0% on log scale)
    ax1.set_xticks(sizes_sorted)
    ax1.set_xticklabels([f'{s:.1f}%\n{labels_sorted[i]}'
                         for i, s in enumerate(sizes_sorted)], fontsize=10, rotation=45, ha='right')

    # Set x-axis range with padding (use min/max of data for log scale)
    min_size = min(sizes_sorted)
    max_size = max(sizes_sorted)
    ax1.set_xlim(min_size * 0.7, max_size * 1.3)

    # Set y-axes to start from 0 with 5% padding on top
    pa_max = max(pas_sorted)
    pc_max = max(pcs_sorted)
    ax1.set_ylim(0, pa_max * 1.05)
    ax2.set_ylim(0, pc_max * 1.05)

    # Add title
    plt.title(title, fontsize=15, fontweight='bold', pad=20)

    # Add legend
    lines = line1 + line2
    labels = ['PA (%)', 'PC (%)']
    ax1.legend(lines, labels, loc='best', fontsize=11, framealpha=0.9)

    # Adjust layout
    fig.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    # Graph 1: Highest PA
    create_pa_pc_plot(
        highest_pa_data,
        'Codec Comparison: Highest PA (Basic Normalization Only)',
        output_dir / 'codec_comparison_highest_pa.png'
    )

    # Graph 2: Best Balance
    create_pa_pc_plot(
        best_balance_data,
        'Codec Comparison: Best Balance Score (Basic Normalization Only)',
        output_dir / 'codec_comparison_best_balance.png'
    )

    print("\nFile size percentages (relative to ZSTD):")
    for codec, pct in sorted(size_percentages.items(), key=lambda x: x[1], reverse=True):
        print(f"  {codec}: {pct:.1f}%")


if __name__ == "__main__":
    main()
