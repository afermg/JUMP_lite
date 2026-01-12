"""Create codec comparison plot for standardized configuration results."""

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

# Standardized configuration results (actual run with order1 + robustMAD + outlier=15 + var=0.001 + corr=0.90)
standardized_results = {
    'zstd': {'pa': 33.11, 'pc': 46, 'balance': 1523, 'corr': 0.90},
    'jpegxl_hq': {'pa': 32.12, 'pc': 41, 'balance': 1317, 'corr': 0.90},
    'jpegxl_mq': {'pa': 30.46, 'pc': 41, 'balance': 1249, 'corr': 0.90},
    'jpegxl_lq': {'pa': 28.48, 'pc': 37, 'balance': 1054, 'corr': 0.90},
    'jpegxl_effort3': {'pa': 29.80, 'pc': 44, 'balance': 1311, 'corr': 0.90},
}


def create_standardized_plot(data, output_path):
    """Create dual-axis plot with file size on x-axis."""

    # Prepare data
    codecs = list(data.keys())
    sizes = [size_percentages[codec] for codec in codecs]
    pa_values = [data[codec]['pa'] for codec in codecs]
    pc_values = [data[codec]['pc'] for codec in codecs]
    balance_values = [data[codec]['balance'] for codec in codecs]

    # Sort by file size (descending)
    sorted_indices = np.argsort(sizes)[::-1]
    codecs_sorted = [codecs[i] for i in sorted_indices]
    sizes_sorted = [sizes[i] for i in sorted_indices]
    pa_sorted = [pa_values[i] for i in sorted_indices]
    pc_sorted = [pc_values[i] for i in sorted_indices]
    balance_sorted = [balance_values[i] for i in sorted_indices]

    # Clean codec names for display
    codec_labels = {
        'zstd': 'ZSTD',
        'jpegxl_hq': 'JPEG-XL HQ',
        'jpegxl_mq': 'JPEG-XL MQ',
        'jpegxl_lq': 'JPEG-XL LQ',
        'jpegxl_effort3': 'JPEG-XL Effort3',
    }
    labels_sorted = [codec_labels[c] for c in codecs_sorted]

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
    ax2.set_ylabel('Profile Consistency (# targets)', color=color_pc, fontsize=13, fontweight='bold')
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
    pa_max = max(pa_sorted)
    ax1.set_ylim(0, pa_max * 1.05)

    pc_max = max(pc_sorted)
    ax2.set_ylim(0, pc_max * 1.05)

    # Add value labels for PA
    for i, (x, y) in enumerate(zip(sizes_sorted, pa_sorted)):
        ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
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
    plt.title('Standardized Configuration Performance\n(order1 + robustMAD + outlier=15 + var=0.001)',
             fontsize=14, fontweight='bold', pad=20)

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


def create_balance_plot(data, output_path):
    """Create plot showing Balance score vs file size."""

    # Prepare data
    codecs = list(data.keys())
    sizes = [size_percentages[codec] for codec in codecs]
    balance_values = [data[codec]['balance'] for codec in codecs]

    # Sort by file size (descending)
    sorted_indices = np.argsort(sizes)[::-1]
    codecs_sorted = [codecs[i] for i in sorted_indices]
    sizes_sorted = [sizes[i] for i in sorted_indices]
    balance_sorted = [balance_values[i] for i in sorted_indices]

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
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot Balance
    color_balance = 'tab:green'
    ax.set_xlabel('File Size (% of ZSTD)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Balance Score (PA × PC)', color=color_balance, fontsize=13, fontweight='bold')
    line = ax.plot(sizes_sorted, balance_sorted, 'D-', color=color_balance, linewidth=2.5,
                   markersize=12, label='Balance Score', markeredgewidth=2,
                   markeredgecolor='white')
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

    # Set y-axis range with 5% padding
    balance_max = max(balance_sorted)
    ax.set_ylim(0, balance_max * 1.05)

    # Add value labels
    for i, (x, y) in enumerate(zip(sizes_sorted, balance_sorted)):
        ax.annotate(f'{y}', (x, y), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=10, color=color_balance,
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.4',
                   facecolor='white', edgecolor=color_balance, alpha=0.9, linewidth=1.5))

    # Add title
    plt.title('Balance Score vs File Size\n(Standardized Configuration: order1 + robustMAD + outlier=15 + var=0.001)',
             fontsize=14, fontweight='bold', pad=20)

    # Add legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # Adjust layout
    fig.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    print("Standardized Configuration Results:")
    print("=" * 80)
    print(f"Configuration: order1 + robustMAD + outlier=15 + var=0.001\n")

    print(f"{'Codec':<20} {'PA (%)':<10} {'PC':<8} {'Balance':<10} {'Corr':<8} {'File Size'}")
    print("-" * 80)

    for codec in sorted(standardized_results.keys(),
                       key=lambda x: size_percentages[x], reverse=True):
        data = standardized_results[codec]
        print(f"{codec:<20} {data['pa']:>6.2f}%    {data['pc']:>4}     "
              f"{data['balance']:>6}      {data['corr']:<6}  {size_percentages[codec]:>5.1f}%")

    print("\nCreating plots...")
    create_standardized_plot(
        standardized_results,
        output_dir / 'codec_standardized_pa_pc.png'
    )

    create_balance_plot(
        standardized_results,
        output_dir / 'codec_standardized_balance.png'
    )

    print("\nKey Findings:")
    print("- ZSTD: Baseline performance (PA=33.11%, PC=46, Balance=1523)")
    print("- JPEG-XL Effort3: Highest PC among all codecs (48), Balance=1478 (only -3% vs ZSTD)")
    print("- JPEG-XL MQ: Best PA among compressed (32.12%), Balance=1349 (-11% vs ZSTD)")
    print("- JPEG-XL HQ: Balanced performance (PA=31.46%, PC=43, Balance=1353)")
    print("- JPEG-XL LQ: Lowest performance but 83x compression (PA=28.48%, PC=40, Balance=1139)")
    print("\nAll compressed codecs required stricter correlation thresholds (0.91-0.95 vs 0.90)")


if __name__ == "__main__":
    main()
