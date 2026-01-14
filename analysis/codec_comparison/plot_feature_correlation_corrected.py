"""Create codec comparison plot for feature correlation median - using correct values from heatmap."""

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

# Correct median correlations calculated by feature_correlation_cp_measure_script.py
median_correlations = {
    'zstd': 1.000000,  # Perfect correlation with itself
    'jpegxl_hq': 0.958786,  # High quality maintains 96% feature correlation
    'jpegxl_effort3': 0.930298,  # Effort 3 maintains 93% feature correlation
    'jpegxl_mq': 0.869118,  # Medium quality maintains 87% feature correlation
    'jpegxl_lq': 0.854788,  # Low quality maintains 85% feature correlation
}


def create_correlation_plot(correlations, output_path):
    """Create plot with file size % on x-axis and median correlation on y-axis."""

    # Prepare data for plotting
    codecs = list(correlations.keys())
    sizes = [size_percentages[codec] for codec in codecs]
    corrs = [correlations[codec] for codec in codecs]

    # Sort by file size (descending)
    sorted_indices = np.argsort(sizes)[::-1]
    codecs_sorted = [codecs[i] for i in sorted_indices]
    sizes_sorted = [sizes[i] for i in sorted_indices]
    corrs_sorted = [corrs[i] for i in sorted_indices]

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

    # Plot correlation
    color_corr = 'tab:purple'
    ax.set_xlabel('File Size (% of ZSTD)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Median Feature Correlation with ZSTD', color=color_corr,
                  fontsize=13, fontweight='bold')
    line = ax.plot(sizes_sorted, corrs_sorted, 'D-', color=color_corr, linewidth=2.5,
                   markersize=12, label='Median Correlation', markeredgewidth=2,
                   markeredgecolor='white')
    ax.tick_params(axis='y', labelcolor=color_corr, labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add correlation value labels
    for i, (x, y) in enumerate(zip(sizes_sorted, corrs_sorted)):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=10, color=color_corr,
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.4',
                   facecolor='white', edgecolor=color_corr, alpha=0.9, linewidth=1.5))

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

    # Set y-axis to start from reasonable minimum (0.8 to 1.0)
    ax.set_ylim(0.0, 1.02)

    # Add title
    plt.title('Feature Correlation vs File Size for Different Codecs',
             fontsize=15, fontweight='bold', pad=20)

    # Add legend
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)

    # Adjust layout
    fig.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    print("Using corrected median feature correlations from compartment heatmap:")
    for codec, corr in sorted(median_correlations.items(),
                              key=lambda x: size_percentages[x[0]],
                              reverse=True):
        print(f"  {codec}: {corr:.3f} (file size: {size_percentages[codec]:.1f}%)")

    print("\nCreating plot...")
    create_correlation_plot(
        median_correlations,
        output_dir / 'codec_feature_correlation.png'
    )


if __name__ == "__main__":
    main()
