"""Create codec comparison plot for segmentation IoU overlap."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# Map to our codec naming
codec_mapping = {
    'jpegxl_lossy_hq': 'jpegxl_hq',
    'jpegxl_lossy_mq': 'jpegxl_mq',
    'jpegxl_lossy_lq': 'jpegxl_lq',
    'jpegxl_lossy_effort_3': 'jpegxl_effort3',
}

def load_segmentation_data(csv_path):
    """Load and process segmentation IoU data from CSV."""
    summary_df = pd.read_csv(csv_path)

    # Extract median IoU values
    median_iou = {
        'zstd': 1.0,  # Perfect overlap with itself
    }

    for _, row in summary_df.iterrows():
        method = row['method'].replace('.zarr', '')
        median_iou[method] = row['iou_median']

    # Remap to match our naming convention
    median_iou_remapped = {'zstd': 1.0}
    for old_name, new_name in codec_mapping.items():
        if old_name in median_iou:
            median_iou_remapped[new_name] = median_iou[old_name]

    return median_iou_remapped

# Load both cell and nuclei segmentation data
cell_iou = load_segmentation_data('../../result_summary/segmentation_comparison_summary.csv')
nuclei_iou = load_segmentation_data('output/nuclei_segmentation_comparison_summary.csv')


def create_iou_plot(cell_iou_values, nuclei_iou_values, output_path):
    """Create plot with file size % on x-axis and median IoU on y-axis for both cell and nuclei."""

    # Prepare data for plotting (use cell data for codec ordering)
    codecs = list(cell_iou_values.keys())
    sizes = [size_percentages[codec] for codec in codecs]

    # Sort by file size (descending)
    sorted_indices = np.argsort(sizes)[::-1]
    codecs_sorted = [codecs[i] for i in sorted_indices]
    sizes_sorted = [sizes[i] for i in sorted_indices]

    # Get IoU values for both cell and nuclei
    cell_ious_sorted = [cell_iou_values[c] for c in codecs_sorted]
    nuclei_ious_sorted = [nuclei_iou_values[c] for c in codecs_sorted]

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

    # Plot Cell IoU
    color_cell = 'tab:green'
    ax.set_xlabel('File Size (% of ZSTD)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Median Segmentation IoU (Intersection over Union)',
                  fontsize=13, fontweight='bold')
    line_cell = ax.plot(sizes_sorted, cell_ious_sorted, 's-', color=color_cell, linewidth=2.5,
                   markersize=12, label='Cell Segmentation', markeredgewidth=2,
                   markeredgecolor='white')

    # Plot Nuclei IoU
    color_nuclei = 'tab:blue'
    line_nuclei = ax.plot(sizes_sorted, nuclei_ious_sorted, 'o-', color=color_nuclei, linewidth=2.5,
                   markersize=12, label='Nuclei Segmentation', markeredgewidth=2,
                   markeredgecolor='white')

    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add IoU value labels for cell segmentation (below, since lower values)
    for i, (x, y) in enumerate(zip(sizes_sorted, cell_ious_sorted)):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                   xytext=(0, -18), ha='center', fontsize=9, color=color_cell,
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                   facecolor='white', edgecolor=color_cell, alpha=0.9, linewidth=1.5))

    # Add IoU value labels for nuclei segmentation (above, since higher values)
    for i, (x, y) in enumerate(zip(sizes_sorted, nuclei_ious_sorted)):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                   xytext=(0, 15), ha='center', fontsize=9, color=color_nuclei,
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                   facecolor='white', edgecolor=color_nuclei, alpha=0.9, linewidth=1.5))

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

    # Set y-axis range (IoU typically 0.9 to 1.0 for good segmentation)
    ax.set_ylim(0.0, 1.02)

    # Add title
    plt.title('Cell & Nuclei Segmentation IoU vs File Size for Different Codecs',
             fontsize=15, fontweight='bold', pad=20)

    # Add legend
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9, frameon=True)

    # Adjust layout
    fig.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    print("Cell Segmentation - Median IoU values:")
    for codec, iou in sorted(cell_iou.items(),
                              key=lambda x: size_percentages[x[0]],
                              reverse=True):
        print(f"  {codec}: {iou:.4f} (file size: {size_percentages[codec]:.1f}%)")

    print("\nNuclei Segmentation - Median IoU values:")
    for codec, iou in sorted(nuclei_iou.items(),
                              key=lambda x: size_percentages[x[0]],
                              reverse=True):
        print(f"  {codec}: {iou:.4f} (file size: {size_percentages[codec]:.1f}%)")

    print("\nCreating plot...")
    create_iou_plot(
        cell_iou,
        nuclei_iou,
        output_dir / 'codec_segmentation_iou.png'
    )


if __name__ == "__main__":
    main()
