"""
Merge compression metrics with quality metrics for comprehensive analysis.

Usage:
    python src/merge_metrics.py
"""

from pathlib import Path
import pandas as pd
import json
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.style import Style


def merge_metrics(results_dir: Path = Path("data/results")):
    """
    Merge compression metrics (from compress_tif.py) with quality metrics (from evaluate_quality.py).

    Creates a comprehensive CSV with all metrics for easy analysis.
    """
    # Load compression metrics
    compression_csv = results_dir / "compression_metrics.csv"
    if not compression_csv.exists():
        print(f"Error: {compression_csv} not found. Run compress_tif.py first.")
        return

    df_compression = pd.read_csv(compression_csv)
    print(f"Loaded compression metrics: {len(df_compression)} codecs")

    # Load quality metrics summary
    quality_csv = results_dir / "metrics_summary.csv"
    if not quality_csv.exists():
        print(f"Error: {quality_csv} not found. Run evaluate_quality.py first.")
        return

    # Read CSV with multi-level headers (header in rows 0 and 1)
    df_quality = pd.read_csv(quality_csv, header=[0, 1])

    # Flatten multi-level columns
    df_quality.columns = ['_'.join(col).strip('_') for col in df_quality.columns.values]

    # The first column is codec (unnamed in the CSV)
    df_quality = df_quality.rename(columns={df_quality.columns[0]: 'codec'})

    # Remove empty rows and the 'codec' header row that got read as data
    df_quality = df_quality[
        df_quality['codec'].notna() &
        (df_quality['codec'] != '') &
        (df_quality['codec'] != 'codec')  # Remove the header row
    ]

    print(f"Loaded quality metrics: {len(df_quality)} codecs")

    # Merge on codec name
    df_merged = pd.merge(
        df_compression,
        df_quality,
        on='codec',
        how='outer'
    )

    # Reorder columns for readability
    column_order = [
        'codec',
        'filesize_ratio',
        'compression_ratio',
        'compression_time_sec',
        'decompression_time_sec',
        'filesize_bytes',
    ]

    # Add quality metrics columns (mean values)
    quality_cols = [col for col in df_merged.columns if 'mean' in col.lower() or col in ['psnr', 'ssim', 'lpips']]
    column_order.extend(sorted(quality_cols))

    # Add remaining columns
    remaining_cols = [col for col in df_merged.columns if col not in column_order]
    column_order.extend(sorted(remaining_cols))

    # Reorder
    df_merged = df_merged[[col for col in column_order if col in df_merged.columns]]

    # Save merged data
    merged_csv = results_dir / "metrics_combined.csv"
    df_merged.to_csv(merged_csv, index=False)
    print(f"\nMerged metrics saved to {merged_csv}")

    # Save as JSON
    merged_json = results_dir / "metrics_combined.json"
    df_merged.to_json(merged_json, orient='records', indent=2)
    print(f"Merged metrics saved to {merged_json}")

    # Print summary with color coding
    print("\n" + "="*100)
    print("COMBINED METRICS SUMMARY (Color coded: Green=Best, Red=Worst)")
    print("="*100)

    # Select key columns for display
    display_cols = ['codec', 'filesize_ratio', 'compression_time_sec', 'decompression_time_sec']

    # Add mean quality metrics if available
    for metric in ['psnr', 'ssim', 'lpips']:
        mean_col = f'{metric}_mean'
        if mean_col in df_merged.columns:
            display_cols.append(mean_col)
        elif metric in df_merged.columns:
            display_cols.append(metric)

    display_df = df_merged[display_cols].copy()

    # Create rich table with color coding
    console = Console()
    table = Table(show_header=True, header_style="bold cyan")

    # Add columns
    for col in display_cols:
        table.add_column(col, justify="right" if col != "codec" else "left")

    # Helper function to get color based on normalized value
    def get_color_gradient(value, min_val, max_val, reverse=False):
        """
        Return RGB color from red (worst) to green (best).
        If reverse=True, lower is better (e.g., filesize, time, lpips).
        """
        if pd.isna(value):
            return "white"

        # If all values are the same, show as neutral (yellow)
        if min_val == max_val:
            return "rgb(255,255,0)"

        # Normalize to 0-1
        normalized = (value - min_val) / (max_val - min_val)

        # Handle any NaN from division
        if np.isnan(normalized):
            return "rgb(255,255,0)"

        if reverse:
            normalized = 1 - normalized  # Invert for "lower is better" metrics

        # Clamp to [0, 1]
        normalized = max(0.0, min(1.0, normalized))

        # Create gradient: red (0) -> yellow (0.5) -> green (1)
        if normalized < 0.5:
            # Red to yellow
            r = 255
            g = int(255 * (normalized * 2))
            b = 0
        else:
            # Yellow to green
            r = int(255 * (1 - (normalized - 0.5) * 2))
            g = 255
            b = 0

        return f"rgb({r},{g},{b})"

    # Determine which columns should be reversed (lower is better)
    reverse_cols = {'filesize_ratio', 'compression_time_sec', 'decompression_time_sec', 'lpips_mean'}

    # Add rows with color coding
    for _, row in display_df.iterrows():
        row_data = []
        for col in display_cols:
            value = row[col]

            if col == 'codec':
                # Codec name without color
                row_data.append(str(value))
            elif pd.isna(value):
                row_data.append("[dim]N/A[/dim]")
            else:
                # Get min/max for this column
                col_values = display_df[col].dropna()
                min_val = col_values.min()
                max_val = col_values.max()

                # Determine if lower is better
                reverse = col in reverse_cols

                # Get color
                color = get_color_gradient(value, min_val, max_val, reverse=reverse)

                # Format value
                if isinstance(value, (int, float)):
                    formatted = f"{value:.4f}"
                else:
                    formatted = str(value)

                row_data.append(f"[{color}]{formatted}[/{color}]")

        table.add_row(*row_data)

    console.print(table)
    print("="*100)

    # Save the table as markdown with color coding
    md_path = results_dir / "metrics_combined.md"
    with open(md_path, 'w') as f:
        f.write("# Combined Compression and Quality Metrics\n\n")
        f.write("Color Legend: 🟢 Best → 🟡 Average → 🔴 Worst\n\n")

        # Create markdown table header
        header = "| " + " | ".join(display_cols) + " |\n"
        separator = "| " + " | ".join([":---:" if col != "codec" else ":---" for col in display_cols]) + " |\n"
        f.write(header)
        f.write(separator)

        # Add rows with color emoji indicators
        for _, row in display_df.iterrows():
            row_data = []
            for col in display_cols:
                value = row[col]

                if col == 'codec':
                    row_data.append(str(value))
                elif pd.isna(value):
                    row_data.append("N/A")
                else:
                    # Get min/max for this column
                    col_values = display_df[col].dropna()
                    min_val = col_values.min()
                    max_val = col_values.max()

                    # Determine if lower is better
                    reverse = col in reverse_cols

                    # Calculate normalized value
                    if min_val == max_val:
                        emoji = "🟡"
                    else:
                        normalized = (value - min_val) / (max_val - min_val)
                        if reverse:
                            normalized = 1 - normalized

                        # Assign emoji based on value
                        if normalized >= 0.66:
                            emoji = "🟢"
                        elif normalized >= 0.33:
                            emoji = "🟡"
                        else:
                            emoji = "🔴"

                    # Format value
                    if isinstance(value, (int, float)):
                        formatted = f"{emoji} {value:.4f}"
                    else:
                        formatted = f"{emoji} {value}"

                    row_data.append(formatted)

            f.write("| " + " | ".join(row_data) + " |\n")

        f.write("\n## Metrics Explanation\n\n")
        f.write("- **filesize_ratio**: Compressed size / raw size (lower is better)\n")
        f.write("- **compression_time_sec**: Time to compress all images (lower is better)\n")
        f.write("- **decompression_time_sec**: Time to decompress all images (lower is better)\n")
        f.write("- **psnr_mean**: Peak Signal-to-Noise Ratio in dB (higher is better, 30+ good, 35+ excellent)\n")
        f.write("- **ssim_mean**: Structural Similarity Index 0-1 (higher is better, 0.9+ good)\n")
        f.write("- **lpips_mean**: Learned Perceptual similarity (lower is better, <0.1 good)\n")

    print(f"\nColor-coded markdown table saved to {md_path}")

    return df_merged


if __name__ == "__main__":
    merge_metrics()
