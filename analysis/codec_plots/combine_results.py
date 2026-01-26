#!/usr/bin/env python3
"""Combine all _results.csv files from codec comparison into a single table."""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Combine codec comparison results")
    parser.add_argument("folder", type=Path, help="Folder containing *_results.csv files")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output CSV path (default: <folder>/combined_results.csv)")
    args = parser.parse_args()

    if not args.folder.exists():
        print(f"Error: Folder {args.folder} does not exist")
        return 1

    # Find all _results.csv files (exclude combined_results.csv)
    csv_files = [f for f in args.folder.glob("*_results.csv")
                 if f.name != "combined_results.csv"]

    if not csv_files:
        print(f"No *_results.csv files found in {args.folder}")
        return 1

    print(f"Found {len(csv_files)} result files:")

    # Read and combine
    dfs = []
    for csv_file in sorted(csv_files):
        # Extract feature type from filename (e.g., "dinov2_results.csv" -> "dinov2")
        feature_type = csv_file.stem.replace("_results", "")
        print(f"  - {csv_file.name} ({feature_type})")

        df = pd.read_csv(csv_file)
        df.insert(0, "feature_type", feature_type)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Sort by feature_type, then by file_size_pct descending
    combined = combined.sort_values(["feature_type", "file_size_pct"], ascending=[True, False])

    # Output path
    output_path = args.output if args.output else args.folder / "combined_results.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nCombined {len(combined)} rows from {len(csv_files)} files")
    print(f"Saved to: {output_path}")

    # Print full combined table
    print("\n" + "=" * 120)
    print("COMBINED RESULTS")
    print("=" * 120)
    print(combined.to_string(index=False))

    # Create multi-metric plot
    plot_path = output_path.parent / "combined_metrics_plot.png"
    create_metrics_plot(combined, plot_path)

    return 0


def create_metrics_plot(df: pd.DataFrame, output_path: Path):
    """Create a 5-panel plot showing metrics vs file size, colored by feature_type."""
    metrics = [
        ("pa", "Phenotypic Activity (%)"),
        ("pc", "Profile Consistency"),
        ("balance", "Balance (PA × PC)"),
        ("silhouette", "Silhouette Score"),
        ("kbet", "kBET Score"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Get unique feature types and assign colors
    feature_types = sorted(df["feature_type"].unique())
    cmap = plt.colormaps.get_cmap("tab10")
    colors = {ft: cmap(i) for i, ft in enumerate(feature_types)}

    for idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[idx]

        for ft in feature_types:
            ft_data = df[df["feature_type"] == ft].sort_values("file_size_pct", ascending=False)

            if metric_col in ft_data.columns:
                ax.plot(
                    ft_data["file_size_pct"],
                    ft_data[metric_col],
                    "o-",
                    color=colors[ft],
                    label=ft,
                    markersize=8,
                    linewidth=2,
                    alpha=0.8,
                )

        ax.set_xlabel("File Size (% of ZSTD)", fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    # Put legend in the 6th subplot space (bottom right)
    axes[5].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc="center", fontsize=10, title="Feature Type", title_fontsize=12)

    plt.suptitle("Codec Comparison: All Metrics vs File Size", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    exit(main())
