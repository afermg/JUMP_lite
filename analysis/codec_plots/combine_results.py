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

    # Create PA vs PC plot
    pa_pc_plot_path = output_path.parent / "pa_vs_pc_plot.png"
    create_pa_pc_plot(combined, pa_pc_plot_path)

    return 0


def create_pa_pc_plot(df: pd.DataFrame, output_path: Path):
    """Create a PA vs PC scatter plot with dot size by codec and color by model."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get unique feature types (models) and assign colors
    feature_types = sorted(df["feature_type"].unique())
    cmap = plt.colormaps.get_cmap("tab10")
    colors = {ft: cmap(i) for i, ft in enumerate(feature_types)}

    # Get unique codecs and assign sizes (ordered by file size: zstd > hq > effort > mq > lq)
    codec_order = ["zstd", "jpegxl_lossy_hq", "jpegxl_lossy_effort_3", "jpegxl_lossy_mq", "jpegxl_lossy_lq"]
    codecs = [c for c in codec_order if c in df["codec"].unique()]
    # Add any codecs not in the predefined order
    codecs += [c for c in df["codec"].unique() if c not in codec_order]
    # Map codecs to sizes (larger size for larger file size codecs, zstd is largest)
    size_min, size_max = 50, 600
    codec_sizes = {codec: size_max - (size_max - size_min) * i / max(1, len(codecs) - 1)
                   for i, codec in enumerate(codecs)}

    # Plot each point
    for _, row in df.iterrows():
        ax.scatter(
            row["pc"],
            row["pa"],
            c=[colors[row["feature_type"]]],
            s=codec_sizes[row["codec"]],
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

    # Create legend for models (colors)
    color_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[ft],
                   markersize=10, label=ft)
        for ft in feature_types
    ]
    legend1 = ax.legend(handles=color_handles, title="Model", loc="upper left",
                        bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.add_artist(legend1)

    # Create legend for codecs (sizes)
    size_handles = [
        plt.Line2D([0], [0], marker="o", color="gray", markersize=codec_sizes[codec] ** 0.5,
                   label=df[df["codec"] == codec]["codec_label"].iloc[0] if "codec_label" in df.columns else codec,
                   linestyle="None")
        for codec in codecs
    ]
    ax.legend(handles=size_handles, title="Codec (size)", loc="lower left",
              bbox_to_anchor=(1.02, 0), fontsize=9)

    ax.set_xlabel("Phenotypic Consistency (PC)", fontsize=12)
    ax.set_ylabel("Phenotypic Activity (PA %)", fontsize=12)
    ax.set_title("PA vs PC by Model and Codec", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.4)
    plt.close()

    print(f"Saved PA vs PC plot to: {output_path}")


def create_metrics_plot(df: pd.DataFrame, output_path: Path):
    """Create a 5-panel plot showing metrics vs file size, colored by feature_type."""
    metrics = [
        ("pa", "Phenotypic Activity (%)"),
        ("pc", "Phenotypic Consistency"),
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
