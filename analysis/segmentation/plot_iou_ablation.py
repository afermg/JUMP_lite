#!/usr/bin/env python3
"""Plot segmentation metrics across IoU thresholds to show consistency.

Reads detailed_results CSVs (already computed at IoU 0.5, 0.7, 0.8, 0.9)
and produces a figure showing that codec rankings are preserved across thresholds.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

THRESHOLDS = [50, 70, 80, 90]
THRESHOLD_LABELS = ["0.5", "0.7", "0.8", "0.9"]

CODEC_DISPLAY = {
    "jpegxl_lossy_hq": "High",
    "jpegxl_lossy_effort_3": "Mid-High",
    "jpegxl_lossy_d2_e8": "D2E8",
    "jpegxl_lossy_mq": "Medium",
    "jpegxl_lossy_lq": "Low",
    "jpegxl_lossy_d10": "D10",
}

CODEC_ORDER = ["High", "Mid-High", "D2E8", "Medium", "Low", "D10"]


def load_detailed_results(results_dir: Path) -> pd.DataFrame:
    """Load all detailed_results CSVs into a single DataFrame."""
    rows = []
    for csv_path in sorted(results_dir.glob("segment_*.csv")):
        name = csv_path.stem  # e.g. segment_cell_jpegxl_lossy_hq
        parts = name.split("_", 2)  # ['segment', 'cell', 'jpegxl_lossy_hq']
        obj_type = parts[1]  # cell or nuclei
        codec_key = parts[2]  # jpegxl_lossy_hq

        df = pd.read_csv(csv_path)
        df["object_type"] = obj_type
        df["codec"] = CODEC_DISPLAY.get(codec_key, codec_key)
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def plot_ablation(data: pd.DataFrame, metric: str, output_path: Path):
    """Plot metric across IoU thresholds, one panel per object type."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, obj_type in zip(axes, ["cell", "nuclei"]):
        sub = data[data["object_type"] == obj_type]

        # Compute mean metric per codec per threshold
        summary = []
        for thresh in THRESHOLDS:
            col = f"inst_{metric}_{thresh}"
            if col not in sub.columns:
                continue
            grouped = sub.groupby("codec")[col].agg(["mean", "std"]).reset_index()
            grouped["threshold"] = f"{thresh / 100:.1f}"
            grouped.rename(columns={"mean": "value", "std": "std"}, inplace=True)
            summary.append(grouped)

        summary = pd.concat(summary, ignore_index=True)

        for codec in CODEC_ORDER:
            codec_data = summary[summary["codec"] == codec]
            if codec_data.empty:
                continue
            ax.errorbar(
                codec_data["threshold"],
                codec_data["value"],
                yerr=codec_data["std"],
                marker="o",
                label=codec,
                capsize=3,
                linewidth=1.5,
                markersize=5,
            )

        ax.set_xlabel("IoU Threshold")
        ax.set_title(f"{obj_type.capitalize()} Segmentation")
        ax.grid(True, alpha=0.3)

    label = "AP" if metric == "accuracy" else f"Instance {metric.upper()}"
    axes[0].set_ylabel(label)
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle(
        f"{label} across IoU Thresholds by Compression Codec",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("analysis/segmentation/output/segmentation_comparison/detailed_results"),
        help="Directory containing segment_*.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/segmentation/output"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy"],
        help="Metrics to plot (default: accuracy, i.e. AP per StarDist/DSB2018)",
    )
    args = parser.parse_args()

    data = load_detailed_results(args.results_dir)
    print(f"Loaded {len(data)} rows, {data['codec'].nunique()} codecs, {data['object_type'].nunique()} object types")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        output_path = args.output_dir / f"iou_ablation_{metric}.png"
        plot_ablation(data, metric, output_path)


if __name__ == "__main__":
    main()
