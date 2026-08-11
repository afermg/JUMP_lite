#!/usr/bin/env python3
"""Plot fixed-recipe held-out PA/PC performance across codecs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CODECS = ["Raw", "HQ", "MQ", "D20"]
FAMILIES = ["morphem", "dinov2", "subcell", "openphenom"]
DISPLAY_NAMES = {
    "morphem": "MorphEM",
    "dinov2": "DINOv2",
    "subcell": "SubCell",
    "openphenom": "OpenPhenom",
}
COLORS = {
    "morphem": "#0072B2",
    "dinov2": "#D55E00",
    "subcell": "#009E73",
    "openphenom": "#CC79A7",
}
MARKERS = {
    "morphem": "o",
    "dinov2": "s",
    "subcell": "^",
    "openphenom": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=here / "results" / "heldout_test_scores.csv",
        help="Held-out score table produced by run_analysis.py.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=here / "results" / "heldout_codec_performance",
        help="Output path without extension; both PDF and PNG are written.",
    )
    return parser.parse_args()


def load_scores(path: Path) -> pd.DataFrame:
    scores = pd.read_csv(path)
    required = {"family", "codec", "status", "test_balanced_nap_product"}
    missing_columns = required.difference(scores.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    scores = scores.loc[
        scores["family"].isin(FAMILIES) & scores["codec"].isin(CODECS)
    ].copy()
    if not (scores["status"] == "ok").all():
        failed = scores.loc[scores["status"] != "ok", ["family", "codec", "status"]]
        raise ValueError(f"Cannot plot failed profiles:\n{failed.to_string(index=False)}")

    observed = set(zip(scores["family"], scores["codec"]))
    expected = {(family, codec) for family in FAMILIES for codec in CODECS}
    if observed != expected or len(scores) != len(expected):
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise ValueError(f"Expected one row per model/codec; missing={missing}, extra={extra}")

    scores["family"] = pd.Categorical(scores["family"], FAMILIES, ordered=True)
    scores["codec"] = pd.Categorical(scores["codec"], CODECS, ordered=True)
    return scores.sort_values(["family", "codec"])


def make_figure(scores: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    x = list(range(len(CODECS)))

    for family in FAMILIES:
        model = scores.loc[scores["family"] == family].copy()
        product = model["test_balanced_nap_product"].to_numpy()
        relative_change = 100.0 * (product / product[0] - 1.0)
        style = {
            "color": COLORS[family],
            "marker": MARKERS[family],
            "linewidth": 1.7,
            "markersize": 5.2,
            "markeredgecolor": "white",
            "markeredgewidth": 0.45,
        }
        axes[0].plot(x, product * 100.0, label=DISPLAY_NAMES[family], **style)
        axes[1].plot(x, relative_change, label=DISPLAY_NAMES[family], **style)

    axes[0].set_title("(a) Absolute held-out score", loc="left", fontweight="bold")
    axes[0].set_ylabel(r"Balanced PA--PC product ($\times 10^{-2}$)")
    axes[0].set_ylim(bottom=0)

    axes[1].set_title("(b) Change from model-specific Raw", loc="left", fontweight="bold")
    axes[1].set_ylabel("Change from Raw (%)")
    axes[1].axhline(0, color="#555555", linewidth=0.9, linestyle="--", zorder=0)
    axes[1].set_ylim(-58, 13)

    for axis in axes:
        axis.set_xticks(x, CODECS)
        axis.set_xlabel("Image codec")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
        axis.tick_params(length=3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.5,
    )
    return fig


def main() -> None:
    args = parse_args()
    scores = load_scores(args.input_csv)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure(scores)
    figure.savefig(
        args.output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"Creator": "plot_heldout_codec_performance.py", "CreationDate": None},
    )
    figure.savefig(
        args.output_prefix.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "plot_heldout_codec_performance.py"},
    )
    plt.close(figure)
    print(args.output_prefix.with_suffix(".pdf"))
    print(args.output_prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
