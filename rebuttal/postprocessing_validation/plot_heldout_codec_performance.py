#!/usr/bin/env python3
"""Plot fixed-recipe held-out PA/PC performance across codecs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
        "--uncertainty-csv",
        type=Path,
        default=here / "results" / "uncertainty" / "heldout_uncertainty.csv",
        help="Conditional paired cluster-bootstrap score intervals.",
    )
    parser.add_argument(
        "--codec-comparisons-csv",
        type=Path,
        default=here / "results" / "uncertainty" / "codec_vs_raw_paired.csv",
        help="Paired codec-vs-Raw changes and intervals.",
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


def load_uncertainty(path: Path, scores: pd.DataFrame) -> pd.DataFrame:
    uncertainty = pd.read_csv(path)
    required = {
        "family",
        "codec",
        "product_point",
        "product_ci_low",
        "product_ci_high",
        "replicates",
    }
    if missing_columns := required.difference(uncertainty.columns):
        raise ValueError(f"Missing uncertainty columns: {sorted(missing_columns)}")
    uncertainty = uncertainty.loc[
        uncertainty["family"].isin(FAMILIES) & uncertainty["codec"].isin(CODECS)
    ].copy()
    observed = set(zip(uncertainty["family"], uncertainty["codec"]))
    expected = {(family, codec) for family in FAMILIES for codec in CODECS}
    if observed != expected or len(uncertainty) != len(expected):
        raise ValueError("Expected one uncertainty row per plotted model/codec")
    if uncertainty["replicates"].nunique() != 1:
        raise ValueError("Uncertainty rows use different bootstrap replicate counts")
    merged = uncertainty.merge(
        scores[["family", "codec", "test_balanced_nap_product"]],
        on=["family", "codec"],
        how="left",
        validate="one_to_one",
    )
    error = (
        merged["product_point"] - merged["test_balanced_nap_product"]
    ).abs().max()
    if error > 1e-12:
        raise ValueError(f"Uncertainty point estimates disagree with score table: {error}")
    if not (
        (merged["product_ci_low"] <= merged["product_point"])
        & (merged["product_point"] <= merged["product_ci_high"])
    ).all():
        raise ValueError("A product point estimate falls outside its interval")
    merged["family"] = pd.Categorical(merged["family"], FAMILIES, ordered=True)
    merged["codec"] = pd.Categorical(merged["codec"], CODECS, ordered=True)
    return merged.sort_values(["family", "codec"])


def load_codec_comparisons(path: Path, scores: pd.DataFrame) -> pd.DataFrame:
    comparisons = pd.read_csv(path)
    required = {
        "family",
        "codec",
        "primary_learned_model",
        "product_relative_pct_point",
        "product_relative_pct_ci_low",
        "product_relative_pct_ci_high",
        "product_relative_pct_suppressed",
    }
    if missing_columns := required.difference(comparisons.columns):
        raise ValueError(f"Missing codec-comparison columns: {sorted(missing_columns)}")
    comparisons = comparisons.loc[
        comparisons["family"].isin(FAMILIES)
        & comparisons["codec"].isin(CODECS[1:])
        & comparisons["primary_learned_model"]
    ].copy()
    observed = set(zip(comparisons["family"], comparisons["codec"]))
    expected = {(family, codec) for family in FAMILIES for codec in CODECS[1:]}
    if observed != expected or len(comparisons) != len(expected):
        raise ValueError("Expected one paired change per plotted model/lossy codec")
    if comparisons["product_relative_pct_suppressed"].any():
        raise ValueError("A plotted relative interval was suppressed as unstable")
    score_lookup = scores.set_index(["family", "codec"])["test_balanced_nap_product"]
    for row in comparisons.itertuples(index=False):
        expected_point = 100.0 * (
            score_lookup.loc[(row.family, row.codec)]
            / score_lookup.loc[(row.family, "Raw")]
            - 1.0
        )
        if abs(expected_point - row.product_relative_pct_point) > 1e-10:
            raise ValueError(
                f"Relative-change point disagrees for {row.family}/{row.codec}"
            )
    comparisons["family"] = pd.Categorical(
        comparisons["family"], FAMILIES, ordered=True
    )
    comparisons["codec"] = pd.Categorical(
        comparisons["codec"], CODECS[1:], ordered=True
    )
    return comparisons.sort_values(["family", "codec"])


def make_figure(
    scores: pd.DataFrame,
    uncertainty: pd.DataFrame,
    codec_comparisons: pd.DataFrame,
) -> plt.Figure:
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
        model_uncertainty = uncertainty.loc[uncertainty["family"] == family].copy()
        product = model["test_balanced_nap_product"].to_numpy()
        product_low = model_uncertainty["product_ci_low"].to_numpy()
        product_high = model_uncertainty["product_ci_high"].to_numpy()
        comparisons = codec_comparisons.loc[
            codec_comparisons["family"] == family
        ].copy()
        relative_change = np.r_[
            0.0, comparisons["product_relative_pct_point"].to_numpy()
        ]
        relative_low = np.r_[
            0.0, comparisons["product_relative_pct_ci_low"].to_numpy()
        ]
        relative_high = np.r_[
            0.0, comparisons["product_relative_pct_ci_high"].to_numpy()
        ]
        style = {
            "color": COLORS[family],
            "marker": MARKERS[family],
            "linewidth": 1.7,
            "markersize": 5.2,
            "markeredgecolor": "white",
            "markeredgewidth": 0.45,
            "capsize": 2.0,
            "elinewidth": 0.9,
        }
        axes[0].errorbar(
            x,
            product * 100.0,
            yerr=[
                (product - product_low) * 100.0,
                (product_high - product) * 100.0,
            ],
            label=DISPLAY_NAMES[family],
            **style,
        )
        axes[1].errorbar(
            x,
            relative_change,
            yerr=[relative_change - relative_low, relative_high - relative_change],
            label=DISPLAY_NAMES[family],
            **style,
        )

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
    uncertainty = load_uncertainty(args.uncertainty_csv, scores)
    codec_comparisons = load_codec_comparisons(args.codec_comparisons_csv, scores)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure(scores, uncertainty, codec_comparisons)
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
