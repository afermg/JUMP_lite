#!/usr/bin/env python3
"""Render a clearer presentation of Figure S16 from immutable N=5 summaries.

This is a presentation-only renderer. It does not recompute any scientific
quantity. Panels c and d horizontally dodge compression settings; panel e
shows the same PA/PC means without duplicating the marginal intervals already
shown in panels c and d.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

FAMILIES = [
    "cell_count",
    "cellprofiler",
    "dinov2",
    "dinov2_random",
    "morphem",
    "openphenom",
    "subcell",
]
LABELS = {
    "cell_count": "Cell Count",
    "cellprofiler": "CellProfiler",
    "dinov2": "DINOv2",
    "dinov2_random": "ViT-rand",
    "morphem": "MorphEM",
    "openphenom": "OpenPhenom",
    "subcell": "SubCell",
}
COLORS = {
    "cell_count": "#AD6892",
    "cellprofiler": "#FC8D62",
    "dinov2": "#E5C494",
    "dinov2_random": "#66C2A5",
    "morphem": "#A6D854",
    "openphenom": "#FFD92F",
    "subcell": "#8DA0CB",
}
SETTINGS = ["Raw", "HQ", "MQ", "D20"]
MARKERS = {"Raw": "o", "HQ": "s", "MQ": "^", "D20": "D"}
DODGE = {"Raw": -0.30, "HQ": -0.10, "MQ": 0.10, "D20": 0.30}
EXPECTED_INPUTS = {
    "development_end_to_end_t_intervals.csv": "8301ff6956564f1ff1e942f84adbbb5f9e06ab7747fa6b7f735b2aff368c6dc5",
    "heldout_end_to_end_t_intervals.csv": "557488bcb4860eaceed71f93d0a0511404c22e90f4ccc0bf2ca5b1ce268864eb",
    "paired_raw_difference_t_intervals.csv": "a1fc7b4d0fc960ba528d55fdfd4fdfd035561befd356d792c63ece77722287f2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asymmetric_error(mean: float, low: float, high: float) -> list[list[float]]:
    return [[mean - low], [high - mean]]


def validate_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for name, expected in EXPECTED_INPUTS.items():
        path = input_dir / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"input identity mismatch: {path}")

    development = pd.read_csv(input_dir / "development_end_to_end_t_intervals.csv")
    heldout = pd.read_csv(input_dir / "heldout_end_to_end_t_intervals.csv")
    paired = pd.read_csv(input_dir / "paired_raw_difference_t_intervals.csv")

    if development["family"].tolist() != FAMILIES:
        raise RuntimeError("development family order/closure mismatch")
    if paired["family"].tolist() != FAMILIES:
        raise RuntimeError("paired family order/closure mismatch")
    expected_pairs = {
        (family, setting) for family in FAMILIES[2:] for setting in SETTINGS
    } | {("cell_count", "Raw"), ("cellprofiler", "Raw")}
    observed_pairs = set(zip(heldout["family"], heldout["codec"], strict=True))
    if len(heldout) != 22 or observed_pairs != expected_pairs:
        raise RuntimeError("held-out family/setting closure mismatch")
    for frame in (development, heldout, paired):
        if not (frame["n"].eq(5).all() and frame["df"].eq(4).all()):
            raise RuntimeError("interval identity mismatch")
    return development, heldout, paired


def style_categorical_axis(axis: plt.Axes) -> None:
    x = np.arange(len(FAMILIES))
    axis.axhline(0, color=".78", linewidth=0.8, zorder=0)
    axis.set_xticks(x, [LABELS[family] for family in FAMILIES], rotation=38, ha="right")
    axis.set_ylabel("Mean normalized AP")
    axis.margins(x=0.035)


def add_panel_letters(axes: np.ndarray) -> None:
    for letter, axis in zip("abcdef", axes, strict=True):
        axis.text(
            -0.105,
            1.075,
            letter,
            transform=axis.transAxes,
            fontsize=15,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )


def render(input_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    development, heldout, paired = validate_inputs(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "strict_selection_heldout_pa_pc.png"
    pdf = output_dir / "strict_selection_heldout_pa_pc.pdf"
    if png.exists() or pdf.exists():
        raise RuntimeError(f"no-clobber output exists under {output_dir}")

    x = np.arange(len(FAMILIES))
    fig, axes_grid = plt.subplots(2, 3, figsize=(15.5, 9.4), constrained_layout=True)
    axes = axes_grid.ravel()

    # Panels a and b: Raw recipe-selection performance.
    for panel, (metric, title) in enumerate(
        [
            ("validation_pa_mean_nap", "PA — recipe-selection split (Raw)"),
            ("validation_pc_mean_nap", "PC — recipe-selection split (Raw)"),
        ]
    ):
        axis = axes[panel]
        for index, family in enumerate(FAMILIES):
            row = development.loc[development["family"].eq(family)].iloc[0]
            mean = float(row[f"{metric}_mean"])
            low = float(row[f"{metric}_ci95_low"])
            high = float(row[f"{metric}_ci95_high"])
            axis.errorbar(
                index,
                mean,
                yerr=asymmetric_error(mean, low, high),
                fmt="o",
                markersize=8,
                color=COLORS[family],
                ecolor=COLORS[family],
                elinewidth=1.2,
                markeredgecolor="black",
                markeredgewidth=0.6,
                capsize=3,
                zorder=3,
            )
        axis.set_title(title, fontweight="bold")
        style_categorical_axis(axis)

    # Panels c and d: wider setting dodge keeps neighboring interval stems apart.
    for panel, (metric, title) in enumerate(
        [
            ("test_pa_mean_nap", "PA — held-out test split"),
            ("test_pc_mean_nap", "PC — held-out test split"),
        ],
        start=2,
    ):
        axis = axes[panel]
        for index, family in enumerate(FAMILIES):
            rows = heldout.loc[heldout["family"].eq(family)]
            for setting in SETTINGS:
                match = rows.loc[rows["codec"].eq(setting)]
                if match.empty:
                    continue
                row = match.iloc[0]
                mean = float(row[f"{metric}_mean"])
                low = float(row[f"{metric}_ci95_low"])
                high = float(row[f"{metric}_ci95_high"])
                axis.errorbar(
                    index + DODGE[setting],
                    mean,
                    yerr=asymmetric_error(mean, low, high),
                    fmt=MARKERS[setting],
                    markersize=7,
                    color=COLORS[family],
                    ecolor="0.18",
                    elinewidth=1.05,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    capsize=2.3,
                    capthick=1.05,
                    zorder=3,
                )
        axis.set_title(title, fontweight="bold")
        style_categorical_axis(axis)

    # Panel e: means only. Marginal intervals are already visible in c and d.
    axis = axes[4]
    for family in FAMILIES:
        rows = heldout.loc[heldout["family"].eq(family)].copy()
        rows["_setting_order"] = rows["codec"].map(
            {s: i for i, s in enumerate(SETTINGS)}
        )
        rows.sort_values("_setting_order", inplace=True)
        if len(rows) > 1:
            axis.plot(
                rows["test_pa_mean_nap_mean"],
                rows["test_pc_mean_nap_mean"],
                color=COLORS[family],
                linewidth=1.4,
                alpha=0.70,
                zorder=1,
            )
        for row in rows.itertuples(index=False):
            axis.plot(
                row.test_pa_mean_nap_mean,
                row.test_pc_mean_nap_mean,
                marker=MARKERS[row.codec],
                markersize=7,
                linestyle="",
                markerfacecolor=COLORS[family],
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=3,
            )
    axis.set_title("PA vs PC — held-out test split", fontweight="bold")
    axis.set_xlabel("Held-out PA mean normalized AP")
    axis.set_ylabel("Held-out PC mean normalized AP")
    axis.axhline(0, color=".82", linewidth=0.7, zorder=0)
    axis.axvline(0, color=".82", linewidth=0.7, zorder=0)
    axis.text(
        0.03,
        0.97,
        "Means shown; intervals are in c and d",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.35",
    )

    # Panel f: paired Raw held-out-minus-selection differences.
    axis = axes[5]
    width = 0.34
    for metric, delta_x, hatch, label in (
        ("raw_heldout_minus_development_pa", -width / 2, None, "PA difference"),
        ("raw_heldout_minus_development_pc", width / 2, "//", "PC difference"),
    ):
        means: list[float] = []
        errors: list[float] = []
        for family in FAMILIES:
            row = paired.loc[paired["family"].eq(family)].iloc[0]
            mean = float(row[f"{metric}_mean"])
            means.append(mean)
            errors.append(float(row[f"{metric}_ci95_high"]) - mean)
        axis.bar(
            x + delta_x,
            means,
            width,
            yerr=errors,
            capsize=3,
            color=[COLORS[family] for family in FAMILIES],
            edgecolor="black",
            error_kw={"ecolor": "0.12", "elinewidth": 1.05, "capthick": 1.05},
            linewidth=0.4,
            hatch=hatch,
            label=label,
            zorder=3,
        )
    axis.axhline(0, color="black", linewidth=0.8, zorder=1)
    axis.set_title(
        "Held-out − recipe-selection (paired within seed; Raw)", fontweight="bold"
    )
    axis.set_ylabel("Difference in mean normalized AP")
    axis.set_xticks(x, [LABELS[family] for family in FAMILIES], rotation=38, ha="right")
    axis.legend(frameon=False)

    setting_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[setting],
            linestyle="",
            color="black",
            label=setting,
        )
        for setting in SETTINGS
    ]
    axes[4].legend(
        handles=setting_handles,
        title="Compression setting (shape)",
        frameon=False,
        loc="lower right",
    )
    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=COLORS[family],
            markeredgecolor="black",
            label=LABELS[family],
        )
        for family in FAMILIES
    ]
    fig.legend(
        handles=family_handles,
        loc="outside lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    add_panel_letters(axes)
    fig.suptitle(
        "Held-out performance across five split replicates (n=5)",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "render_s16_clear_intervals.py"},
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "render_s16_clear_intervals.py",
            "CreationDate": None,
        },
    )
    plt.close(fig)
    return png, pdf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    png, pdf = render(args.input_dir.resolve(), args.output_dir.resolve())
    print(f"{png}\t{sha256(png)}")
    print(f"{pdf}\t{sha256(pdf)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
