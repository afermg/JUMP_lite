#!/usr/bin/env python3
"""Plot MOTIVE codec delta from the raw / lossless baseline per family.

Mirrors the absolute-only variant of
``src/norm_3/gather_sweep_results.py:generate_codec_delta_from_raw_groups_plot``:

- Per family, identify a baseline codec (preference order: ``raw`` →
  ``jpegxl_lossy_raw`` (= lossless JpegXL) → ``zstd`` → ``lz4hc``).
- For each *other* codec in that family, pair configs by ``config`` name and
  compute ``delta = codec_recall - baseline_recall``.
- Codecs run along the x-axis grouped by family (gaps between families). The
  y-axis shows ``Δ recall@k%`` in percentage points. A horizontal ``y=0``
  line marks "no difference vs baseline"; negative = codec is worse.
- Each codec column has translucent dots (one per config) plus a black-edged
  diamond at the mean delta, with a stem connecting the mean back to ``y=0``.
- Family colours are explained by a single legend at the bottom of the figure
  (no per-panel family brackets).
- One subplot per MOTIVE metric (8 total), arranged in a 2×4 grid.

Reads the long-format ``motive_sweep_summary.csv`` produced by
``analysis/plot_motive_results.py``. Writes a single PNG per recall@k%.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


SET2 = sns.color_palette("Set2", 8)
FAMILY_COLOR = {
    "cell_count": SET2[3],
    "cellprofiler": SET2[1],
    "cp_measure": SET2[1],
    "dinov2": SET2[6],
    "dinov2_random": SET2[0],
    "morphem": SET2[4],
    "openphenom": SET2[5],
    "subcell": SET2[2],
    "subcell__clip01": SET2[2],
}
FAMILY_DISPLAY = {
    "cell_count": "CellCount",
    "cellprofiler": "CellProfiler",
    "cp_measure": "CP-Measure",
    "dinov2": "DINOv2",
    "dinov2_random": "ViT-rand",
    "morphem": "MorphEm",
    "openphenom": "OpenPhenom",
    "subcell": "SubCell",
    "subcell__clip01": "SubCell",
}
FAMILY_PLOT_ORDER = [
    "cell_count", "cellprofiler", "cp_measure",
    "dinov2", "morphem", "openphenom",
    "subcell", "subcell__clip01",
    "dinov2_random",
]
CODEC_DISPLAY = {
    "raw": "raw",
    "zstd": "zstd",
    "lz4hc": "lz4hc",
    "blosc_zstd": "blosc-zstd",
    "jpegxl_lossy_raw": "lossless",
    "jpegxl_lossy_hq": "hq",
    "jpegxl_lossy_mq": "mq",
    "jpegxl_lossy_lq": "lq",
    "jpegxl_lossy_d20": "d20",
    "jpegxl_lossy_d50": "d50",
}
CODEC_ORDER = [
    "raw",
    "zstd",
    "lz4hc",
    "blosc_zstd",
    "jpegxl_lossy_raw",
    "jpegxl_lossy_hq",
    "jpegxl_lossy_mq",
    "jpegxl_lossy_lq",
    "jpegxl_lossy_d20",
    "jpegxl_lossy_d50",
]
BASELINE_PREFERENCE = ("raw", "jpegxl_lossy_raw", "zstd", "lz4hc", "blosc_zstd")


# (task, modality, group, panel_label, panel_subtitle)
# ``panel_subtitle`` is what appears above each subplot. Multi-word subtitles
# are split onto two lines for compactness, mirroring the reference script.
PANELS: list[tuple[str, str | None, str | None, str]] = [
    ("CC", None, "group_high", "CC\nDivs↔Divs"),
    ("CC", None, "group_low", "CC\nBioact↔Bioact"),
    ("GG", "orf", None, "GG\nORF"),
    ("GG", "crispr", None, "GG\nCRISPR"),
    ("CG", "orf", "group_high", "CG\nDivs→ORF"),
    ("CG", "crispr", "group_high", "CG\nDivs→CRISPR"),
    ("CG", "orf", "group_low", "CG\nBioact→ORF"),
    ("CG", "crispr", "group_low", "CG\nBioact→CRISPR"),
]


def _codec_sort_key(codec: str) -> tuple[int, str]:
    if codec in CODEC_ORDER:
        return CODEC_ORDER.index(codec), codec
    return len(CODEC_ORDER), codec


def filter_panel(df: pl.DataFrame, task, modality, group) -> pl.DataFrame:
    f = df.filter(pl.col("task") == task)
    f = f.filter(pl.col("modality").is_null() if modality is None else pl.col("modality") == modality)
    f = f.filter(pl.col("group").is_null() if group is None else pl.col("group") == group)
    return f


def pick_baseline_codec(codecs: list[str]) -> str | None:
    """Return the highest-quality codec available in this family for use as baseline."""
    for c in BASELINE_PREFERENCE:
        if c in codecs:
            return c
    return None


def build_deltas(
    df: pl.DataFrame, recall_col: str
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Pair every (family, codec) config with the family's baseline by ``config``,
    return long-format ``delta`` rows and the family→baseline_codec map.
    """
    baseline_per_family: dict[str, str] = {}
    delta_rows: list[dict] = []

    for family in df.select("family").unique().to_series().to_list():
        codecs = df.filter(pl.col("family") == family).select("codec").unique().to_series().to_list()
        baseline = pick_baseline_codec(codecs)
        if baseline is None:
            continue
        baseline_per_family[family] = baseline
        baseline_df = df.filter(
            (pl.col("family") == family) & (pl.col("codec") == baseline)
        ).select("config", "task", "modality", "group", recall_col).rename(
            {recall_col: "baseline_recall"}
        )
        for codec in codecs:
            if codec == baseline:
                continue
            other_df = df.filter(
                (pl.col("family") == family) & (pl.col("codec") == codec)
            ).select("config", "task", "modality", "group", recall_col).rename(
                {recall_col: "codec_recall"}
            )
            paired = other_df.join(
                baseline_df,
                on=["config", "task", "modality", "group"],
                how="inner",
                # CC has null modality; GG has null group. Treat nulls as
                # equal so the join matches across baseline/codec on those.
                nulls_equal=True,
            )
            for r in paired.iter_rows(named=True):
                delta_rows.append(
                    dict(
                        family=family,
                        codec=codec,
                        baseline=baseline,
                        config=r["config"],
                        task=r["task"],
                        modality=r["modality"],
                        group=r["group"],
                        baseline_recall=r["baseline_recall"],
                        codec_recall=r["codec_recall"],
                        delta=r["codec_recall"] - r["baseline_recall"],
                    )
                )
    return pl.DataFrame(delta_rows), baseline_per_family


def build_x_layout(
    delta_df: pl.DataFrame,
) -> tuple[
    dict[tuple[str, str], float],
    list[float],
    list[str],
    list[str],
    float,
]:
    """Codecs along the x-axis, grouped by family with gaps between families.

    Returns:
        x_positions: (family, codec) → x position
        x_ticks:      x positions (one per (family, codec))
        x_tick_labels: codec display labels matching ``x_ticks``
        family_order: family names in plot order (for legend)
        total_width:  rightmost extent (used for ``set_xlim``)
    """
    families = delta_df.select("family").unique().to_series().to_list()
    family_order = sorted(
        families,
        key=lambda f: FAMILY_PLOT_ORDER.index(f) if f in FAMILY_PLOT_ORDER else 999,
    )

    x_positions: dict[tuple[str, str], float] = {}
    x_ticks: list[float] = []
    x_tick_labels: list[str] = []

    GAP = 0.6
    cursor = 0.0
    for fam in family_order:
        codecs = sorted(
            delta_df.filter(pl.col("family") == fam).select("codec").unique().to_series().to_list(),
            key=_codec_sort_key,
        )
        if not codecs:
            continue
        for codec in codecs:
            x_positions[(fam, codec)] = cursor
            x_ticks.append(cursor)
            x_tick_labels.append(CODEC_DISPLAY.get(codec, codec))
            cursor += 1.0
        cursor += GAP
    total_width = cursor
    return x_positions, x_ticks, x_tick_labels, family_order, total_width


def plot_codec_delta(
    summary_csv: Path, output_path: Path, k_pct: int = 5,
    excluded_families: list[str] | None = None,
    shared_y: bool = False,
    drop_orf: bool = False,
) -> None:
    df = pl.read_csv(summary_csv)
    if df.is_empty():
        raise SystemExit(f"summary CSV is empty: {summary_csv}")
    recall_col = f"recall_{k_pct}"
    if recall_col not in df.columns:
        raise SystemExit(f"recall column '{recall_col}' missing from summary")

    if excluded_families:
        before = df.height
        df = df.filter(~pl.col("family").is_in(excluded_families))
        print(
            f"[filter] dropped families {excluded_families}: "
            f"{before:,} → {df.height:,} rows"
        )

    delta_df, baseline_per_family = build_deltas(df, recall_col)
    if delta_df.is_empty():
        raise SystemExit(
            "no deltas computed — does any family have multiple codecs paired by config?"
        )

    print(f"[delta] {delta_df.height:,} rows across "
          f"{delta_df.select('family','codec').n_unique()} (family, codec) pairs")
    for fam in sorted(baseline_per_family):
        n = delta_df.filter(pl.col("family") == fam).height
        print(f"  {fam}: baseline={baseline_per_family[fam]}, n_paired_rows={n:,}")

    x_positions, x_ticks, x_tick_labels, family_order, total_width = build_x_layout(delta_df)

    panels = [p for p in PANELS if not (drop_orf and p[1] == "orf")]
    n_metrics = len(panels)
    if n_metrics <= 4:
        n_cols = n_metrics
    elif n_metrics <= 6:
        n_cols = 3
    else:
        n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols

    cat_extent_in = max(5.0, 0.7 * len(x_ticks))
    delta_extent_in = 4.0
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cat_extent_in * n_cols / 2, delta_extent_in * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    fs_title = 14
    fs_axis = 11
    fs_tick = 9
    fs_family = 10
    panel_labels = "abcdefghijklmnop"
    rng = np.random.default_rng(0)

    # Compute global symmetric y-range across panels (in percentage points).
    if shared_y:
        all_vals = (delta_df["delta"].drop_nulls().to_numpy() * 100)
        if all_vals.size:
            extreme = float(np.max(np.abs(all_vals)))
            pad = 0.05 * extreme if extreme > 0 else 0.1
            global_ylim: tuple[float, float] | None = (-(extreme + pad), extreme + pad)
        else:
            global_ylim = None
    else:
        global_ylim = None

    for i, (task, modality, group, label) in enumerate(panels):
        ax = axes_flat[i]
        sub = filter_panel(delta_df, task, modality, group)

        for (fam, codec), xpos in x_positions.items():
            cell = sub.filter((pl.col("family") == fam) & (pl.col("codec") == codec))
            vals = cell["delta"].drop_nulls().to_list()
            if not vals:
                continue
            color = FAMILY_COLOR.get(fam, (0.5, 0.5, 0.5))
            x_jit = rng.normal(xpos, 0.12, len(vals))
            ax.scatter(x_jit, np.array(vals) * 100, c=[color], s=15, alpha=0.35,
                       edgecolors="white", linewidths=0.15, zorder=3)
            mean_val = float(np.mean(vals)) * 100
            ax.scatter(xpos, mean_val, c=[color], s=90, alpha=1.0,
                       edgecolors="black", linewidths=0.6, marker="D", zorder=5)
            ax.plot([xpos, xpos], [0, mean_val], color=color, linewidth=1.2,
                    alpha=0.7, zorder=2)

        ax.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.6, zorder=1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels, fontsize=fs_tick, rotation=45, ha="right")
        ax.set_xlim(-0.5, total_width - 0.5)
        ax.set_ylabel(f"Δ recall@{k_pct}% (pp)", fontsize=fs_axis + 3,
                      fontweight="bold")
        ax.set_title(label, fontsize=fs_title, fontweight="bold")
        if global_ylim is not None:
            ax.set_ylim(global_ylim)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
        ax.tick_params(axis="y", labelsize=fs_tick + 3)
        ax.grid(True, alpha=0.15, axis="y", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.02, 1.05, panel_labels[i], transform=ax.transAxes,
                fontsize=fs_title + 4, fontweight="bold", va="bottom", ha="right")

    for j in range(n_metrics, len(axes_flat)):
        axes_flat[j].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=FAMILY_COLOR.get(fam, (0.5, 0.5, 0.5)),
               markeredgecolor="black", markeredgewidth=0.8, markersize=18,
               label=FAMILY_DISPLAY.get(fam, fam))
        for fam in family_order
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), fontsize=fs_family + 6,
               frameon=False, ncol=max(1, len(family_order)),
               title="Model family", title_fontsize=fs_family + 8,
               handletextpad=0.5, columnspacing=1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv", type=Path, required=True,
        help="motive_sweep_summary.csv produced by plot_motive_results.py",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write the delta plots into.",
    )
    parser.add_argument(
        "--k-pcts", type=str, default="1,5,10",
        help="Comma-separated k%% values (one figure per value). Default: 1,5,10.",
    )
    parser.add_argument(
        "--exclude-families", type=str, default="dinov2_random",
        help=(
            "Comma-separated family names to drop before plotting. Default: "
            "dinov2_random. Pass an empty string to keep everything."
        ),
    )
    args = parser.parse_args()

    excluded = [f.strip() for f in args.exclude_families.split(",") if f.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for k_str in args.k_pcts.split(","):
        k_pct = int(k_str.strip())
        out = args.output_dir / f"motive_codec_delta_from_raw_groups_abs_only_{k_pct}pct.png"
        plot_codec_delta(args.summary_csv, out, k_pct,
                         excluded_families=excluded, shared_y=False)
        out_shared = args.output_dir / (
            f"motive_codec_delta_from_raw_groups_abs_only_{k_pct}pct_sharedy.png"
        )
        plot_codec_delta(args.summary_csv, out_shared, k_pct,
                         excluded_families=excluded, shared_y=True)
        out_noorf = args.output_dir / (
            f"motive_codec_delta_from_raw_groups_abs_only_{k_pct}pct_noORF.png"
        )
        plot_codec_delta(args.summary_csv, out_noorf, k_pct,
                         excluded_families=excluded, shared_y=False, drop_orf=True)
        out_noorf_sy = args.output_dir / (
            f"motive_codec_delta_from_raw_groups_abs_only_{k_pct}pct_noORF_sharedy.png"
        )
        plot_codec_delta(args.summary_csv, out_noorf_sy, k_pct,
                         excluded_families=excluded, shared_y=True, drop_orf=True)


if __name__ == "__main__":
    main()
