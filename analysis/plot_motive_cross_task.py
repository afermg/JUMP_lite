#!/usr/bin/env python3
"""Cross-task scatter plots from MOTIVE sweep results.

For each requested recall@k%%, emits one figure with two scatter panels:

- **CC vs GG** — x = mean CC recall, y = mean GG recall
- **CG-Divs vs CG-Bioact** — x = mean CG-Divs recall, y = mean CG-Bioact recall

Each dot is the *best (family, codec)* config selected by the panel's balanced
score (``x * y``). Family colour matches ``plot_motive_results.py``. A dashed
random-baseline line is drawn on each axis (k%%/100); a dotted parity line
(``y = x``) marks where the two recalls would match. With ``--show-all-points``
every config is also drawn as a faint cloud underneath.

``--no-orf`` (passed through the ``noORF`` filename suffix) drops rows whose
target modality is ``orf`` — matching the convention used by
``plot_motive_results.py`` and ``plot_motive_codec_delta.py``. In that mode the
GG axis becomes CRISPR-only and the CG axes become CRISPR-only.

Reads the long-format ``motive_sweep_summary.csv`` produced by
``analysis/plot_motive_results.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.lines import Line2D


# Kept in sync with plot_motive_results.py / plot_motive_codec_delta.py.
SET2 = sns.color_palette("Set2", 8)
# CellCount uses a darkened pink (matches FAMILY_SET2_COLOR in JUMP_core/
# gather_sweep_results.py line 590) so the dot is visible at low alpha and
# pairs visually with panel A.
_CELLCOUNT_PINK = tuple(c * 0.75 for c in SET2[3])
FAMILY_COLOR = {
    "cell_count": _CELLCOUNT_PINK,
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

# Per-codec compression level — lower = less lossy. Mirrors the convention in
# JUMP_core/src/norm_3/gather_sweep_results.COMPRESSION_LEVEL, adapted to the
# MOTIVE codec names emitted by analysis/plot_motive_results.py:CODEC_ORDER.
CODEC_LEVEL = {
    "raw":               0,
    "zstd":              1,
    "lz4hc":             1,
    "blosc_zstd":        1,
    "jpegxl_lossy_raw":  1,   # JpegXL lossless
    "jpegxl_lossy_hq":   2,
    "jpegxl_lossy_mq":   5,
    "jpegxl_lossy_lq":   7,
    "jpegxl_lossy_d20": 10,
    "jpegxl_lossy_d50": 12,
}
_MAX_CODEC_LEVEL = max(CODEC_LEVEL.values())


def _codec_size(codec: str) -> float:
    """Dot size for a codec — lossless = big, heavy-lossy = small.

    Same formula as ``generate_nap_pa_vs_pc_combined`` in JUMP_core:
    ``30 + 950 * exp(-2.5 * level / max_level)``. Unknown codecs get the
    mid-range size.
    """
    level = CODEC_LEVEL.get(codec, _MAX_CODEC_LEVEL // 2)
    return 30 + 950 * float(np.exp(-2.5 * level / max(_MAX_CODEC_LEVEL, 1)))


# Codec display label — matches plot_motive_results.CODEC_DISPLAY so size-legend
# entries read the same as the recall-sweep figure's x-tick labels.
CODEC_DISPLAY = {
    "raw":              "Raw",
    "zstd":             "zstd",
    "lz4hc":            "lz4hc",
    "blosc_zstd":       "blosc-zstd",
    "jpegxl_lossy_raw": "Lossless",
    "jpegxl_lossy_hq":  "HQ",
    "jpegxl_lossy_mq":  "MQ",
    "jpegxl_lossy_lq":  "LQ",
    "jpegxl_lossy_d20": "D20",
    "jpegxl_lossy_d50": "D50",
}


def _add_family_legend(ax, families_in_data, fs_legend,
                       bbox_anchor=(0.25, 0.98)):
    """Family legend with uniform marker size, ordered by FAMILY_PLOT_ORDER.

    Default anchor places the legend's upper-left corner at axes (0.25, 0.98)
    — i.e., just to the right of the codec legend at the top-left of the axes.
    """
    fam_order = sorted(
        set(families_in_data),
        key=lambda f: FAMILY_PLOT_ORDER.index(f) if f in FAMILY_PLOT_ORDER else 999,
    )
    handles = []
    for f in fam_order:
        h = ax.scatter([], [], c=[FAMILY_COLOR.get(f, (0.5, 0.5, 0.5))],
                       s=200, alpha=0.85,
                       edgecolors="black", linewidths=0.8)
        handles.append(h)
    labels = [FAMILY_DISPLAY.get(f, f) for f in fam_order]
    leg = ax.legend(handles=handles, labels=labels,
                    loc="upper left", bbox_to_anchor=bbox_anchor,
                    bbox_transform=ax.transAxes,
                    fontsize=fs_legend, framealpha=0.9, edgecolor="gray",
                    title="family", title_fontsize=fs_legend,
                    labelspacing=1.0, borderpad=0.8)
    leg.get_title().set_fontweight("bold")
    return leg


def _add_codec_size_legend(ax, codecs_in_data, fs_legend,
                           bbox_anchor=(0.01, 0.98)):
    """Codec→size legend — markers sized identically to the scatter dots.

    Default anchor places the legend's upper-left corner at axes (0.01, 0.98)
    — i.e., at the top-left of the axes, with the family legend immediately
    to its right.
    """
    codecs_sorted = sorted(
        set(codecs_in_data),
        key=lambda c: CODEC_LEVEL.get(c, _MAX_CODEC_LEVEL // 2),
    )
    handles = []
    labels = []
    for codec in codecs_sorted:
        h = ax.scatter([], [], c=["lightgray"], s=_codec_size(codec),
                       edgecolors="black", linewidths=0.8, alpha=0.85)
        handles.append(h)
        labels.append(CODEC_DISPLAY.get(codec, codec))
    leg = ax.legend(handles=handles, labels=labels,
                    loc="upper left", bbox_to_anchor=bbox_anchor,
                    bbox_transform=ax.transAxes,
                    fontsize=fs_legend, framealpha=0.9, edgecolor="gray",
                    title="codec", title_fontsize=fs_legend,
                    labelspacing=1.6, borderpad=0.8)
    leg.get_title().set_fontweight("bold")
    return leg


def _agg_axis(df: pl.DataFrame, task: str, group_filter: pl.Expr | None,
              drop_orf: bool, recall_col: str) -> pl.DataFrame:
    """Average ``recall_col`` per (family, codec, config) over a task/group slice.

    ``group_filter`` is an explicit polars expression (or ``None`` to keep every
    group for the chosen task). CC rows carry ``group ∈ {group_high, group_low}``
    so the CC axis passes ``group_filter=None`` to average over both subsets;
    GG rows carry ``group=null`` so the GG axis passes ``pl.col("group").is_null()``.

    ``drop_orf=True`` removes rows whose modality is ``orf`` before aggregation
    (modalities that are ``null`` are kept, matching ``plot_motive_results.py``'s
    ``--no-orf`` semantics).
    """
    f = df.filter(pl.col("task") == task)
    if group_filter is not None:
        f = f.filter(group_filter)
    if drop_orf:
        f = f.filter((pl.col("modality") != "orf") | pl.col("modality").is_null())
    return (
        f.group_by(["family", "codec", "config"])
         .agg(pl.col(recall_col).drop_nulls().mean().alias("val"))
         .filter(pl.col("val").is_not_null())
    )


def _build_wide(df: pl.DataFrame, drop_orf: bool, recall_col: str) -> pl.DataFrame:
    """Build a wide table with all four axis means per (family, codec, config).

    Columns: family, codec, config, cc, gg, cg_div, cg_bio, score.
    ``score = cc * gg * cg_div * cg_bio`` is the balanced product across all four
    cross-task axes — the analogue of the ``PA*PC`` balanced score used by
    ``gather_sweep_results._add_best_column``. Configs without data for one of
    the four slices are dropped by the inner join.
    """
    cc  = _agg_axis(df, "CC", group_filter=None,
                    drop_orf=False, recall_col=recall_col).rename({"val": "cc"})
    gg  = _agg_axis(df, "GG", group_filter=pl.col("group").is_null(),
                    drop_orf=drop_orf, recall_col=recall_col).rename({"val": "gg"})
    div = _agg_axis(df, "CG", group_filter=pl.col("group") == "group_high",
                    drop_orf=drop_orf, recall_col=recall_col).rename({"val": "cg_div"})
    bio = _agg_axis(df, "CG", group_filter=pl.col("group") == "group_low",
                    drop_orf=drop_orf, recall_col=recall_col).rename({"val": "cg_bio"})
    wide = (cc.join(gg,  on=["family", "codec", "config"], how="inner")
              .join(div, on=["family", "codec", "config"], how="inner")
              .join(bio, on=["family", "codec", "config"], how="inner"))
    return wide.with_columns(
        (pl.col("cc") * pl.col("gg") * pl.col("cg_div") * pl.col("cg_bio")).alias("score")
    )


def _apply_best_selection(wide: pl.DataFrame, best_selection: str) -> pl.DataFrame:
    """Pick one (family, codec, config) row per (family, codec).

    Mirrors ``gather_sweep_results._compute_best_idx`` semantics:

    - ``per_codec``     — each (family, codec) keeps its own ``score``-max config.
    - ``best_avg_codec``— for each family, pick the config that maximises the
                         mean score across that family's codecs; pin every
                         codec to that config. Falls back to per-codec best
                         when the pinned config does not exist for a codec.
    - ``best_any_codec``— for each family, pick the single (codec, config) with
                         max score; pin every codec in the family to that
                         config (same fallback).
    - ``zstd_reference``— for each family, pick zstd's best config; pin every
                         codec (same fallback).
    """
    if best_selection == "per_codec":
        return (wide.sort("score", descending=True)
                    .group_by(["family", "codec"], maintain_order=True)
                    .head(1))

    if best_selection == "best_any_codec":
        family_pin = (wide.sort("score", descending=True)
                          .group_by("family", maintain_order=True)
                          .head(1)
                          .select("family", "config"))
    elif best_selection == "best_avg_codec":
        avg = (wide.group_by(["family", "config"])
                   .agg(pl.col("score").mean().alias("avg_score")))
        family_pin = (avg.sort("avg_score", descending=True)
                         .group_by("family", maintain_order=True)
                         .head(1)
                         .select("family", "config"))
    elif best_selection == "zstd_reference":
        zstd = wide.filter(pl.col("codec") == "zstd")
        family_pin = (zstd.sort("score", descending=True)
                          .group_by("family", maintain_order=True)
                          .head(1)
                          .select("family", "config"))
    else:
        raise ValueError(f"Unknown best_selection: {best_selection}")

    family_pin = family_pin.rename({"config": "pinned_config"})
    pinned = (wide.join(family_pin, on="family", how="left")
                  .filter(pl.col("config") == pl.col("pinned_config"))
                  .drop("pinned_config"))

    matched_keys = pinned.select("family", "codec").unique()
    unmatched = wide.join(matched_keys, on=["family", "codec"], how="anti")
    fallback = (unmatched.sort("score", descending=True)
                         .group_by(["family", "codec"], maintain_order=True)
                         .head(1))
    return pl.concat([pinned, fallback]) if fallback.height > 0 else pinned


_SELECTION_SUFFIX = {
    "per_codec":       "",
    "zstd_reference":  "_zstd_pinned",
    "best_any_codec":  "_best_any_codec",
    "best_avg_codec":  "_best_avg_codec",
}


def _plot_panel(ax, best_df: pl.DataFrame, cloud_df: pl.DataFrame | None,
                x_label: str, y_label: str, title: str,
                random_baseline: float, *,
                fs_title: int, fs_axis: int, fs_tick: int, fs_legend: int,
                show_parity: bool = True,
                show_legends: bool = True) -> None:
    # Background cloud (every config) — family-coloured, faint.
    if cloud_df is not None and cloud_df.height > 0:
        for fam in cloud_df["family"].unique().to_list():
            sub = cloud_df.filter(pl.col("family") == fam)
            color = FAMILY_COLOR.get(fam, (0.5, 0.5, 0.5))
            ax.scatter(sub["x"].to_list(), sub["y"].to_list(),
                       c=[color], s=40, alpha=0.30,
                       edgecolors="none", zorder=2)

    # Best-per-codec dots — one legend entry per family.
    families_seen: set[str] = set()
    # Order families so the legend follows FAMILY_PLOT_ORDER.
    fams_in_data = best_df["family"].unique().to_list()
    fam_order = sorted(
        fams_in_data,
        key=lambda f: FAMILY_PLOT_ORDER.index(f) if f in FAMILY_PLOT_ORDER else 999,
    )
    codecs_seen: set[str] = set()
    for fam in fam_order:
        fam_rows = best_df.filter(pl.col("family") == fam)
        if fam_rows.height == 0:
            continue
        color = FAMILY_COLOR.get(fam, (0.5, 0.5, 0.5))
        for row in fam_rows.iter_rows(named=True):
            ax.scatter(
                row["x"], row["y"], c=[color],
                s=_codec_size(row["codec"]), alpha=0.85,
                edgecolors="black", linewidths=0.8, zorder=5,
            )
            families_seen.add(fam)
            codecs_seen.add(row["codec"])

    # Axis limits anchored at 0 with a 15% pad above the data.
    xs = best_df["x"].to_list()
    ys = best_df["y"].to_list()
    if cloud_df is not None and cloud_df.height > 0:
        xs = xs + cloud_df["x"].to_list()
        ys = ys + cloud_df["y"].to_list()
    x_max = max(xs + [random_baseline]) if xs else random_baseline
    y_max = max(ys + [random_baseline]) if ys else random_baseline
    ax.set_xlim(0, x_max * 1.15 if x_max > 0 else 0.15)
    ax.set_ylim(0, y_max * 1.15 if y_max > 0 else 0.15)

    # Random-baseline reference lines (one per axis).
    ax.axhline(random_baseline, color="gray", linestyle="--", linewidth=1.0,
               alpha=0.6, zorder=1)
    ax.axvline(random_baseline, color="gray", linestyle="--", linewidth=1.0,
               alpha=0.6, zorder=1)

    if show_parity:
        hi = min(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, hi], [0, hi], color="gray", linestyle=":",
                linewidth=1.0, alpha=0.6, zorder=1)

    ax.set_xlabel(x_label, fontsize=fs_axis, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=fs_axis, fontweight="bold")
    ax.tick_params(axis="both", labelsize=fs_tick)
    ax.set_title(title, fontsize=fs_title, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legends and families_seen:
        # Two legends side-by-side: codec→size on the left, family color on the
        # right. Both panels in this figure pass show_legends=False — the
        # paired figure (sweep_nap_pa_vs_pc_panel_a) carries the shared legend.
        fam_leg = _add_family_legend(ax, families_seen, fs_legend,
                                     bbox_anchor=(0.25, 0.98))
        ax.add_artist(fam_leg)
        if codecs_seen:
            _add_codec_size_legend(ax, codecs_seen, fs_legend,
                                   bbox_anchor=(0.01, 0.98))


def plot_cross_task(df: pl.DataFrame, k_pct: int, output_path: Path, *,
                    drop_orf: bool = False, show_all_points: bool = False,
                    best_selection: str = "best_avg_codec",
                    codec_filter: list[str] | None = None,
                    only_panel: str | None = None) -> None:
    """When ``only_panel='b'``, render only the CC vs GG panel at panel A's
    aspect (figsize matches sweep_nap_pa_vs_pc_panel_a's saved shape)."""
    recall_col = f"recall_{k_pct}"
    if recall_col not in df.columns:
        raise SystemExit(f"recall column '{recall_col}' missing from summary")

    if codec_filter is not None:
        keep = set(codec_filter)
        before = df.height
        df = df.filter(pl.col("codec").is_in(list(keep)))
        print(f"[filter] codecs {sorted(keep)}: {before:,} -> {df.height:,} rows")

    wide = _build_wide(df, drop_orf=drop_orf, recall_col=recall_col)
    if wide.height == 0:
        raise SystemExit(f"no joined data for recall_{k_pct} (drop_orf={drop_orf})")
    best = _apply_best_selection(wide, best_selection)

    panel_b_best  = best.select("family", "codec", "config",
                                pl.col("cc").alias("x"),     pl.col("gg").alias("y"))
    panel_b_cloud = wide.select("family", "codec", "config",
                                pl.col("cc").alias("x"),     pl.col("gg").alias("y"))
    panel_c_best  = best.select("family", "codec", "config",
                                pl.col("cg_div").alias("x"), pl.col("cg_bio").alias("y"))
    panel_c_cloud = wide.select("family", "codec", "config",
                                pl.col("cg_div").alias("x"), pl.col("cg_bio").alias("y"))

    # Font sizes mirror sweep_nap_pa_vs_pc_panel_a (gather_sweep_results.py
    # generate_nap_pa_vs_pc_panel_a, lines ~3325-3328 + fs_title+4 for the
    # panel label) so b/c sit visually consistent next to panel a.
    fs_title, fs_axis, fs_tick, fs_legend, fs_panel = 32, 24, 18, 17, 36
    rand = k_pct / 100.0

    # Title/label format mirrors sweep_nap_pa_vs_pc_panel_a:
    #   title  = "<Metric>: <Y> vs <X>"   (e.g. "Mean NAP: PA vs PC")
    #   xlabel = "<X> <Metric>"           (e.g. "PC Mean NAP")
    #   ylabel = "<Y> <Metric>"           (e.g. "PA Mean NAP")
    metric = f"Mean Recall@{k_pct}%"
    gg_tag = " (CRISPR)" if drop_orf else ""
    cg_tag = " (CRISPR)" if drop_orf else ""

    if only_panel == "b":
        # Standalone panel b at panel A's saved aspect (~0.696 w/h). Using
        # figsize=(12, 17.2) matches the stacked b+c canvas; with a single
        # subplot it produces a portrait panel-b in the same external shape
        # as sweep_nap_pa_vs_pc_panel_a.
        fig, ax_b = plt.subplots(1, 1, figsize=(12, 17.2))
        _plot_panel(
            ax_b, panel_b_best, panel_b_cloud if show_all_points else None,
            x_label=f"CC {metric}",
            y_label=f"GG {metric}{gg_tag}",
            title=f"{metric}: GG vs CC",
            random_baseline=rand,
            fs_title=fs_title, fs_axis=fs_axis, fs_tick=fs_tick, fs_legend=fs_legend,
            show_parity=False,
            show_legends=False,
        )
        ax_b.text(-0.02, 1.02, "b", transform=ax_b.transAxes,
                  fontsize=fs_panel, fontweight="bold", va="bottom", ha="right")
    else:
        # Stacked 2-panel layout. figsize aspect (12 / 17.2 ≈ 0.698) matches
        # the saved aspect of sweep_nap_pa_vs_pc_panel_a (≈0.696) so this
        # figure pairs cleanly with the panel-A reference.
        fig, axes = plt.subplots(2, 1, figsize=(12, 17.2))
        _plot_panel(
            axes[0], panel_b_best, panel_b_cloud if show_all_points else None,
            x_label=f"CC {metric}",
            y_label=f"GG {metric}{gg_tag}",
            title=f"{metric}: GG vs CC",
            random_baseline=rand,
            fs_title=fs_title, fs_axis=fs_axis, fs_tick=fs_tick, fs_legend=fs_legend,
            show_parity=False,
            show_legends=False,
        )
        axes[0].text(-0.02, 1.02, "b", transform=axes[0].transAxes,
                     fontsize=fs_panel, fontweight="bold", va="bottom", ha="right")

        _plot_panel(
            axes[1], panel_c_best, panel_c_cloud if show_all_points else None,
            x_label=f"CG-Divs {metric}{cg_tag}",
            y_label=f"CG-Bioact {metric}{cg_tag}",
            title=f"{metric}: CG-Bioact vs CG-Divs",
            random_baseline=rand,
            fs_title=fs_title, fs_axis=fs_axis, fs_tick=fs_tick, fs_legend=fs_legend,
            show_parity=True,
            show_legends=False,
        )
        axes[1].text(-0.02, 1.02, "c", transform=axes[1].transAxes,
                     fontsize=fs_panel, fontweight="bold", va="bottom", ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv", type=Path, required=True,
        help="motive_sweep_summary.csv produced by analysis/plot_motive_results.py",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write the PNGs into.",
    )
    parser.add_argument(
        "--k-pcts", type=str, default="1,5,10",
        help="Comma-separated k%% values (one figure per value). Default: 1,5,10.",
    )
    parser.add_argument(
        "--show-all-points", action="store_true",
        help="Also save a `_with_all_points` variant per k%% (background cloud of every config).",
    )
    parser.add_argument(
        "--best-selection",
        choices=sorted(_SELECTION_SUFFIX.keys()),
        default="best_avg_codec",
        help=(
            "How to pick the (family, codec, config) row each dot represents. "
            "Mirrors gather_sweep_results.py's --best-selection. Default: "
            "best_avg_codec (matches sweep_nap_pa_vs_pc_panel_a_best_avg_codec)."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_csv(args.summary_csv)
    print(f"[load] {df.height:,} rows from {args.summary_csv}")

    sel_suffix = _SELECTION_SUFFIX[args.best_selection]
    print(f"[select] best_selection={args.best_selection} (suffix='{sel_suffix}')")

    # (codec_filter, filename_suffix) passes. None = all codecs; the raw+mq
    # variant mirrors the JUMP_core panel A "_raw_mq" output. Note: tabular
    # families (cell_count, cellprofiler) carry `raw` while DL families carry
    # `jpegxl_lossy_raw` (JpegXL lossless) as their uncompressed baseline — we
    # include both so every family appears.
    codec_passes: list[tuple[list[str] | None, str]] = [
        (None, ""),
        (["raw", "jpegxl_lossy_raw", "jpegxl_lossy_mq"], "_raw_mq"),
    ]
    # (only_panel, filename_suffix) passes. None = full stacked b+c layout;
    # "b" = standalone panel b at the same shape as panel A.
    layout_passes: list[tuple[str | None, str]] = [
        (None, ""),
        ("b", "_b_only"),
    ]

    for k_str in args.k_pcts.split(","):
        k_pct = int(k_str.strip())
        for drop_orf in (False, True):
            orf_tag = "_noORF" if drop_orf else ""
            for codec_filter, codec_suffix in codec_passes:
                for only_panel, layout_suffix in layout_passes:
                    base = (
                        f"motive_cross_task_at_{k_pct}pct{sel_suffix}"
                        f"{codec_suffix}{layout_suffix}{orf_tag}"
                    )
                    plot_cross_task(
                        df, k_pct, args.output_dir / f"{base}.png",
                        drop_orf=drop_orf, show_all_points=False,
                        best_selection=args.best_selection,
                        codec_filter=codec_filter,
                        only_panel=only_panel,
                    )
                    if args.show_all_points:
                        plot_cross_task(
                            df, k_pct, args.output_dir / f"{base}_with_all_points.png",
                            drop_orf=drop_orf, show_all_points=True,
                            best_selection=args.best_selection,
                            codec_filter=codec_filter,
                            only_panel=only_panel,
                        )


if __name__ == "__main__":
    main()
