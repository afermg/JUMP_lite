#!/usr/bin/env python3
"""Gather and plot MOTIVE sweep results.

Walks ``<sweep_dir>/**/motive/metrics.json`` (the layout produced by
``just motive-eval-sweep``), aggregates the ``full`` recall numbers across
all configs, and emits one figure per recall@k% laid out like
``src/norm_3/gather_sweep_results.py:generate_group_nap_plot``:

- One subplot per (task, modality, compound_group) combination
- X-axis: codec, grouped under model-family brackets
- Y-axis: recall@k%
- Scatter dots = each normalisation/clip config; stars = best config per
  (family, codec) cell

Also writes ``motive_sweep_summary.csv`` (long-format) for downstream use.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns


# Family colour and display name — kept in sync with
# src/norm_3/gather_sweep_results.py:FAMILY_SET2_COLOR / FAMILY_DISPLAY.
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
    # jpegxl_lossy with no distance == lossless jpegxl (the dir is named
    # ``..._jpegxl_lossy_raw_raw_features``); display as "lossless".
    "jpegxl_lossy_raw": "lossless",
    "jpegxl_lossy_hq": "hq",
    "jpegxl_lossy_mq": "mq",
    "jpegxl_lossy_lq": "lq",
    "jpegxl_lossy_d20": "d20",
    "jpegxl_lossy_d50": "d50",
}

# X-axis codec ordering: best quality first (raw / lossless) → most lossy.
# Codecs not in this list are appended alphabetically after the known ones.
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


def _codec_sort_key(codec: str) -> tuple[int, str]:
    """Key for sorting codecs by canonical quality order, then alphabetic."""
    if codec in CODEC_ORDER:
        return (CODEC_ORDER.index(codec), codec)
    return (len(CODEC_ORDER), codec)

# (task, modality, group, panel_title) — kept in fixed order so panel
# layout is reproducible.
# Panel layout — 8 panels, 2x4 grid. CC split into high↔high and low↔low to
# match the post-2025-05-07 evaluate_motive.py output schema.
PANELS: list[tuple[str, str | None, str | None, str]] = [
    ("CC", None, "group_high", "CC Divs↔Divs (compound→compound)"),
    ("CC", None, "group_low", "CC Bioact↔Bioact (compound→compound)"),
    ("GG", "orf", None, "GG ORF (gene→gene)"),
    ("GG", "crispr", None, "GG CRISPR (gene→gene)"),
    ("CG", "orf", "group_high", "CG Divs→ORF (compound→gene)"),
    ("CG", "crispr", "group_high", "CG Divs→CRISPR (compound→gene)"),
    ("CG", "orf", "group_low", "CG Bioact→ORF (compound→gene)"),
    ("CG", "crispr", "group_low", "CG Bioact→CRISPR (compound→gene)"),
]

FEAT_DIR_RE = re.compile(r"(.+?)_jump_lite_updated_(.+?)_raw_features")
FEAT_DIR_RAW_RE = re.compile(r"(.+?)_jump_lite_raw_features")


# Filename suffix per best-selection mode. Matches plot_motive_cross_task.py's
# _SELECTION_SUFFIX so output paths stay consistent across the motive figures.
_SELECTION_SUFFIX = {
    "per_codec":       "",
    "zstd_reference":  "_zstd_pinned",
    "best_any_codec":  "_best_any_codec",
    "best_avg_codec":  "_best_avg_codec",
}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def parse_metrics_path(p: Path) -> tuple[str | None, str | None, str]:
    """Return (family, codec, config) for a ``.../<feat>/<config>/motive/metrics.json``.

    Supports two feat-dir conventions:
    - ``<family>_jump_lite_updated_<codec>_raw_features`` (compressed)
    - ``<family>_jump_lite_raw_features`` (uncompressed baseline → codec="raw")
    """
    parts = p.parts
    feat_dir = parts[-4]
    config = parts[-3]
    m = FEAT_DIR_RE.match(feat_dir)
    if m:
        return m.group(1), m.group(2), config
    m = FEAT_DIR_RAW_RE.match(feat_dir)
    if m:
        family = m.group(1)
        # Strip a trailing "_raw" suffix so e.g. "cellprofiler_raw" maps to the
        # "cellprofiler" family in FAMILY_COLOR / FAMILY_DISPLAY.
        if family.endswith("_raw"):
            family = family[: -len("_raw")]
        return family, "raw", config
    return None, None, config


def load_results(sweep_dir: Path) -> pl.DataFrame:
    """Walk the sweep dir, collect long-format recall rows for every panel."""
    rows: list[dict] = []
    n_files = 0
    for p in sweep_dir.rglob("motive/metrics.json"):
        n_files += 1
        family, codec, config = parse_metrics_path(p)
        if family is None:
            continue
        try:
            m = json.loads(p.read_text())
        except json.JSONDecodeError:
            print(f"  warning: bad JSON at {p}")
            continue

        for s in m.get("MOTIVE_CC", {}).get("summary", []):
            # Older runs emitted a single un-grouped CC row; newer runs split
            # high↔high and low↔low. Honour ``compound_group`` if present.
            rows.append(
                dict(
                    family=family, codec=codec, config=config,
                    task="CC", modality=None,
                    group=s.get("compound_group"),
                    recall_1=s.get("recall@1%"),
                    recall_5=s.get("recall@5%"),
                    recall_10=s.get("recall@10%"),
                )
            )
        for r in m.get("MOTIVE_GG", {}).get("summary", []):
            if r.get("setting") and r["setting"] != "full":
                continue
            rows.append(
                dict(
                    family=family, codec=codec, config=config,
                    task="GG", modality=r["target_modality"], group=None,
                    recall_1=r.get("recall@1%"),
                    recall_5=r.get("recall@5%"),
                    recall_10=r.get("recall@10%"),
                )
            )
        for r in m.get("MOTIVE_CG", {}).get("summary", []):
            if r.get("setting") and r["setting"] != "full":
                continue
            rows.append(
                dict(
                    family=family, codec=codec, config=config,
                    task="CG", modality=r["target_modality"],
                    group=r["compound_group"],
                    recall_1=r.get("recall@1%"),
                    recall_5=r.get("recall@5%"),
                    recall_10=r.get("recall@10%"),
                )
            )
    print(f"  scanned {n_files} metrics.json files, kept {len(rows)} rows")
    return pl.DataFrame(rows)


def filter_panel(df: pl.DataFrame, task, modality, group) -> pl.DataFrame:
    f = df.filter(pl.col("task") == task)
    if modality is None:
        f = f.filter(pl.col("modality").is_null())
    else:
        f = f.filter(pl.col("modality") == modality)
    if group is None:
        f = f.filter(pl.col("group").is_null())
    else:
        f = f.filter(pl.col("group") == group)
    return f


# ---------------------------------------------------------------------------
# layout helpers (mirrors generate_group_nap_plot)
# ---------------------------------------------------------------------------


def build_xpos(df: pl.DataFrame):
    """Return (xpos dict, family_groups, tick_positions, tick_labels, total_width).

    xpos: {(family, codec) → x-coordinate}
    family_groups: list of (display_name, x_start, x_end) for bracket annotation
    """
    families = df.select("family").unique().to_series().to_list()
    family_order = sorted(
        families,
        key=lambda f: FAMILY_PLOT_ORDER.index(f) if f in FAMILY_PLOT_ORDER else 999,
    )
    family_codecs = {
        f: sorted(
            df.filter(pl.col("family") == f).select("codec").unique().to_series().to_list(),
            key=_codec_sort_key,
        )
        for f in family_order
    }

    MIN_SLOTS = 3
    GAP = 1.5
    xpos: dict[tuple[str, str], float] = {}
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    family_groups: list[tuple[str, float, float]] = []
    cursor = 0.0
    for fam in family_order:
        codecs = family_codecs[fam]
        n = len(codecs)
        slots = max(MIN_SLOTS, n)
        offset = (slots - n) / 2.0
        slot_start = cursor
        slot_end = cursor + slots - 1
        for i, c in enumerate(codecs):
            x = cursor + offset + i
            xpos[(fam, c)] = x
            tick_positions.append(x)
            tick_labels.append(CODEC_DISPLAY.get(c, c))
        family_groups.append((FAMILY_DISPLAY.get(fam, fam), slot_start, slot_end))
        cursor += slots + GAP
    total_width = cursor - GAP
    return xpos, family_groups, tick_positions, tick_labels, total_width


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


def _compute_pinned_configs(
    df: pl.DataFrame,
    panels: list[tuple[str, str | None, str | None, str]],
    recall_col: str,
    best_selection: str,
) -> dict[tuple[str, str], str | None]:
    """Return ``{(family, codec): pinned_config or None}`` for the selection.

    Selection score per ``(family, codec, config)`` is the mean of ``recall_col``
    across the panels actually plotted (drop_orf-aware). Modes mirror
    ``plot_motive_cross_task._apply_best_selection``:

    - ``per_codec``      — no pinning; caller falls back to per-cell max.
    - ``best_avg_codec`` — per family, pin the config maximising the mean
                           selection score across codecs.
    - ``best_any_codec`` — per family, pin the config of the single
                           (codec, config) with the highest selection score.
    - ``zstd_reference`` — per family, pin zstd's best-score config (None if
                           the family has no zstd row → fallback at draw time).
    """
    if best_selection == "per_codec":
        return {}

    # Restrict df to the (task, modality, group) slices that are actually
    # plotted, so the selection score only reflects visible panels.
    panel_masks = []
    for task, modality, group, _ in panels:
        f = pl.col("task") == task
        f = f & (pl.col("modality").is_null() if modality is None
                 else pl.col("modality") == modality)
        f = f & (pl.col("group").is_null() if group is None
                 else pl.col("group") == group)
        panel_masks.append(f)
    mask = panel_masks[0]
    for f in panel_masks[1:]:
        mask = mask | f

    agg = (df.filter(mask)
             .group_by(["family", "codec", "config"])
             .agg(pl.col(recall_col).drop_nulls().mean().alias("score"))
             .filter(pl.col("score").is_not_null()))
    if agg.height == 0:
        return {}

    fam_pin: dict[str, str | None] = {}
    for fam in agg["family"].unique().to_list():
        fam_rows = agg.filter(pl.col("family") == fam)
        if fam_rows.height == 0:
            continue
        if best_selection == "best_any_codec":
            top = fam_rows.sort("score", descending=True).head(1)
            fam_pin[fam] = top["config"][0]
        elif best_selection == "best_avg_codec":
            cfg_mean = (fam_rows.group_by("config")
                                .agg(pl.col("score").mean().alias("avg")))
            top = cfg_mean.sort("avg", descending=True).head(1)
            fam_pin[fam] = top["config"][0]
        elif best_selection == "zstd_reference":
            zstd = fam_rows.filter(pl.col("codec") == "zstd")
            if zstd.height == 0:
                fam_pin[fam] = None
            else:
                fam_pin[fam] = zstd.sort("score", descending=True).head(1)["config"][0]
        else:
            raise ValueError(f"unknown best_selection: {best_selection}")

    out: dict[tuple[str, str], str | None] = {}
    seen_pairs = agg.select("family", "codec").unique()
    for fam, codec in seen_pairs.iter_rows():
        pinned = fam_pin.get(fam)
        if pinned is None:
            out[(fam, codec)] = None
            continue
        present = agg.filter(
            (pl.col("family") == fam)
            & (pl.col("codec") == codec)
            & (pl.col("config") == pinned)
        ).height > 0
        out[(fam, codec)] = pinned if present else None
    return out


def plot_motive_sweep(
    df: pl.DataFrame,
    output_path: Path,
    k_pct: int = 5,
    shared_y: bool = False,
    drop_orf: bool = False,
    best_selection: str = "per_codec",
) -> None:
    if df.height == 0:
        raise SystemExit("no rows to plot — was --sweep-dir correct?")

    recall_col = f"recall_{k_pct}"
    if recall_col not in df.columns:
        raise SystemExit(f"recall column '{recall_col}' missing from data")
    random_baseline = k_pct / 100.0

    panels = [p for p in PANELS if not (drop_orf and p[1] == "orf")]
    n_panels = len(panels)

    pinned = _compute_pinned_configs(df, panels, recall_col, best_selection)

    xpos, family_groups, tick_positions, tick_labels, total_width = build_xpos(df)

    if n_panels <= 4:
        n_cols = n_panels
    elif n_panels <= 6:
        n_cols = 3
    else:
        n_cols = 4
    n_rows = (n_panels + n_cols - 1) // n_cols
    col_width = max(8.0, total_width * 0.45)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(col_width * n_cols, 11 * n_rows),
                             squeeze=False)
    axes = axes.flatten()

    fs_title = 28
    fs_subtitle = 20
    fs_axis = 18
    fs_tick = 16
    fs_xtick = 13
    fs_family = 13

    # Compute the shared y-range across all panels (raw recall_5/etc values).
    # Pad slightly so dots near the extremes aren't clipped.
    if shared_y:
        all_vals = df[recall_col].drop_nulls().to_list()
        if all_vals:
            y_lo = min(all_vals + [random_baseline])
            y_hi = max(all_vals + [random_baseline])
            pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.005
            global_ylim = (max(0.0, y_lo - pad), y_hi + pad)
        else:
            global_ylim = None
    else:
        global_ylim = None

    rng = np.random.default_rng(0)

    for i, (task, modality, group, title) in enumerate(panels):
        ax = axes[i]
        sub = filter_panel(df, task, modality, group)

        # Random-baseline reference line
        ax.axhline(random_baseline, color="gray", linestyle="--", linewidth=1.0,
                   alpha=0.6, zorder=1)

        for (fam, codec), x in xpos.items():
            cell = sub.filter((pl.col("family") == fam) & (pl.col("codec") == codec))
            vals = cell[recall_col].drop_nulls().to_list()
            if not vals:
                continue
            color = FAMILY_COLOR.get(fam, (0.5, 0.5, 0.5))
            x_jit = rng.normal(x, 0.12, len(vals))
            ax.scatter(x_jit, vals, c=[color], s=50, alpha=0.45,
                       edgecolors="white", linewidths=0.3)

            # Star = pinned config's value when one was picked, else per-cell
            # max (current per_codec behaviour and the fallback path for the
            # other selection modes when the pinned config has no row for
            # this (family, codec) slice).
            star_val: float | None = None
            pinned_cfg = pinned.get((fam, codec))
            if pinned_cfg is not None:
                pin_vals = (cell.filter(pl.col("config") == pinned_cfg)
                                [recall_col].drop_nulls().to_list())
                if pin_vals:
                    star_val = max(pin_vals)
            if star_val is None:
                star_val = max(vals)
            ax.scatter(x, star_val, c=[color], s=400, alpha=1.0,
                       edgecolors="black", linewidths=1.0, marker="*", zorder=10)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=fs_xtick)
        ax.set_xlim(-0.5, total_width + 0.5)
        if global_ylim is not None:
            ax.set_ylim(global_ylim)
        ax.set_ylabel(f"recall@{k_pct}%", fontsize=fs_axis, fontweight="bold")
        ax.set_title(title, fontsize=fs_subtitle, fontweight="bold")
        ax.tick_params(axis="y", labelsize=fs_tick)
        ax.grid(True, alpha=0.2, axis="y", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Family bracket labels at the bottom
        trans = ax.get_xaxis_transform()
        for fam_disp, x_start, x_end in family_groups:
            mid = (x_start + x_end) / 2.0
            ax.text(mid, -0.22, fam_disp, transform=trans,
                    ha="center", va="top", fontsize=fs_family, fontweight="bold",
                    rotation=45, rotation_mode="anchor")
            ax.plot([x_start - 0.3, x_end + 0.3], [-0.14, -0.14], transform=trans,
                    color="gray", linewidth=0.8, clip_on=False)

    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.05, 1, 1.0])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {output_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir", type=Path, required=True,
        help="Top-level dir containing <feat_dir>/<config>/motive/metrics.json files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("aux_figures/motive"),
        help="Where to write summary CSV and plots (default: aux_figures/motive).",
    )
    parser.add_argument(
        "--k-pcts", type=str, default="1,5,10",
        help="Comma-separated k%% values to plot (one figure per value). Default: 1,5,10.",
    )
    parser.add_argument(
        "--best-selections", type=str,
        default="per_codec,best_avg_codec,best_any_codec,zstd_reference",
        help=(
            "Comma-separated star-highlight selection modes; one set of "
            "figures per mode. Suffix per mode mirrors plot_motive_cross_task.py: "
            "per_codec='', best_avg_codec='_best_avg_codec', "
            "best_any_codec='_best_any_codec', zstd_reference='_zstd_pinned'. "
            "Default: all four."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] scanning {args.sweep_dir}")
    df = load_results(args.sweep_dir)
    if df.height == 0:
        raise SystemExit(f"no metrics.json files found under {args.sweep_dir}")
    n_configs = df.select("family", "codec", "config").unique().height
    print(f"[load] {df.height:,} rows from {n_configs:,} configs")

    summary_csv = args.output_dir / "motive_sweep_summary.csv"
    df.write_csv(summary_csv)
    print(f"[write] {summary_csv}")

    selections = [s.strip() for s in args.best_selections.split(",") if s.strip()]
    for sel in selections:
        if sel not in _SELECTION_SUFFIX:
            raise SystemExit(
                f"unknown best_selection '{sel}' — choose from {list(_SELECTION_SUFFIX)}"
            )

    for k_str in args.k_pcts.split(","):
        k_pct = int(k_str.strip())
        for sel in selections:
            sfx = _SELECTION_SUFFIX[sel]
            base = f"motive_sweep_recall_at_{k_pct}pct{sfx}"
            plot_motive_sweep(df, args.output_dir / f"{base}.png",
                              k_pct, shared_y=False, best_selection=sel)
            plot_motive_sweep(df, args.output_dir / f"{base}_sharedy.png",
                              k_pct, shared_y=True, best_selection=sel)
            plot_motive_sweep(df, args.output_dir / f"{base}_noORF.png",
                              k_pct, shared_y=False, drop_orf=True,
                              best_selection=sel)
            plot_motive_sweep(df, args.output_dir / f"{base}_noORF_sharedy.png",
                              k_pct, shared_y=True, drop_orf=True,
                              best_selection=sel)


if __name__ == "__main__":
    main()
