#!/usr/bin/env python3
"""Per-codec model-rank bump chart across the 11 combined-delta-table tasks.

For one codec at a time, produce a figure that ranks each model (1 = best) on
every task that appears in the combined RefChem + MOTIVE codec-delta table
(see ``generate_combined_codec_delta_table.py``).

Tasks (left to right):
  PA: CRISPR · ORF · Divs · Bioact          (sweep_results.csv)
  PC: Divs · Bioact                         (sweep_results.csv)
  CC: Divs · Bioact                         (motive_sweep_summary.csv)
  GG: CRISPR                                (motive_sweep_summary.csv)
  CG: Divs · Bioact                         (motive_sweep_summary.csv)

For each (family, codec) we pick a single config per source using
``--best-selection`` (same modes as ``gather_sweep_results.py`` and
``plot_motive_cross_task.py``):
  - ``per_codec``      — each codec's own best config (selection score =
                         per-source mean across that source's tasks).
  - ``best_avg_codec`` — per family, pin the config that maximises the mean
                         selection score across codecs.
  - ``best_any_codec`` — per family, pin the config of the single (codec,
                         config) with the highest selection score.
  - ``zstd_reference`` — per family, pin to zstd's best config (if zstd
                         exists in that source).

Ranks are computed per task across the 4 shared families
(DINOv2, MorphEm, OpenPhenom, SubCell — the same set the combined delta
table reports on). Each figure is saved as
``model_task_rank_{best_selection}_{codec}.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns


# ---------------------------------------------------------------------------
# Family colour table — kept in sync with FAMILY_SET2_COLOR in
# JUMP_core/src/norm_3/gather_sweep_results.py (line 591). CellCount is
# darkened (factor 0.75) for visibility, mirroring the panel-A convention.
# ---------------------------------------------------------------------------

SET2 = sns.color_palette("Set2", 8)
_CELLCOUNT_PINK = tuple(c * 0.75 for c in SET2[3])
FAMILY_COLOR = {
    "CellCount":   _CELLCOUNT_PINK,
    "CellProfiler": SET2[1],
    "DINOv2":      SET2[6],
    "ViT-rand":    SET2[0],
    "MorphEm":     SET2[4],
    "OpenPhenom":  SET2[5],
    "SubCell":     SET2[2],
}

# All families ranked. The combined delta table only shows the 4 DL families
# (DINOv2/MorphEm/OpenPhenom/SubCell) because the delta is undefined without
# a codec sweep, but for rank-per-task we can include every family — the
# tabular families (CellCount, CellProfiler) simply reuse their single
# available config at every codec slot since image compression doesn't apply
# to pre-extracted features.
MODELS = [
    "DINOv2", "MorphEm", "OpenPhenom", "SubCell",
    "ViT-rand", "CellProfiler", "CellCount",
]
# Tabular families with no codec sweep — their score is the same at every
# codec figure (read from the single uncompressed row).
TABULAR_FAMILIES = {"CellCount", "CellProfiler"}

# Codec aliases. The "canonical" name is the human-facing display label used
# in titles + filenames; the per-source values are how each codec appears in
# its respective CSV (RefChem uses the suffix after "jpegxl_lossy_"; MOTIVE
# uses the full "jpegxl_lossy_<tier>" string).
#
# Note: only DL families have jpegxl_lossy_raw in MOTIVE; the tabular cell_count/
# cellprofiler families carry plain "raw" instead. Since this script only ranks
# the 4 DL families we resolve to "jpegxl_lossy_raw".
CODEC_CHOICES: dict[str, dict[str, str]] = {
    "raw": {"display": "lossless", "refchem": "raw", "motive": "jpegxl_lossy_raw"},
    "hq":  {"display": "hq",       "refchem": "hq",  "motive": "jpegxl_lossy_hq"},
    "mq":  {"display": "mq",       "refchem": "mq",  "motive": "jpegxl_lossy_mq"},
    "d20": {"display": "d20",      "refchem": "d20", "motive": "jpegxl_lossy_d20"},
}

# RefChem (family substring -> display) + reverse-map for parsing model rows.
REFCAM_FAMILY_DISPLAY = {
    "dinov2_lite":           "DINOv2",
    "dinov2_random_lite":    "ViT-rand",
    "morphem_lite":          "MorphEm",
    "openphenom_lite":       "OpenPhenom",
    "subcell__clip01_lite":  "SubCell",
    "subcell_lite":          "SubCell",
}
# Tabular RefChem model names — no jpegxl_lossy slot. Each row in the CSV is
# the single available config; we treat it as the family's score at every
# codec slot.
REFCAM_TABULAR_MODEL = {
    "cell_count_lite_raw":   "CellCount",
    "cellprofiler_lite_raw": "CellProfiler",
}
MOTIVE_FAMILY_DISPLAY = {
    "dinov2":           "DINOv2",
    "dinov2_random":    "ViT-rand",
    "morphem":          "MorphEm",
    "openphenom":       "OpenPhenom",
    "subcell":          "SubCell",
    "subcell__clip01":  "SubCell",
    "cell_count":       "CellCount",
    "cellprofiler":     "CellProfiler",
}

# (task_label, source, metric_extractor). metric_extractor returns the per-row
# score given a pandas Series (RefChem) or polars row dict (MOTIVE filter).
REFCAM_TASKS = [
    ("PA CRISPR", "PA_group_crispr_mean_normalized_average_precision"),
    ("PA ORF",    "PA_group_orf_mean_normalized_average_precision"),
    ("PA Divs",   "PA_group_high_mean_normalized_average_precision"),
    ("PA Bioact", "PA_group_low_mean_normalized_average_precision"),
    ("PC Divs",   "PC_group_high_mean_normalized_average_precision"),
    ("PC Bioact", "PC_group_low_mean_normalized_average_precision"),
]

# MOTIVE tasks expressed as (label, task, modality, group). modality/group
# may be None.
MOTIVE_TASKS = [
    ("CC Divs",     "CC", None,     "group_high"),
    ("CC Bioact",   "CC", None,     "group_low"),
    ("GG CRISPR",   "GG", "crispr", None),
    ("CG Divs",     "CG", "crispr", "group_high"),
    ("CG Bioact",   "CG", "crispr", "group_low"),
]

TASK_ORDER = [t[0] for t in REFCAM_TASKS] + [t[0] for t in MOTIVE_TASKS]

# Group separators (drawn as faint vertical lines) — between the four blocks
# PA / PC / CC / GG / CG.
GROUP_BOUNDARIES = [4, 6, 8, 9]  # indices *after* which to draw a divider

# Bottom-tier (sub-task) labels, dropping the bench prefix so the bracket
# header above can carry it. Order matches TASK_ORDER.
SUBTASK_LABELS = [
    "CRISPR", "ORF", "Divs", "Bioact",   # PA
    "Divs", "Bioact",                    # PC
    "Divs", "Bioact",                    # CC
    "CRISPR",                            # GG
    "Divs", "Bioact",                    # CG → CRISPR
]

# Mid-tier (bench) and top-tier (source) bracket spans for the LaTeX-style
# hierarchical header. Each entry: (label, start_col_idx, end_col_idx) using
# zero-based indices into the heatmap's columns (last column = "Mean").
MID_HEADER_SPANS = [
    ("PA",            0, 3),
    ("PC",            4, 5),
    ("CC",            6, 7),
    ("GG",            8, 8),
    (r"CG $\to$ CRISPR", 9, 10),
    ("Mean",          11, 11),
]
# Top-tier follows the LaTeX table exactly: PA columns carry no super-header,
# "RefChem" sits only over PC, "MOTIVE" spans CC+GG+CG, "Mean" trails.
TOP_HEADER_SPANS = [
    ("RefChem",  4, 5),
    ("MOTIVE",   6, 10),
    ("Mean",     11, 11),
]


# ---------------------------------------------------------------------------
# RefChem helpers
# ---------------------------------------------------------------------------

def _refcam_parse_model(model_name: str) -> tuple[str, str] | None:
    """Return (family_display, refchem_codec) if the row matches one of the
    shared families' jpegxl variants; else None.

    Model strings look like ``dinov2_lite_jpegxl_lossy_<tier>_raw``. The "raw"
    suffix is the data-format tag; the codec tier is the substring just before
    it. For the lossless baseline tier=`raw` so the model name ends with
    ``_jpegxl_lossy_raw_raw``.
    """
    for fk, fd in REFCAM_FAMILY_DISPLAY.items():
        prefix = fk + "_jpegxl_lossy_"
        if model_name.startswith(prefix) and model_name.endswith("_raw"):
            tier = model_name[len(prefix):-len("_raw")]
            return fd, tier
    return None


def _refcam_scores(df: pd.DataFrame, codec: str, best_selection: str
                   ) -> dict[tuple[str, str], float | None]:
    """Return ``{(family_display, task_label): score}`` for the given codec.

    ``codec`` is the RefChem-side codec name (e.g. ``"raw"``, ``"mq"``).
    The config picked per family follows ``best_selection`` (computed from the
    mean of the 6 RefChem task scores). Tabular families (CellCount,
    CellProfiler) have no codec sweep — their single row is reused at every
    codec slot via the synthetic key ``"_tabular"``.
    """
    # Build per-(family, codec_key) frames keyed by config. DL families use
    # their codec tier; tabular families use the synthetic "_tabular" key.
    by_fc: dict[tuple[str, str], pd.DataFrame] = {}
    for model_name, sub in df.groupby("model"):
        tab_fam = REFCAM_TABULAR_MODEL.get(model_name)
        if tab_fam is not None and tab_fam in MODELS:
            by_fc[(tab_fam, "_tabular")] = sub.set_index("config")
            continue
        parsed = _refcam_parse_model(model_name)
        if parsed is None:
            continue
        fam, tier = parsed
        if fam not in MODELS:
            continue
        by_fc[(fam, tier)] = sub.set_index("config")

    task_cols = [c for _, c in REFCAM_TASKS]
    # Selection score per (family, codec, config) = mean of the 6 task NAPs.
    sel_score: dict[tuple[str, str, str], float] = {}
    for (fam, tier), frame in by_fc.items():
        for cfg in frame.index.unique():
            row = frame.loc[cfg]
            if isinstance(row, pd.DataFrame):  # duplicate config — average
                vals = row[task_cols].mean(axis=0)
            else:
                vals = row[task_cols]
            vals = vals.astype(float).dropna()
            if vals.empty:
                continue
            sel_score[(fam, tier, cfg)] = float(vals.mean())

    pinned_cfg = _pick_config_per_family(sel_score, best_selection)

    out: dict[tuple[str, str], float | None] = {}
    for fam in MODELS:
        # Tabular families use their fixed key regardless of requested codec.
        lookup_codec = "_tabular" if fam in TABULAR_FAMILIES else codec
        cfg = pinned_cfg.get(fam, {}).get(lookup_codec)
        if cfg is None:
            cands = {k: v for k, v in sel_score.items()
                     if k[0] == fam and k[1] == lookup_codec}
            if cands:
                cfg = max(cands.items(), key=lambda kv: kv[1])[0][2]
        frame = by_fc.get((fam, lookup_codec))
        if frame is None or cfg is None or cfg not in frame.index:
            for label, _ in REFCAM_TASKS:
                out[(fam, label)] = None
            continue
        row = frame.loc[cfg]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        for label, col in REFCAM_TASKS:
            val = row.get(col, None)
            out[(fam, label)] = float(val) if val is not None and not pd.isna(val) else None
    return out


# ---------------------------------------------------------------------------
# MOTIVE helpers
# ---------------------------------------------------------------------------

def _motive_scores(df: pl.DataFrame, codec: str, best_selection: str,
                   recall_col: str) -> dict[tuple[str, str], float | None]:
    """Return ``{(family_display, task_label): score}`` for the given codec.

    ``codec`` is the MOTIVE-side codec name (e.g. ``"jpegxl_lossy_raw"``).
    Selection score per (family, codec, config) = mean of the 5 MOTIVE task
    recalls.
    """
    # Map (task, modality, group) -> task_label
    motive_tasks_keyed = {(t, m, g): label for label, t, m, g in MOTIVE_TASKS}

    # Aggregate to one row per (family, codec, config, task, modality, group).
    agg = (
        df.group_by(["family", "codec", "config", "task", "modality", "group"])
          .agg(pl.col(recall_col).drop_nulls().mean().alias("val"))
          .filter(pl.col("val").is_not_null())
    )

    # Selection score per (family, codec, config) = mean across the 5 tasks.
    # Filter to rows that match one of our 5 task slices.
    task_filters = [
        ((pl.col("task") == t) &
         (pl.col("modality").is_null() if m is None else pl.col("modality") == m) &
         (pl.col("group").is_null()    if g is None else pl.col("group") == g))
        for _, t, m, g in MOTIVE_TASKS
    ]
    task_mask = task_filters[0]
    for f in task_filters[1:]:
        task_mask = task_mask | f
    agg_tasks = agg.filter(task_mask)

    sel = (
        agg_tasks.group_by(["family", "codec", "config"])
                 .agg(pl.col("val").mean().alias("sel_score"))
    )

    sel_score: dict[tuple[str, str, str], float] = {}
    for r in sel.iter_rows(named=True):
        fam_disp = MOTIVE_FAMILY_DISPLAY.get(r["family"])
        if fam_disp not in MODELS:
            continue
        sel_score[(fam_disp, r["codec"], r["config"])] = float(r["sel_score"])

    pinned_cfg = _pick_config_per_family(sel_score, best_selection)

    out: dict[tuple[str, str], float | None] = {label: None for label in
                                                 [t[0] for t in MOTIVE_TASKS]}
    out = {}  # actually keyed by (family, label)

    for fam in MODELS:
        # Tabular families ship only `codec="raw"` in MOTIVE — fall back to it
        # whatever codec the caller asked for.
        lookup_codec = "raw" if fam in TABULAR_FAMILIES else codec
        cfg = pinned_cfg.get(fam, {}).get(lookup_codec)
        if cfg is None:
            cands = {k: v for k, v in sel_score.items()
                     if k[0] == fam and k[1] == lookup_codec}
            if cands:
                cfg = max(cands.items(), key=lambda kv: kv[1])[0][2]
        if cfg is None:
            for label, *_ in MOTIVE_TASKS:
                out[(fam, label)] = None
            continue

        # Resolve back to family key in MOTIVE to filter agg_tasks.
        fam_key_candidates = [k for k, v in MOTIVE_FAMILY_DISPLAY.items() if v == fam]
        fam_rows = agg_tasks.filter(
            pl.col("family").is_in(fam_key_candidates) &
            (pl.col("codec") == lookup_codec) &
            (pl.col("config") == cfg)
        )
        # Index rows by (task, modality, group)
        idx: dict[tuple, float] = {}
        for r in fam_rows.iter_rows(named=True):
            idx[(r["task"], r["modality"], r["group"])] = float(r["val"])

        for label, t, m, g in MOTIVE_TASKS:
            out[(fam, label)] = idx.get((t, m, g))
    return out


# ---------------------------------------------------------------------------
# Config selection shared by both sources
# ---------------------------------------------------------------------------

def _pick_config_per_family(sel_score: dict[tuple[str, str, str], float],
                            best_selection: str
                            ) -> dict[str, dict[str, str]]:
    """Return ``{family: {codec: config}}`` chosen per ``best_selection``.

    Mirrors the semantics of ``gather_sweep_results._compute_best_idx`` /
    ``plot_motive_cross_task._apply_best_selection`` but operates on a
    pre-aggregated selection score (mean across that source's tasks).
    """
    out: dict[str, dict[str, str]] = {}

    # Group keys by family
    fams = {k[0] for k in sel_score}
    for fam in fams:
        fam_keys = {(k[1], k[2]): v for k, v in sel_score.items() if k[0] == fam}
        if not fam_keys:
            continue

        if best_selection == "per_codec":
            # Each codec picks its own best config.
            by_codec: dict[str, list[tuple[str, float]]] = {}
            for (codec, cfg), v in fam_keys.items():
                by_codec.setdefault(codec, []).append((cfg, v))
            out[fam] = {c: max(items, key=lambda x: x[1])[0] for c, items in by_codec.items()}
            continue

        if best_selection == "best_any_codec":
            best_codec, best_cfg = max(fam_keys.items(), key=lambda x: x[1])[0]
            pinned = best_cfg
        elif best_selection == "best_avg_codec":
            # Per (family, config), avg across codecs.
            by_cfg: dict[str, list[float]] = {}
            for (codec, cfg), v in fam_keys.items():
                by_cfg.setdefault(cfg, []).append(v)
            cfg_means = {cfg: float(np.mean(vs)) for cfg, vs in by_cfg.items()}
            pinned = max(cfg_means.items(), key=lambda x: x[1])[0]
        elif best_selection == "zstd_reference":
            zstd_keys = [(cfg, v) for (codec, cfg), v in fam_keys.items() if codec == "zstd"]
            if not zstd_keys:
                # Fall back to per-codec when zstd absent
                by_codec = {}
                for (codec, cfg), v in fam_keys.items():
                    by_codec.setdefault(codec, []).append((cfg, v))
                out[fam] = {c: max(items, key=lambda x: x[1])[0] for c, items in by_codec.items()}
                continue
            pinned = max(zstd_keys, key=lambda x: x[1])[0]
        else:
            raise ValueError(f"unknown best_selection: {best_selection}")

        # Pin every codec in the family to that config; per-codec fallback if absent.
        codecs_present = {c for c, _ in fam_keys}
        fam_pinned: dict[str, str] = {}
        for codec in codecs_present:
            if (codec, pinned) in fam_keys:
                fam_pinned[codec] = pinned
            else:
                # Per-codec fallback.
                cands = [(cfg, v) for (c, cfg), v in fam_keys.items() if c == codec]
                fam_pinned[codec] = max(cands, key=lambda x: x[1])[0]
        out[fam] = fam_pinned

    return out


# ---------------------------------------------------------------------------
# Bump-chart renderer
# ---------------------------------------------------------------------------

def _plot_rank(scores: dict[tuple[str, str], float | None], *,
               codec_display: str, codec_key: str, best_selection: str,
               output_path: Path) -> None:
    """Render a heatmap of per-task-normalised scores and save to ``output_path``.

    Rows = models (sorted top→bottom by mean normalised score, best on top).
    Columns = tasks. Cell colour = raw value / per-task max across models.
    Each cell is annotated with its raw value. Vertical separators divide
    PA / PC / CC / GG / CG.
    """
    rows = []
    for fam in MODELS:
        rows.append({"model": fam, **{lbl: scores.get((fam, lbl)) for lbl in TASK_ORDER}})
    score_df = pd.DataFrame(rows).set_index("model")

    task_max = score_df.max(axis=0)
    norm_df = score_df.div(task_max.replace(0, np.nan), axis=1)

    row_score = norm_df.mean(axis=1, skipna=True)
    model_order = row_score.sort_values(ascending=False).index.tolist()
    norm_df = norm_df.loc[model_order, TASK_ORDER]
    score_df = score_df.loc[model_order, TASK_ORDER]
    row_score = row_score.loc[model_order]

    n_tasks = len(TASK_ORDER)
    n_models = len(model_order)

    # Append the per-model mean normalised score as an extra column on the
    # right. Both the colour cell and the annotation use this same value
    # (which is already in [0, 1] since each spoke was normalised).
    display_cols = TASK_ORDER + ["Mean"]
    norm_arr = np.concatenate(
        [norm_df.to_numpy(dtype=float), row_score.to_numpy().reshape(-1, 1)], axis=1
    )
    n_cols = len(display_cols)

    fig, ax = plt.subplots(figsize=(0.95 * n_cols + 4, 0.75 * n_models + 4))
    fs_title, fs_axis, fs_tick, fs_legend, fs_anno = 22, 16, 14, 13, 11
    fs_mid_hdr, fs_top_hdr = 14, 16

    # Sequential colormap: white (worst-on-task = 0) → medium red. Truncate
    # the Reds palette to its lower 70% so the darkest cells don't read as
    # near-black. pcolormesh edges give a small white gap between cells.
    from matplotlib.colors import LinearSegmentedColormap
    _reds = plt.get_cmap("Reds")
    cmap = LinearSegmentedColormap.from_list(
        "Reds_truncated",
        _reds(np.linspace(0.0, 0.7, 256)),
    )

    # Shift the Mean column right by ``mean_gap`` to create a wider white
    # break between the task block and the summary column. The shift applies
    # to text annotations, tick positions, and the bracket-span x coords.
    mean_gap = 0.35
    col_x: list[float] = [float(i) for i in range(n_tasks)] + [n_tasks + mean_gap]

    y_edges = np.arange(n_models + 1) - 0.5
    # Task block (first n_tasks columns) — uniform width 1.
    task_x_edges = np.arange(n_tasks + 1) - 0.5
    im = ax.pcolormesh(
        task_x_edges, y_edges, norm_arr[:, :n_tasks],
        cmap=cmap, vmin=0, vmax=1,
        edgecolors="none", zorder=2,
    )
    # Mean column — separately drawn at the shifted x position.
    mean_x_edges = np.array([col_x[-1] - 0.5, col_x[-1] + 0.5])
    ax.pcolormesh(
        mean_x_edges, y_edges, norm_arr[:, n_tasks:n_tasks + 1],
        cmap=cmap, vmin=0, vmax=1,
        edgecolors="none", zorder=2,
    )

    # Column separators only (no row separators): draw thick white verticals
    # between adjacent task columns. The gap between the last task column and
    # the Mean column is already produced by ``mean_gap``.
    y_top = n_models - 0.5
    y_bot = -0.5
    for ci in range(1, n_tasks):
        x = task_x_edges[ci]
        ax.plot([x, x], [y_bot, y_top], color="white", linewidth=3,
                solid_capstyle="butt", zorder=3)

    # Black horizontal separators isolate the full CellProfiler feature row
    # in every panel. CellProfiler and the CellCount baseline both inherit
    # archived six-to-nine-site well aggregation, whereas the image-model
    # representations use four sites. A thick white line underneath each
    # black line creates visible whitespace around the full-feature reference.
    if "CellProfiler" in model_order:
        cp_idx = model_order.index("CellProfiler")
        task_left = task_x_edges[0]
        task_right = task_x_edges[-1]
        mean_left = mean_x_edges[0]
        mean_right = mean_x_edges[-1]
        for y in (cp_idx - 0.5, cp_idx + 0.5):
            ax.plot([task_left, task_right], [y, y], color="white",
                    linewidth=14.0, solid_capstyle="butt", zorder=4)
            ax.plot([mean_left, mean_right], [y, y], color="white",
                    linewidth=14.0, solid_capstyle="butt", zorder=4)
            ax.plot([task_left, task_right], [y, y], color="black",
                    linewidth=2.0, solid_capstyle="butt", zorder=5)
            ax.plot([mean_left, mean_right], [y, y], color="black",
                    linewidth=2.0, solid_capstyle="butt", zorder=5)

    # Identify the strongest non-CellProfiler representation independently
    # for every task and for the normalized Mean, as requested for the
    # emphasized values. CellCount remains eligible but is explicitly labelled
    # as a baseline derived from the same six-to-nine-site CellProfiler wells.
    eligible_models = [m for m in model_order if m != "CellProfiler"]
    winner_by_col = {
        task: score_df.loc[eligible_models, task].idxmax()
        for task in TASK_ORDER
    }
    winner_by_col["Mean"] = row_score.loc[eligible_models].idxmax()

    # Annotate raw values for tasks; for the Mean column, annotate the
    # normalised value directly (it's the row mean of [0,1]-scaled scores).
    # Only the non-CellProfiler winner is bold within each column.
    for yi, fam in enumerate(model_order):
        for xi, col in enumerate(display_cols):
            x_pos = col_x[xi]
            norm = norm_arr[yi, xi]
            if col == "Mean":
                anno = "—" if np.isnan(norm) else f"{norm:.3f}"
            else:
                raw = score_df.loc[fam, col]
                anno = "—" if np.isnan(raw) else f"{raw:.3f}"
            if np.isnan(norm):
                ax.text(x_pos, yi, "—", fontsize=fs_anno, ha="center",
                        va="center", color="gray", zorder=4)
                continue
            # Truncated Reds (top capped at 0.7 of full range) is lighter, so
            # white text is only needed for the very top of the scale.
            txt_color = "white" if norm > 0.88 else "black"
            ax.text(x_pos, yi, anno, fontsize=fs_anno,
                    ha="center", va="center", color=txt_color,
                    fontweight="bold" if fam == winner_by_col[col] else "normal",
                    zorder=4)
    # Group dividers are handled by the white gap from pcolormesh edges + the
    # wider mean_gap offset — no explicit separator line needed.

    ax.set_xticks(col_x)
    # Bottom-tier labels: just the sub-task name + per-task max, mirroring the
    # innermost row of the LaTeX delta-table header.
    bottom_labels = [
        f"{name}\nmax={task_max[task]:.3f}" if not np.isnan(task_max[task])
        else name
        for task, name in zip(TASK_ORDER, SUBTASK_LABELS)
    ] + ["All\n(norm.)"]
    ax.set_xticklabels(bottom_labels, fontsize=fs_tick - 1, rotation=30, ha="right")

    # Hierarchical column headers above the heatmap — same structure as the
    # LaTeX table's \multicolumn{...} rows. Drawn in data coordinates above
    # the first data row (y = -0.5 is the top edge of the heatmap). Span
    # indices are mapped through ``col_x`` so the Mean column's gap is
    # honoured.
    def _draw_bracket(y_line: float, y_text: float, spans, fontsize: int,
                      *, bold: bool = False) -> None:
        for label, c_start, c_end in spans:
            x_start = col_x[c_start] - 0.4
            x_end = col_x[c_end] + 0.4
            ax.plot([x_start, x_end], [y_line, y_line],
                    color="black", linewidth=1.2, clip_on=False, zorder=6)
            ax.text((x_start + x_end) / 2.0, y_text, label,
                    fontsize=fontsize, ha="center", va="bottom",
                    fontweight="bold" if bold else "normal",
                    clip_on=False, zorder=6)

    # Mid tier (bench) at y ≈ -0.9, top tier (source) at y ≈ -1.7.
    _draw_bracket(y_line=-0.7, y_text=-0.85, spans=MID_HEADER_SPANS,
                  fontsize=fs_mid_hdr)
    _draw_bracket(y_line=-1.4, y_text=-1.55, spans=TOP_HEADER_SPANS,
                  fontsize=fs_top_hdr, bold=True)
    ax.set_xlim(-0.6, col_x[-1] + 0.6)
    ax.set_ylim(n_models - 0.5, -2.0)

    ax.set_yticks(np.arange(n_models))
    # Mark the inherited site aggregation for both tabular representations;
    # for lossy codecs, also state that CellProfiler remains Raw-only.
    ytick_labels = []
    for model in model_order:
        if model == "CellProfiler":
            suffix = "raw; 6–9 sites" if codec_key in {"hq", "mq", "d20"} else "6–9 sites"
            ytick_labels.append(f"{model}\n({suffix})")
        elif model == "CellCount":
            ytick_labels.append(f"{model}\n(CP-derived; 6–9 sites)")
        else:
            ytick_labels.append(model)
    ax.set_yticklabels(ytick_labels, fontsize=fs_tick)
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # Remove the outer border (all spines).
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    # Horizontal colorbar tucked into the empty top-left band above the PA
    # columns (PA has no top-tier label, so that strip is free). Position is
    # given in data coordinates via ax.inset_axes(transform=ax.transData).
    pa_span = MID_HEADER_SPANS[0]  # ("PA", 0, 3)
    pa_left = col_x[pa_span[1]] - 0.4
    pa_right = col_x[pa_span[2]] + 0.4
    pa_centre = (pa_left + pa_right) / 2.0
    # Shrink the bar to ~70% of the PA span and centre it under the band.
    cbar_w = (pa_right - pa_left) * 0.7
    cbar_x0 = pa_centre - cbar_w / 2.0
    cbar_y0 = -1.82
    cbar_h = 0.22
    cax = ax.inset_axes(
        [cbar_x0, cbar_y0, cbar_w, cbar_h], transform=ax.transData,
    )
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("raw value / per-task max",
                   fontsize=fs_legend, labelpad=4)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.tick_params(labelsize=fs_legend - 2, length=2, pad=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refchem-csv", type=Path,
        default=Path("data/intermediate/sweep_v11_lite/sweep_results.csv"),
        help="RefChem sweep_results.csv (per-config NAP metrics).",
    )
    parser.add_argument(
        "--motive-csv", type=Path,
        default=Path("data/results/figures/motive_large_strict/motive_sweep_summary.csv"),
        help="MOTIVE motive_sweep_summary.csv (per-config recall@k%%).",
    )
    parser.add_argument(
        "--motive-recall-col", default="recall_1",
        help="Recall column in motive_sweep_summary.csv. Default: recall_1.",
    )
    parser.add_argument(
        "--codecs", default="raw,hq,mq,d20",
        help=(
            "Comma-separated codec keys (raw / hq / mq / d20). One PNG per "
            "codec. 'raw' resolves to the lossless baseline "
            "(jpegxl_lossy_raw on MOTIVE)."
        ),
    )
    parser.add_argument(
        "--best-selection",
        choices=["per_codec", "best_avg_codec", "best_any_codec", "zstd_reference"],
        default="best_avg_codec",
        help="How to pin each family's config. Default: best_avg_codec.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write the PNGs into.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    refchem = pd.read_csv(args.refchem_csv)
    motive = pl.read_csv(args.motive_csv)
    print(f"[load] refchem={len(refchem):,} rows; motive={motive.height:,} rows")

    for codec in args.codecs.split(","):
        codec = codec.strip()
        if codec not in CODEC_CHOICES:
            raise SystemExit(f"unknown codec '{codec}' — choose from {list(CODEC_CHOICES)}")
        spec = CODEC_CHOICES[codec]
        refchem_scores = _refcam_scores(refchem, spec["refchem"], args.best_selection)
        motive_scores = _motive_scores(motive, spec["motive"], args.best_selection,
                                       args.motive_recall_col)
        scores = {**refchem_scores, **motive_scores}

        # Diagnostic
        missing = [k for k, v in scores.items() if v is None]
        if missing:
            print(f"[warn] codec={codec}: {len(missing)} missing (model, task) cells")

        fname = f"model_task_rank_{args.best_selection}_{codec}.png"
        _plot_rank(scores, codec_display=spec["display"], codec_key=codec,
                   best_selection=args.best_selection,
                   output_path=args.output_dir / fname)


if __name__ == "__main__":
    main()
