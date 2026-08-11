#!/usr/bin/env python3
"""Combined RefChem + MOTIVE codec-delta LaTeX table.

Builds a single narrow table covering, per codec tier (hq / mq / d20), the
mean %-delta from each family's baseline codec across four benchmarks:

  - RefChem PA: CRISPR, ORF, Divs, Bioact         (sweep_results.csv)
  - RefChem PC: Divs, Bioact                      (sweep_results.csv)
  - MOTIVE CC : Divs, Bioact                      (motive_sweep_summary.csv)
  - MOTIVE GG : CRISPR                            (motive_sweep_summary.csv)
  - MOTIVE CG : Divs -> CRISPR, Bioact -> CRISPR  (motive_sweep_summary.csv)

Plus a trailing "Mean" column = unweighted mean across the 11 task cells.

Outputs two .tex files:
  combined_codec_delta_pct_table.tex          full per-model rows plus a
                                              clearly labeled Mean row
  combined_codec_delta_pct_table_summary.tex  per-tier aggregate only

Assumptions:
  - Top-tier header: RefChem spans PA + PC; MOTIVE spans CC/GG/CG; Mean trails.
    Mid-tier: PA, PC, CC, GG, CG -> CRISPR.
  - Models are the four shared by both data sources: DINOv2, MorphEm,
    OpenPhenom, SubCell. Other families in the CSVs are dropped.
  - Codec tiers shown: hq, mq, d20 (matches the existing per-source tables).
  - Negative NAP values are floored to 0 before computing the RefChem %-delta
    (mirroring gather_sweep_results.generate_codec_delta_from_raw_groups_plot).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


# ---------------------------------------------------------------------------
# Shared config: which families/codecs/metrics make it into this table
# ---------------------------------------------------------------------------

MODEL_DISPLAY_ORDER = ["DINOv2", "MorphEm", "OpenPhenom", "SubCell"]

CODEC_TIERS = ["hq", "mq", "d20"]
# Map (family-from-CSV, codec-from-CSV) cells to the codec tier above.
# RefCam model strings end with `_jpegxl_lossy_<codec>_raw`; MOTIVE codec
# column already holds `jpegxl_lossy_<codec>`. We normalize both sides into
# the short tier names below.
CODEC_NORMALIZE = {
    "hq": "hq",
    "mq": "mq",
    "d20": "d20",
}

# Reader-visible labels only; internal codec tiers and filenames stay lowercase.
CODEC_DISPLAY = {
    "hq": "JXL-HQ",
    "mq": "JXL-MQ",
    "d20": "JXL-D20",
}

# RefCam family substrings to display labels.
REFCAM_FAMILY_DISPLAY = {
    "dinov2_lite":           "DINOv2",
    "morphem_lite":          "MorphEm",
    "openphenom_lite":       "OpenPhenom",
    "subcell__clip01_lite":  "SubCell",
    "subcell_lite":          "SubCell",
}
# MOTIVE family values to display labels.
MOTIVE_FAMILY_DISPLAY = {
    "dinov2":           "DINOv2",
    "morphem":          "MorphEm",
    "openphenom":       "OpenPhenom",
    "subcell":          "SubCell",
    "subcell__clip01":  "SubCell",
}

# Per-group NAP columns we surface in PA / PC.
REFCAM_PA_METRICS = [
    ("CRISPR", "PA_group_crispr_mean_normalized_average_precision"),
    ("ORF",    "PA_group_orf_mean_normalized_average_precision"),
    ("Divs",   "PA_group_high_mean_normalized_average_precision"),
    ("Bioact", "PA_group_low_mean_normalized_average_precision"),
]
REFCAM_PC_METRICS = [
    ("Divs",   "PC_group_high_mean_normalized_average_precision"),
    ("Bioact", "PC_group_low_mean_normalized_average_precision"),
]

# MOTIVE panels. Each panel name is one numeric column in the table.
# panel_name -> (task, modality, group). modality / group can be None.
MOTIVE_PANELS = [
    ("CC_Divs",     ("CC", None, "group_high")),
    ("CC_Bioact",   ("CC", None, "group_low")),
    ("GG_CRISPR",   ("GG", "crispr", None)),
    ("CG_Divs",     ("CG", "crispr", "group_high")),   # Divs -> CRISPR
    ("CG_Bioact",   ("CG", "crispr", "group_low")),    # Bioact -> CRISPR
]

# Baseline preference for MOTIVE (first that appears in the family wins).
MOTIVE_BASELINE_PREFERENCE = ("raw", "jpegxl_lossy_raw", "zstd", "lz4hc", "blosc_zstd")


# ---------------------------------------------------------------------------
# RefCam: parse sweep_results.csv -> per-(model, codec_tier, metric) mean/std
# ---------------------------------------------------------------------------

def _refcam_family_codec(model_name: str) -> tuple[str, str] | None:
    """Map a sweep_results row's `model` string to (family_display, codec_tier).

    Returns None if the row isn't a {family} x {hq/mq/d20} cell (e.g. it's the
    `raw` baseline, or a family we don't show).
    """
    fam_disp = None
    fam_key = None
    for fk, fd in REFCAM_FAMILY_DISPLAY.items():
        if model_name.startswith(fk + "_"):
            fam_disp, fam_key = fd, fk
            break
    if fam_disp is None:
        return None
    suffix = model_name[len(fam_key) + 1:]  # e.g. "jpegxl_lossy_hq_raw"
    for tier in CODEC_TIERS:
        if suffix == f"jpegxl_lossy_{tier}_raw":
            return fam_disp, tier
    return None


def _refcam_is_baseline(model_name: str) -> tuple[str, str] | None:
    """Return (family_display, family_key) if this row is the family's raw baseline."""
    for fk, fd in REFCAM_FAMILY_DISPLAY.items():
        if model_name == f"{fk}_jpegxl_lossy_raw_raw":
            return fd, fk
    return None


def compute_refcam_stats(csv_path: Path) -> pd.DataFrame:
    """Per (family_display, codec_tier, bench, metric_label) -> (mean, std) in %.

    Mirrors generate_codec_delta_from_raw_groups_plot's aggregation:
      - pair codec vs baseline by `config` within a family
      - cap NAP values at 0
      - delta_pct = (codec - baseline) / |baseline| * 100
    """
    df = pd.read_csv(csv_path)

    # Build per-family baseline lookup (config -> row) keyed by family display.
    baseline_by_family: dict[str, pd.DataFrame] = {}
    for model_name, sub in df.groupby("model"):
        b = _refcam_is_baseline(model_name)
        if b is None:
            continue
        fam_disp, _ = b
        if fam_disp in baseline_by_family:
            # Multiple baselines per family shouldn't happen here, but if it
            # does we keep the first.
            continue
        baseline_by_family[fam_disp] = sub.set_index("config")

    metrics_all = REFCAM_PA_METRICS + REFCAM_PC_METRICS
    metric_to_bench = {col: "PA" for _, col in REFCAM_PA_METRICS}
    metric_to_bench.update({col: "PC" for _, col in REFCAM_PC_METRICS})
    metric_to_label = {col: lbl for lbl, col in metrics_all}

    rows: list[dict] = []
    for model_name, sub in df.groupby("model"):
        cell = _refcam_family_codec(model_name)
        if cell is None:
            continue
        fam_disp, tier = cell
        base_df = baseline_by_family.get(fam_disp)
        if base_df is None:
            continue
        codec_df = sub.set_index("config")
        shared = codec_df.index.intersection(base_df.index)
        if len(shared) == 0:
            continue
        for _, metric_col in metrics_all:
            if metric_col not in df.columns:
                continue
            base_vals = base_df.loc[shared, metric_col].astype(float).clip(lower=0)
            codec_vals = codec_df.loc[shared, metric_col].astype(float).clip(lower=0)
            mask = (~np.isnan(base_vals)) & (~np.isnan(codec_vals))
            if not mask.any():
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                delta_pct = np.where(
                    np.abs(base_vals[mask]) > 1e-12,
                    (codec_vals[mask] - base_vals[mask])
                    / np.abs(base_vals[mask]) * 100,
                    np.nan,
                )
            valid = delta_pct[~np.isnan(delta_pct)]
            if valid.size == 0:
                continue
            rows.append({
                "model": fam_disp,
                "codec": tier,
                "bench": metric_to_bench[metric_col],
                "metric": metric_to_label[metric_col],
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MOTIVE: parse motive_sweep_summary.csv -> per-(model, codec_tier, panel) stats
# ---------------------------------------------------------------------------

def _motive_codec_tier(codec_str: str) -> str | None:
    for tier in CODEC_TIERS:
        if codec_str == f"jpegxl_lossy_{tier}":
            return tier
    return None


def _pick_motive_baseline(codecs: list[str]) -> str | None:
    for c in MOTIVE_BASELINE_PREFERENCE:
        if c in codecs:
            return c
    return None


def compute_motive_stats(csv_path: Path, recall_col: str = "recall_1") -> pd.DataFrame:
    """Per (family_display, codec_tier, panel) -> (mean, std) of delta_pct.

    Pairs (config, task, modality, group) inside each family vs the family's
    baseline codec; delta_pct = (codec - baseline) / |baseline| * 100.
    """
    df = pl.read_csv(csv_path)
    if recall_col not in df.columns:
        raise SystemExit(f"recall column '{recall_col}' missing from {csv_path}")

    rows: list[dict] = []
    for fam in df.select("family").unique().to_series().to_list():
        fam_disp = MOTIVE_FAMILY_DISPLAY.get(fam)
        if fam_disp is None:
            continue
        codecs = (
            df.filter(pl.col("family") == fam)
              .select("codec").unique().to_series().to_list()
        )
        baseline = _pick_motive_baseline(codecs)
        if baseline is None:
            continue
        base = (
            df.filter((pl.col("family") == fam) & (pl.col("codec") == baseline))
              .select("config", "task", "modality", "group", recall_col)
              .rename({recall_col: "baseline_recall"})
        )

        for codec in codecs:
            tier = _motive_codec_tier(codec)
            if tier is None:
                continue
            other = (
                df.filter((pl.col("family") == fam) & (pl.col("codec") == codec))
                  .select("config", "task", "modality", "group", recall_col)
                  .rename({recall_col: "codec_recall"})
            )
            paired = other.join(
                base,
                on=["config", "task", "modality", "group"],
                how="inner",
                nulls_equal=True,
            )
            if paired.is_empty():
                continue
            paired = paired.with_columns(
                pl.when(pl.col("baseline_recall").abs() > 1e-12)
                  .then((pl.col("codec_recall") - pl.col("baseline_recall"))
                        / pl.col("baseline_recall").abs() * 100)
                  .otherwise(None)
                  .alias("delta_pct")
            )

            for panel_name, (task, modality, group) in MOTIVE_PANELS:
                sub = paired.filter(pl.col("task") == task)
                sub = sub.filter(
                    pl.col("modality").is_null() if modality is None
                    else pl.col("modality") == modality
                )
                sub = sub.filter(
                    pl.col("group").is_null() if group is None
                    else pl.col("group") == group
                )
                vals = sub["delta_pct"].drop_nulls().to_numpy()
                if vals.size == 0:
                    continue
                rows.append({
                    "model": fam_disp,
                    "codec": tier,
                    "panel": panel_name,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compose the combined per-cell value table (model x codec x column)
# ---------------------------------------------------------------------------

# Column order in the final table: (bench_label_in_header, column_id, fetch_fn)
# fetch_fn(model, codec, refcam_df, motive_df) -> (mean, std) or (None, None)
COLUMN_DEFS = [
    ("PA",   "CRISPR", "refcam"),
    ("PA",   "ORF",    "refcam"),
    ("PA",   "Divs",   "refcam"),
    ("PA",   "Bioact", "refcam"),
    ("PC",   "Divs",   "refcam"),
    ("PC",   "Bioact", "refcam"),
    ("CC",   "Divs",   "motive"),
    ("CC",   "Bioact", "motive"),
    ("GG",   "CRISPR", "motive"),
    ("CG",   "Divs",   "motive"),
    ("CG",   "Bioact", "motive"),
]

MOTIVE_PANEL_FOR_COL = {
    ("CC", "Divs"):   "CC_Divs",
    ("CC", "Bioact"): "CC_Bioact",
    ("GG", "CRISPR"): "GG_CRISPR",
    ("CG", "Divs"):   "CG_Divs",
    ("CG", "Bioact"): "CG_Bioact",
}


def _fetch(model: str, codec: str, bench: str, metric: str, source: str,
           refcam: pd.DataFrame, motive: pd.DataFrame) -> tuple[float | None, float | None]:
    if source == "refcam":
        row = refcam[
            (refcam["model"] == model)
            & (refcam["codec"] == codec)
            & (refcam["bench"] == bench)
            & (refcam["metric"] == metric)
        ]
    else:
        panel = MOTIVE_PANEL_FOR_COL[(bench, metric)]
        row = motive[
            (motive["model"] == model)
            & (motive["codec"] == codec)
            & (motive["panel"] == panel)
        ]
    if row.empty:
        return None, None
    return float(row["mean"].iloc[0]), float(row["std"].iloc[0])


def _fmt_with_std(m: float | None, s: float | None) -> str:
    """Cell with std, brace-wrapped for an S column."""
    if m is None or np.isnan(m):
        return "{--}"
    if s is None or np.isnan(s) or s == 0:
        return "{" + f"{m:+.1f}" + "}"
    return "{" + f"{m:+.1f} $\\pm$ {s:.1f}" + "}"


def _fmt_bare(m: float | None) -> str:
    """Bare numeric for the active aggregate row (no braces, no std)."""
    if m is None or np.isnan(m):
        return "{--}"
    return f"{m:+.1f}"


# ---------------------------------------------------------------------------
# LaTeX emission
# ---------------------------------------------------------------------------

def _header_lines() -> list[str]:
    """Four-tier header for the 14-column table (Mean column trails).

    Columns: 1-2 = labels, 3-6 = PA (compound consistency, no top-tier
    super-header), 7-8 = PC (under RefChem), 9-10 = CC, 11 = GG, 12-13 = CG,
    14 = Mean.

    Tier 1 (family):  blank | RefChem | MOTIVE | Mean
    Tier 2 (metric):  NAP $\\Delta$ | Recall@1 $\\Delta$
    Tier 3 (sub):     PA | PC | CC | GG | CG -> CRISPR
    Tier 4 (cols):    CRISPR, ORF, Divs, ...
    """
    top = (
        r" & & \multicolumn{4}{c}{} & "
        r"\multicolumn{2}{c}{\textbf{RefChem}} & "
        r"\multicolumn{5}{c}{\textbf{MOTIVE}} & "
        r"\multicolumn{1}{c}{\textbf{Mean}} \\"
    )
    top_rules = (
        r"\cmidrule(lr){7-8} \cmidrule(lr){9-13} \cmidrule(lr){14-14}"
    )
    metric = (
        " & & "
        r"\multicolumn{6}{c}{NAP $\Delta$} & "
        r"\multicolumn{5}{c}{Recall@1 $\Delta$} & \\"
    )
    metric_rules = r"\cmidrule(lr){3-8} \cmidrule(lr){9-13}"
    mid = (
        " & & "
        r"\multicolumn{4}{c}{PA} & "
        r"\multicolumn{2}{c}{PC} & "
        r"\multicolumn{2}{c}{CC} & "
        r"\multicolumn{1}{c}{GG} & "
        r"\multicolumn{2}{c}{CG $\to$ CRISPR} & \\"
    )
    mid_rules = (
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-8} "
        r"\cmidrule(lr){9-10} \cmidrule(lr){11-11} \cmidrule(lr){12-13}"
    )
    bottom_cells = [
        "{CRISPR}", "{ORF}", "{Divs}", "{Bioact}",  # PA
        "{Divs}", "{Bioact}",                       # PC
        "{Divs}", "{Bioact}",                       # CC
        "{CRISPR}",                                 # GG
        "{Divs}", "{Bioact}",                       # CG -> CRISPR
        "{All}",                                    # Mean across all tasks
    ]
    bottom = r"\textbf{Codec} & & " + " & ".join(bottom_cells) + r" \\"
    return [top, top_rules, metric, metric_rules, mid, mid_rules, bottom]


def _model_task_means(model: str, codec: str,
                      refcam: pd.DataFrame, motive: pd.DataFrame) -> list[float]:
    """Per-task mean values for one (model, codec). Skips missing entries."""
    out: list[float] = []
    for bench, metric, src in COLUMN_DEFS:
        mv, _ = _fetch(model, codec, bench, metric, src, refcam, motive)
        if mv is not None and not np.isnan(mv):
            out.append(mv)
    return out


def _row_cells_with_std(model: str, codec: str,
                        refcam: pd.DataFrame, motive: pd.DataFrame) -> list[str]:
    task_means = _model_task_means(model, codec, refcam, motive)
    mean_cell = (
        _fmt_with_std(float(np.mean(task_means)), None)
        if task_means else "{--}"
    )
    return [
        _fmt_with_std(*_fetch(model, codec, bench, metric, src, refcam, motive))
        for bench, metric, src in COLUMN_DEFS
    ] + [mean_cell]


def _aggregate_cells(codec: str,
                     refcam: pd.DataFrame, motive: pd.DataFrame,
                     models: list[str]) -> list[str]:
    """Per-column unweighted mean across models for the active aggregate row.

    Trailing 'Mean' cell is the unweighted mean of the 11 per-task aggregates.
    """
    task_cells_num: list[float] = []
    task_cells_str: list[str] = []
    for bench, metric, src in COLUMN_DEFS:
        vals: list[float] = []
        for m in models:
            mv, _ = _fetch(m, codec, bench, metric, src, refcam, motive)
            if mv is not None and not np.isnan(mv):
                vals.append(mv)
        agg = float(np.mean(vals)) if vals else float("nan")
        task_cells_num.append(agg)
        task_cells_str.append(_fmt_bare(agg if not np.isnan(agg) else None))
    valid = [v for v in task_cells_num if not np.isnan(v)]
    mean_cell = _fmt_bare(float(np.mean(valid)) if valid else None)
    return task_cells_str + [mean_cell]


def build_combined_table(
    refcam: pd.DataFrame, motive: pd.DataFrame,
    *, summary_only: bool, caption: str, label: str,
) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\tiny")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    # Full table is wider than \textwidth even at \tiny — shrink it to fit.
    # Summary table fits without scaling, so leave it unscaled.
    if not summary_only:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{ll NNNN NN NN N NN N}")
    lines.append(r"\toprule")
    lines.extend(_header_lines())
    lines.append(r"\midrule")

    models = MODEL_DISPLAY_ORDER
    for tier_idx, codec in enumerate(CODEC_TIERS):
        codec_display = CODEC_DISPLAY[codec]
        if not summary_only:
            # Show the codec once in each five-row block, centered on the
            # OpenPhenom row; the final row is the aggregate across models.
            for m in models:
                tier_cell = codec_display if m == "OpenPhenom" else ""
                cells = _row_cells_with_std(m, codec, refcam, motive)
                lines.append(
                    f"{tier_cell} & {m} & " + " & ".join(cells) + r" \\"
                )
        agg_cells = _aggregate_cells(codec, refcam, motive, models)
        if summary_only:
            lines.append(
                f"{codec_display} & \\textit{{Mean}} & "
                + " & ".join(agg_cells)
                + r" \\"
            )
        else:
            lines.append(
                r" & \textit{Mean} & " + " & ".join(agg_cells) + r" \\"
            )
        if tier_idx < len(CODEC_TIERS) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if not summary_only:
        lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_CAPTION = (
    r"Mean percentage performance change from baseline (raw/lossless, zstd) "
    r"codec ($\pm$ std across normalization configs). RefChem reports NAP; "
    r"MOTIVE reports recall@1\%."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refchem-csv",
        type=Path,
        default=Path("data/intermediate/sweep_v11_lite/sweep_results.csv"),
        help="RefChem sweep_results.csv (per-config NAP metrics).",
    )
    parser.add_argument(
        "--motive-csv",
        type=Path,
        default=Path("data/results/figures/motive_large_strict/motive_sweep_summary.csv"),
        help="MOTIVE motive_sweep_summary.csv (per-config recall@k%%).",
    )
    parser.add_argument(
        "--motive-recall-col", default="recall_1",
        help="Recall column in motive_sweep_summary.csv. Default: recall_1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/tables/combined_codec_delta"),
        help="Directory to write the .tex files into.",
    )
    parser.add_argument(
        "--caption", default=DEFAULT_CAPTION,
        help="Override the caption text.",
    )
    parser.add_argument(
        "--label-full", default="tab:codec_delta_pct_combined",
        help="LaTeX label for the full table.",
    )
    parser.add_argument(
        "--label-summary", default="tab:codec_delta_pct_combined_summary",
        help="LaTeX label for the summary-only table.",
    )
    args = parser.parse_args()

    refchem_df = compute_refcam_stats(args.refchem_csv)
    motive_df = compute_motive_stats(args.motive_csv, args.motive_recall_col)
    print(f"[refchem] {len(refchem_df):,} (model, codec, bench, metric) cells")
    print(f"[motive]  {len(motive_df):,} (model, codec, panel) cells")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    full = build_combined_table(
        refchem_df, motive_df,
        summary_only=False,
        caption=args.caption,
        label=args.label_full,
    )
    summary = build_combined_table(
        refchem_df, motive_df,
        summary_only=True,
        caption=args.caption + " Per-tier aggregate only.",
        label=args.label_summary,
    )

    full_path = args.output_dir / "combined_codec_delta_pct_table.tex"
    summary_path = args.output_dir / "combined_codec_delta_pct_table_summary.tex"
    full_path.write_text(full)
    summary_path.write_text(summary)
    print(f"[write] {full_path}")
    print(f"[write] {summary_path}")


if __name__ == "__main__":
    main()
