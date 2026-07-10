#!/usr/bin/env python3
"""LaTeX percentage-delta table for MOTIVE results.

Mirrors ``gather_sweep_results.generate_codec_delta_from_raw_groups_plot``'s
``codec_delta_pct_table.tex`` output, but for MOTIVE retrieval recall@k%.

Layout:
- Each row = (codec, family). Codec spans a block; only the middle row of the
  block shows the codec label.
- Within a codec block, an italic "Mean" row aggregates across families.
- Columns grouped by MOTIVE task — **CC**, **GG**, **CG** — with a "Mean"
  sub-column per group followed by the per-panel sub-columns.
- Each cell is ``mean ± std`` of the **percentage delta** vs the family's
  baseline codec (preference order: raw → jpegxl_lossy_raw → zstd → lz4hc →
  blosc_zstd) across configs paired by ``config`` name.

Reads the long-format ``motive_sweep_summary.csv`` produced by
``analysis/plot_motive_results.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Re-use the shared logic + dictionaries from the delta plot script.
sys.path.insert(0, str(Path(__file__).parent))
from plot_motive_codec_delta import (  # noqa: E402
    CODEC_DISPLAY,
    CODEC_ORDER,
    FAMILY_DISPLAY,
    FAMILY_PLOT_ORDER,
    build_deltas,
)


# (group_label, [(panel_name, (task, modality, group))...]).
# Panel names are LaTeX-safe (arrows escaped to $\to$).
TASK_GROUPS: list[tuple[str, list[tuple[str, tuple[str, str | None, str | None]]]]] = [
    ("CC", [
        ("Divs",   ("CC", None, "group_high")),
        ("Bioact", ("CC", None, "group_low")),
    ]),
    ("GG", [
        ("ORF",    ("GG", "orf",    None)),
        ("CRISPR", ("GG", "crispr", None)),
    ]),
    ("CG", [
        (r"Divs $\to$ ORF",      ("CG", "orf",    "group_high")),
        (r"Divs $\to$ CRISPR",   ("CG", "crispr", "group_high")),
        (r"Bioact $\to$ ORF",    ("CG", "orf",    "group_low")),
        (r"Bioact $\to$ CRISPR", ("CG", "crispr", "group_low")),
    ]),
]


def _codec_sort_key(codec: str) -> tuple[int, str]:
    if codec in CODEC_ORDER:
        return CODEC_ORDER.index(codec), codec
    return len(CODEC_ORDER), codec


def _family_sort_key(family: str) -> int:
    return FAMILY_PLOT_ORDER.index(family) if family in FAMILY_PLOT_ORDER else 999


def _filtered_task_groups(drop_orf: bool):
    """Return TASK_GROUPS with the three ORF-containing panels removed when
    ``drop_orf`` is set. Empty groups (e.g. GG with only ORF) are dropped."""
    if not drop_orf:
        return TASK_GROUPS
    out = []
    for group_label, panels in TASK_GROUPS:
        kept = [p for p in panels if p[1][1] != "orf"]
        if kept:
            out.append((group_label, kept))
    return out


def aggregate_pct(
    delta_df: pl.DataFrame,
    drop_orf: bool = False,
) -> pl.DataFrame:
    """Aggregate ``delta_pct`` mean/std per (family, codec, panel_name).

    ``delta_pct = delta / |baseline_recall| * 100`` with a safe-divide guard
    matching the reference implementation.
    """
    df = delta_df.with_columns(
        pl.when(pl.col("baseline_recall").abs() > 1e-12)
          .then((pl.col("delta") / pl.col("baseline_recall").abs()) * 100)
          .otherwise(None)
          .alias("delta_pct"),
    )

    rows: list[dict] = []
    for _group_label, panels in _filtered_task_groups(drop_orf):
        for panel_name, (task, modality, group) in panels:
            sub = df.filter(pl.col("task") == task)
            sub = sub.filter(
                pl.col("modality").is_null() if modality is None
                else pl.col("modality") == modality
            )
            sub = sub.filter(
                pl.col("group").is_null() if group is None
                else pl.col("group") == group
            )
            agg = sub.group_by(["family", "codec"]).agg([
                pl.col("delta_pct").drop_nulls().mean().alias("mean"),
                pl.col("delta_pct").drop_nulls().std().alias("std"),
                pl.col("delta_pct").drop_nulls().len().alias("n"),
            ])
            for r in agg.iter_rows(named=True):
                rows.append({
                    "panel": panel_name,
                    "family": r["family"],
                    "codec": r["codec"],
                    "mean": r["mean"],
                    "std": r["std"],
                    "n": r["n"],
                })
    return pl.DataFrame(rows)


def _fmt_cell(mean_v: float | None, std_v: float | None) -> str:
    if mean_v is None or (isinstance(mean_v, float) and np.isnan(mean_v)):
        return "--"
    if std_v is None or (isinstance(std_v, float) and np.isnan(std_v)) or std_v == 0:
        return f"{mean_v:+.1f}"
    return f"{mean_v:+.1f} $\\pm$ {std_v:.1f}"


def _fmt_mean(mean_v: float | None) -> str:
    if mean_v is None or (isinstance(mean_v, float) and np.isnan(mean_v)):
        return "--"
    return f"{mean_v:+.1f}"


def _lookup(
    agg_df: pl.DataFrame, family: str, codec: str, panel: str,
) -> tuple[float | None, float | None]:
    row = agg_df.filter(
        (pl.col("family") == family)
        & (pl.col("codec") == codec)
        & (pl.col("panel") == panel)
    )
    if row.is_empty():
        return None, None
    r = row.row(0, named=True)
    return r["mean"], r["std"]


def build_latex_table(
    agg_df: pl.DataFrame, k_pct: int, drop_orf: bool = False,
) -> str:
    """Compose the LaTeX table string."""
    # Codecs that have at least one row in the aggregation, ordered by
    # CODEC_ORDER (= compression-level order).
    present_codecs = (
        agg_df.select("codec").unique().to_series().to_list()
    )
    present_codecs = sorted(present_codecs, key=_codec_sort_key)

    # Families, ordered by FAMILY_PLOT_ORDER.
    present_families = (
        agg_df.select("family").unique().to_series().to_list()
    )
    present_families = sorted(present_families, key=_family_sort_key)

    # Column counts per task group: 1 ("Mean") + n_panels. When the only panel
    # left after ORF-dropping is a single one, skip the redundant "Mean" col.
    group_specs = []  # (group_label, panel_names, col_count, has_mean)
    for group_label, panels in _filtered_task_groups(drop_orf):
        panel_names = [p[0] for p in panels]
        has_mean = len(panel_names) > 1
        col_count = (1 if has_mean else 0) + len(panel_names)
        group_specs.append((group_label, panel_names, col_count, has_mean))
    n_data_cols = sum(g[2] for g in group_specs)

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Mean percentage performance change in MOTIVE recall@"
        + str(k_pct)
        + r"\% from baseline (raw / lossless) codec ($\pm$ std across "
        + r"normalization configs).}"
    )
    lines.append(r"\label{tab:motive_codec_delta_pct_" + str(k_pct) + "pct}")
    col_spec = "ll" + "c" * n_data_cols
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    # Group header row.
    group_header_parts = [" ", " "]  # empty cells for Codec & Model
    cmidrule_parts: list[str] = []
    col_pos = 3  # 1-indexed; columns 1-2 are Codec & Model
    for group_label, _panel_names, col_count, _has_mean in group_specs:
        group_header_parts.append(
            r"\multicolumn{" + str(col_count)
            + r"}{c}{\textbf{" + group_label + r" \% $\Delta$}}"
        )
        cmidrule_parts.append(
            r"\cmidrule(lr){" + str(col_pos) + "-" + str(col_pos + col_count - 1) + "}"
        )
        col_pos += col_count
    lines.append(" & ".join(group_header_parts) + r" \\")
    lines.append(" ".join(cmidrule_parts))

    # Sub-header row.
    sub_names: list[str] = []
    for _group_label, panel_names, _col_count, has_mean in group_specs:
        if has_mean:
            sub_names.append("Mean")
        sub_names.extend(panel_names)
    lines.append(
        r"\textbf{Codec} & \textbf{Model} & "
        + " & ".join(sub_names)
        + r" \\"
    )
    lines.append(r"\midrule")

    def _row_cells(family: str, codec: str) -> tuple[list[str], list[float]]:
        """Return ([formatted cells], [numeric column means]) for a family×codec row."""
        cells: list[str] = []
        col_means: list[float] = []
        for _group_label, panel_names, _col_count, has_mean in group_specs:
            panel_means: list[float] = []
            panel_cells: list[str] = []
            for panel in panel_names:
                mv, sv = _lookup(agg_df, family, codec, panel)
                panel_means.append(mv if mv is not None else float("nan"))
                panel_cells.append(_fmt_cell(mv, sv))
            if has_mean:
                valid = [v for v in panel_means if not np.isnan(v)]
                group_mean = float(np.mean(valid)) if valid else float("nan")
                cells.append(_fmt_mean(group_mean))
                col_means.append(group_mean)
            cells.extend(panel_cells)
            col_means.extend(panel_means)
        return cells, col_means

    # Data rows: blocks by codec, with families sub-grouped inside each block.
    for codec_idx, codec in enumerate(present_codecs):
        codec_label = CODEC_DISPLAY.get(codec, codec)
        # Families that actually have data for this codec.
        families_for_codec = (
            agg_df.filter(pl.col("codec") == codec)
                  .select("family")
                  .unique()
                  .to_series()
                  .to_list()
        )
        families_for_codec = sorted(families_for_codec, key=_family_sort_key)
        if not families_for_codec:
            continue

        n_data_rows = len(families_for_codec)
        n_block_rows = n_data_rows + (1 if n_data_rows > 1 else 0)
        mid_row = n_block_rows // 2

        block_means: list[list[float]] = []
        for row_idx, family in enumerate(families_for_codec):
            cells, col_means = _row_cells(family, codec)
            block_means.append(col_means)
            cl_cell = codec_label if row_idx == mid_row else ""
            fam_disp = FAMILY_DISPLAY.get(family, family)
            lines.append(
                f"{cl_cell} & {fam_disp} & " + " & ".join(cells) + r" \\"
            )

        # Mean row across families within this codec block.
        if n_data_rows > 1:
            arr = np.array(block_means, dtype=float)
            mean_cells: list[str] = []
            for col_i in range(arr.shape[1]):
                col_vals = arr[:, col_i]
                valid = col_vals[~np.isnan(col_vals)]
                if valid.size > 0:
                    mean_cells.append(f"\\textbf{{{np.mean(valid):+.1f}}}")
                else:
                    mean_cells.append("--")
            cl_cell = codec_label if mid_row == n_data_rows else ""
            lines.append(
                f"{cl_cell} & \\textit{{Mean}} & "
                + " & ".join(mean_cells)
                + r" \\"
            )

        if codec_idx < len(present_codecs) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def generate(
    summary_csv: Path,
    output_path: Path,
    k_pct: int,
    excluded_families: list[str] | None,
    force: bool,
    drop_orf: bool = False,
) -> None:
    if output_path.exists() and not force:
        raise SystemExit(
            f"refusing to overwrite {output_path} — pass --force to override"
        )

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
        raise SystemExit("no deltas computed")

    print(f"[delta] {delta_df.height:,} rows across "
          f"{delta_df.select('family', 'codec').n_unique()} (family, codec) pairs")
    for fam in sorted(baseline_per_family):
        n = delta_df.filter(pl.col("family") == fam).height
        print(f"  {fam}: baseline={baseline_per_family[fam]}, n_paired_rows={n:,}")

    agg_df = aggregate_pct(delta_df, drop_orf=drop_orf)
    if agg_df.is_empty():
        raise SystemExit("no aggregated rows — empty input?")
    print(f"[agg] {agg_df.height:,} (family, codec, panel) cells"
          + (" (ORF dropped)" if drop_orf else ""))

    latex = build_latex_table(agg_df, k_pct, drop_orf=drop_orf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex)
    print(f"[write] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv", type=Path, required=True,
        help="motive_sweep_summary.csv produced by plot_motive_results.py",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write the .tex file(s) into.",
    )
    parser.add_argument(
        "--k-pcts", type=str, default="1,5,10",
        help="Comma-separated k%% values (one .tex file per value). Default: 1,5,10.",
    )
    parser.add_argument(
        "--exclude-families", type=str, default="dinov2_random",
        help=(
            "Comma-separated family names to drop before computing deltas. "
            "Default: dinov2_random. Pass an empty string to keep everything."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing .tex files instead of refusing.",
    )
    args = parser.parse_args()

    excluded = [f.strip() for f in args.exclude_families.split(",") if f.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for k_str in args.k_pcts.split(","):
        k_pct = int(k_str.strip())
        out = args.output_dir / f"motive_codec_delta_pct_table_{k_pct}pct.tex"
        generate(args.summary_csv, out, k_pct,
                 excluded_families=excluded, force=args.force, drop_orf=False)
        out_noorf = args.output_dir / f"motive_codec_delta_pct_table_{k_pct}pct_noORF.tex"
        generate(args.summary_csv, out_noorf, k_pct,
                 excluded_families=excluded, force=args.force, drop_orf=True)


if __name__ == "__main__":
    main()
