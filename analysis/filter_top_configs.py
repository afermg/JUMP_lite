#!/usr/bin/env python3
"""Filter a norm_3 ``sweep_results.csv`` to top-N configs per (family, codec).

Reads the long-format sweep results CSV, parses each row's ``model`` column
into ``(family, codec)``, sorts by a ranking metric, takes the top-N per
group, and writes one ``output.parquet`` path per kept config to a text file.

Use the resulting list with ``just motive-eval-list`` to run MOTIVE eval on
the filtered subset only.

Naming conventions handled (all observed under variance_first_v11_lite):
- CSV ``<family>_lite_<codec>_raw`` → dir ``<family>_jump_lite_updated_<codec>_raw_features``
- CSV ``cell_count_lite_raw``       → dir ``cell_count_jump_lite_raw_features``
- CSV ``cellprofiler_lite_raw``     → dir ``cellprofiler_raw_jump_lite_raw_features``
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl


CSV_LOSSY_RE = re.compile(r"^(.+?)_lite_(.+?)_raw$")
CSV_RAW_BASELINES = {"cell_count", "cellprofiler"}


def parse_csv_model(model: str) -> tuple[str | None, str | None]:
    if model in {f"{f}_lite_raw" for f in CSV_RAW_BASELINES}:
        return model.removesuffix("_lite_raw"), "raw"
    m = CSV_LOSSY_RE.match(model)
    if m:
        return m.group(1), m.group(2)
    return None, None


def csv_model_to_featdir(model: str, sweep_dir: Path) -> Path | None:
    """Resolve a CSV ``model`` value to the actual feat-dir under sweep_dir."""
    family, codec = parse_csv_model(model)
    if family is None:
        return None
    if codec == "raw":
        candidates = [
            sweep_dir / f"{family}_jump_lite_raw_features",
            sweep_dir / f"{family}_raw_jump_lite_raw_features",
        ]
    else:
        candidates = [
            sweep_dir / f"{family}_jump_lite_updated_{codec}_raw_features",
        ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-results",
        type=Path,
        required=True,
        help="Path to the long-format sweep_results.csv produced by gather_sweep_results.py.",
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help="Top-level dir containing the feat-dirs that hold each config's output.parquet.",
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--metric",
        type=str,
        default="PA_mean_nap",
        help="Column to rank by, descending. Default: PA_mean_nap.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where to write the resulting list of absolute output.parquet paths.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output list.",
    )
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite existing {args.out}. Pass --force to opt in."
        )

    print(f"[load] {args.sweep_results}")
    df = pl.read_csv(args.sweep_results)
    if args.metric not in df.columns:
        raise SystemExit(
            f"metric column '{args.metric}' not found in CSV. "
            f"Available metric-ish columns include: "
            f"{[c for c in df.columns if c in ('PA','PC','PA_mean_nap','PC_mean_nap')]}"
        )

    parsed = [parse_csv_model(m) for m in df["model"].to_list()]
    df = df.with_columns(
        pl.Series("family", [fc[0] for fc in parsed]),
        pl.Series("codec", [fc[1] for fc in parsed]),
    )
    n_total = df.height
    df = df.filter(
        pl.col("family").is_not_null()
        & pl.col(args.metric).is_not_null()
        & pl.col(args.metric).is_not_nan()
    )
    print(
        f"[load] {df.height:,}/{n_total:,} rows usable "
        f"(non-null family + non-null/non-NaN {args.metric})"
    )
    print(f"[load] unique (family, codec) combos: {df.select('family','codec').n_unique()}")

    top = (
        df.sort(args.metric, descending=True)
        .group_by(["family", "codec"], maintain_order=True)
        .head(args.top_n)
    )
    print(
        f"[filter] top-{args.top_n} per group: {top.height:,} rows total "
        f"({top.select('family','codec').n_unique()} groups)"
    )

    print()
    print("=== rows kept per group ===")
    print(top.group_by(["family", "codec"]).len().sort(["family", "codec"]))

    paths: list[str] = []
    missing: list[str] = []
    for row in top.iter_rows(named=True):
        feat_dir = csv_model_to_featdir(row["model"], args.sweep_dir)
        if feat_dir is None:
            missing.append(f"{row['model']} (no matching feat-dir)")
            continue
        p = feat_dir / row["config"] / "output.parquet"
        if not p.exists():
            missing.append(f"{row['model']}/{row['config']} (parquet missing: {p})")
            continue
        paths.append(str(p))

    if missing:
        print(
            f"\n[warn] {len(missing)} configs unresolved (showing up to 10):",
            file=sys.stderr,
        )
        for m in missing[:10]:
            print(f"  {m}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(paths) + "\n")
    print(f"\n[write] {args.out} ({len(paths):,} paths)")


if __name__ == "__main__":
    main()
