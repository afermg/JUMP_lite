#!/usr/bin/env python3
"""Render the release-aligned JUMP-lite target-overlap SuperVenn."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from supervenn import supervenn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = REPO_ROOT / "metadata/jump_lite_v1_perturbation_metadata.parquet"
DEFAULT_REFCHEM = REPO_ROOT / "data/refchemdb/refchemdb_conf_jump_matched.parquet"
EXPECTED_METADATA_SHA256 = (
    "bbedb37f12fdeb9a09e72abaa166159d286052dc1201166d811c5310db5cd7e1"
)
EXPECTED_REFCHEM_SHA256 = (
    "34d4b6eae4c87cfcc135a5d91523388475b1482ebfcbb02ac243f6d13fb7d5a0"
)
SET_SPECS = [
    (
        "Diversity compounds\n(759 RefChemDB targets)",
        ["source_2", "source_6", "source_8"],
    ),
    ("Bioactive compounds\n(981 RefChemDB targets)", ["source_7"]),
    ("CRISPR\n(7,975 target symbols)", ["source_13"]),
    ("ORF\n(12,598 target symbols)", ["source_4"]),
]
EXPECTED_SET_SIZES = [759, 981, 7_975, 12_598]
EXPECTED_PATTERN_COUNTS = {
    (0, 1, 2, 3): 533,
    (0, 1, 2): 201,
    (0, 1, 3): 14,
    (0, 2, 3): 8,
    (0, 2): 3,
    (1, 2, 3): 135,
    (1, 2): 75,
    (1, 3): 23,
    (2, 3): 4_569,
    (2,): 2_451,
    (3,): 7_316,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_sets(
    metadata_path: Path, refchem_path: Path
) -> tuple[list[set[str]], Counter[tuple[int, ...]]]:
    for path, expected in (
        (metadata_path, EXPECTED_METADATA_SHA256),
        (refchem_path, EXPECTED_REFCHEM_SHA256),
    ):
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"Input drift for {path}: {observed} != {expected}")

    metadata = pl.read_parquet(metadata_path)
    if metadata.height != 163_776:
        raise RuntimeError(f"Expected 163,776 metadata rows, found {metadata.height:,}")

    refchem = (
        pl.read_parquet(refchem_path)
        .filter(pl.col("WithinModalityTier").is_in(["Tier1", "Tier2", "Tier3"]))
        .filter(pl.col("target").is_not_null())
        .group_by("Metadata_JCP2022")
        .agg(pl.col("target").unique().sort().str.join("|").alias("RefChem_target"))
    )
    metadata = metadata.join(refchem, on="Metadata_JCP2022", how="left").with_columns(
        pl.when(pl.col("Metadata_pert_type") == "negcon")
        .then(pl.lit("unknown"))
        .when(pl.col("Metadata_Perturbation_Type").is_in(["orf", "crispr"]))
        .then(pl.col("Metadata_Symbol"))
        .when(pl.col("RefChem_target").is_not_null())
        .then(pl.col("RefChem_target"))
        .otherwise(pl.lit("unknown"))
        .alias("target_symbol")
    )

    sets: list[set[str]] = []
    for _, sources in SET_SPECS:
        values = metadata.filter(
            pl.col("Metadata_Source").is_in(sources)
            & pl.col("target_symbol").is_not_null()
            & (pl.col("target_symbol") != "unknown")
        )["target_symbol"].to_list()
        sets.append(
            {
                symbol.strip()
                for value in values
                for symbol in value.split("|")
                if symbol.strip().lower() not in {"", "unknown", "nan", "none", "null"}
            }
        )

    observed_sizes = [len(items) for items in sets]
    if observed_sizes != EXPECTED_SET_SIZES:
        raise RuntimeError(
            f"Set-size mismatch: {observed_sizes} != {EXPECTED_SET_SIZES}"
        )

    patterns = Counter(
        tuple(index for index, items in enumerate(sets) if symbol in items)
        for symbol in set.union(*sets)
    )
    if patterns != Counter(EXPECTED_PATTERN_COUNTS):
        raise RuntimeError(
            f"Intersection mismatch: {patterns} != {EXPECTED_PATTERN_COUNTS}"
        )
    return sets, patterns


def write_summaries(
    sets: list[set[str]], patterns: Counter[tuple[int, ...]], outdir: Path
) -> None:
    with (outdir / "set_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["set_index", "label", "count"])
        for index, ((label, _), items) in enumerate(zip(SET_SPECS, sets, strict=True)):
            writer.writerow([index, label.replace("\n", " "), len(items)])

    with (outdir / "intersection_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["diversity", "bioactive", "crispr", "orf", "count"])
        for membership, count in sorted(patterns.items()):
            writer.writerow([*(int(i in membership) for i in range(4)), count])


def render(sets: list[set[str]], outdir: Path) -> None:
    labels = [label for label, _ in SET_SPECS]
    colors = ["#d95f5f", "#71bc73", "#f2a154", "#6fa6ce"]
    fig = plt.figure(figsize=(16, 7.6))
    result = supervenn(
        sets,
        labels,
        side_plots=False,
        chunks_ordering="size",
        sets_ordering=None,
        widths_minmax_ratio=0.05,
        min_width_for_annotation=0,
        rotate_col_annotations=True,
        col_annotations_area_height=1.25,
        color_cycle=colors,
    )
    axis = result.axes["main"]
    axis.set_xlabel("Unique target symbols in each membership pattern", fontsize=13)
    axis.set_ylabel("JUMP-lite perturbation group", fontsize=13)
    axis.tick_params(axis="both", labelsize=11)
    plt.title(
        "JUMP-lite target overlap across perturbation groups "
        "($\\geq$4 wells for compounds)",
        fontsize=17,
        pad=12,
    )
    fig.savefig(
        outdir / "target_all_sources_filtered_supervenn.png",
        dpi=180,
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "target_all_sources_filtered_supervenn.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "paper_artifacts/target_overlap/render.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--refchem", type=Path, default=DEFAULT_REFCHEM)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata_path = args.metadata.resolve(strict=True)
    refchem_path = args.refchem.resolve(strict=True)
    outdir = args.output_dir.resolve()
    if outdir.exists():
        raise RuntimeError(f"Output directory already exists: {outdir}")
    outdir.mkdir(parents=True)

    sets, patterns = build_sets(metadata_path, refchem_path)
    write_summaries(sets, patterns, outdir)
    render(sets, outdir)
    provenance = {
        "metadata": {
            "path": provenance_path(metadata_path),
            "sha256": sha256(metadata_path),
            "rows": 163_776,
        },
        "refchemdb": {
            "path": provenance_path(refchem_path),
            "sha256": sha256(refchem_path),
        },
        "set_sizes": EXPECTED_SET_SIZES,
        "union_size": len(set.union(*sets)),
        "png_sha256": sha256(outdir / "target_all_sources_filtered_supervenn.png"),
        "pdf_sha256": sha256(outdir / "target_all_sources_filtered_supervenn.pdf"),
        "method": (
            "RefChemDB WithinModalityTier Tier1-3 targets for compounds; "
            "Metadata_Symbol for CRISPR/ORF; negative controls and missing-value "
            "placeholders excluded; positive controls retained"
        ),
    }
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
