#!/usr/bin/env python3
"""Export the compact SI cluster-control package from canonical results only.

This utility copies and summarizes committed current-release artifacts. It never
fits clusters, rescoring models, recomputes scientific metrics, or refits UMAP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent.parent
CANONICAL_ROOT = HERE / "outputs/profile_cluster_representativeness_release_v1"
FIT_ROOT = HERE / "outputs/profile_cluster_representativeness_v1/fit"
EXPORTER_VERSION = "cluster-selection-control-export-v1"
CANONICAL_SOURCE_COMMIT = "d8d81b213b1219bd5fa7fb6aa501716cff5f1c16"
CENTRAL_CLAIM = (
    "Broad but non-uniform coverage of operational CellProfiler "
    "compound-profile space in JUMP."
)

# Exact identities of the reviewed canonical artifacts consumed by this export.
SOURCE_SPECS: dict[str, tuple[Path, int, str]] = {
    "cluster_selection_compound_map.pdf": (
        CANONICAL_ROOT / "cluster_selection_compound_map.pdf", 956_319,
        "59c639a30940bcce5fbbd94669049998ba9f3181164888c839748b35c8b14b3c",
    ),
    "cluster_selection_compound_map.png": (
        CANONICAL_ROOT / "cluster_selection_compound_map.png", 1_525_856,
        "a436610f578c770913c5e7e14ea25d504ebfc264c3e94bfc278ea1bf4c0103ea",
    ),
    "cluster_selection_summary_table.csv": (
        CANONICAL_ROOT / "cluster_selection_summary_table.csv", 1_496,
        "b7312089803aebecb273f799f4f0d2d96151f1fd69fed2b79341de21d44ec84a",
    ),
    "cluster_selection_table.csv": (
        CANONICAL_ROOT / "cluster_selection_table.csv", 38_582,
        "1a8dcae43d2e0f1ed6da976a7b764978733a2d2fb2c27aa92523858f910e0e4c",
    ),
    "retrieval_metrics.csv": (
        CANONICAL_ROOT / "retrieval_metrics.csv", 1_784,
        "5184ceec065a2bba05cda0da52cc41126f7c6353daa61ecb33f24dc3be8a6959",
    ),
    "current_release_treatment_manifest.parquet": (
        CANONICAL_ROOT / "current_release_treatment_manifest.parquet", 32_941,
        "3d3ca04b51a47d23b9c8f20b62fdc24295cd0c8e44b678b2a123c353fd5ee56b",
    ),
    "clustering_diagnostics.csv": (
        FIT_ROOT / "clustering_diagnostics.csv", 4_708,
        "9df79ab9f92e6c3c1ca0b268e6544a9d59647a5c1e7bb345f86ce0b07e628709",
    ),
}

PACKAGE_FILES = {
    "README.md",
    "RESULTS_PROVENANCE.json",
    "SHA256SUMS",
    "cluster_selection_compound_map.pdf",
    "cluster_selection_compound_map.png",
    "cluster_selection_summary_table.csv",
    "cluster_selection_table.csv",
    "clustering_diagnostics.csv",
    "release_selected_compounds.csv",
    "retrieval_metrics.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def repository_record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    root = REPOSITORY_ROOT.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Source escapes repository root: {resolved}") from error
    if relative == Path(".") or ".." in relative.parts:
        raise RuntimeError(f"Unsafe repository record: {relative}")
    return {
        "path": relative.as_posix(),
        "path_scope": "repository_root",
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def package_record(path: Path, output_dir: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    root = output_dir.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Package file escapes output root: {resolved}") from error
    if relative == Path(".") or relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe package record: {relative}")
    return {
        "path": relative.as_posix(),
        "path_scope": "package_root",
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_sources(specs: dict[str, tuple[Path, int, str]] = SOURCE_SPECS) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for name, (path, expected_bytes, expected_sha256) in sorted(specs.items()):
        record = repository_record(path)
        if (record["bytes"], record["sha256"]) != (expected_bytes, expected_sha256):
            raise RuntimeError(f"Canonical source drift for {name}: {record}")
        records[name] = record
    return records


def validate_scientific_contract() -> dict[str, object]:
    summary = pd.read_csv(CANONICAL_ROOT / "cluster_selection_summary_table.csv").set_index("metric")["value"]
    expected = {
        "total_compounds": 115_721,
        "fit_eligible_compounds": 95_426,
        "selected_compounds": 3_775,
        "occupied_clusters": 120,
        "total_clusters": 128,
        "eligible_mass_in_occupied_clusters": 0.962368746463228,
        "total_variation_selected_vs_eligible": 0.474989347870955,
        "average_precision_acquisition_structure": 0.513536652643861,
        "average_precision_cluster_only": 0.273104769075815,
        "average_precision_structure_plus_cluster": 0.573313325285965,
        "combined_to_structure_ap_ratio": 1.11640195949861,
        "conditional_permutation_p": 0.000499750124937531,
        "mean_seed_adjusted_rand_index": 0.162414221608182,
    }
    missing = sorted(set(expected) - set(summary.index))
    if missing:
        raise RuntimeError(f"Canonical summary metrics absent: {missing}")
    for metric, value in expected.items():
        actual = float(summary[metric])
        tolerance = 0 if isinstance(value, int) else 1e-14
        if abs(actual - value) > tolerance:
            raise RuntimeError(f"Scientific contract drift for {metric}: {actual} != {value}")
    table = pd.read_csv(CANONICAL_ROOT / "cluster_selection_table.csv")
    if len(table) != 128 or int(table["n_selected"].sum()) != 3_775:
        raise RuntimeError("Canonical cluster table count drift")
    if int((table["n_selected"] > 0).sum()) != 120:
        raise RuntimeError("Canonical occupied-cluster count drift")
    return {key: float(summary[key]) for key in expected}


def readme_text() -> str:
    return f"""# Broader-JUMP compound-profile coverage control

This compact Supplementary Information package is exported from the reviewed,
committed current-release results. The complete scientific fit and rescore
pipeline lives in `rebuttal/representativeness/`:
`analyze_cluster_representativeness.py` fits the label-blind historical
partition, and `rescore_cluster_representativeness_release_v1.py` derives and
scores the current 3,775-treatment release cohort without refitting.
`export_cluster_selection_control.py` only verifies and copies canonical
artifacts; it does not fit, rescore, or recompute scientific metrics.

The tracked release contains **3,776 compound identities**: **3,775 treatment
compounds** plus one excluded shared DMSO negative-control identity. The selected
treatments occupy **120/128 operational clusters**, whose eligible compounds
comprise **96.24%** of fit-eligible compound mass. Coverage is non-uniform
(TV **0.4750**). Eligible out-of-fold average precision is **0.5135** for
acquisition structure, **0.2731** for cluster alone, and **0.5733** for structure
plus cluster; the combined/structure ratio is **1.116**, below the predeclared
**1.25x** materiality gate. The 2,000-permutation finite-cohort conditional test
has `p=0.000500`, and mean seed ARI is **0.162**.

**Bounded interpretation:** {CENTRAL_CLAIM} This does not establish proportional
or random sampling, stable biological classes, coverage of every phenotype,
representativeness under other feature models, or genetic-perturbation
representativeness. The UMAP is visualization only.

## Files

- `cluster_selection_compound_map.pdf` / `.png`: corrected SI figure.
- `cluster_selection_summary_table.csv`: compact reported metrics.
- `cluster_selection_table.csv`: all 128 operational clusters.
- `retrieval_metrics.csv`: eligible and all-compound OOF metrics.
- `clustering_diagnostics.csv`: frozen partition diagnostics.
- `release_selected_compounds.csv`: sorted current 3,775 treatment identifiers.
- `RESULTS_PROVENANCE.json`: source and package records.
- `SHA256SUMS`: checksums for every other package file.

Large consensus, assignment, UMAP-coordinate, and model files are intentionally
excluded from this compact export and remain in the full scientific package.
"""


def export_package(output_dir: Path, *, source_specs: dict[str, tuple[Path, int, str]] = SOURCE_SPECS) -> None:
    output_dir = output_dir.resolve(strict=False)
    if output_dir.exists():
        raise RuntimeError(f"Output directory must be absent: {output_dir}")
    source_records = verify_sources(source_specs)
    scientific = validate_scientific_contract()

    output_dir.mkdir(parents=True)
    try:
        for name in (
            "cluster_selection_compound_map.pdf",
            "cluster_selection_compound_map.png",
            "cluster_selection_summary_table.csv",
            "cluster_selection_table.csv",
            "retrieval_metrics.csv",
            "clustering_diagnostics.csv",
        ):
            shutil.copyfile(source_specs[name][0], output_dir / name)

        manifest = (
            pl.read_parquet(source_specs["current_release_treatment_manifest.parquet"][0])
            .select("Metadata_JCP2022")
            .sort("Metadata_JCP2022")
        )
        if manifest.height != 3_775 or manifest["Metadata_JCP2022"].n_unique() != 3_775:
            raise RuntimeError("Canonical current-release manifest count/uniqueness drift")
        manifest.write_csv(output_dir / "release_selected_compounds.csv")
        (output_dir / "README.md").write_text(readme_text())

        preliminary = [
            package_record(output_dir / name, output_dir)
            for name in sorted(PACKAGE_FILES - {"RESULTS_PROVENANCE.json", "SHA256SUMS"})
        ]
        provenance = {
            "version": EXPORTER_VERSION,
            "canonical_source_commit": CANONICAL_SOURCE_COMMIT,
            "claim": CENTRAL_CLAIM,
            "excluded_claims": [
                "proportional or random sampling",
                "stable biological classes",
                "coverage of every phenotype",
                "representativeness under other feature models",
                "genetic-perturbation representativeness",
            ],
            "pipeline": {
                "fit": "rebuttal/representativeness/analyze_cluster_representativeness.py",
                "current_release_rescore": "rebuttal/representativeness/rescore_cluster_representativeness_release_v1.py",
                "export": "rebuttal/representativeness/export_cluster_selection_control.py",
                "export_only_no_fit_or_rescore": True,
            },
            "exporter": repository_record(Path(__file__)),
            "sources": source_records,
            "scientific_contract": scientific,
            "release_compound_identities": 3_776,
            "selected_treatments": 3_775,
            "package_files_excluding_provenance_and_checksums": preliminary,
            "path_base_definition": {
                "repository_root": "checkout root containing pyproject.toml",
                "package_root": "compact export output directory",
            },
        }
        (output_dir / "RESULTS_PROVENANCE.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        checksummed = sorted(PACKAGE_FILES - {"SHA256SUMS"})
        checksum_text = "".join(
            f"{sha256_file(output_dir / name)}  {name}\n" for name in checksummed
        )
        (output_dir / "SHA256SUMS").write_text(checksum_text)
        actual = {path.name for path in output_dir.iterdir() if path.is_file()}
        if actual != PACKAGE_FILES or any(path.is_dir() for path in output_dir.iterdir()):
            raise RuntimeError(f"Export inventory drift: {sorted(actual)}")
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_package(args.output_dir)
    print(f"Exported {len(PACKAGE_FILES)} files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
