#!/usr/bin/env python3
"""Fail-closed preflight validation for the JUMP-Lite CPG release.

Run this immediately before every CPG upload. A nonzero exit means no upload
should occur. Validation is read-only unless ``--json-output`` is specified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb

DEFAULT_IMAGE_ROOT = Path(
    "/work/datasets/jump_lite/images/compressed/compressed_test/jump_lite_updated"
)
DEFAULT_PROFILE_ROOT = Path(
    "/work/datasets/jump_lite/aliby_output/jump_lite_rerun/jump_lite_updated"
)
DEFAULT_METADATA_ROOT = Path("/work/datasets/jump_lite/cpg_release/metadata")
DEFAULT_ZSTD_ROOT = Path("/work/datasets/jump_lite/zstd_rebuild/v1.0/zstd.zarr")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SITE_MANIFEST = REPOSITORY_ROOT / "metadata/jump_lite_v1_site_manifest.parquet"
DEFAULT_WELL_MANIFEST = REPOSITORY_ROOT / "metadata/jump_lite_v1_well_manifest.parquet"
CANONICAL_CODEC = "jpegxl_lossy_mq.zarr"
IMAGE_CODECS = (
    "jpegxl_lossy_mq.zarr",
    "jpegxl_lossy_hq.zarr",
    "jpegxl_lossy_d20.zarr",
)
EXPECTED_SITE_COUNT = 655_101
EXPECTED_WELL_COUNT = 163_776
EXPECTED_BENCHMARK_PERTURBATION_COUNT = 24_356
EXPECTED_PLATE_COUNT = 551
EXPECTED_SOURCE_COUNT = 6
EXPECTED_SITE_DIGEST = "4ea6ea3f5457c33a1412a80a89d8696d4f8e77474cf449e75db7ce6ba98685e2"
EXPECTED_PROFILE_VARIANTS = {
    "dinov2": set(IMAGE_CODECS),
    "dinov2_random": set(IMAGE_CODECS),
    "morphem": set(IMAGE_CODECS),
    "openphenom_confusing": set(IMAGE_CODECS),
    "subcell": {CANONICAL_CODEC},
    "subcell__clip01": set(IMAGE_CODECS),
}
METADATA_FILES = {
    "site_index": "jump_lite_site_index.parquet",
    "image_index": "jump_lite_image_index.parquet",
    "perturbation_metadata": "jump_lite_perturbation_metadata.parquet",
    "plate_manifest": "jump_lite_plate_manifest.parquet",
    "annotations": "jump_lite_refchem_annotations.parquet",
    "manifest": "metadata_manifest.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--zstd-root", type=Path, default=DEFAULT_ZSTD_ROOT)
    parser.add_argument("--site-manifest", type=Path, default=DEFAULT_SITE_MANIFEST)
    parser.add_argument("--well-manifest", type=Path, default=DEFAULT_WELL_MANIFEST)
    parser.add_argument(
        "--require-zstd",
        action="store_true",
        help="fail unless the finalized lossless v1.0 Zstd store is present",
    )
    parser.add_argument("--expected-sites", type=int, default=EXPECTED_SITE_COUNT)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optionally write the complete validation report as JSON",
    )
    return parser.parse_args()


def directory_keys(path: Path) -> set[str]:
    return {
        entry.name
        for entry in os.scandir(path)
        if entry.is_dir(follow_symlinks=False)
    }


def parquet_keys(path: Path) -> set[str]:
    return {
        entry.name.removesuffix(".parquet")
        for entry in os.scandir(path)
        if entry.is_file(follow_symlinks=False)
        and entry.name.endswith(".parquet")
    }


def key_digest(keys: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def zarr_v3_array_complete(path: Path) -> bool:
    metadata_path = path / "zarr.json"
    chunk_root = path / "c"
    if not metadata_path.is_file() or not chunk_root.is_dir():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return metadata.get("node_type") == "array" and any(
        entry.is_file() for entry in chunk_root.rglob("*")
    )


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def actual_profile_variants(profile_root: Path) -> dict[str, set[str]]:
    variants: dict[str, set[str]] = {}
    for model in profile_root.iterdir():
        if not model.is_dir():
            continue
        codecs = {
            codec.name
            for codec in model.iterdir()
            if codec.is_dir() and (codec / "profiles").is_dir()
        }
        if codecs:
            variants[model.name] = codecs
    return variants


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, object] = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "image_root": str(args.image_root),
        "zstd_root": str(args.zstd_root),
        "profile_root": str(args.profile_root),
        "metadata_root": str(args.metadata_root),
    }

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    canonical_path = args.image_root / CANONICAL_CODEC
    require(canonical_path.is_dir(), f"Missing canonical image dataset: {canonical_path}")
    require(
        args.site_manifest.is_file(),
        f"Missing frozen release site manifest: {args.site_manifest}",
    )
    require(
        args.well_manifest.is_file(),
        f"Missing frozen release well manifest: {args.well_manifest}",
    )
    if args.site_manifest.is_file():
        con = duckdb.connect()
        site_manifest = sql_path(args.site_manifest)
        manifest_rows, manifest_distinct, coordinate_mismatches = con.execute(
            f"""
            SELECT
              count(*),
              count(DISTINCT Metadata_Site_Key),
              count(*) FILTER (
                WHERE Metadata_Site_Key != concat_ws(
                  '__', Metadata_Source, Metadata_Batch, Metadata_Plate,
                  Metadata_Well, cast(Metadata_Site AS VARCHAR)
                )
              )
            FROM read_parquet('{site_manifest}')
            """
        ).fetchone()
        canonical_keys = {
            str(row[0])
            for row in con.execute(
                f"SELECT Metadata_Site_Key FROM read_parquet('{site_manifest}')"
            ).fetchall()
        }
        require(
            manifest_rows == manifest_distinct == len(canonical_keys),
            "Frozen site manifest contains duplicate keys",
        )
        require(
            coordinate_mismatches == 0,
            "Frozen site-manifest keys disagree with coordinate columns",
        )
        if args.well_manifest.is_file():
            well_manifest = sql_path(args.well_manifest)
            well_rows, distinct_wells = con.execute(
                f"""
                SELECT count(*), count(DISTINCT (
                  Metadata_Source, Metadata_Batch, Metadata_Plate, Metadata_Well
                ))
                FROM read_parquet('{well_manifest}')
                """
            ).fetchone()
            missing_wells, extra_wells = con.execute(
                f"""
                WITH site_wells AS (
                  SELECT DISTINCT Metadata_Source, Metadata_Batch,
                                  Metadata_Plate, Metadata_Well
                  FROM read_parquet('{site_manifest}')
                ), frozen_wells AS (
                  SELECT Metadata_Source, Metadata_Batch,
                         Metadata_Plate, Metadata_Well
                  FROM read_parquet('{well_manifest}')
                )
                SELECT
                  (SELECT count(*) FROM (
                    SELECT * FROM site_wells EXCEPT SELECT * FROM frozen_wells
                  )),
                  (SELECT count(*) FROM (
                    SELECT * FROM frozen_wells EXCEPT SELECT * FROM site_wells
                  ))
                """
            ).fetchone()
            require(
                well_rows == distinct_wells,
                "Frozen well manifest contains duplicate identities",
            )
            require(
                not missing_wells and not extra_wells,
                "Frozen well manifest does not exactly match site-manifest wells",
            )
        con.close()
    else:
        canonical_keys = set()
    source_canonical_keys = directory_keys(canonical_path) if canonical_path.is_dir() else set()
    missing_source_sites = canonical_keys - source_canonical_keys
    extra_source_sites = source_canonical_keys - canonical_keys
    require(not missing_source_sites, f"MQ source is missing {len(missing_source_sites):,} release sites")
    if extra_source_sites:
        warnings.append(f"MQ source contains {len(extra_source_sites):,} non-release sites; uploaders must filter by the frozen manifest")

    malformed_keys = [key for key in canonical_keys if len(key.split("__")) != 5]
    require(not malformed_keys, f"Malformed canonical site keys: {malformed_keys[:5]}")
    require(
        len(canonical_keys) == args.expected_sites,
        f"Canonical site count is {len(canonical_keys):,}; expected {args.expected_sites:,}",
    )
    canonical_digest = key_digest(canonical_keys)
    require(canonical_digest == EXPECTED_SITE_DIGEST, "Frozen site-manifest digest differs from v1.0")

    image_report: dict[str, object] = {}
    for codec in IMAGE_CODECS:
        path = args.image_root / codec
        require(path.is_dir(), f"Missing image dataset: {path}")
        if not path.is_dir():
            continue
        keys = directory_keys(path)
        missing = canonical_keys - keys
        extra = keys - canonical_keys
        require(not missing, f"Image source is missing release keys for {codec}: missing={len(missing):,}")
        if extra:
            warnings.append(f"{codec} contains {len(extra):,} non-release source sites")
        require((path / ".zgroup").is_file(), f"Missing Zarr v2 .zgroup: {path}")
        image_report[codec] = {
            "site_count": len(canonical_keys),
            "source_site_count": len(keys),
            "site_key_sha256": canonical_digest,
            "missing_canonical_sites": len(missing),
            "extra_sites": len(extra),
        }

    if args.require_zstd or args.zstd_root.exists():
        require(args.zstd_root.is_dir(), f"Missing finalized Zstd dataset: {args.zstd_root}")
        if args.zstd_root.is_dir():
            zstd_keys = directory_keys(args.zstd_root)
            missing = canonical_keys - zstd_keys
            extra = zstd_keys - canonical_keys
            require(not missing, f"Zstd source is missing {len(missing):,} release sites")
            if extra:
                warnings.append(f"Zstd source contains {len(extra):,} non-release source sites")
            require(
                (args.zstd_root / "zarr.json").is_file(),
                f"Missing Zarr v3 group metadata: {args.zstd_root}",
            )
            incomplete = [
                key for key in canonical_keys if not zarr_v3_array_complete(args.zstd_root / key)
            ]
            require(
                not incomplete,
                f"Zstd contains incomplete arrays: {incomplete[:10]}",
            )
            image_report["zstd.zarr"] = {
                "zarr_format": 3,
                "site_count": len(canonical_keys),
                "source_site_count": len(zstd_keys),
                "site_key_sha256": canonical_digest,
                "missing_canonical_sites": len(missing),
                "extra_sites": len(extra),
                "incomplete_sites": len(incomplete),
            }
    report["images"] = image_report

    discovered = actual_profile_variants(args.profile_root)
    require(
        discovered == EXPECTED_PROFILE_VARIANTS,
        "Profile variant inventory differs from the frozen release definition: "
        f"expected={EXPECTED_PROFILE_VARIANTS}, actual={discovered}",
    )

    profile_report: dict[str, object] = {}
    for model, codecs in sorted(EXPECTED_PROFILE_VARIANTS.items()):
        for codec in sorted(codecs):
            profiles = args.profile_root / model / codec / "profiles"
            require(profiles.is_dir(), f"Missing per-site Parquet directory: {profiles}")
            if not profiles.is_dir():
                continue
            keys = parquet_keys(profiles)
            missing = canonical_keys - keys
            extra = keys - canonical_keys
            require(not missing, f"Per-site Parquets are missing release keys for {model}/{codec}: missing={len(missing):,}")
            if extra:
                warnings.append(f"{model}/{codec} contains {len(extra):,} non-release source Parquets")
            profile_report[f"{model}/{codec}"] = {
                "file_count": len(canonical_keys),
                "source_file_count": len(keys),
                "site_key_sha256": canonical_digest,
                "missing_canonical_sites": len(missing),
                "extra_sites": len(extra),
            }
    report["per_site_parquets"] = profile_report

    paths = {name: args.metadata_root / filename for name, filename in METADATA_FILES.items()}
    dataset_readme = args.metadata_root.parent / "README.md"
    require(dataset_readme.is_file(), f"Missing uploadable dataset README: {dataset_readme}")
    for name, path in paths.items():
        require(path.is_file(), f"Missing release metadata {name}: {path}")

    metadata_report: dict[str, object] = {}
    parquet_paths = [
        paths[name]
        for name in (
            "site_index",
            "image_index",
            "perturbation_metadata",
            "plate_manifest",
            "annotations",
        )
    ]
    if all(path.is_file() for path in parquet_paths):
        con = duckdb.connect()
        site_index = sql_path(paths["site_index"])
        image_index = sql_path(paths["image_index"])
        perturbation = sql_path(paths["perturbation_metadata"])
        plate_manifest = sql_path(paths["plate_manifest"])
        annotations = sql_path(paths["annotations"])

        index_rows, index_distinct, index_coordinate_mismatches = con.execute(
            f"""
            SELECT
              count(*),
              count(DISTINCT Metadata_Site_Key),
              count(*) FILTER (
                WHERE Metadata_Site_Key != concat_ws(
                  '__', Metadata_Source, Metadata_Batch, Metadata_Plate,
                  Metadata_Well, cast(Metadata_Site AS VARCHAR)
                )
              )
            FROM read_parquet('{site_index}')
            """
        ).fetchone()
        index_keys = {
            row[0]
            for row in con.execute(
                f"SELECT Metadata_Site_Key FROM read_parquet('{site_index}')"
            ).fetchall()
        }
        require(
            index_rows == len(canonical_keys)
            and index_distinct == len(canonical_keys)
            and index_keys == canonical_keys,
            "Frozen site index does not exactly match the canonical image keys",
        )
        require(
            index_coordinate_mismatches == 0,
            "Frozen site-index keys disagree with coordinate columns",
        )
        null_urls = con.execute(
            f"""
            SELECT count(*) FROM read_parquet('{site_index}')
            WHERE URL_OrigAGP IS NULL OR URL_OrigDNA IS NULL OR URL_OrigER IS NULL
               OR URL_OrigMito IS NULL OR URL_OrigRNA IS NULL
            """
        ).fetchone()[0]
        require(null_urls == 0, f"Site index has {null_urls:,} rows with missing image URLs")

        image_rows, image_sites, image_channels, null_uris = con.execute(
            f"""
            SELECT count(*), count(DISTINCT Metadata_Site_Key),
                   count(DISTINCT Metadata_Channel),
                   count(*) FILTER (WHERE uri IS NULL)
            FROM read_parquet('{image_index}')
            """
        ).fetchone()
        require(
            image_rows == 5 * len(canonical_keys)
            and image_sites == len(canonical_keys)
            and image_channels == 5
            and null_uris == 0,
            "Tidy image index is not a complete five-channel expansion of the site index",
        )

        well_rows, distinct_wells, benchmark_perturbations = con.execute(
            f"""
            SELECT
              count(*),
              count(DISTINCT Metadata_Well_Key),
              count(DISTINCT Metadata_JCP2022) FILTER (
                WHERE Metadata_Perturbation_Type = 'compound'
              ) +
              count(DISTINCT Metadata_Symbol) FILTER (
                WHERE Metadata_Perturbation_Type = 'crispr'
              ) +
              count(DISTINCT Metadata_Symbol) FILTER (
                WHERE Metadata_Perturbation_Type = 'orf'
              )
            FROM read_parquet('{perturbation}')
            """
        ).fetchone()
        require(
            well_rows == distinct_wells,
            "Perturbation metadata contains duplicate source/batch/plate/well keys",
        )
        metadata_key_mismatches = con.execute(
            f"""
            SELECT count(*) FROM read_parquet('{perturbation}')
            WHERE Metadata_Well_Key != concat_ws(
              '__', Metadata_Source, Metadata_Batch, Metadata_Plate, Metadata_Well
            )
            """
        ).fetchone()[0]
        missing_metadata_wells, extra_metadata_wells = con.execute(
            f"""
            WITH site_wells AS (
              SELECT DISTINCT Metadata_Source, Metadata_Batch,
                              Metadata_Plate, Metadata_Well
              FROM read_parquet('{site_index}')
            ), metadata_wells AS (
              SELECT Metadata_Source, Metadata_Batch,
                     Metadata_Plate, Metadata_Well
              FROM read_parquet('{perturbation}')
            )
            SELECT
              (SELECT count(*) FROM (
                SELECT * FROM site_wells EXCEPT SELECT * FROM metadata_wells
              )),
              (SELECT count(*) FROM (
                SELECT * FROM metadata_wells EXCEPT SELECT * FROM site_wells
              ))
            """
        ).fetchone()
        require(
            metadata_key_mismatches == 0,
            "Perturbation Metadata_Well_Key values disagree with coordinates",
        )
        require(
            not missing_metadata_wells and not extra_metadata_wells,
            "Perturbation metadata does not exactly cover frozen site-index wells",
        )
        require(
            benchmark_perturbations == EXPECTED_BENCHMARK_PERTURBATION_COUNT,
            "Benchmark perturbation count does not match the frozen release",
        )

        plate_rows, summed_sites = con.execute(
            f"SELECT count(*), sum(site_count) FROM read_parquet('{plate_manifest}')"
        ).fetchone()
        require(
            plate_rows == EXPECTED_PLATE_COUNT and summed_sites == len(canonical_keys),
            "Plate manifest totals do not match the frozen site index",
        )

        source_count, well_count = con.execute(
            f"""
            SELECT count(DISTINCT Metadata_Source),
                   count(DISTINCT concat_ws('__', Metadata_Source, Metadata_Batch,
                                            Metadata_Plate, Metadata_Well))
            FROM read_parquet('{site_index}')
            """
        ).fetchone()
        require(source_count == EXPECTED_SOURCE_COUNT, "Unexpected source count")
        require(well_count == EXPECTED_WELL_COUNT, "Unexpected well count")

        annotation_rows, annotation_ids, out_of_subset_ids = con.execute(
            f"""
            SELECT
              count(*),
              count(DISTINCT Metadata_JCP2022),
              count(DISTINCT Metadata_JCP2022) FILTER (
                WHERE Metadata_JCP2022 NOT IN (
                  SELECT Metadata_JCP2022 FROM read_parquet('{perturbation}')
                )
              )
            FROM read_parquet('{annotations}')
            """
        ).fetchone()
        require(
            out_of_subset_ids == 0,
            f"Annotations contain {out_of_subset_ids:,} perturbations absent from the release",
        )

        metadata_report = {
            "site_rows": index_rows,
            "image_rows": image_rows,
            "perturbation_metadata_rows": well_rows,
            "benchmark_perturbations": benchmark_perturbations,
            "annotated_wells": distinct_wells,
            "annotation_rows": annotation_rows,
            "annotated_perturbations": annotation_ids,
            "plate_count": plate_rows,
            "well_count": well_count,
            "source_count": source_count,
            "site_key_sha256": key_digest(index_keys),
        }

    if paths["manifest"].is_file():
        manifest = json.loads(paths["manifest"].read_text())
        require(
            manifest.get("site_key_sha256") == canonical_digest,
            "Metadata manifest site-key digest differs from canonical images",
        )
        require(
            manifest.get("site_count") == len(canonical_keys),
            "Metadata manifest site count differs from canonical images",
        )
    report["metadata"] = metadata_report

    if warnings:
        report["warnings"] = warnings
    report["errors"] = errors
    report["status"] = "ready" if not errors else "not_ready"
    report["canonical_site_count"] = len(canonical_keys)
    report["canonical_site_key_sha256"] = canonical_digest

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"CPG release status: {report['status']}")
    print(f"Canonical sites: {len(canonical_keys):,}; SHA-256: {canonical_digest}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Image datasets: {len(image_report)}")
    print(f"Per-site Parquet variants: {len(profile_report)}")
    print(
        f"Metadata: {metadata_report.get('well_count', 0):,} wells, "
        f"{metadata_report.get('plate_count', 0):,} plates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
