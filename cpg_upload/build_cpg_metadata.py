#!/usr/bin/env python3
"""Build frozen CPG metadata tables for the JUMP-Lite paper cohort.

This does not resample sites. It validates the committed v1.0 site manifest
against the MQ source Zarr and joins those exact keys to the full JUMP index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import polars as pl

DEFAULT_CANONICAL_ZARR = Path(
    "/work/datasets/jump_lite/images/compressed/compressed_test/"
    "jump_lite_updated/jpegxl_lossy_mq.zarr"
)
DEFAULT_FULL_INDEX = Path("/work/datasets/jump_lite/misc/jl_index.parquet")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SITE_MANIFEST = REPOSITORY_ROOT / "metadata/jump_lite_v1_site_manifest.parquet"
DEFAULT_WELL_MANIFEST = REPOSITORY_ROOT / "metadata/jump_lite_v1_well_manifest.parquet"
DEFAULT_PERTURBATION_METADATA = REPOSITORY_ROOT / "metadata/jump_lite_v1_perturbation_metadata.parquet"
DEFAULT_ANNOTATIONS = REPOSITORY_ROOT / "metadata/jump_lite_v1_refchem_annotations.parquet"
DEFAULT_OUTPUT_DIR = Path("/work/datasets/jump_lite/cpg_release/metadata")
EXPECTED_SITE_COUNT = 655_101
CHANNELS = ("AGP", "DNA", "ER", "Mito", "RNA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-zarr", type=Path, default=DEFAULT_CANONICAL_ZARR)
    parser.add_argument("--site-manifest", type=Path, default=DEFAULT_SITE_MANIFEST)
    parser.add_argument("--well-manifest", type=Path, default=DEFAULT_WELL_MANIFEST)
    parser.add_argument("--full-index", type=Path, default=DEFAULT_FULL_INDEX)
    parser.add_argument(
        "--perturbation-metadata", type=Path, default=DEFAULT_PERTURBATION_METADATA
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-sites", type=int, default=EXPECTED_SITE_COUNT)
    return parser.parse_args()


def canonical_key_table(
    zarr_path: Path,
    site_manifest: Path,
    well_manifest: Path,
) -> tuple[pl.DataFrame, list[str]]:
    identity_columns = [
        "Metadata_Source",
        "Metadata_Batch",
        "Metadata_Plate",
        "Metadata_Well",
    ]
    table = pl.read_parquet(site_manifest).select(
        "Metadata_Site_Key",
        *identity_columns,
        "Metadata_Site",
    ).sort("Metadata_Site_Key")
    keys = table["Metadata_Site_Key"].to_list()
    if len(keys) != len(set(keys)):
        raise RuntimeError("frozen site manifest contains duplicate keys")
    expected_keys = pl.concat_str(
        *[pl.col(column) for column in identity_columns],
        pl.col("Metadata_Site").cast(pl.String),
        separator="__",
    )
    malformed = table.filter(pl.col("Metadata_Site_Key") != expected_keys)
    if not malformed.is_empty():
        raise RuntimeError(
            "frozen site-manifest keys disagree with coordinate columns: "
            f"{malformed.head(5).to_dicts()}"
        )

    wells = pl.read_parquet(well_manifest).select(identity_columns)
    if wells.height != wells.unique().height:
        raise RuntimeError("frozen well manifest contains duplicate identities")
    site_wells = table.select(identity_columns).unique()
    missing_wells = site_wells.join(wells, on=identity_columns, how="anti")
    extra_wells = wells.join(site_wells, on=identity_columns, how="anti")
    if not missing_wells.is_empty() or not extra_wells.is_empty():
        raise RuntimeError(
            "frozen well manifest does not exactly match site-manifest wells: "
            f"missing={missing_wells.height:,}, extra={extra_wells.height:,}"
        )

    missing = [key for key in keys if not (zarr_path / key).is_dir()]
    if missing:
        raise RuntimeError(f"canonical MQ source is missing release sites: {missing[:5]}")
    return table, keys


def key_digest(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def atomic_copy(con: duckdb.DuckDBPyConnection, query: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    con.execute(
        f"COPY ({query}) TO '{sql_path(temporary)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    temporary.replace(destination)


def main() -> int:
    args = parse_args()
    key_table, keys = canonical_key_table(
        args.canonical_zarr,
        args.site_manifest,
        args.well_manifest,
    )
    if len(keys) != args.expected_sites:
        raise RuntimeError(
            f"Canonical Zarr has {len(keys):,} sites; expected {args.expected_sites:,}"
        )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_readme_source = Path(__file__).with_name("JUMP_LITE_README.md")
    dataset_readme = output_dir.parent / "README.md"
    dataset_readme.write_text(dataset_readme_source.read_text())
    site_index = output_dir / "jump_lite_site_index.parquet"
    image_index = output_dir / "jump_lite_image_index.parquet"
    perturbation_metadata = output_dir / "jump_lite_perturbation_metadata.parquet"
    annotations = output_dir / "jump_lite_refchem_annotations.parquet"
    plate_manifest = output_dir / "jump_lite_plate_manifest.parquet"

    con = duckdb.connect()
    con.register("canonical_keys", key_table)
    full_index = sql_path(args.full_index)
    source_metadata = sql_path(args.perturbation_metadata)
    source_annotations = sql_path(args.annotations)

    join_columns = (
        "Metadata_Source, Metadata_Batch, Metadata_Plate, "
        "Metadata_Well, Metadata_Site"
    )
    matched, distinct_matched = con.execute(
        f"""
        SELECT count(*), count(DISTINCT Metadata_Site_Key)
        FROM read_parquet('{full_index}') AS source
        INNER JOIN canonical_keys USING ({join_columns})
        """
    ).fetchone()
    if matched != len(keys) or distinct_matched != len(keys):
        raise RuntimeError(
            "Canonical keys do not map one-to-one to the full index: "
            f"rows={matched:,}, distinct_keys={distinct_matched:,}, "
            f"expected={len(keys):,}"
        )

    site_query = f"""
        SELECT
          canonical.Metadata_Site_Key,
          source.*
        FROM read_parquet('{full_index}') AS source
        INNER JOIN canonical_keys AS canonical USING ({join_columns})
        ORDER BY
          Metadata_Source, Metadata_Batch, Metadata_Plate,
          Metadata_Well, Metadata_Site
    """
    atomic_copy(con, site_query, site_index)

    channel_queries = []
    for order, channel in enumerate(CHANNELS):
        channel_queries.append(
            f"""
            SELECT
              Metadata_Site_Key,
              Metadata_Source,
              Metadata_Batch,
              Metadata_Plate,
              Metadata_Well,
              Metadata_Site,
              {order} AS channel_order,
              '{channel}' AS Metadata_Channel,
              URL_Orig{channel} AS uri
            FROM read_parquet('{sql_path(site_index)}')
            """
        )
    image_query = " UNION ALL ".join(channel_queries)
    atomic_copy(
        con,
        f"""
        SELECT * EXCLUDE channel_order
        FROM ({image_query})
        ORDER BY
          Metadata_Source, Metadata_Batch, Metadata_Plate,
          Metadata_Well, Metadata_Site, channel_order
        """,
        image_index,
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW canonical_wells AS
        SELECT DISTINCT
          Metadata_Source, Metadata_Batch, Metadata_Plate, Metadata_Well
        FROM canonical_keys
        """
    )
    canonical_well_count = key_table.select(
        "Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"
    ).unique().height
    metadata_rows, metadata_distinct_wells = con.execute(
        f"""
        SELECT count(*), count(DISTINCT (
          metadata.Metadata_Source, metadata.Metadata_Batch,
          metadata.Metadata_Plate, metadata.Metadata_Well
        ))
        FROM read_parquet('{source_metadata}') AS metadata
        INNER JOIN canonical_wells USING (
          Metadata_Source, Metadata_Batch, Metadata_Plate, Metadata_Well
        )
        """
    ).fetchone()
    if metadata_rows != canonical_well_count or metadata_distinct_wells != canonical_well_count:
        raise RuntimeError(
            "perturbation metadata does not map one-to-one to every frozen well: "
            f"rows={metadata_rows:,}, distinct_wells={metadata_distinct_wells:,}, "
            f"expected={canonical_well_count:,}"
        )

    perturbation_query = f"""
        SELECT
          concat_ws('__', Metadata_Source, Metadata_Batch,
                    Metadata_Plate, Metadata_Well) AS Metadata_Well_Key,
          metadata.Metadata_Source,
          metadata.Metadata_Batch,
          metadata.Metadata_Plate,
          metadata.Metadata_Well,
          metadata.Metadata_JCP2022,
          metadata.Metadata_broad_sample,
          metadata.Metadata_Symbol,
          metadata.Metadata_pert_type,
          metadata.Metadata_Perturbation_Type,
          metadata.Metadata_Group
        FROM read_parquet('{source_metadata}') AS metadata
        SEMI JOIN canonical_wells USING (
          Metadata_Source, Metadata_Batch, Metadata_Plate, Metadata_Well
        )
        ORDER BY Metadata_Source, Metadata_Batch, Metadata_Plate, Metadata_Well
    """
    atomic_copy(con, perturbation_query, perturbation_metadata)

    # Keep only annotations whose query perturbation occurs in this release.
    # The unfiltered source table is broader than the frozen JUMP-Lite subset.
    annotation_query = f"""
        SELECT annotations.*
        FROM read_parquet('{source_annotations}') AS annotations
        SEMI JOIN (
          SELECT DISTINCT Metadata_JCP2022
          FROM read_parquet('{sql_path(perturbation_metadata)}')
        ) AS release_perturbations USING (Metadata_JCP2022)
        ORDER BY Metadata_JCP2022, target, mode
    """
    atomic_copy(con, annotation_query, annotations)

    plate_query = f"""
        SELECT
          Metadata_Source,
          Metadata_Batch,
          Metadata_Plate,
          count(DISTINCT Metadata_Well) AS well_count,
          count(*) AS site_count
        FROM read_parquet('{sql_path(site_index)}')
        GROUP BY ALL
        ORDER BY Metadata_Source, Metadata_Batch, Metadata_Plate
    """
    atomic_copy(con, plate_query, plate_manifest)

    site_rows = con.execute(
        f"SELECT count(*) FROM read_parquet('{sql_path(site_index)}')"
    ).fetchone()[0]
    image_rows = con.execute(
        f"SELECT count(*) FROM read_parquet('{sql_path(image_index)}')"
    ).fetchone()[0]
    metadata_rows, benchmark_perturbations = con.execute(
        f"""
        SELECT
          count(*),
          count(DISTINCT Metadata_JCP2022) FILTER (
            WHERE Metadata_Perturbation_Type = 'compound'
          ) +
          count(DISTINCT Metadata_Symbol) FILTER (
            WHERE Metadata_Perturbation_Type = 'crispr'
          ) +
          count(DISTINCT Metadata_Symbol) FILTER (
            WHERE Metadata_Perturbation_Type = 'orf'
          )
        FROM read_parquet('{sql_path(perturbation_metadata)}')
        """
    ).fetchone()
    plate_rows = con.execute(
        f"SELECT count(*) FROM read_parquet('{sql_path(plate_manifest)}')"
    ).fetchone()[0]
    annotation_rows, annotation_ids = con.execute(
        f"SELECT count(*), count(DISTINCT Metadata_JCP2022) "
        f"FROM read_parquet('{sql_path(annotations)}')"
    ).fetchone()

    source_counts = dict(
        con.execute(
            f"""
            SELECT Metadata_Source, count(*)
            FROM read_parquet('{sql_path(site_index)}')
            GROUP BY 1 ORDER BY 1
            """
        ).fetchall()
    )
    well_count = con.execute(
        f"""
        SELECT count(*) FROM (
          SELECT DISTINCT Metadata_Source, Metadata_Batch,
                          Metadata_Plate, Metadata_Well
          FROM read_parquet('{sql_path(site_index)}')
        )
        """
    ).fetchone()[0]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canonical_image_dataset": str(args.canonical_zarr),
        "canonical_codec": args.canonical_zarr.name,
        "release_site_manifest": str(args.site_manifest),
        "release_well_manifest": str(args.well_manifest),
        "site_key_sha256": key_digest(keys),
        "site_count": site_rows,
        "well_count": well_count,
        "plate_count": plate_rows,
        "source_count": len(source_counts),
        "site_count_by_source": source_counts,
        "channel_order": list(CHANNELS),
        "image_index_row_count": image_rows,
        "perturbation_metadata_row_count": metadata_rows,
        "benchmark_perturbation_count": benchmark_perturbations,
        "annotation_row_count": annotation_rows,
        "annotated_perturbation_count": annotation_ids,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size}
            for path in (
                dataset_readme,
                site_index,
                image_index,
                perturbation_metadata,
                annotations,
                plate_manifest,
            )
        },
        "notes": [
            "The paper-cohort site set is frozen in the committed v1.0 manifest and validated against MQ; it is not resampled.",
            "The image index points to the original public JUMP TIFF images.",
            "Benchmark perturbations are counted by modality and biological entity.",
        ],
    }
    manifest_path = output_dir / "metadata_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote release README to {dataset_readme}")
    print(f"Wrote CPG metadata to {output_dir}")
    print(f"  sites: {site_rows:,}; images: {image_rows:,}; wells: {well_count:,}")
    print(f"  plates: {plate_rows:,}; sources: {len(source_counts):,}")
    print(
        f"  perturbation metadata rows: {metadata_rows:,}; "
        f"benchmark perturbations: {benchmark_perturbations:,}"
    )
    print(
        f"  annotation rows: {annotation_rows:,}; "
        f"annotated perturbations: {annotation_ids:,}"
    )
    print(f"  site-key SHA-256: {manifest['site_key_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
