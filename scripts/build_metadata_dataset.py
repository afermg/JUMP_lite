#!/usr/bin/env python
"""
Build the metadata_dataset_filtered_4reps.parquet file.

This script combines the full 8-step pipeline for generating the final
metadata dataset used in JUMP_core experiments. Each step corresponds
to a previously separate script, now unified into a single reference
implementation.

Pipeline Steps:
    Step 1: InChIKey -> JCP2022 mapping        (from standardize_annotations.py)
    Step 2: Generate raw JUMP metadata         (from download_images.py, metadata-only)
    Step 3: Filter metadata (25% fill rate)    (from analyze_metadata.py)
    Step 4: Prepare negative controls          (from prepare_negative_controls.py)
    Step 5: Match metadata to profiles         (from compare_metadata_profiles.py)
    Step 6: Filter to >=4 replicates           (from compare_compound_overlap.py)
    Step 7: Build target lists                 (MOTIVE + RefChemDB annotations)
    Step 8: Filter raw CP profiles             (match wells to metadata)

Dependencies:
    - polars: DataFrame operations
    - duckdb: SQL queries on jump_metadata.duckdb (well, plate, compound, crispr, orf tables)

Usage:
    # Full pipeline from scratch:
    python build_metadata_dataset.py \\
        --annotations-db /work/datasets/jump_core/annotations/jump_metadata.duckdb \\
        --annotations-cc /work/datasets/jump_core/annotations/annotations_compound_compound.parquet \\
        --annotations-cg /work/datasets/jump_core/annotations/annotations_compound_gene.parquet \\
        --profiles /work/datasets/jump_core_annotated/raw_jump_CP_profiles/profiles.parquet \\
        --refchemdb /path/to/refchemdb_conf_jump_matched.parquet \\
        --output-dir /home/jfredinh/projects/JUMP_core/metadata \\
        --save-intermediates

    # Resume from step 5 (if intermediates already exist):
    python build_metadata_dataset.py \\
        --skip-to 5 \\
        --profiles /work/datasets/jump_core_annotated/raw_jump_CP_profiles/profiles.parquet \\
        --refchemdb /path/to/refchemdb_conf_jump_matched.parquet \\
        --output-dir /home/jfredinh/projects/JUMP_core/metadata
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import duckdb
import polars as pl


# All JUMP metadata (well, plate, compound, crispr, orf) is loaded from the
# local jump_metadata.duckdb database, eliminating the need for broad_babel,
# jump_portrait, or pooch network downloads.


# ============================================================================
# Constants
# ============================================================================

WELLS_PER_384_PLATE = 384

# Known JUMP negative control JCP2022 IDs.
# These are excluded from the compound perturbation list in Step 2 because
# they are not treatments -- they are added back via the dedicated negative
# control sampling in Step 4.
NEGATIVE_CONTROL_JCPS = [
    "JCP2022_999999",
    "JCP2022_033924",   # DMSO
    "JCP2022_037716",
    "JCP2022_025848",
    "JCP2022_046054",
    "JCP2022_035095",
    "JCP2022_064022",
    "JCP2022_050797",
    "JCP2022_012818",
    "JCP2022_085227",
    "JCP2022_800001",   # Non-targeting guide (CRISPR)
    "JCP2022_800002",
    "JCP2022_915128",   # BFP (ORF negcon)
    "JCP2022_915129",   # HcRed (ORF negcon)
    "JCP2022_915130",   # Luciferase (ORF negcon)
    "JCP2022_915131",   # LacZ (ORF negcon)
]

# Negative control JCP IDs by modality (used in Step 4).
# Each plate type has its own appropriate negative control.
# ORF negcons are non-human reporter genes (per JUMP perturbation_control.csv).
# Note: JCP2022_805264 was previously listed here but is actually PLK1 (CRISPR poscon).
MODALITY_NEGCONS = {
    "COMPOUND": ["JCP2022_033924"],                                    # DMSO
    "CRISPR":   ["JCP2022_800001"],                                    # Non-targeting guide
    "ORF":      ["JCP2022_915128", "JCP2022_915129",                   # BFP, HcRed
                 "JCP2022_915130", "JCP2022_915131"],                  # Luciferase, LacZ
}

# Fraction of negative controls to sample per plate, by modality.
# Default: keep all negative controls (fraction=1.0).  Override with
# --negcon-fraction to sample a subset (e.g. 0.5 for 50%).
MODALITY_FRACTION = {
    "COMPOUND": 1.0,
    "CRISPR":   1.0,
    "ORF":      1.0,
}

# Redlisted plates: known-bad plates that should be excluded from analysis.
# Sourced from scripts/utils/redlisted_plates.py in the original repo.
# Each entry maps plate ID -> human-readable reason for exclusion.
EXCLUDED_PLATES = {
    # Source 3 - batches without DMSO negative controls
    # Batch: CP_35_all_Phenix1
    "BAY5871b":  "SOURCE3_BATCH_REDLIST (CP_35_all_Phenix1) - no DMSO negative controls",
    "BAY5871c":  "SOURCE3_BATCH_REDLIST (CP_35_all_Phenix1) - no DMSO negative controls",
    "BAY5871d":  "SOURCE3_BATCH_REDLIST (CP_35_all_Phenix1) - no DMSO negative controls",
    "BAY5873a":  "SOURCE3_BATCH_REDLIST (CP_35_all_Phenix1) - no DMSO negative controls",
    "BAY5873b":  "SOURCE3_BATCH_REDLIST (CP_35_all_Phenix1) - no DMSO negative controls",
    "BAY5873c":  "SOURCE3_BATCH_REDLIST (CP_35_all_Phenix1) - no DMSO negative controls",
    "BAY5873d":  "SOURCE3_BATCH_REDLIST (CP_35_all_Phenix1) - no DMSO negative controls",
    "BAY5875a":  "SOURCE3_BATCH_REDLIST (CP_35_all_Phenix1) - no DMSO negative controls",
    "BAY5875b":  "SOURCE3_BATCH_REDLIST (CP_35_all_Phenix1) - no DMSO negative controls",
    "BAY5875c":  "SOURCE3_BATCH_REDLIST (CP_35_all_Phenix1) - no DMSO negative controls",
    "BAY5875d":  "SOURCE3_BATCH_REDLIST (CP_35_all_Phenix1) - no DMSO negative controls",
    # Batch: CP_36_all_Phenix1
    "BAY5872a":  "SOURCE3_BATCH_REDLIST (CP_36_all_Phenix1) - no DMSO negative controls",
    "BAY5872b":  "SOURCE3_BATCH_REDLIST (CP_36_all_Phenix1) - no DMSO negative controls",
    "BAY5872c":  "SOURCE3_BATCH_REDLIST (CP_36_all_Phenix1) - no DMSO negative controls",
    "BAY5872d":  "SOURCE3_BATCH_REDLIST (CP_36_all_Phenix1) - no DMSO negative controls",
    "BAY5874a":  "SOURCE3_BATCH_REDLIST (CP_36_all_Phenix1) - no DMSO negative controls",
    "BAY5874b":  "SOURCE3_BATCH_REDLIST (CP_36_all_Phenix1) - no DMSO negative controls",
    "BAY5874c":  "SOURCE3_BATCH_REDLIST (CP_36_all_Phenix1) - no DMSO negative controls",
    "BAY5874d":  "SOURCE3_BATCH_REDLIST (CP_36_all_Phenix1) - no DMSO negative controls",
    "BAY5876a":  "SOURCE3_BATCH_REDLIST (CP_36_all_Phenix1) - no DMSO negative controls",
    "BAY5876b":  "SOURCE3_BATCH_REDLIST (CP_36_all_Phenix1) - no DMSO negative controls",
    "BAY5876c":  "SOURCE3_BATCH_REDLIST (CP_36_all_Phenix1) - no DMSO negative controls",
    "BAY5876d":  "SOURCE3_BATCH_REDLIST (CP_36_all_Phenix1) - no DMSO negative controls",
    # Batch: CP59
    "BR5871b3":  "SOURCE3_BATCH_REDLIST (CP59) - no DMSO negative controls",
    "BR5871c3":  "SOURCE3_BATCH_REDLIST (CP59) - no DMSO negative controls",
    "BR5871d3":  "SOURCE3_BATCH_REDLIST (CP59) - no DMSO negative controls",
    "BR5872a3":  "SOURCE3_BATCH_REDLIST (CP59) - no DMSO negative controls",
    "BR5872b3":  "SOURCE3_BATCH_REDLIST (CP59) - no DMSO negative controls",
    "BR5872c3":  "SOURCE3_BATCH_REDLIST (CP59) - no DMSO negative controls",
    "BR5872d3":  "SOURCE3_BATCH_REDLIST (CP59) - no DMSO negative controls",
    "BR5875a3":  "SOURCE3_BATCH_REDLIST (CP59) - no DMSO negative controls",
    "BR5875b3":  "SOURCE3_BATCH_REDLIST (CP59) - no DMSO negative controls",
    "BR5875c3":  "SOURCE3_BATCH_REDLIST (CP59) - no DMSO negative controls",
    "BR5875d3":  "SOURCE3_BATCH_REDLIST (CP59) - no DMSO negative controls",
    "BR5876a3":  "SOURCE3_BATCH_REDLIST (CP59) - no DMSO negative controls",
    "BR5876b3":  "SOURCE3_BATCH_REDLIST (CP59) - no DMSO negative controls",
    "BR5876c3":  "SOURCE3_BATCH_REDLIST (CP59) - no DMSO negative controls",
    # Batch: CP60
    "BR5873a3":  "SOURCE3_BATCH_REDLIST (CP60) - no DMSO negative controls",
    "BR5873b3":  "SOURCE3_BATCH_REDLIST (CP60) - no DMSO negative controls",
    "BR5873c3":  "SOURCE3_BATCH_REDLIST (CP60) - no DMSO negative controls",
    "BR5873d3W": "SOURCE3_BATCH_REDLIST (CP60) - no DMSO negative controls",
    "BR5874a3":  "SOURCE3_BATCH_REDLIST (CP60) - no DMSO negative controls",
    "BR5874b3":  "SOURCE3_BATCH_REDLIST (CP60) - no DMSO negative controls",
    "BR5874c3":  "SOURCE3_BATCH_REDLIST (CP60) - no DMSO negative controls",
    "BR5874d3":  "SOURCE3_BATCH_REDLIST (CP60) - no DMSO negative controls",
    # Source 4 - ORF Batch12 redlist (dye anomaly)
    "BR00126706": "ORF_BATCH12_REDLIST - dye anomaly",
    "BR00126708": "ORF_BATCH12_REDLIST - dye anomaly",
    "BR00126709": "ORF_BATCH12_REDLIST - dye anomaly",
    "BR00126710": "ORF_BATCH12_REDLIST - dye anomaly",
    "BR00126711": "ORF_BATCH12_REDLIST - dye anomaly",
    "BR00126712": "ORF_BATCH12_REDLIST - dye anomaly",
    "BR00126714": "ORF_BATCH12_REDLIST - dye anomaly",
    "BR00126715": "ORF_BATCH12_REDLIST - dye anomaly",
    "BR00126716": "ORF_BATCH12_REDLIST - dye anomaly",
    "BR00126717": "ORF_BATCH12_REDLIST - dye anomaly",
    "BR00126718": "ORF_BATCH12_REDLIST - dye anomaly",
    # Source 4 - Explicit redlist
    "BR00123528A": "EXPLICIT_REDLIST - github.com/jump-cellpainting/aws/issues/70",
    # Source 4 - TARGET1 plates not in config plate_types
    "BR00123523": "PLATE_TYPE_FILTER - TARGET1 not in config plate_types",
    "BR00123524": "PLATE_TYPE_FILTER - TARGET1 not in config plate_types",
    "BR00125181": "PLATE_TYPE_FILTER - TARGET1 not in config plate_types",
    "BR00125638": "PLATE_TYPE_FILTER - TARGET1 not in config plate_types",
    # Source 2 - Batch_13 has only 1 plate (32 negcons) - too few samples for
    # reliable batch correction
    "1086289792": "SMALL_BATCH_REDLIST (20211003_Batch_13) - only 1 plate in batch",
    # Source 7 - Run6/7/9M have 1-2 plates each (32-72 negcons) - too few
    # samples for reliable batch correction
    "CP7-SC1-01":  "SMALL_BATCH_REDLIST (20211121_Run6) - only 1 plate in batch",
    "CP7-SC1-02":  "SMALL_BATCH_REDLIST (20211126_Run7) - only 2 plates in batch",
    "CP8-SC1-02":  "SMALL_BATCH_REDLIST (20211126_Run7) - only 2 plates in batch",
    "CP6-SC1-18":  "SMALL_BATCH_REDLIST (20211211_Run9M) - only 2 plates in batch",
    "CP7-SC1-03":  "SMALL_BATCH_REDLIST (20211211_Run9M) - only 2 plates in batch",
}


# ============================================================================
# Step 1: InChIKey -> JCP2022 Mapping
# ============================================================================
# Source: src/standardize_annotations.py
#
# This step translates InChIKeys from annotation databases (compound-compound
# and compound-gene interactions from MOTIVE) into JUMP JCP2022 identifiers
# using the JUMP metadata DuckDB database.  Matching is done on the InChIKey
# connectivity layer (first 14 characters before the first hyphen) to handle
# stereoisomer variations.
# ============================================================================


def step1_inchikey_to_jcp_mapping(
    db_path: str,
    annotations_cc_path: str,
    annotations_cg_path: str,
    output_dir: Path,
    save_intermediates: bool = False,
) -> pl.DataFrame:
    """Translate InChIKeys from annotation databases to JCP2022 IDs.

    Args:
        db_path: Path to jump_metadata.duckdb containing the compound table.
        annotations_cc_path: Path to compound-compound annotations parquet.
        annotations_cg_path: Path to compound-gene annotations parquet.
        output_dir: Directory for intermediate outputs.
        save_intermediates: Whether to write intermediate CSV files.

    Returns:
        Polars DataFrame with columns:
            [Metadata_InChIKey, InChIKey_Connectivity, Metadata_JCP2022]
        containing the deduplicated combined mapping.
    """
    print("=" * 70)
    print("STEP 1: InChIKey -> JCP2022 Mapping")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load the InChIKey -> JCP2022 mapping from DuckDB
    # ------------------------------------------------------------------
    print("  Loading InChIKey -> JCP2022 mapping from DuckDB...")
    con = duckdb.connect(db_path, read_only=True)
    mapping_pd = con.execute("""
        SELECT Metadata_InChIKey, Metadata_JCP2022
        FROM compound
        WHERE Metadata_InChIKey IS NOT NULL
          AND Metadata_JCP2022 IS NOT NULL
    """).df()
    con.close()
    mapping = pl.from_pandas(mapping_pd)

    # Extract connectivity layer (first part of InChIKey before the hyphen).
    # The connectivity layer captures molecular identity independent of
    # stereochemistry, charge, or isotope labelling.
    mapping = mapping.with_columns(
        pl.col("Metadata_InChIKey")
        .str.split("-")
        .list.first()
        .alias("InChIKey_Connectivity")
    )
    # Keep unique (connectivity, JCP2022) pairs for the join
    connectivity_mapping = mapping.select(
        "InChIKey_Connectivity", "Metadata_JCP2022"
    ).unique()

    print(f"  Loaded {mapping.height:,} InChIKey -> JCP2022 entries")
    print(f"  Unique connectivity-layer mappings: {connectivity_mapping.height:,}")

    # ------------------------------------------------------------------
    # Process compound-compound annotations
    # ------------------------------------------------------------------
    print("\n  Processing compound-compound annotations...")
    df_cc = pl.read_parquet(annotations_cc_path)
    inchikeys_a = (
        df_cc.select("inchikey_a")
        .drop_nulls()
        .unique()
        .rename({"inchikey_a": "Metadata_InChIKey"})
    )
    inchikeys_b = (
        df_cc.select("inchikey_b")
        .drop_nulls()
        .unique()
        .rename({"inchikey_b": "Metadata_InChIKey"})
    )
    inchikeys_cc = pl.concat([inchikeys_a, inchikeys_b]).unique()
    print(f"    Unique InChIKeys in compound-compound: {inchikeys_cc.height:,}")

    result_cc = _match_inchikeys_to_jcp(inchikeys_cc, connectivity_mapping)
    result_cc = result_cc.drop_nulls(subset=["Metadata_JCP2022"])
    print(f"    Mapped to JCP2022: {result_cc.height:,}")

    if save_intermediates:
        path = output_dir / "inchikey_to_jcp2022_mapping_compound_compound.csv"
        result_cc.write_csv(path)
        print(f"    Saved: {path}")

    # ------------------------------------------------------------------
    # Process compound-gene annotations
    # ------------------------------------------------------------------
    print("\n  Processing compound-gene annotations...")
    df_cg = pl.read_parquet(annotations_cg_path)
    inchikeys_cg = (
        df_cg.select("inchikey")
        .drop_nulls()
        .unique()
        .rename({"inchikey": "Metadata_InChIKey"})
    )
    print(f"    Unique InChIKeys in compound-gene: {inchikeys_cg.height:,}")

    result_cg = _match_inchikeys_to_jcp(inchikeys_cg, connectivity_mapping)
    result_cg = result_cg.drop_nulls(subset=["Metadata_JCP2022"])
    print(f"    Mapped to JCP2022: {result_cg.height:,}")

    if save_intermediates:
        path = output_dir / "inchikey_to_jcp2022_mapping_compound_gene.csv"
        result_cg.write_csv(path)
        print(f"    Saved: {path}")

    # ------------------------------------------------------------------
    # Combine and deduplicate
    # ------------------------------------------------------------------
    combined = pl.concat([result_cc, result_cg]).unique()
    # Deduplicate by Metadata_JCP2022 to get one row per unique JUMP compound
    combined = combined.unique(subset=["Metadata_JCP2022"])
    print(f"\n  Combined unique JCP2022 mappings: {combined.height:,}")

    if save_intermediates:
        path = output_dir / "inchikey_to_jcp2022_mapping_combined.csv"
        combined.write_csv(path)
        print(f"  Saved: {path}")

    return combined


def _match_inchikeys_to_jcp(
    inchikeys_df: pl.DataFrame,
    connectivity_mapping: pl.DataFrame,
) -> pl.DataFrame:
    """Match InChIKeys to JCP2022 IDs via the connectivity layer.

    Args:
        inchikeys_df: DataFrame with a single column ``Metadata_InChIKey``.
        connectivity_mapping: DataFrame with columns
            ``[InChIKey_Connectivity, Metadata_JCP2022]``.

    Returns:
        DataFrame with columns:
            ``[Metadata_InChIKey, InChIKey_Connectivity, Metadata_JCP2022]``
    """
    df = inchikeys_df.with_columns(
        pl.col("Metadata_InChIKey")
        .str.split("-")
        .list.first()
        .alias("InChIKey_Connectivity")
    ).unique(subset=["InChIKey_Connectivity"])

    result = df.join(connectivity_mapping, on="InChIKey_Connectivity", how="left")
    return result.select("Metadata_InChIKey", "InChIKey_Connectivity", "Metadata_JCP2022")


# ============================================================================
# Step 2: Generate Raw JUMP Metadata
# ============================================================================
# This step builds the raw metadata file that maps each JCP2022 ID to its
# physical location in the Cell Painting Gallery (source, batch, plate, well).
# It reads perturbation lists (CRISPR, ORF) from the DuckDB and resolves
# locations via a well+plate join — all local, no network access needed.
# ============================================================================


def step2_generate_raw_metadata(
    jcp_mapping: pl.DataFrame,
    db_path: str,
    output_dir: Path,
    save_intermediates: bool = False,
) -> pl.DataFrame:
    """Build the raw JUMP metadata mapping JCP2022 IDs to plate/well locations.

    This resolves every perturbation (CRISPR genes, ORF genes, annotated
    compounds) to its physical address in the Cell Painting Gallery using
    the local DuckDB database.

    Args:
        jcp_mapping: DataFrame from Step 1 with column ``Metadata_JCP2022``
            containing the annotated compound JCP IDs.
        db_path: Path to jump_metadata.duckdb.
        output_dir: Directory for intermediate outputs.
        save_intermediates: Whether to write the raw metadata parquet.

    Returns:
        Polars DataFrame with columns:
            ``[Metadata_Source, Metadata_Batch, Metadata_Plate,
              Metadata_Well, Metadata_JCP2022]``
    """
    print("\n" + "=" * 70)
    print("STEP 2: Generate Raw JUMP Metadata")
    print("=" * 70)

    con = duckdb.connect(db_path, read_only=True)

    # ------------------------------------------------------------------
    # Load CRISPR perturbation list from DuckDB
    # ------------------------------------------------------------------
    print("  Loading CRISPR perturbation list from DuckDB...")
    crispr_jcps = pl.from_pandas(
        con.execute("SELECT DISTINCT Metadata_JCP2022 FROM crispr").df()
    ).to_series()
    print(f"    CRISPR perturbations: {len(crispr_jcps):,}")

    # ------------------------------------------------------------------
    # Load ORF perturbation list from DuckDB
    # ------------------------------------------------------------------
    print("  Loading ORF perturbation list from DuckDB...")
    orf_jcps = pl.from_pandas(
        con.execute("SELECT DISTINCT Metadata_JCP2022 FROM orf").df()
    ).to_series()
    print(f"    ORF perturbations: {len(orf_jcps):,}")

    # ------------------------------------------------------------------
    # Get compound JCP IDs from Step 1 mapping, excluding controls
    # ------------------------------------------------------------------
    compound_jcps = jcp_mapping.select("Metadata_JCP2022").to_series().unique()
    compound_jcps = compound_jcps.filter(~compound_jcps.is_in(NEGATIVE_CONTROL_JCPS))
    print(f"    Compound perturbations (after excluding controls): {len(compound_jcps):,}")

    # ------------------------------------------------------------------
    # Collect all JCP IDs to resolve
    # ------------------------------------------------------------------
    all_jcps = list(set(crispr_jcps.to_list() + orf_jcps.to_list() + compound_jcps.to_list()))
    print(f"\n  Resolving locations for {len(all_jcps):,} unique JCP IDs via DuckDB...")

    # ------------------------------------------------------------------
    # Resolve all JCP2022 IDs to (source, batch, plate, well) via a
    # single well+plate join in DuckDB — replaces the slow sequential
    # jump_portrait lookups.
    # ------------------------------------------------------------------
    metadata_all_pd = con.execute("""
        SELECT
            w.Metadata_Source,
            p.Metadata_Batch,
            w.Metadata_Plate,
            w.Metadata_Well,
            w.Metadata_JCP2022
        FROM well w
        JOIN plate p
            ON w.Metadata_Source = p.Metadata_Source
            AND w.Metadata_Plate = p.Metadata_Plate
        WHERE w.Metadata_JCP2022 IN (SELECT UNNEST($1::VARCHAR[]))
    """, [all_jcps]).df()
    con.close()

    metadata_all = pl.from_pandas(metadata_all_pd)

    print(f"\n  Total metadata rows: {metadata_all.height:,}")
    print(f"  Unique plates: {metadata_all.select('Metadata_Plate').n_unique():,}")
    print(f"  Unique JCP IDs: {metadata_all.select('Metadata_JCP2022').n_unique():,}")

    if save_intermediates:
        path = output_dir / "metadata.parquet"
        metadata_all.write_parquet(path)
        print(f"  Saved: {path}")

    return metadata_all


# ============================================================================
# Step 3: Filter Metadata (25% Plate Fill Rate)
# ============================================================================
# Source: analysis/annotated_data_selection/well_downloading/analyze_metadata.py
#
# This step performs three filters on the raw metadata:
#   1. Remove source_9 (1536-well plates, incompatible format)
#   2. Remove TARGET2 plates (identified via DuckDB plate metadata)
#   3. Remove plates with < 25% fill rate (fewer than 96 of 384 wells present)
#
# It also classifies each well by perturbation type (CRISPR, ORF, COMPOUND)
# using the DuckDB lookup tables.
# ============================================================================


def step3_filter_metadata(
    metadata_all: pl.DataFrame,
    db_path: str,
    output_dir: Path,
    min_fill_rate: float = 25.0,
    save_intermediates: bool = False,
) -> pl.DataFrame:
    """Filter raw metadata by source, plate type, and fill rate.

    Args:
        metadata_all: Raw metadata from Step 2.
        db_path: Path to jump_metadata.duckdb.
        output_dir: Directory for intermediate outputs.
        min_fill_rate: Minimum plate fill rate percentage (default 25%).
        save_intermediates: Whether to write the filtered metadata parquet.

    Returns:
        Filtered Polars DataFrame with an added ``Perturbation_Type`` column.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Filter Metadata (Plate Fill Rate)")
    print("=" * 70)

    con = duckdb.connect(db_path, read_only=True)

    # ------------------------------------------------------------------
    # 3a. Remove source_9 (1536-well plates)
    # ------------------------------------------------------------------
    sources_to_exclude = ["source_9"]
    df = metadata_all.filter(~pl.col("Metadata_Source").is_in(sources_to_exclude))
    print(f"  After removing {sources_to_exclude}: {df.height:,} rows")

    # ------------------------------------------------------------------
    # 3b. Remove TARGET2 plates using DuckDB plate metadata
    # ------------------------------------------------------------------
    print("  Loading plate metadata from DuckDB...")
    plates_target2 = set(
        con.execute("""
            SELECT Metadata_Plate FROM plate
            WHERE Metadata_PlateType ILIKE '%TARGET2%'
        """).df()["Metadata_Plate"]
    )
    print(f"  TARGET2 plates to exclude: {len(plates_target2):,}")

    df = df.filter(~pl.col("Metadata_Plate").is_in(list(plates_target2)))
    print(f"  After removing TARGET2: {df.height:,} rows")

    # ------------------------------------------------------------------
    # 3c. Classify perturbation types
    # ------------------------------------------------------------------
    print("  Classifying perturbation types via DuckDB...")
    crispr_jcps = set(
        con.execute("SELECT DISTINCT Metadata_JCP2022 FROM crispr WHERE Metadata_JCP2022 IS NOT NULL").df()["Metadata_JCP2022"]
    )
    orf_jcps = set(
        con.execute("SELECT DISTINCT Metadata_JCP2022 FROM orf WHERE Metadata_JCP2022 IS NOT NULL").df()["Metadata_JCP2022"]
    )
    compound_jcps = set(
        con.execute("SELECT DISTINCT Metadata_JCP2022 FROM compound WHERE Metadata_JCP2022 IS NOT NULL").df()["Metadata_JCP2022"]
    )
    con.close()

    # Build a mapping from JCP2022 -> perturbation type for all unique IDs.
    # The order of checks matters: CRISPR > ORF > COMPOUND > UNKNOWN.
    unique_jcp_ids = df.select("Metadata_JCP2022").unique().to_series().to_list()
    type_map: dict[str, str] = {}
    for jcp_id in unique_jcp_ids:
        if jcp_id in crispr_jcps:
            type_map[jcp_id] = "CRISPR"
        elif jcp_id in orf_jcps:
            type_map[jcp_id] = "ORF"
        elif jcp_id in compound_jcps:
            type_map[jcp_id] = "COMPOUND"
        else:
            type_map[jcp_id] = "UNKNOWN"

    type_mapping_df = pl.DataFrame({
        "Metadata_JCP2022": list(type_map.keys()),
        "Perturbation_Type": list(type_map.values()),
    })
    df = df.join(type_mapping_df, on="Metadata_JCP2022", how="left")

    counts = (
        df.group_by("Perturbation_Type")
        .agg(pl.col("Metadata_JCP2022").n_unique().alias("unique_jcp_ids"))
        .sort("Perturbation_Type")
    )
    print("  Perturbation type counts:")
    for row in counts.iter_rows(named=True):
        print(f"    {row['Perturbation_Type']}: {row['unique_jcp_ids']:,} unique JCP IDs")

    # ------------------------------------------------------------------
    # 3d. Calculate plate fill rates and filter
    #
    # Fill rate = (number of unique wells present on a plate) / 384 * 100.
    # Plates below the threshold are likely partial uploads or failures.
    # ------------------------------------------------------------------
    print(f"\n  Calculating plate fill rates (min threshold: {min_fill_rate}%)...")
    plate_fills = (
        df.group_by(["Metadata_Source", "Metadata_Plate"])
        .agg(pl.col("Metadata_Well").n_unique().alias("Wells_Present"))
        .with_columns(
            (pl.col("Wells_Present") / WELLS_PER_384_PLATE * 100)
            .round(2)
            .alias("Fill_Rate_Percent")
        )
        .sort("Fill_Rate_Percent", descending=True)
    )

    fill_stats = plate_fills.select("Fill_Rate_Percent").describe()
    print(f"  Fill rate statistics:\n{fill_stats}")

    # Filter to plates meeting the minimum fill rate
    valid_plates = plate_fills.filter(
        pl.col("Fill_Rate_Percent") >= min_fill_rate
    ).select("Metadata_Source", "Metadata_Plate")

    n_plates_before = df.select("Metadata_Plate").n_unique()
    df = df.join(valid_plates, on=["Metadata_Source", "Metadata_Plate"], how="inner")
    n_plates_after = df.select("Metadata_Plate").n_unique()

    print(f"  Plates before fill-rate filter: {n_plates_before:,}")
    print(f"  Plates after fill-rate filter:  {n_plates_after:,}")
    print(f"  Rows after filtering: {df.height:,}")

    if save_intermediates:
        path = output_dir / "metadata_filtered.parquet"
        df.write_parquet(path)
        print(f"  Saved: {path}")

    return df


# ============================================================================
# Step 4: Prepare Negative Controls
# ============================================================================
# Source: analysis/annotated_data_selection/well_downloading/prepare_negative_controls.py
#
# For each plate in the filtered metadata, this step retrieves its negative
# control wells from the full JUMP well metadata (via DuckDB).  Negative
# controls are modality-specific:
#   - COMPOUND plates: DMSO (JCP2022_033924)
#   - CRISPR plates:   Non-targeting guide (JCP2022_800001)
#   - ORF plates:      LacZ + untreated (JCP2022_805264, JCP2022_915128)
#
# A fraction of negative controls is sampled per plate to keep the dataset
# balanced (50% for compound/CRISPR, 100% for ORF).
# ============================================================================


def step4_prepare_negative_controls(
    filtered_metadata: pl.DataFrame,
    db_path: str,
    output_dir: Path,
    seed: int = 42,
    negcon_fraction: Optional[float] = None,
    save_intermediates: bool = False,
) -> pl.DataFrame:
    """Sample negative control wells for each plate in the filtered metadata.

    Args:
        filtered_metadata: Filtered metadata from Step 3, must include
            ``Perturbation_Type`` column.
        output_dir: Directory for intermediate outputs.
        seed: Random seed for reproducible sampling.
        negcon_fraction: If provided, overrides MODALITY_FRACTION for all
            modalities.  Use 1.0 to keep all negative controls.
        save_intermediates: Whether to write the negative controls parquet.

    Returns:
        Polars DataFrame of sampled negative control wells with columns:
            ``[Metadata_Source, Metadata_Batch, Metadata_Plate,
              Metadata_Well, Metadata_JCP2022]``
    """
    import pandas as pd

    print("\n" + "=" * 70)
    print("STEP 4: Prepare Negative Controls")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 4a. Get the set of plates per modality from the filtered metadata
    # ------------------------------------------------------------------
    plates_by_modality: dict[str, set[str]] = {}
    for modality in ["COMPOUND", "CRISPR", "ORF"]:
        plates = set(
            filtered_metadata
            .filter(pl.col("Perturbation_Type") == modality)
            .select("Metadata_Plate")
            .unique()
            .to_series()
            .to_list()
        )
        plates_by_modality[modality] = plates
        print(f"  {modality} plates: {len(plates):,}")

    # ------------------------------------------------------------------
    # 4b. Load full JUMP well + plate metadata from DuckDB
    #
    # We need the full well table (not just our filtered subset) because
    # negative control wells were deliberately excluded from the treatment
    # metadata built in Steps 1-2.
    # ------------------------------------------------------------------
    print("\n  Loading full JUMP well metadata from DuckDB...")
    con = duckdb.connect(db_path, read_only=True)
    full_metadata_pd = con.execute("""
        SELECT
            w.Metadata_Source,
            p.Metadata_Batch,
            w.Metadata_Plate,
            w.Metadata_Well,
            w.Metadata_JCP2022
        FROM well w
        JOIN plate p
            ON w.Metadata_Source = p.Metadata_Source
            AND w.Metadata_Plate = p.Metadata_Plate
    """).df()
    con.close()
    full_metadata = pl.from_pandas(full_metadata_pd)
    print(f"  Full well metadata: {full_metadata.height:,} rows")

    # ------------------------------------------------------------------
    # 4c. Extract negative control wells, matched by modality
    #
    # For each modality, find wells on its plates whose JCP2022 ID is one
    # of the known negative control IDs for that modality.
    # ------------------------------------------------------------------
    neg_control_dfs: list[pl.DataFrame] = []
    for modality, plates in plates_by_modality.items():
        if modality not in MODALITY_NEGCONS:
            continue
        neg_jcps = MODALITY_NEGCONS[modality]
        modality_negcons = (
            full_metadata
            .filter(
                pl.col("Metadata_Plate").is_in(list(plates))
                & pl.col("Metadata_JCP2022").is_in(neg_jcps)
            )
            .with_columns(pl.lit(modality).alias("Modality"))
        )
        neg_control_dfs.append(modality_negcons)

    if neg_control_dfs:
        neg_controls = pl.concat(neg_control_dfs)
    else:
        print("  WARNING: No negative controls found.")
        return pl.DataFrame(schema={
            "Metadata_Source": pl.Utf8,
            "Metadata_Batch": pl.Utf8,
            "Metadata_Plate": pl.Utf8,
            "Metadata_Well": pl.Utf8,
            "Metadata_JCP2022": pl.Utf8,
        })

    print(f"\n  Total negative control wells found: {neg_controls.height:,}")
    for modality in ["COMPOUND", "CRISPR", "ORF"]:
        mod_data = neg_controls.filter(pl.col("Modality") == modality)
        if mod_data.height > 0:
            for jcp in MODALITY_NEGCONS.get(modality, []):
                jcp_data = mod_data.filter(pl.col("Metadata_JCP2022") == jcp)
                if jcp_data.height > 0:
                    n_plates = jcp_data.select("Metadata_Plate").n_unique()
                    print(
                        f"    {modality} / {jcp}: "
                        f"{jcp_data.height:,} wells across {n_plates} plates"
                    )

    # ------------------------------------------------------------------
    # 4d. Sample a fraction per plate (modality-specific fraction)
    #
    # This uses pandas groupby for per-plate random sampling because polars
    # sample operates on the full DataFrame rather than per-group.
    # ------------------------------------------------------------------
    # Build effective fraction map, applying override if provided
    effective_fractions = dict(MODALITY_FRACTION)
    if negcon_fraction is not None:
        for mod in effective_fractions:
            effective_fractions[mod] = negcon_fraction
        print(f"\n  Using override negcon_fraction={negcon_fraction} for all modalities")

    print(f"\n  Sampling negative controls (seed={seed})...")
    for mod, frac in effective_fractions.items():
        print(f"    {mod}: fraction={frac}")
    neg_controls_pd = neg_controls.to_pandas()

    sampled_dfs: list[pd.DataFrame] = []
    for _plate, group in neg_controls_pd.groupby("Metadata_Plate"):
        modality = group["Modality"].iloc[0]
        fraction = effective_fractions.get(modality, 1.0)

        if fraction >= 1.0:
            sampled_dfs.append(group)
        else:
            n_samples = max(1, int(len(group) * fraction))
            sampled_dfs.append(group.sample(n=n_samples, random_state=seed))

    sampled_pd = (
        pd.concat(sampled_dfs, ignore_index=True)
        if sampled_dfs
        else pd.DataFrame()
    )
    sampled = pl.from_pandas(sampled_pd)

    print(f"  Sampled negative controls: {sampled.height:,}")
    print(f"  Plates represented: {sampled.select('Metadata_Plate').n_unique():,}")

    # Select output columns (drop the Modality helper column)
    output_cols = [
        "Metadata_Source", "Metadata_Batch", "Metadata_Plate",
        "Metadata_Well", "Metadata_JCP2022",
    ]
    result = sampled.select([c for c in output_cols if c in sampled.columns])

    if save_intermediates:
        path = output_dir / "metadata_negative_controls.parquet"
        result.write_parquet(path)
        print(f"  Saved: {path}")

    return result


# ============================================================================
# Step 5: Match Metadata to Profiles
# ============================================================================
# Source: scripts/compare_metadata_profiles.py
#
# This step intersects the filtered metadata (treatment wells + negative
# controls) with the actual profile data to find wells that exist in both.
# Wells on redlisted plates are excluded.  The result is enriched with
# additional metadata columns from the profiles (Metadata_JCP2022,
# Metadata_broad_sample, Metadata_Symbol, Metadata_pert_type,
# Metadata_Perturbation_Type).
# ============================================================================


def step5_match_metadata_to_profiles(
    filtered_metadata: pl.DataFrame,
    negative_controls: pl.DataFrame,
    profiles_path: str,
    output_dir: Path,
    save_intermediates: bool = False,
) -> pl.DataFrame:
    """Intersect metadata with profiles and enrich with profile columns.

    Args:
        filtered_metadata: Treatment metadata from Step 3.
        negative_controls: Negative control metadata from Step 4.
        profiles_path: Path to the raw JUMP CP profiles parquet file.
        output_dir: Directory for intermediate outputs.
        save_intermediates: Whether to write the matched metadata parquet.

    Returns:
        Polars DataFrame of wells present in both metadata and profiles,
        enriched with additional metadata columns from the profile file.
    """
    print("\n" + "=" * 70)
    print("STEP 5: Match Metadata to Profiles")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 5a. Combine treatment metadata with negative controls
    # ------------------------------------------------------------------
    common_cols = [
        "Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well",
    ]
    # Use only columns that exist in both DataFrames
    filtered_cols = [c for c in common_cols if c in filtered_metadata.columns]
    negcon_cols = [c for c in common_cols if c in negative_controls.columns]
    shared_cols = sorted(set(filtered_cols) & set(negcon_cols))

    df_metadata = pl.concat([
        filtered_metadata.select(shared_cols),
        negative_controls.select(shared_cols),
    ])
    print(f"  Combined metadata: {df_metadata.height:,} rows")

    # ------------------------------------------------------------------
    # 5b. Load profile metadata columns (lazy scan for efficiency)
    # ------------------------------------------------------------------
    print(f"  Loading profiles from: {profiles_path}")
    df_profiles = (
        pl.scan_parquet(profiles_path)
        .select(["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"])
        .collect()
    )
    print(f"  Profile rows: {df_profiles.height:,}")

    # ------------------------------------------------------------------
    # 5c. Remove redlisted plates from metadata
    # ------------------------------------------------------------------
    n_before = df_metadata.height
    df_metadata = df_metadata.filter(
        ~pl.col("Metadata_Plate").is_in(list(EXCLUDED_PLATES.keys()))
    )
    n_removed = n_before - df_metadata.height
    print(f"  Removed {n_removed:,} rows from {len(EXCLUDED_PLATES)} redlisted plates")

    # ------------------------------------------------------------------
    # 5d. Compute intersection using full key (Source + Plate + Well)
    #
    # The full key uniquely identifies a well across the entire JUMP
    # dataset.  We find wells that appear in both our curated metadata
    # and the actual profile file.
    # ------------------------------------------------------------------
    metadata_keys = set(
        df_metadata.with_columns(
            pl.concat_str(
                ["Metadata_Source", "Metadata_Plate", "Metadata_Well"],
                separator="_",
            ).alias("full_key")
        )
        .select("full_key")
        .unique()
        .to_series()
        .to_list()
    )
    profiles_keys = set(
        df_profiles.with_columns(
            pl.concat_str(
                ["Metadata_Source", "Metadata_Plate", "Metadata_Well"],
                separator="_",
            ).alias("full_key")
        )
        .select("full_key")
        .unique()
        .to_series()
        .to_list()
    )

    intersection_keys = metadata_keys & profiles_keys
    only_in_metadata = metadata_keys - profiles_keys
    only_in_profiles = profiles_keys - metadata_keys

    print(f"\n  Unique keys in metadata:  {len(metadata_keys):,}")
    print(f"  Unique keys in profiles:  {len(profiles_keys):,}")
    print(f"  Keys in BOTH:             {len(intersection_keys):,}")
    print(f"  Keys ONLY in metadata:    {len(only_in_metadata):,}")
    print(f"  Keys ONLY in profiles:    {len(only_in_profiles):,}")

    if len(metadata_keys) > 0:
        coverage = len(intersection_keys) / len(metadata_keys) * 100
        print(f"  Metadata coverage:        {coverage:.1f}%")

    # ------------------------------------------------------------------
    # 5e. Filter both metadata sources to the intersection, then combine
    # ------------------------------------------------------------------
    df_filtered_matched = (
        filtered_metadata
        .with_columns(
            pl.concat_str(
                ["Metadata_Source", "Metadata_Plate", "Metadata_Well"],
                separator="_",
            ).alias("full_key")
        )
        .filter(pl.col("full_key").is_in(intersection_keys))
        .drop("full_key")
    )
    df_negative_matched = (
        negative_controls
        .with_columns(
            pl.concat_str(
                ["Metadata_Source", "Metadata_Plate", "Metadata_Well"],
                separator="_",
            ).alias("full_key")
        )
        .filter(pl.col("full_key").is_in(intersection_keys))
        .drop("full_key")
    )

    # Combine on shared columns and deduplicate
    df_matched = pl.concat([
        df_filtered_matched.select(shared_cols),
        df_negative_matched.select(shared_cols),
    ]).unique()

    print(f"\n  Combined matched metadata: {df_matched.height:,} rows")

    # Verify well count matches the set intersection
    matched_keys = set(
        df_matched.with_columns(
            pl.concat_str(
                ["Metadata_Source", "Metadata_Plate", "Metadata_Well"],
                separator="_",
            ).alias("full_key")
        )
        .select("full_key")
        .unique()
        .to_series()
        .to_list()
    )
    assert len(matched_keys) == len(intersection_keys), (
        f"Well count mismatch: {len(matched_keys)} vs {len(intersection_keys)}"
    )
    print(f"  Well count verified: {len(matched_keys):,} unique wells")

    # ------------------------------------------------------------------
    # 5f. Enrich with additional columns from profiles
    #
    # The profile parquet contains per-well annotation columns that are
    # not in our curated metadata.  We left-join to bring them in.
    # ------------------------------------------------------------------
    print("  Enriching with profile metadata columns...")
    df_profiles_for_join = (
        pl.scan_parquet(profiles_path)
        .select([
            "Metadata_Source", "Metadata_Plate", "Metadata_Well",
            "Metadata_JCP2022", "Metadata_broad_sample",
            "Metadata_Symbol", "Metadata_pert_type",
            "Metadata_Perturbation_Type",
        ])
        .unique()
        .collect()
    )

    # Cast join columns to string to avoid type mismatch (cat vs str)
    join_cols = ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]
    df_profiles_for_join = df_profiles_for_join.with_columns([
        pl.col(c).cast(pl.Utf8) for c in join_cols
    ])

    # Normalize Metadata_Perturbation_Type:
    #   - Fill nulls with "compound"
    #   - Replace any value that is not "orf" or "crispr" with "compound"
    # This ensures a clean three-category classification.
    df_profiles_for_join = df_profiles_for_join.with_columns(
        pl.col("Metadata_Perturbation_Type")
        .cast(pl.Utf8)
        .fill_null("compound")
        .alias("Metadata_Perturbation_Type")
    ).with_columns(
        pl.when(pl.col("Metadata_Perturbation_Type").is_in(["orf", "crispr"]))
        .then(pl.col("Metadata_Perturbation_Type"))
        .otherwise(pl.lit("compound"))
        .alias("Metadata_Perturbation_Type")
    )

    # Left join to add profile columns to matched metadata
    df_matched_with_jcp = df_matched.join(
        df_profiles_for_join,
        on=join_cols,
        how="left",
    )

    # Enforce a deterministic column order so the output is stable
    # regardless of which side of the join each column came from.
    canonical_order = [
        "Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well",
        "Metadata_JCP2022", "Metadata_broad_sample", "Metadata_Symbol",
        "Metadata_pert_type", "Metadata_Perturbation_Type",
    ]
    # Keep only columns that actually exist, in the canonical order
    ordered_cols = [c for c in canonical_order if c in df_matched_with_jcp.columns]
    df_matched_with_jcp = df_matched_with_jcp.select(ordered_cols)

    print(
        f"  Matched metadata with profile columns: "
        f"{df_matched_with_jcp.height:,} rows, {df_matched_with_jcp.shape[1]} columns"
    )
    print(f"  Columns: {df_matched_with_jcp.columns}")

    if save_intermediates:
        path = output_dir / "metadata_dataset.parquet"
        df_matched_with_jcp.write_parquet(path)
        print(f"  Saved: {path}")

    return df_matched_with_jcp


# ============================================================================
# Step 6: Filter to >= 4 Replicates
# ============================================================================
# Source: scripts/compare_compound_overlap.py
#
# This step applies the final replicate-count filter to the matched metadata.
# For compound sources (source_2, source_6, source_8 and source_7), only
# JCP2022 IDs with >= 4 replicates (wells) across their respective source
# group are kept.  Non-compound sources (ORF, CRISPR) and negative controls
# are passed through unchanged.
#
# The two compound source groups are treated separately because they have
# different concentration protocols:
#   - source_2/6/8: "high-concentration" compound experiments
#   - source_7:     "low-concentration" compound experiments
#
# It also optionally uses RefChemDB annotations to compute compound-target
# overlap statistics (informational only, not required for the output).
# ============================================================================


def step6_filter_by_replicates(
    metadata_dataset: pl.DataFrame,
    output_dir: Path,
    min_replicates: int = 4,
    refchemdb_path: Optional[str] = None,
    save_intermediates: bool = False,
) -> pl.DataFrame:
    """Filter metadata to keep only compounds with sufficient replicates.

    For compound sources (source_2/6/8 and source_7), retains only JCP2022
    IDs that have >= ``min_replicates`` wells.  ORF/CRISPR sources and
    negative controls are kept regardless.

    Args:
        metadata_dataset: Matched metadata from Step 5.
        output_dir: Directory for the final output.
        min_replicates: Minimum number of replicates required (default 4).
        refchemdb_path: Optional path to refchemdb_conf_jump_matched.parquet
            for computing overlap statistics.  Not required for filtering.
        save_intermediates: Whether to write additional intermediate files.

    Returns:
        Final filtered Polars DataFrame, saved as
        ``metadata_dataset_filtered_4reps.parquet``.
    """
    print("\n" + "=" * 70)
    print(f"STEP 6: Filter to >= {min_replicates} Replicates")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 6a. Identify JCP IDs with >= min_replicates in compound sources
    #
    # source_2/6/8 are pooled together (same high-concentration protocol),
    # source_7 is treated independently (lower concentration).
    # ------------------------------------------------------------------
    cpg_sources = ["source_2", "source_6", "source_8"]
    s7_sources = ["source_7"]

    cpg_jcps = _get_jcpids_with_min_replicates(
        metadata_dataset, cpg_sources, min_replicates
    )
    s7_jcps = _get_jcpids_with_min_replicates(
        metadata_dataset, s7_sources, min_replicates
    )

    print(f"  source_2/6/8 JCPIDs with >= {min_replicates} reps: {len(cpg_jcps):,}")
    print(f"  source_7 JCPIDs with >= {min_replicates} reps:     {len(s7_jcps):,}")
    print(f"  Overlap (in both groups): {len(cpg_jcps & s7_jcps):,}")

    if save_intermediates:
        overlap_dir = output_dir / "dataset_overlaps"
        overlap_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"Metadata_JCP2022": sorted(cpg_jcps)}).write_parquet(
            overlap_dir / "jcpids_source_2_6_8_4reps.parquet"
        )
        pl.DataFrame({"Metadata_JCP2022": sorted(s7_jcps)}).write_parquet(
            overlap_dir / "jcpids_source_7_4reps.parquet"
        )
        print(f"  Saved JCP ID lists to: {overlap_dir}")

    # ------------------------------------------------------------------
    # 6b. Apply the replicate filter
    #
    # Keep a row if ANY of these conditions is true:
    #   - It is from source_2/6/8 AND its JCP2022 is in cpg_jcps
    #   - It is from source_7 AND its JCP2022 is in s7_jcps
    #   - It is from any non-compound source (ORF, CRISPR, etc.)
    #   - It is a negative control (Metadata_Perturbation_Type ~ negcon)
    # ------------------------------------------------------------------
    compound_sources_all = cpg_sources + s7_sources

    df_filtered = metadata_dataset.filter(
        # Compound wells in source_2/6/8 meeting the replicate threshold
        (
            pl.col("Metadata_Source").cast(pl.Utf8).is_in(cpg_sources)
            & pl.col("Metadata_JCP2022").is_in(list(cpg_jcps))
        )
        # Compound wells in source_7 meeting the replicate threshold
        | (
            (pl.col("Metadata_Source").cast(pl.Utf8) == "source_7")
            & pl.col("Metadata_JCP2022").is_in(list(s7_jcps))
        )
        # Non-compound sources: keep all (ORF source_4, CRISPR source_13, etc.)
        | ~pl.col("Metadata_Source").cast(pl.Utf8).is_in(compound_sources_all)
        # Negative controls from any source
        | pl.col("Metadata_Perturbation_Type").str.contains("(?i)negcon")
    )

    # ------------------------------------------------------------------
    # 6c. Add a source-group column for downstream convenience
    #
    # This allows easy group_by operations in downstream analyses without
    # needing to remember which sources belong to which protocol.
    # ------------------------------------------------------------------
    df_filtered = df_filtered.with_columns(
        pl.when(pl.col("Metadata_Source").cast(pl.Utf8).is_in(cpg_sources))
        .then(pl.lit("group_high"))
        .when(pl.col("Metadata_Source").cast(pl.Utf8) == "source_7")
        .then(pl.lit("group_low"))
        .when(pl.col("Metadata_Source").cast(pl.Utf8) == "source_4")
        .then(pl.lit("group_orf"))
        .when(pl.col("Metadata_Source").cast(pl.Utf8) == "source_13")
        .then(pl.lit("group_crispr"))
        .otherwise(pl.lit("group_other"))
        .alias("Metadata_Group")
    )

    print(f"\n  Original rows:  {metadata_dataset.height:,}")
    print(f"  Filtered rows:  {df_filtered.height:,}")
    print(f"  Removed rows:   {metadata_dataset.height - df_filtered.height:,}")

    group_counts = (
        df_filtered.group_by("Metadata_Group")
        .agg(pl.len().alias("count"))
        .sort("Metadata_Group")
    )
    print("\n  Group counts:")
    for row in group_counts.iter_rows(named=True):
        print(f"    {row['Metadata_Group']}: {row['count']:,}")

    # ------------------------------------------------------------------
    # 6d. Optional: RefChemDB overlap statistics
    # ------------------------------------------------------------------
    if refchemdb_path and Path(refchemdb_path).exists():
        _print_refchemdb_overlap(
            df_filtered, refchemdb_path, cpg_jcps, s7_jcps, min_replicates
        )

    # ------------------------------------------------------------------
    # 6e. Save the final output
    # ------------------------------------------------------------------
    output_path = output_dir / "metadata_dataset_filtered_4reps.parquet"
    df_filtered.write_parquet(output_path)
    print(f"\n  FINAL OUTPUT: {output_path}")
    print(f"  Rows: {df_filtered.height:,}")
    print(f"  Columns: {df_filtered.columns}")

    return df_filtered


def _get_jcpids_with_min_replicates(
    df: pl.DataFrame,
    sources: list[str],
    min_reps: int,
) -> set[str]:
    """Get JCP2022 IDs with >= min_reps replicates across given sources.

    Args:
        df: Metadata DataFrame with ``Metadata_Source`` and
            ``Metadata_JCP2022`` columns.
        sources: List of source identifiers to filter on.
        min_reps: Minimum replicate count.

    Returns:
        Set of JCP2022 IDs meeting the replicate threshold.
    """
    counts = (
        df.filter(pl.col("Metadata_Source").cast(pl.Utf8).is_in(sources))
        .group_by("Metadata_JCP2022")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") >= min_reps)
    )
    return set(counts["Metadata_JCP2022"].drop_nulls().to_list())


def _print_refchemdb_overlap(
    df_filtered: pl.DataFrame,
    refchemdb_path: str,
    cpg_jcps: set[str],
    s7_jcps: set[str],
    min_replicates: int,
) -> None:
    """Print overlap statistics between the filtered metadata and RefChemDB.

    This is informational only and does not modify the output dataset.
    It helps quantify how many annotated compound-target relationships
    are represented in the final filtered dataset.

    Args:
        df_filtered: The final filtered metadata DataFrame.
        refchemdb_path: Path to the RefChemDB parquet file.
        cpg_jcps: Set of JCP IDs passing the replicate filter in source_2/6/8.
        s7_jcps: Set of JCP IDs passing the replicate filter in source_7.
        min_replicates: The replicate threshold used.
    """
    print("\n  --- RefChemDB Overlap Statistics ---")
    df_refchemdb = pl.read_parquet(refchemdb_path)

    refchemdb_compounds = set(
        df_refchemdb["Metadata_JCP2022"].drop_nulls().unique().to_list()
    )

    # Compound overlap
    metadata_compounds = set(
        df_filtered
        .filter(~pl.col("Metadata_Perturbation_Type").is_in(["orf", "crispr"]))
        .select("Metadata_JCP2022")
        .drop_nulls()
        .unique()
        .to_series()
        .to_list()
    )
    cmpd_overlap = metadata_compounds & refchemdb_compounds
    print(f"  Metadata compounds: {len(metadata_compounds):,}")
    print(f"  RefChemDB compounds: {len(refchemdb_compounds):,}")
    if refchemdb_compounds:
        pct = len(cmpd_overlap) / len(refchemdb_compounds) * 100
        print(f"  Overlap: {len(cmpd_overlap):,} ({pct:.1f}% of RefChemDB)")

    # Target overlap via JCP2022 -> target mapping in RefChemDB
    if "target" in df_refchemdb.columns:
        jcp_to_targets = (
            df_refchemdb.select(["Metadata_JCP2022", "target"])
            .drop_nulls()
            .unique()
        )

        cpg_targets = set(
            jcp_to_targets
            .filter(pl.col("Metadata_JCP2022").is_in(list(cpg_jcps)))
            ["target"]
            .to_list()
        )
        s7_targets = set(
            jcp_to_targets
            .filter(pl.col("Metadata_JCP2022").is_in(list(s7_jcps)))
            ["target"]
            .to_list()
        )
        print(
            f"\n  Targets reachable from source_2/6/8 "
            f"(>= {min_replicates} reps): {len(cpg_targets):,}"
        )
        print(
            f"  Targets reachable from source_7 "
            f"(>= {min_replicates} reps): {len(s7_targets):,}"
        )
        print(f"  Target overlap: {len(cpg_targets & s7_targets):,}")


# ============================================================================
# Step 7: Build Target Lists (MOTIVE + RefChemDB)
# ============================================================================
# This step adds two target annotation columns to the filtered metadata:
#   - Metadata_MOTIVE_target: Gene targets from the MOTIVE compound-gene
#     annotation database (annotations_compound_gene.parquet).  For ORF/CRISPR
#     wells, the gene symbol (Metadata_Symbol) is used directly.
#   - Metadata_RefChemDB_target: Gene targets from the RefChemDB curated
#     database, filtered to Tier1-3 confidence.  Higher confidence than MOTIVE
#     but lower coverage.
#
# Compounds are mapped via: JCP2022 -> InChIKey (from Step 1 mapping) ->
# gene targets (from annotation databases).  Targets are aggregated as
# pipe-separated sorted strings per JCP2022 ID.
#
# Wells without target annotations (negcons, unannotated compounds) receive
# the value "unknown".
# ============================================================================


def step7_build_target_list(
    df_filtered: pl.DataFrame,
    annotations_cg_path: str,
    output_dir: Path,
    refchemdb_path: Optional[str] = None,
    save_intermediates: bool = False,
) -> pl.DataFrame:
    """Add target annotation columns to the filtered metadata.

    Args:
        df_filtered: Filtered metadata from Step 6.
        annotations_cg_path: Path to compound-gene annotations parquet
            (MOTIVE database).
        output_dir: Directory containing inchikey_to_jcp2022_mapping_combined.csv
            and for saving outputs.
        refchemdb_path: Optional path to refchemdb_conf_jump_matched.parquet.
        save_intermediates: Whether to write intermediate files.

    Returns:
        DataFrame with added Metadata_MOTIVE_target and optionally
        Metadata_RefChemDB_target columns.
    """
    print("\n" + "=" * 70)
    print("STEP 7: Build Target Lists (MOTIVE + RefChemDB)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 7a. Load InChIKey -> JCP2022 mapping from Step 1 output
    # ------------------------------------------------------------------
    mapping_path = output_dir / "inchikey_to_jcp2022_mapping_combined.csv"
    print(f"  Loading InChIKey -> JCP2022 mapping from: {mapping_path}")
    jcp_mapping = pl.read_csv(str(mapping_path))
    print(f"    Entries: {jcp_mapping.height:,}")

    # ------------------------------------------------------------------
    # 7b. Build MOTIVE compound targets
    #
    # Join: annotations_cg (inchikey -> gene target)
    #   via jcp_mapping (InChIKey_Connectivity -> JCP2022)
    #   to get: JCP2022 -> gene targets
    # ------------------------------------------------------------------
    print("\n  Building MOTIVE compound targets...")
    df_cg = pl.read_parquet(annotations_cg_path)
    print(f"    Compound-gene annotations: {df_cg.height:,} rows")

    # Extract connectivity layer from annotation InChIKeys
    df_cg = df_cg.filter(pl.col("inchikey").is_not_null()).with_columns(
        pl.col("inchikey")
        .str.split("-")
        .list.first()
        .alias("InChIKey_Connectivity")
    )

    # Join annotations with JCP mapping via connectivity layer
    cg_with_jcp = df_cg.join(
        jcp_mapping.select("InChIKey_Connectivity", "Metadata_JCP2022"),
        on="InChIKey_Connectivity",
        how="inner",
    )
    print(f"    Annotations matched to JCP2022: {cg_with_jcp.height:,}")

    # Aggregate unique sorted targets per JCP2022 as pipe-separated string
    motive_targets = (
        cg_with_jcp
        .filter(pl.col("target").is_not_null())
        .group_by("Metadata_JCP2022")
        .agg(
            pl.col("target")
            .unique()
            .sort()
            .str.join("|")
            .alias("Metadata_MOTIVE_target")
        )
    )
    print(f"    Unique JCP2022 IDs with MOTIVE targets: {motive_targets.height:,}")

    # ------------------------------------------------------------------
    # 7c. Merge MOTIVE targets into metadata
    #
    # For ORF/CRISPR: use Metadata_Symbol directly as the target
    # For compounds: use the MOTIVE mapping
    # For negcons/unknown: "unknown"
    # ------------------------------------------------------------------
    print("\n  Merging MOTIVE targets into metadata...")

    # Drop any pre-existing target columns (idempotency for --skip-to 7 reruns)
    drop_cols = [
        c for c in df_filtered.columns
        if c in ("Metadata_MOTIVE_target", "Metadata_RefChemDB_target",
                 "Metadata_MOTIVE_target_right", "Metadata_RefChemDB_target_right")
    ]
    if drop_cols:
        df_filtered = df_filtered.drop(drop_cols)
        print(f"    Dropped pre-existing columns: {drop_cols}")

    # Cast JCP2022 columns to string for joining (metadata may have categorical)
    df = df_filtered.with_columns(
        pl.col("Metadata_JCP2022").cast(pl.Utf8).alias("_jcp_str")
    )
    motive_targets = motive_targets.with_columns(
        pl.col("Metadata_JCP2022").cast(pl.Utf8)
    )

    # Join compound targets
    df = df.join(
        motive_targets,
        left_on="_jcp_str",
        right_on="Metadata_JCP2022",
        how="left",
    )

    # Priority: negcons -> "unknown", ORF/CRISPR treatments -> gene symbol,
    # compounds with annotation -> MOTIVE target, else -> "unknown"
    df = df.with_columns(
        pl.when(pl.col("Metadata_pert_type").cast(pl.Utf8) == "negcon")
        .then(pl.lit("unknown"))
        .when(
            pl.col("Metadata_Perturbation_Type").cast(pl.Utf8).is_in(["orf", "crispr"])
        )
        .then(pl.col("Metadata_Symbol").cast(pl.Utf8))
        .when(pl.col("Metadata_MOTIVE_target").is_not_null())
        .then(pl.col("Metadata_MOTIVE_target"))
        .otherwise(pl.lit("unknown"))
        .alias("Metadata_MOTIVE_target")
    ).drop("_jcp_str")

    # Report coverage
    n_with_target = df.filter(pl.col("Metadata_MOTIVE_target") != "unknown").height
    n_total = df.height
    print(f"    Wells with MOTIVE target: {n_with_target:,} / {n_total:,}")

    # Per perturbation type breakdown
    for ptype in ["orf", "crispr", "compound"]:
        subset = df.filter(
            pl.col("Metadata_Perturbation_Type").cast(pl.Utf8) == ptype
        )
        annotated = subset.filter(
            pl.col("Metadata_MOTIVE_target") != "unknown"
        ).height
        unique_jcps = subset.filter(
            pl.col("Metadata_MOTIVE_target") != "unknown"
        ).select("Metadata_JCP2022").n_unique()
        print(
            f"    {ptype}: {annotated:,} wells annotated "
            f"({unique_jcps:,} unique JCPs)"
        )

    # ------------------------------------------------------------------
    # 7d. Build RefChemDB targets (if provided)
    # ------------------------------------------------------------------
    if refchemdb_path and Path(refchemdb_path).exists():
        print(f"\n  Building RefChemDB targets from: {refchemdb_path}")
        df_refchemdb = pl.read_parquet(refchemdb_path)
        print(f"    Total RefChemDB rows: {df_refchemdb.height:,}")

        # Filter to Tier1-3 (exclude "Excluded")
        df_ref_filtered = df_refchemdb.filter(
            pl.col("WithinModalityTier").is_in(["Tier1", "Tier2", "Tier3"])
        )
        print(f"    After Tier1-3 filter: {df_ref_filtered.height:,}")

        # Aggregate unique sorted targets per JCP2022
        refchemdb_targets = (
            df_ref_filtered
            .filter(pl.col("target").is_not_null())
            .group_by("Metadata_JCP2022")
            .agg(
                pl.col("target")
                .unique()
                .sort()
                .str.join("|")
                .alias("Metadata_RefChemDB_target")
            )
        )
        print(
            f"    Unique JCP2022 IDs with RefChemDB targets: "
            f"{refchemdb_targets.height:,}"
        )

        # Merge into metadata
        df = df.with_columns(
            pl.col("Metadata_JCP2022").cast(pl.Utf8).alias("_jcp_str")
        )
        refchemdb_targets = refchemdb_targets.with_columns(
            pl.col("Metadata_JCP2022").cast(pl.Utf8)
        )

        df = df.join(
            refchemdb_targets,
            left_on="_jcp_str",
            right_on="Metadata_JCP2022",
            how="left",
        )

        # Priority: negcons -> "unknown", ORF/CRISPR treatments -> gene symbol,
        # compounds with annotation -> RefChemDB target, else -> "unknown"
        df = df.with_columns(
            pl.when(pl.col("Metadata_pert_type").cast(pl.Utf8) == "negcon")
            .then(pl.lit("unknown"))
            .when(
                pl.col("Metadata_Perturbation_Type")
                .cast(pl.Utf8)
                .is_in(["orf", "crispr"])
            )
            .then(pl.col("Metadata_Symbol").cast(pl.Utf8))
            .when(pl.col("Metadata_RefChemDB_target").is_not_null())
            .then(pl.col("Metadata_RefChemDB_target"))
            .otherwise(pl.lit("unknown"))
            .alias("Metadata_RefChemDB_target")
        ).drop("_jcp_str")

        # Report coverage
        n_with_ref = df.filter(
            pl.col("Metadata_RefChemDB_target") != "unknown"
        ).height
        print(f"    Wells with RefChemDB target: {n_with_ref:,} / {n_total:,}")

        for ptype in ["orf", "crispr", "compound"]:
            subset = df.filter(
                pl.col("Metadata_Perturbation_Type").cast(pl.Utf8) == ptype
            )
            annotated = subset.filter(
                pl.col("Metadata_RefChemDB_target") != "unknown"
            ).height
            unique_jcps = subset.filter(
                pl.col("Metadata_RefChemDB_target") != "unknown"
            ).select("Metadata_JCP2022").n_unique()
            print(
                f"    {ptype}: {annotated:,} wells annotated "
                f"({unique_jcps:,} unique JCPs)"
            )
    else:
        if refchemdb_path:
            print(f"\n  WARNING: RefChemDB file not found: {refchemdb_path}")
        else:
            print("\n  Skipping RefChemDB targets (--refchemdb not provided)")

    # ------------------------------------------------------------------
    # 7e. Add boolean negcon column for copairs compatibility
    # ------------------------------------------------------------------
    if "Metadata_pert_type" in df.columns:
        df = df.with_columns(
            (pl.col("Metadata_pert_type").cast(pl.Utf8) == "negcon").alias("Metadata_negcon")
        )
        n_neg = df["Metadata_negcon"].sum()
        print(f"\n  Added Metadata_negcon (boolean): {n_neg:,} negcons")

    # ------------------------------------------------------------------
    # 7f. Save the updated metadata
    # ------------------------------------------------------------------
    output_path = output_dir / "metadata_dataset_filtered_4reps.parquet"
    df.write_parquet(output_path)
    print(f"\n  Saved updated metadata: {output_path}")
    print(f"  Columns: {df.columns}")
    print(f"  Rows: {df.height:,}")

    return df


# ============================================================================
# Step 8: Filter Raw CellProfiler Profiles to Match Metadata
# ============================================================================
# This step takes the full JUMP CellProfiler profiles (well-level, ~3700
# features) and filters them to the exact wells present in the final
# metadata.  The output is formatted to match the DL feature parquets
# (Metadata_id as first column, features in the middle, metadata at end).
#
# This ensures the CP baseline uses the same wells as the DL models,
# enabling fair comparison in the normalization sweeps.
# ============================================================================


def step8_filter_cp_profiles(
    metadata_path: Path,
    cp_profiles_path: str,
    output_path: Path,
) -> None:
    """Filter raw CellProfiler profiles to wells in the metadata.

    Args:
        metadata_path: Path to the final metadata parquet (from Step 7).
        cp_profiles_path: Path to the raw JUMP CP profiles parquet.
        output_path: Path for the filtered output parquet.
    """
    print("\n" + "=" * 70)
    print("STEP 8: Filter Raw CellProfiler Profiles")
    print("=" * 70)

    join_cols = ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]

    # ------------------------------------------------------------------
    # 8a. Load metadata (wells of interest)
    # ------------------------------------------------------------------
    print(f"  Loading metadata: {metadata_path}")
    meta_df = pl.read_parquet(str(metadata_path))
    meta_df = meta_df.with_columns([pl.col(c).cast(pl.Utf8) for c in join_cols])
    print(f"    Metadata wells: {meta_df.height:,}")

    # ------------------------------------------------------------------
    # 8b. Load source CP profiles (feature + join columns only)
    # ------------------------------------------------------------------
    print(f"  Loading CP profiles: {cp_profiles_path}")
    source_df = pl.read_parquet(cp_profiles_path)
    print(f"    Source shape: {source_df.shape}")

    # Keep only feature columns (non-Metadata_) + join columns
    feature_cols = [c for c in source_df.columns if not c.startswith("Metadata_")]
    source_df = source_df.select(join_cols + feature_cols)
    source_df = source_df.with_columns([pl.col(c).cast(pl.Utf8) for c in join_cols])
    print(f"    Feature columns: {len(feature_cols):,}")

    # ------------------------------------------------------------------
    # 8c. Inner join to filter to metadata wells
    # ------------------------------------------------------------------
    print("  Joining source features with metadata...")
    filtered_df = source_df.join(meta_df, on=join_cols, how="inner")
    print(f"    Matched wells: {filtered_df.height:,}")

    if len(filtered_df) == 0:
        print("  ERROR: No matching wells found!")
        return

    # ------------------------------------------------------------------
    # 8d. Create Metadata_id and add model/dataset/compression columns
    # ------------------------------------------------------------------
    filtered_df = filtered_df.with_columns([
        (
            pl.col("Metadata_Source") + "__" +
            pl.col("Metadata_Batch") + "__" +
            pl.col("Metadata_Plate") + "__" +
            pl.col("Metadata_Well")
        ).alias("Metadata_id"),
        pl.lit("cellprofiler_raw").alias("Metadata_model"),
        pl.lit("jump_lite").alias("Metadata_dataset"),
        pl.lit("none").alias("Metadata_compression"),
    ])

    # ------------------------------------------------------------------
    # 8e. Arrange columns: Metadata_id, features, then trailing metadata
    # ------------------------------------------------------------------
    trailing_metadata = [
        "Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well",
        "Metadata_pert_type", "Metadata_Perturbation_Type",
        "Metadata_JCP2022", "Metadata_broad_sample", "Metadata_Symbol",
        "Metadata_model", "Metadata_dataset", "Metadata_compression",
    ]
    final_order = ["Metadata_id"] + feature_cols + [
        c for c in trailing_metadata if c in filtered_df.columns
    ]
    filtered_df = filtered_df.select(final_order)

    # ------------------------------------------------------------------
    # 8f. Save output
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.write_parquet(output_path)

    n_meta = len([c for c in filtered_df.columns if c.startswith("Metadata_")])
    n_feat = len(filtered_df.columns) - n_meta
    print(f"\n  Saved: {output_path}")
    print(f"  Rows: {filtered_df.height:,}")
    print(f"  Feature columns: {n_feat:,}")
    print(f"  Metadata columns: {n_meta}")


# ============================================================================
# CLI and Main Orchestration
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build metadata_dataset_filtered_4reps.parquet via the "
            "8-step JUMP pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input paths ---
    input_group = parser.add_argument_group("Input paths")
    input_group.add_argument(
        "--annotations-db",
        type=str,
        default="/work/datasets/jump_core/annotations/jump_metadata.duckdb",
        help="Path to jump_metadata.duckdb (Step 1). Default: %(default)s",
    )
    input_group.add_argument(
        "--annotations-cc",
        type=str,
        default="/work/datasets/jump_core/annotations/annotations_compound_compound.parquet",
        help=(
            "Path to compound-compound annotations parquet (Step 1). "
            "Default: %(default)s"
        ),
    )
    input_group.add_argument(
        "--annotations-cg",
        type=str,
        default="/work/datasets/jump_core/annotations/annotations_compound_gene.parquet",
        help=(
            "Path to compound-gene annotations parquet (Step 1). "
            "Default: %(default)s"
        ),
    )
    input_group.add_argument(
        "--profiles",
        type=str,
        default="/work/datasets/jump_core_annotated/raw_jump_CP_profiles/profiles.parquet",
        help="Path to raw JUMP CP profiles parquet (Step 5). Default: %(default)s",
    )
    input_group.add_argument(
        "--refchemdb",
        type=str,
        default="/home/jfredinh/projects/JUMP_core/metadata/refchemdb_conf_jump_matched.parquet",
        help=(
            "Path to refchemdb_conf_jump_matched.parquet (Steps 6-7). "
            "Default: %(default)s"
        ),
    )
    input_group.add_argument(
        "--cp-output",
        type=str,
        default=None,
        help=(
            "Output path for filtered CP profiles parquet (Step 8). "
            "Default: <output-dir>/../data/features/jump_lite/"
            "cellprofiler_raw_jump_lite_raw_features.parquet"
        ),
    )

    # --- Intermediate input paths (for --skip-to) ---
    intermediate_group = parser.add_argument_group(
        "Intermediate paths (for --skip-to)"
    )
    intermediate_group.add_argument(
        "--jcp-mapping-csv",
        type=str,
        default=None,
        help=(
            "Path to existing inchikey_to_jcp2022_mapping_combined.csv. "
            "Used when --skip-to >= 2 to avoid re-running Step 1."
        ),
    )
    intermediate_group.add_argument(
        "--raw-metadata",
        type=str,
        default=None,
        help=(
            "Path to existing metadata.parquet (raw JUMP metadata). "
            "Used when --skip-to >= 3 to avoid re-running Steps 1-2."
        ),
    )
    intermediate_group.add_argument(
        "--filtered-metadata",
        type=str,
        default=None,
        help=(
            "Path to existing metadata_filtered.parquet. "
            "Used when --skip-to >= 4 to avoid re-running Steps 1-3."
        ),
    )
    intermediate_group.add_argument(
        "--negative-controls",
        type=str,
        default=None,
        help=(
            "Path to existing metadata_negative_controls.parquet. "
            "Used when --skip-to >= 5 to avoid re-running Steps 1-4."
        ),
    )
    intermediate_group.add_argument(
        "--metadata-dataset",
        type=str,
        default=None,
        help=(
            "Path to existing metadata_dataset.parquet. "
            "Used when --skip-to >= 6 to avoid re-running Steps 1-5."
        ),
    )

    # --- Output configuration ---
    output_group = parser.add_argument_group("Output configuration")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="/home/jfredinh/projects/JUMP_core/metadata",
        help="Output directory for all files. Default: %(default)s",
    )

    # --- Pipeline control ---
    control_group = parser.add_argument_group("Pipeline control")
    control_group.add_argument(
        "--skip-to",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help=(
            "Resume from a specific step (1-8). Requires that intermediate "
            "files for prior steps exist (via --save-intermediates or "
            "explicit --*-metadata paths). Default: %(default)s"
        ),
    )
    control_group.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save intermediate outputs after each step.",
    )

    # --- Thresholds ---
    threshold_group = parser.add_argument_group("Thresholds")
    threshold_group.add_argument(
        "--min-fill-rate",
        type=float,
        default=25.0,
        help="Minimum plate fill rate %% for Step 3. Default: %(default)s",
    )
    threshold_group.add_argument(
        "--min-replicates",
        type=int,
        default=4,
        help=(
            "Minimum replicates for compound filtering in Step 6. "
            "Default: %(default)s"
        ),
    )
    threshold_group.add_argument(
        "--negcon-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of negative controls to sample per plate in Step 4 "
            "(0.0-1.0). Overrides the per-modality defaults in "
            "MODALITY_FRACTION. Use 1.0 to keep all negcons, 0.5 to sample "
            "half. If not provided, uses the defaults (all 1.0)."
        ),
    )
    threshold_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for negative control sampling in Step 4. "
            "Default: %(default)s"
        ),
    )

    return parser.parse_args()


def _load_intermediate_or_fail(
    path: Optional[str],
    step_name: str,
) -> pl.DataFrame:
    """Load a parquet intermediate file, or exit with a helpful error."""
    if path is None or not Path(path).exists():
        resolved = path if path else "(not provided)"
        print(
            f"\nERROR: --skip-to requires intermediate file for {step_name}.\n"
            f"  Expected: {resolved}\n"
            f"  Either run earlier steps first with --save-intermediates, "
            f"or supply the path explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  Loading {step_name} from: {path}")
    return pl.read_parquet(path)


def main() -> None:
    """Orchestrate the 6-step metadata pipeline."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skip_to = args.skip_to

    print("=" * 70)
    print("JUMP Metadata Dataset Builder")
    print("=" * 70)
    print(f"  Output directory:    {output_dir}")
    print(f"  Starting from step:  {skip_to}")
    print(f"  Save intermediates:  {args.save_intermediates}")
    print(f"  Min fill rate:       {args.min_fill_rate}%")
    print(f"  Min replicates:      {args.min_replicates}")
    print(f"  Negcon fraction:     {args.negcon_fraction or 'default (all 1.0)'}")
    print(f"  Random seed:         {args.seed}")
    print()

    # ---- Step 1: InChIKey -> JCP2022 Mapping ----
    if skip_to <= 1:
        jcp_mapping = step1_inchikey_to_jcp_mapping(
            db_path=args.annotations_db,
            annotations_cc_path=args.annotations_cc,
            annotations_cg_path=args.annotations_cg,
            output_dir=output_dir,
            save_intermediates=args.save_intermediates,
        )
    else:
        # Try to load from explicit path or default location
        csv_path = (
            args.jcp_mapping_csv
            or str(output_dir / "inchikey_to_jcp2022_mapping_combined.csv")
        )
        if Path(csv_path).exists():
            print(f"  Skipping Step 1: loading JCP mapping from {csv_path}")
            jcp_mapping = pl.read_csv(csv_path)
        else:
            # For skip_to >= 3 we do not need the Step 1 output
            jcp_mapping = None

    # ---- Step 2: Generate Raw JUMP Metadata ----
    if skip_to <= 2:
        if jcp_mapping is None:
            print(
                "\nERROR: Step 2 requires the JCP mapping from Step 1. "
                "Provide --jcp-mapping-csv or run from Step 1.",
                file=sys.stderr,
            )
            sys.exit(1)
        metadata_all = step2_generate_raw_metadata(
            jcp_mapping=jcp_mapping,
            db_path=args.annotations_db,
            output_dir=output_dir,
            save_intermediates=args.save_intermediates,
        )
    else:
        raw_meta_path = args.raw_metadata or str(output_dir / "metadata.parquet")
        if Path(raw_meta_path).exists() and skip_to == 3:
            print(f"  Skipping Step 2: loading raw metadata from {raw_meta_path}")
            metadata_all = pl.read_parquet(raw_meta_path)
        else:
            metadata_all = None

    # ---- Step 3: Filter Metadata (Plate Fill Rate) ----
    if skip_to <= 3:
        if metadata_all is None:
            meta_path = args.raw_metadata or str(output_dir / "metadata.parquet")
            metadata_all = _load_intermediate_or_fail(
                meta_path, "raw metadata (Step 2)"
            )
        filtered_metadata = step3_filter_metadata(
            metadata_all=metadata_all,
            db_path=args.annotations_db,
            output_dir=output_dir,
            min_fill_rate=args.min_fill_rate,
            save_intermediates=args.save_intermediates,
        )
    else:
        filt_path = (
            args.filtered_metadata
            or str(output_dir / "metadata_filtered.parquet")
        )
        if Path(filt_path).exists():
            print(
                f"  Skipping Step 3: loading filtered metadata from {filt_path}"
            )
            filtered_metadata = pl.read_parquet(filt_path)
        else:
            filtered_metadata = _load_intermediate_or_fail(
                filt_path, "filtered metadata (Step 3)"
            )

    # ---- Step 4: Prepare Negative Controls ----
    if skip_to <= 4:
        negative_controls = step4_prepare_negative_controls(
            filtered_metadata=filtered_metadata,
            db_path=args.annotations_db,
            output_dir=output_dir,
            seed=args.seed,
            negcon_fraction=args.negcon_fraction,
            save_intermediates=args.save_intermediates,
        )
    else:
        negcon_path = (
            args.negative_controls
            or str(output_dir / "metadata_negative_controls.parquet")
        )
        if Path(negcon_path).exists():
            print(
                f"  Skipping Step 4: loading negative controls from {negcon_path}"
            )
            negative_controls = pl.read_parquet(negcon_path)
        else:
            negative_controls = _load_intermediate_or_fail(
                negcon_path, "negative controls (Step 4)"
            )

    # ---- Step 5: Match Metadata to Profiles ----
    if skip_to <= 5:
        metadata_dataset = step5_match_metadata_to_profiles(
            filtered_metadata=filtered_metadata,
            negative_controls=negative_controls,
            profiles_path=args.profiles,
            output_dir=output_dir,
            save_intermediates=args.save_intermediates,
        )
    else:
        dataset_path = (
            args.metadata_dataset
            or str(output_dir / "metadata_dataset.parquet")
        )
        if Path(dataset_path).exists():
            print(
                f"  Skipping Step 5: loading metadata_dataset from {dataset_path}"
            )
            metadata_dataset = pl.read_parquet(dataset_path)
        else:
            metadata_dataset = _load_intermediate_or_fail(
                dataset_path, "metadata_dataset (Step 5)"
            )

    # ---- Step 6: Filter to >= N Replicates ----
    if skip_to <= 6:
        df_filtered_final = step6_filter_by_replicates(
            metadata_dataset=metadata_dataset,
            output_dir=output_dir,
            min_replicates=args.min_replicates,
            refchemdb_path=args.refchemdb,
            save_intermediates=args.save_intermediates,
        )
    else:
        filtered_path = str(
            output_dir / "metadata_dataset_filtered_4reps.parquet"
        )
        print(f"  Skipping Step 6: loading from {filtered_path}")
        df_filtered_final = pl.read_parquet(filtered_path)

    # ---- Step 7: Build Target Lists ----
    if skip_to <= 7:
        df_final = step7_build_target_list(
            df_filtered=df_filtered_final,
            annotations_cg_path=args.annotations_cg,
            output_dir=output_dir,
            refchemdb_path=args.refchemdb,
            save_intermediates=args.save_intermediates,
        )
    else:
        filtered_path = str(
            output_dir / "metadata_dataset_filtered_4reps.parquet"
        )
        print(f"  Skipping Step 7: loading from {filtered_path}")
        df_final = pl.read_parquet(filtered_path)

    # ---- Step 8: Filter Raw CellProfiler Profiles ----
    cp_output = args.cp_output or str(
        output_dir.parent / "data" / "features" / "jump_lite"
        / "cellprofiler_raw_jump_lite_raw_features.parquet"
    )
    metadata_path = output_dir / "metadata_dataset_filtered_4reps.parquet"
    step8_filter_cp_profiles(
        metadata_path=metadata_path,
        cp_profiles_path=args.profiles,
        output_path=Path(cp_output),
    )

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    final_path = output_dir / "metadata_dataset_filtered_4reps.parquet"
    print(f"  Final output: {final_path}")
    print(f"  Total rows:   {df_final.height:,}")
    print(f"  Columns:      {df_final.columns}")
    print(
        f"  Unique JCP2022 IDs: "
        f"{df_final.select('Metadata_JCP2022').drop_nulls().n_unique():,}"
    )
    print(
        f"  Unique plates:      "
        f"{df_final.select('Metadata_Plate').n_unique():,}"
    )

    # Per-group summary
    group_summary = (
        df_final.group_by("Metadata_Group")
        .agg([
            pl.len().alias("wells"),
            pl.col("Metadata_Plate").n_unique().alias("plates"),
            pl.col("Metadata_JCP2022").n_unique().alias("jcp_ids"),
        ])
        .sort("Metadata_Group")
    )
    print("\n  Per-group summary:")
    print(group_summary)


if __name__ == "__main__":
    main()
