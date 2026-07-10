#!/usr/bin/env python3
"""Create cell-count baseline features from CellProfiler parquets.

For each CP parquet, extracts cell-count proxy columns as the only two
features, sets Metadata_model to "cell_count", and writes a new parquet.

This serves as a lower-bound baseline: any biological signal recovered by
the normalization pipeline from cell counts alone is attributable to cell-count
variation rather than morphological features.

Supports two datasets:
  - target2: Uses Metadata_n_cells / Metadata_n_nuclei (7 codec variants)
  - jump_lite: Uses Cytoplasm_Number_Object_Number / Nuclei_Number_Object_Number
               as proxies (single file, compression="none")
"""

from pathlib import Path

import polars as pl

# ── target2 configuration ──────────────────────────────────────────
TARGET2_CODECS = [
    "zstd",
    "jpegxl_lossy_hq",
    "jpegxl_lossy_mq",
    "jpegxl_lossy_lq",
    "jpegxl_lossy_d2_e8",
    "jpegxl_lossy_d10",
    "jpegxl_lossy_effort_3",
]

TARGET2_FEATURE_DIR = Path("data/features/jump_target2_4plate")

TARGET2_METADATA_COLS = [
    "Metadata_id",
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Site",
    "Metadata_model",
    "Metadata_dataset",
    "Metadata_compression",
]

TARGET2_FEATURE_RENAMES = {
    "Metadata_n_cells": "cell_count",
    "Metadata_n_nuclei": "nuclei_count",
}

# ── jump_lite configuration ────────────────────────────────────────
JUMP_LITE_FEATURE_DIR = Path("data/features/jump_lite")

JUMP_LITE_METADATA_COLS = [
    "Metadata_id",
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_control_type",
    "Metadata_Perturbation_Type",
    "Metadata_JCP2022",
    "Metadata_broad_sample",
    "Metadata_Symbol",
    "Metadata_model",
    "Metadata_dataset",
    "Metadata_compression",
]

JUMP_LITE_FEATURE_RENAMES = {
    "Cytoplasm_Number_Object_Number": "cell_count",
    "Nuclei_Number_Object_Number": "nuclei_count",
}


def create_target2():
    """Create cell-count features for target2 (7 codec variants)."""
    print("=== target2 ===")
    for codec in TARGET2_CODECS:
        src = TARGET2_FEATURE_DIR / f"cp_measure_jump_target2_4plate_{codec}_raw_features.parquet"
        dst = TARGET2_FEATURE_DIR / f"cell_count_jump_target2_4plate_{codec}_raw_features.parquet"

        if not src.exists():
            print(f"  SKIP (not found): {src}")
            continue

        df = pl.read_parquet(src)

        # Keep metadata cols + the two cell-count columns (renamed as features)
        # Cast to float64 so downstream PCA/GPU ops don't choke on int64
        df_out = (
            df.select(TARGET2_METADATA_COLS + list(TARGET2_FEATURE_RENAMES.keys()))
            .rename(TARGET2_FEATURE_RENAMES)
            .with_columns(
                pl.lit("cell_count").alias("Metadata_model"),
                pl.col("cell_count").cast(pl.Float64),
                pl.col("nuclei_count").cast(pl.Float64),
            )
        )

        print(f"  {codec}: {df_out.shape} -> {dst.name}")
        df_out.write_parquet(dst)


def create_jump_lite():
    """Create cell-count features for jump_lite (single file, no codec variants)."""
    print("\n=== jump_lite ===")
    src = JUMP_LITE_FEATURE_DIR / "cellprofiler_raw_jump_lite_raw_features.parquet"
    dst = JUMP_LITE_FEATURE_DIR / "cell_count_jump_lite_raw_features.parquet"

    if not src.exists():
        print(f"  SKIP (not found): {src}")
        return

    df = pl.read_parquet(src)

    df_out = (
        df.select(JUMP_LITE_METADATA_COLS + list(JUMP_LITE_FEATURE_RENAMES.keys()))
        .rename(JUMP_LITE_FEATURE_RENAMES)
        .with_columns(
            pl.lit("cell_count").alias("Metadata_model"),
            pl.col("cell_count").cast(pl.Float64),
            pl.col("nuclei_count").cast(pl.Float64),
        )
    )

    print(f"  jump_lite: {df_out.shape} -> {dst.name}")
    df_out.write_parquet(dst)


def main():
    create_target2()
    create_jump_lite()
    print("\nDone.")


if __name__ == "__main__":
    main()
