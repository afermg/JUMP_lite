"""Data loading and saving utilities for morphological profiles.

This module provides functions for:
- Loading profiles from parquet/csv files
- Saving profiles with compression options
- Inferring feature and metadata columns
- Loading JUMP metadata
"""

from pathlib import Path
from typing import Literal

import polars as pl


def load_profiles(
    path: str | Path,
    format: Literal["parquet", "csv", "csv.gz"] | None = None,
) -> pl.DataFrame:
    """
    Load morphological profiles from file.

    Args:
        path: File path to profiles
        format: File format (parquet, csv, csv.gz). If None, inferred from extension.

    Returns:
        Polars DataFrame with profiles

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If format is not supported
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    # Infer format from extension if not provided
    if format is None:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            format = "parquet"
        elif suffix == ".csv":
            format = "csv"
        elif suffix == ".gz" and path.stem.endswith(".csv"):
            format = "csv.gz"
        else:
            format = "parquet"  # Default

    if format == "parquet":
        return pl.read_parquet(path)
    elif format in ("csv", "csv.gz"):
        return pl.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_profiles(
    df: pl.DataFrame,
    path: str | Path,
    format: Literal["parquet", "csv", "csv.gz"] = "parquet",
    compression: str | None = None,
) -> None:
    """
    Save processed profiles to disk.

    Args:
        df: DataFrame to save
        path: Output file path
        format: File format (parquet, csv, csv.gz)
        compression: Compression method (for parquet: snappy, gzip, lz4, zstd)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        compression = compression or "zstd"
        df.write_parquet(path, compression=compression)
    elif format == "csv":
        df.write_csv(path)
    elif format == "csv.gz":
        df.write_csv(path, compression="gzip")
    else:
        raise ValueError(f"Unsupported format: {format}")


def infer_columns(
    df: pl.DataFrame,
    metadata_prefix: str | list[str] = "Metadata_",
) -> tuple[list[str], list[str]]:
    """
    Infer feature and metadata columns based on prefix patterns.

    Args:
        df: Input DataFrame
        metadata_prefix: Prefix(es) for metadata columns (e.g., "Metadata_" or ["Metadata_", "^Meta"])

    Returns:
        Tuple of (feature_cols, metadata_cols)
    """
    if isinstance(metadata_prefix, str):
        prefixes = [metadata_prefix]
    else:
        prefixes = metadata_prefix

    metadata_cols = []
    for prefix in prefixes:
        # Remove ^ anchor if present
        clean_prefix = prefix.strip("^")
        matched = [c for c in df.columns if c.startswith(clean_prefix)]
        metadata_cols.extend(matched)

    metadata_cols = list(set(metadata_cols))
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    return feature_cols, metadata_cols


def get_numeric_features(df: pl.DataFrame, features: list[str]) -> list[str]:
    """
    Filter feature list to only include numeric columns.

    Args:
        df: Input DataFrame
        features: Feature columns to filter

    Returns:
        List of numeric feature column names
    """
    numeric_types = (
        pl.Float32, pl.Float64,
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    )
    return [f for f in features if df[f].dtype in numeric_types]


def load_metadata(
    metadata_dir: str | Path,
    plate_col: str = "Metadata_Plate",
    well_col: str = "Metadata_Well",
) -> pl.DataFrame:
    """
    Load and merge well metadata for Target-2 plates.

    Args:
        metadata_dir: Directory containing metadata TSV files
        plate_col: Column name for plate identifier
        well_col: Column name for well identifier

    Returns:
        Polars DataFrame with merged metadata
    """
    import logging
    import pandas as pd

    metadata_dir = Path(metadata_dir)

    try:
        compound_meta = pd.read_csv(
            metadata_dir / "JUMP-Target-2_compound_metadata.tsv", sep="\t"
        ).drop_duplicates()
        plate_map = pd.read_csv(
            metadata_dir / "JUMP-Target-2_compound_platemap.tsv", sep="\t"
        )

        plate_map_annotated = plate_map.merge(compound_meta, on="broad_sample", how="left")

        # Rename columns to have Metadata_ prefix
        rename_cols = {
            col: f"Metadata_{col}"
            for col in plate_map_annotated.columns
            if not col.startswith("Metadata_") and col not in ["well_position", "broad_sample"]
        }
        rename_cols["well_position"] = "Metadata_Well"
        rename_cols["broad_sample"] = "Metadata_broad_sample"

        plate_map_annotated = plate_map_annotated.rename(columns=rename_cols)

        logging.info(f"Loaded metadata for {len(plate_map_annotated)} wells")
        return pl.from_pandas(plate_map_annotated)

    except Exception as e:
        logging.error(f"Failed to load metadata: {e}")
        raise
