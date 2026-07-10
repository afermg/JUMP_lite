"""Data loading and saving utilities for norm_3.

Uses polars for DataFrame operations (efficient CPU library).
Only numeric arrays are transferred to GPU for compute operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl


def load_profiles(
    path: str | Path,
    format: Literal["parquet", "csv", "csv.gz"] | None = None,
) -> pl.DataFrame:
    """Load morphological profiles from file.

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
    """Save processed profiles to disk.

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
    """Infer feature and metadata columns based on prefix patterns.

    Args:
        df: Input DataFrame
        metadata_prefix: Prefix(es) for metadata columns

    Returns:
        Tuple of (feature_cols, metadata_cols)
    """
    if isinstance(metadata_prefix, str):
        prefixes = [metadata_prefix]
    else:
        prefixes = metadata_prefix

    metadata_cols = []
    for prefix in prefixes:
        clean_prefix = prefix.strip("^")
        matched = [c for c in df.columns if c.startswith(clean_prefix)]
        metadata_cols.extend(matched)

    metadata_cols = list(set(metadata_cols))
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    return feature_cols, metadata_cols


def get_numeric_features(df: pl.DataFrame, features: list[str]) -> list[str]:
    """Filter feature list to only include numeric columns.

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


def load_metadata_parquet(
    metadata_path: str | Path | list[str | Path],
    merge_how: str = "left",
) -> pl.DataFrame:
    """Load metadata from one or more parquet files.

    Files are merged sequentially on auto-detected common columns.

    Args:
        metadata_path: Path to metadata parquet file, or list of paths to merge
        merge_how: How to merge multiple files ("left", "inner", "outer")

    Returns:
        Polars DataFrame with metadata
    """
    # Handle single path or list of paths
    if isinstance(metadata_path, (str, Path)):
        paths = [Path(metadata_path)]
    else:
        paths = [Path(p) for p in metadata_path]

    # Load first file
    if not paths[0].exists():
        raise FileNotFoundError(f"Metadata file not found: {paths[0]}")

    df = pl.read_parquet(paths[0])
    print(f"  Loaded metadata[0]: {paths[0].name} -> {df.shape}")

    # Load and merge additional files
    for i, path in enumerate(paths[1:], start=1):
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")

        df2 = pl.read_parquet(path)
        print(f"  Loaded metadata[{i}]: {path.name} -> {df2.shape}")

        # Auto-detect common columns for joining
        common_cols = [c for c in df.columns if c in df2.columns]
        if not common_cols:
            raise ValueError(f"No common columns between merged data and {path.name}")

        # Cast join columns to string for consistent joining
        df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in common_cols])
        df2 = df2.with_columns([pl.col(c).cast(pl.Utf8) for c in common_cols])

        # Merge (avoid duplicating columns that already exist)
        existing_cols = set(df.columns)
        new_cols = [c for c in df2.columns if c not in existing_cols or c in common_cols]
        df2 = df2.select(new_cols)

        df = df.join(df2, on=common_cols, how=merge_how)
        print(f"  Merged on {common_cols} ({merge_how}) -> {df.shape}")

    # Rename Metadata_pert_type to Metadata_control_type if present
    if "Metadata_pert_type" in df.columns and "Metadata_control_type" not in df.columns:
        df = df.rename({"Metadata_pert_type": "Metadata_control_type"})

    # Rename non-Metadata columns to have Metadata_ prefix
    rename_map = {}
    for col in df.columns:
        if not col.startswith("Metadata_"):
            new_name = f"Metadata_{col}"
            rename_map[col] = new_name
    if rename_map:
        df = df.rename(rename_map)
        print(f"  Renamed {len(rename_map)} columns to have Metadata_ prefix")

    print(f"  Final metadata: {df.shape}")
    return df


def drop_na_columns(
    df: pl.DataFrame,
    features: list[str],
    na_cutoff: float = 0.05,
) -> tuple[list[str], list[str]]:
    """Remove features with too many missing or infinite values.

    Args:
        df: Input DataFrame
        features: Feature columns to evaluate
        na_cutoff: Maximum fraction of NaN/inf values allowed

    Returns:
        Tuple of (features_to_keep, features_dropped)
    """
    n_samples = len(df)
    keep_features = []
    dropped_features = []

    for feat in features:
        n_null = df[feat].null_count()
        n_nan = df[feat].is_nan().sum()
        n_inf = df[feat].is_infinite().sum()
        n_invalid = n_null + n_nan + n_inf

        if n_invalid / n_samples <= na_cutoff:
            keep_features.append(feat)
        else:
            dropped_features.append(feat)

    return keep_features, dropped_features
