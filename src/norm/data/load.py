"""Data loading and saving utilities for morphological profiles."""

from pathlib import Path
import polars as pl


def load_profiles(
    path: str | Path, format: str = "parquet", metadata_patterns: list[str] = None
) -> pl.DataFrame:
    """
    Load morphological profiles from file.

    Args:
        path: File path to profiles
        format: File format (parquet, csv, csv.gz)
        metadata_patterns: Patterns to identify metadata columns

    Returns:
        Polars DataFrame with profiles
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    if format == "parquet":
        df = pl.read_parquet(path)
    elif format == "csv":
        df = pl.read_csv(path)
    elif format == "csv.gz":
        df = pl.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {format}")

    return df


def save_profiles(
    df: pl.DataFrame,
    path: str | Path,
    format: str = "parquet",
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
        compression = compression or "snappy"
        df.write_parquet(path, compression=compression)
    elif format == "csv":
        df.write_csv(path)
    elif format == "csv.gz":
        df.write_csv(path, compression="gzip")
    else:
        raise ValueError(f"Unsupported format: {format}")


def infer_columns(
    df: pl.DataFrame, exclude_patterns: list[str]
) -> tuple[list[str], list[str]]:
    """
    Infer feature and metadata columns based on patterns.

    Args:
        df: Input DataFrame
        exclude_patterns: Patterns for metadata columns (e.g., ["Metadata_"])

    Returns:
        Tuple of (feature_cols, metadata_cols)
    """
    metadata_cols = []

    for pattern in exclude_patterns:
        # Remove ^ anchor if present
        clean_pattern = pattern.strip("^")
        matched = [c for c in df.columns if c.startswith(clean_pattern)]
        metadata_cols.extend(matched)

    metadata_cols = list(set(metadata_cols))
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    return feature_cols, metadata_cols
