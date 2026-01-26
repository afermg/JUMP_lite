"""
Feature Extraction Script

Extracts features from aliby_output profiles and creates raw_features.parquet files.
Supports both CellProfiler measurements and embedding-based features (e.g., dinov2).

Usage:
    # Auto-discover and process all MODEL/COMPRESSION combinations
    python src/extract_features.py --input /work/datasets/aliby_output --output ./output

    # Process specific model only
    python src/extract_features.py --input /work/datasets/aliby_output --model cp_measure --output ./output

    # Process specific model and compression
    python src/extract_features.py --input /work/datasets/aliby_output --model dinov2 --compression zstd.zarr --output ./output
"""

import argparse
import warnings
from pathlib import Path

import duckdb
import polars as pl
from polars import selectors as cs
from trommel.core import basic_cleanup


def discover_profile_directories(
    base_dir: Path,
    model_filter: str | None = None,
    compression_filter: str | None = None,
    dataset_filter: str | None = None,
) -> list[dict]:
    """
    Discover all MODEL/DATASET/COMPRESSION/profiles directories.

    Args:
        base_dir: Base aliby_output directory
        model_filter: Optional filter for specific model (e.g., cp_measure, dinov2)
        compression_filter: Optional filter for specific compression (e.g., zstd.zarr)
        dataset_filter: Optional filter for specific dataset (e.g., jump_target2_4plate)

    Returns:
        List of dicts with keys: model, dataset, compression, profiles_path
    """
    results = []

    # Iterate over model directories (cp_measure, dinov2, etc.)
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
        # Skip cache directories
        if "cache" in model_dir.name.lower():
            continue

        model_name = model_dir.name
        if model_filter and model_name != model_filter:
            continue

        # Iterate over dataset directories
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name
            if dataset_filter and dataset_name != dataset_filter:
                continue

            # Iterate over compression directories (*.zarr)
            for compression_dir in dataset_dir.iterdir():
                if not compression_dir.is_dir():
                    continue
                if not compression_dir.name.endswith(".zarr"):
                    continue

                compression_name = compression_dir.name
                if compression_filter and compression_name != compression_filter:
                    continue

                # Check for profiles subdirectory
                profiles_path = compression_dir / "profiles"
                if profiles_path.exists() and profiles_path.is_dir():
                    results.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "compression": compression_name,
                        "profiles_path": profiles_path,
                    })

    return results


def get_features(profiles_dir: Path, cache_dir: Path | None = None) -> pl.DataFrame:
    """
    Extract and process features from parquet files.

    Adapted from feature_correlation_cp_measure_script.py to handle
    both CellProfiler measures and embedding-based features.

    Args:
        profiles_dir: Path to profiles directory containing parquet files
        cache_dir: Optional path to cache directory for intermediate databases

    Returns:
        Polars DataFrame with pivoted well-level features
    """
    parquet_files = profiles_dir / "*.parquet"
    steps_dir = profiles_dir / ".." / "steps"

    site_col = "site"
    metric_name = "metric"
    branch_name = "branch"
    value_name = "values"
    tp_name = "tp"

    # Set up database connection (in-memory if no cache_dir)
    if cache_dir:
        cache_dir.mkdir(exist_ok=True, parents=True)
        db_file = cache_dir / f"features_{profiles_dir.parent.name}.db"
        con = duckdb.connect(str(db_file))
    else:
        con = duckdb.connect()

    try:
        raw = con.sql(f"""
            SELECT *, parse_filename(filename, true) AS {site_col}
            FROM read_parquet('{parquet_files}', filename=true)
        """)

        # Check if this is CellProfiler-style data (has branch, metric, object columns)
        columns = [col[0] for col in raw.description]

        if branch_name in columns and metric_name in columns and "object" in columns:
            # CellProfiler measure format
            # Handle old datasets with column name "values" as list type
            value_dtype = [x[1] for x in raw.description if x[0] == value_name]
            if len(value_dtype) and value_dtype[0] == "list":
                raw = con.sql(f"SELECT *, UNNEST({value_name}) AS value FROM raw")

            # Create well-level dataset with aggregation
            con.sql(f"""
                CREATE OR REPLACE TABLE well_level AS (
                    SELECT
                        {site_col},
                        {branch_name} || {metric_name} AS full_metric_name,
                        object,
                        mean(value) AS cvalue
                    FROM raw
                    GROUP BY {tp_name}, {site_col}, {branch_name}, {metric_name}, object
                )
            """)

            pivoted = con.sql(
                f"PIVOT well_level ON object, full_metric_name USING any_value(cvalue)"
            )
            pivoted_pl = pivoted.pl()

        else:
            # Embedding-based format (e.g., dinov2) - simpler structure
            # Just aggregate to well/site level
            # Get feature columns (numeric columns that aren't metadata)
            feature_cols = [
                col for col in columns
                if col not in [site_col, "filename", tp_name]
            ]

            if feature_cols:
                # Average features per site
                agg_expressions = ", ".join([f"median({col}) AS {col}" for col in feature_cols])
                pivoted = con.sql(f"""
                    SELECT {site_col}, {agg_expressions}
                    FROM raw
                    GROUP BY {site_col}
                """)
                pivoted_pl = pivoted.pl()
            else:
                # Just return raw data grouped by site
                pivoted_pl = raw.pl()

    finally:
        con.close()

    return pivoted_pl


def extract_metadata_from_site(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract metadata columns from site string.

    Site format: source__batch__plate__well__site

    Args:
        df: Polars DataFrame with 'site' column

    Returns:
        Polars DataFrame with metadata columns added
    """
    if "site" not in df.columns:
        return df

    # Split site column into metadata components
    df = df.with_columns([
        pl.col("site").str.split("__").list.get(0).alias("Metadata_Source"),
        pl.col("site").str.split("__").list.get(1).alias("Metadata_Batch"),
        pl.col("site").str.split("__").list.get(2).alias("Metadata_Plate"),
        pl.col("site").str.split("__").list.get(3).alias("Metadata_Well"),
        pl.col("site").str.split("__").list.get(4).alias("Metadata_Site"),
    ])

    # Rename site to Metadata_id
    df = df.rename({"site": "Metadata_id"})

    return df


def process_profiles(
    profiles_info: dict,
    output_dir: Path,
    cache_dir: Path | None = None,
) -> Path | None:
    """
    Process a single profiles directory and save raw_features.parquet.

    Args:
        profiles_info: Dict with model, dataset, compression, profiles_path
        output_dir: Directory to save output parquet file
        cache_dir: Optional cache directory for intermediate databases

    Returns:
        Path to output file, or None if processing failed
    """
    model = profiles_info["model"]
    dataset = profiles_info["dataset"]
    compression = profiles_info["compression"]
    profiles_path = profiles_info["profiles_path"]

    # Create output filename
    compression_clean = compression.replace(".zarr", "")
    output_filename = f"{model}_{dataset}_{compression_clean}_raw_features.parquet"
    output_path = output_dir / output_filename

    print(f"\nProcessing: {model}/{dataset}/{compression}")
    print(f"  Input: {profiles_path}")

    try:
        # Extract features
        df = get_features(profiles_path, cache_dir)
        print(f"  Extracted features: {df.shape}")

        # Extract metadata from site column
        df = extract_metadata_from_site(df)

        # Add model/compression info as metadata
        df = df.with_columns([
            pl.lit(model).alias("Metadata_model"),
            pl.lit(dataset).alias("Metadata_dataset"),
            pl.lit(compression).alias("Metadata_compression"),
        ])

        # Save to parquet
        df.write_parquet(output_path)
        print(f"  Output: {output_path}")
        print(f"  Shape: {df.shape}")

        return output_path

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    """Main entry point for feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract features from aliby_output profiles and create raw_features.parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover and process all combinations
  python src/extract_features.py --input /work/datasets/aliby_output --output ./output

  # Process specific model
  python src/extract_features.py --input /work/datasets/aliby_output --model cp_measure --output ./output

  # Process specific model and compression
  python src/extract_features.py --input /work/datasets/aliby_output --model dinov2 --compression zstd.zarr --output ./output
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Base aliby_output directory containing MODEL/DATASET/COMPRESSION/profiles structure",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for raw_features.parquet files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter for specific model (e.g., cp_measure, dinov2)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=None,
        help="Filter for specific compression (e.g., zstd.zarr)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter for specific dataset (e.g., jump_target2_4plate)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for intermediate database files",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input.exists():
        print(f"Error: Input directory does not exist: {args.input}")
        return 1

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Discover profile directories
    print(f"Discovering profile directories in: {args.input}")
    profiles_list = discover_profile_directories(
        args.input,
        model_filter=args.model,
        compression_filter=args.compression,
        dataset_filter=args.dataset,
    )

    if not profiles_list:
        print("No profile directories found matching the specified filters.")
        return 1

    print(f"\nFound {len(profiles_list)} profile directories:")
    for info in profiles_list:
        print(f"  - {info['model']}/{info['dataset']}/{info['compression']}")

    # Process each profiles directory
    warnings.filterwarnings("ignore")
    successful = 0
    failed = 0

    for profiles_info in profiles_list:
        result = process_profiles(profiles_info, args.output, args.cache_dir)
        if result:
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {args.output}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
