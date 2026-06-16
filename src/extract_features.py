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
    Discover profile directories in two supported layouts:

    3-level: base_dir/MODEL/DATASET/COMPRESSION.zarr/profiles/
    2-level: base_dir/DATASET/MODEL/COMPRESSION.zarr/profiles/

    The function tries 3-level first, then falls back to 2-level.

    Args:
        base_dir: Base aliby_output directory
        model_filter: Optional filter for specific model (e.g., cp_measure, dinov2)
        compression_filter: Optional filter for specific compression (e.g., zstd.zarr)
        dataset_filter: Optional filter for specific dataset (e.g., jump_target2_4plate)

    Returns:
        List of dicts with keys: model, dataset, compression, profiles_path
    """
    results = []

    # Try 3-level layout: base_dir/MODEL/DATASET/COMPRESSION.zarr/profiles/
    for level1_dir in base_dir.iterdir():
        if not level1_dir.is_dir() or "cache" in level1_dir.name.lower():
            continue

        for level2_dir in level1_dir.iterdir():
            if not level2_dir.is_dir():
                continue

            for compression_dir in level2_dir.iterdir():
                if not compression_dir.is_dir() or not compression_dir.name.endswith(".zarr"):
                    continue

                profiles_path = compression_dir / "profiles"
                if not profiles_path.exists() or not profiles_path.is_dir():
                    continue

                # Determine which layout: check if level1 is model or dataset
                # In 3-level: level1=model, level2=dataset
                # In 2-level: level1=dataset, level2=model
                # Heuristic: if model_filter matches level1, it's 3-level
                #            if model_filter matches level2, it's 2-level
                if model_filter:
                    if level1_dir.name == model_filter:
                        model_name, dataset_name = level1_dir.name, level2_dir.name
                    elif level2_dir.name == model_filter:
                        model_name, dataset_name = level2_dir.name, level1_dir.name
                    else:
                        continue
                else:
                    # No model filter — try to detect layout by checking
                    # if level2 contains .zarr dirs (then level1=model/dataset, level2=dataset/model)
                    # Default to 3-level (model/dataset)
                    model_name, dataset_name = level1_dir.name, level2_dir.name

                if model_filter and model_name != model_filter:
                    continue
                if dataset_filter and dataset_name != dataset_filter:
                    continue
                if compression_filter and compression_dir.name != compression_filter:
                    continue

                results.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "compression": compression_dir.name,
                    "profiles_path": profiles_path,
                })

    return results


def get_features(profiles_dir: Path, cache_dir: Path | None = None, filter_border_cells: bool = False) -> pl.DataFrame:
    """
    Extract and process features from parquet files.

    Adapted from feature_correlation_cp_measure_script.py to handle
    both CellProfiler measures and embedding-based features.

    Args:
        profiles_dir: Path to profiles directory containing parquet files
        cache_dir: Optional path to cache directory for intermediate databases
        filter_border_cells: If True, exclude cells touching image borders (default: False)

    Returns:
        Polars DataFrame with pivoted well-level features and cell counts
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
        # Use full path to create unique cache filename
        # profiles_dir structure: MODEL/DATASET/COMPRESSION/profiles
        model_name = profiles_dir.parent.parent.parent.name
        dataset_name = profiles_dir.parent.parent.name
        compression_name = profiles_dir.parent.name
        db_file = cache_dir / f"features_{model_name}_{dataset_name}_{compression_name}.db"
        con = duckdb.connect(str(db_file))
    else:
        con = duckdb.connect()

    try:
        # First parse filename to get site string
        raw = con.sql(f"""
            SELECT *,
                parse_filename(filename, true) AS {site_col}
            FROM read_parquet('{parquet_files}', filename=true)
        """)

        # Extract well_id by splitting on __ and taking first 4 parts
        raw = con.sql(f"""
            SELECT *,
                string_split({site_col}, '__')[1] || '__' ||
                string_split({site_col}, '__')[2] || '__' ||
                string_split({site_col}, '__')[3] || '__' ||
                string_split({site_col}, '__')[4] AS well_id
            FROM raw
        """)

        # Check if this is CellProfiler-style data (has branch, metric, object columns)
        columns = [col[0] for col in raw.description]

        # Determine if data is CellProfiler by checking actual object values,
        # not just column presence. Aliby outputs both CP and DL models in the
        # same long format (branch/metric/object/value), but CP has
        # object='cell'/'nuclei' while DL models have object='morphem'/'dinov2'/etc.
        is_cellprofiler = False
        if branch_name in columns and metric_name in columns and "object" in columns:
            known_cp_objects = {"cell", "nuclei"}
            actual_objects = set(
                row[0] for row in con.sql("SELECT DISTINCT object FROM raw").fetchall()
            )
            is_cellprofiler = actual_objects.issubset(known_cp_objects)
            if not is_cellprofiler:
                print(f"  Detected DL model (objects: {actual_objects}), using median aggregation")

        if is_cellprofiler:
            # CellProfiler measure format
            # Handle old datasets with column name "values" as list type
            value_dtype = [x[1] for x in raw.description if x[0] == value_name]
            if len(value_dtype) and value_dtype[0] == "list":
                raw = con.sql(f"SELECT *, UNNEST({value_name}) AS value FROM raw")

            # Optionally filter border cells (objects touching image edge)
            # Get bounding box coordinates to identify border cells
            if filter_border_cells and "label" in columns:
                # Create table with bbox info per label
                con.sql("""
                    CREATE OR REPLACE TABLE bbox_info AS (
                        SELECT
                            well_id,
                            filename,
                            label,
                            object,
                            MAX(CASE WHEN metric = 'BoundingBoxMinimum_X' THEN value END) AS min_x,
                            MAX(CASE WHEN metric = 'BoundingBoxMinimum_Y' THEN value END) AS min_y,
                            MAX(CASE WHEN metric = 'BoundingBoxMaximum_X' THEN value END) AS max_x,
                            MAX(CASE WHEN metric = 'BoundingBoxMaximum_Y' THEN value END) AS max_y
                        FROM raw
                        WHERE metric LIKE 'BoundingBox%'
                        GROUP BY well_id, filename, label, object
                    )
                """)

                # Get image dimensions (per file since different sites may have different sizes)
                con.sql("""
                    CREATE OR REPLACE TABLE image_dims AS (
                        SELECT
                            filename,
                            MAX(max_x) AS img_width,
                            MAX(max_y) AS img_height
                        FROM bbox_info
                        GROUP BY filename
                    )
                """)

                # Filter interior cells (not touching borders)
                con.sql("""
                    CREATE OR REPLACE TABLE interior_labels AS (
                        SELECT DISTINCT b.well_id, b.filename, b.label, b.object
                        FROM bbox_info b
                        JOIN image_dims d ON b.filename = d.filename
                        WHERE b.min_x > 0
                        AND b.min_y > 0
                        AND b.max_x < d.img_width
                        AND b.max_y < d.img_height
                    )
                """)

                # Filter raw data to only interior cells
                con.sql("""
                    CREATE OR REPLACE TABLE raw_filtered AS (
                        SELECT r.*
                        FROM raw r
                        JOIN interior_labels i
                        ON r.well_id = i.well_id
                        AND r.filename = i.filename
                        AND r.label = i.label
                        AND r.object = i.object
                    )
                """)

                # Use filtered data for aggregation
                raw_table = "raw_filtered"
            else:
                # No filtering or no label column
                raw_table = "raw"

            # Count cells and nuclei per well (if label column exists)
            if "label" in columns:
                cell_counts = con.sql(f"""
                    SELECT
                        well_id,
                        COUNT(DISTINCT CASE WHEN object = 'cell' THEN label END) AS Metadata_n_cells,
                        COUNT(DISTINCT CASE WHEN object = 'nuclei' THEN label END) AS Metadata_n_nuclei
                    FROM {raw_table}
                    GROUP BY well_id
                """)
                cell_counts_pl = cell_counts.pl()
            else:
                cell_counts_pl = None

            # Create well-level dataset with aggregation (using median for cp_measure)
            # Aggregates across all cells and sites within each well
            con.sql(f"""
                CREATE OR REPLACE TABLE well_level AS (
                    SELECT
                        well_id,
                        {branch_name} || {metric_name} AS full_metric_name,
                        object,
                        median(value) AS cvalue
                    FROM {raw_table}
                    GROUP BY {tp_name}, well_id, {branch_name}, {metric_name}, object
                )
            """)

            pivoted = con.sql(
                f"PIVOT well_level ON object, full_metric_name USING any_value(cvalue)"
            )
            pivoted_pl = pivoted.pl()

            # Add cell counts if available
            if cell_counts_pl is not None:
                pivoted_pl = pivoted_pl.join(cell_counts_pl, on="well_id", how="left")

        elif branch_name in columns and metric_name in columns and "object" in columns:
            # DL model in long format (e.g., morphem, dinov2 from aliby)
            # Same schema as CellProfiler but with model-specific object values
            # Use median aggregation (consistent with CellProfiler branch)

            # Handle old datasets with column name "values" as list type
            value_dtype = [x[1] for x in raw.description if x[0] == value_name]
            if len(value_dtype) and value_dtype[0] == "list":
                raw = con.sql(f"SELECT *, UNNEST({value_name}) AS value FROM raw")

            con.sql(f"""
                CREATE OR REPLACE TABLE well_level AS (
                    SELECT
                        well_id,
                        {branch_name} || {metric_name} AS full_metric_name,
                        object,
                        median(value) AS cvalue
                    FROM raw
                    GROUP BY {tp_name}, well_id, {branch_name}, {metric_name}, object
                )
            """)

            pivoted = con.sql(
                f"PIVOT well_level ON object, full_metric_name USING any_value(cvalue)"
            )
            pivoted_pl = pivoted.pl()

        else:
            # Wide embedding format (no branch/metric/object columns)
            # Just aggregate to well level
            feature_cols = [
                col for col in columns
                if col not in [site_col, "filename", tp_name, "well_id"]
            ]

            if feature_cols:
                # Average features per well across all sites (using mean for embeddings)
                agg_expressions = ", ".join([f"mean({col}) AS {col}" for col in feature_cols])
                pivoted = con.sql(f"""
                    SELECT well_id, {agg_expressions}
                    FROM raw
                    GROUP BY well_id
                """)
                pivoted_pl = pivoted.pl()
            else:
                # Just return raw data grouped by well
                pivoted_pl = raw.pl()

    finally:
        con.close()

    return pivoted_pl


def extract_metadata_from_site(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract metadata columns from well_id string.

    Well ID format: source__batch__plate__well

    Since data is aggregated at well level, Metadata_Site is set to "1".

    Args:
        df: Polars DataFrame with 'well_id' column

    Returns:
        Polars DataFrame with metadata columns added
    """
    if "well_id" not in df.columns:
        return df

    # Split well_id column into metadata components
    df = df.with_columns([
        pl.col("well_id").str.split("__").list.get(0).alias("Metadata_Source"),
        pl.col("well_id").str.split("__").list.get(1).alias("Metadata_Batch"),
        pl.col("well_id").str.split("__").list.get(2).alias("Metadata_Plate"),
        pl.col("well_id").str.split("__").list.get(3).alias("Metadata_Well"),
        pl.lit("1").alias("Metadata_Site"),  # Set to "1" since aggregated at well level
    ])

    # Rename well_id and tp to have Metadata_ prefix
    df = df.rename({"well_id": "Metadata_id"})
    if "tp" in df.columns:
        df = df.rename({"tp": "Metadata_tp"})

    return df


def process_profiles(
    profiles_info: dict,
    output_dir: Path,
    cache_dir: Path | None = None,
    filter_border_cells: bool = False,
) -> Path | None:
    """
    Process a single profiles directory and save raw_features.parquet.

    Args:
        profiles_info: Dict with model, dataset, compression, profiles_path
        output_dir: Directory to save output parquet file
        cache_dir: Optional cache directory for intermediate databases
        filter_border_cells: If True, exclude cells touching image borders (default: False)

    Returns:
        Path to output file, or None if processing failed
    """
    model = profiles_info["model"]
    dataset = profiles_info["dataset"]
    compression = profiles_info["compression"]
    profiles_path = profiles_info["profiles_path"]

    # Create output filename
    compression_clean = compression.replace(".zarr", "")
    filter_suffix = "_filtered" if filter_border_cells else ""
    output_filename = f"{model}_{dataset}_{compression_clean}{filter_suffix}_raw_features.parquet"
    output_path = output_dir / output_filename

    print(f"\nProcessing: {model}/{dataset}/{compression}")
    print(f"  Input: {profiles_path}")

    try:
        # Extract features
        df = get_features(profiles_path, cache_dir, filter_border_cells)
        print(f"  Extracted features: {df.shape}")
        if filter_border_cells:
            print(f"  Border cells filtered: enabled")

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
    parser.add_argument(
        "--filter-border-cells",
        action="store_true",
        help="Exclude cells/objects touching image borders (CellProfiler FilterObjects behavior)",
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
        result = process_profiles(profiles_info, args.output, args.cache_dir, args.filter_border_cells)
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
