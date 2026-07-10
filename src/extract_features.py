"""
Feature Extraction Script (Fast / Parallel)

Drop-in replacement for extract_features.py with the following optimizations:
1. Parallel processing of profile directories via joblib
2. Combined SQL queries (fewer DuckDB materializations)
3. Single-query aggregation+pivot (no intermediate CREATE TABLE)
4. Built-in cache-dir default for faster re-runs

Usage:
    # Auto-discover and process all MODEL/COMPRESSION combinations
    python src/extract_features.py --input data/aliby_output --output ./output

    # Process specific model only
    python src/extract_features.py --input data/aliby_output --model cp_measure --output ./output

    # Control parallelism (default: all cores)
    python src/extract_features.py --input data/aliby_output --output ./output --n-jobs 4
"""

import argparse
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path

import duckdb
import polars as pl
from joblib import Parallel, delayed

# Directories with more parquet files than this use plate-chunked processing
_CHUNK_FILE_THRESHOLD = 5000


def discover_profile_directories(
    base_dir: Path,
    model_filter: str | None = None,
    compression_filter: str | None = None,
    dataset_filter: str | None = None,
) -> list[dict]:
    """
    Discover profile directories by recursively scanning for *.zarr/profiles/.

    Handles any nesting depth. The two path segments immediately above the
    .zarr directory are treated as (model, dataset) or (dataset, model),
    resolved via the model_filter heuristic.

    Supported layouts (any prefix depth):
        .../MODEL/DATASET/COMPRESSION.zarr/profiles/
        .../DATASET/MODEL/COMPRESSION.zarr/profiles/

    Args:
        base_dir: Base aliby_output directory
        model_filter: Optional filter for specific model (e.g., cp_measure, dinov2)
        compression_filter: Optional filter for specific compression (e.g., zstd.zarr)
        dataset_filter: Optional filter for specific dataset (e.g., jump_target2_4plate)

    Returns:
        List of dicts with keys: model, dataset, compression, profiles_path
    """
    results = []

    # Recursively find all dirs that contain a profiles/ subdirectory.
    # This handles both compressed (*.zarr/profiles/) and raw (raw/profiles/) layouts.
    for profiles_path in base_dir.rglob("profiles"):
        if not profiles_path.is_dir():
            continue
        compression_dir = profiles_path.parent

        # Skip cache directories anywhere in the path
        rel = compression_dir.relative_to(base_dir)
        if any("cache" in part.lower() for part in rel.parts):
            continue

        if compression_filter and compression_dir.name != compression_filter:
            continue

        # The two directory levels above the .zarr are the model and dataset
        # (in either order). We need at least 2 parent levels above base_dir.
        parent1 = compression_dir.parent  # one level above .zarr
        parent2 = parent1.parent          # two levels above .zarr

        if parent1 == base_dir or parent2 == base_dir:
            # Only one level above base_dir — treat parent1 as model, no dataset
            # (shouldn't normally happen, but be defensive)
            continue

        # parent1.name and parent2.name are the two candidates for model/dataset
        name_a = parent2.name  # further from .zarr
        name_b = parent1.name  # closer to .zarr

        # Determine which is model and which is dataset
        if model_filter:
            if name_a == model_filter:
                model_name, dataset_name = name_a, name_b
            elif name_b == model_filter:
                model_name, dataset_name = name_b, name_a
            else:
                continue
        elif dataset_filter:
            if name_a == dataset_filter:
                model_name, dataset_name = name_b, name_a
            elif name_b == dataset_filter:
                model_name, dataset_name = name_a, name_b
            else:
                continue
        else:
            # Default: assume .../MODEL/DATASET/COMPRESSION.zarr
            model_name, dataset_name = name_a, name_b

        if model_filter and model_name != model_filter:
            continue
        if dataset_filter and dataset_name != dataset_filter:
            continue

        results.append({
            "model": model_name,
            "dataset": dataset_name,
            "compression": compression_dir.name,
            "profiles_path": profiles_path,
        })

    return results


# ---------------------------------------------------------------------------
# Chunked processing for large directories (855K+ small parquet files)
# ---------------------------------------------------------------------------

_WELL_ID_EXPR = (
    "string_split(parse_filename(filename, true), '__')[1] || '__' || "
    "string_split(parse_filename(filename, true), '__')[2] || '__' || "
    "string_split(parse_filename(filename, true), '__')[3] || '__' || "
    "string_split(parse_filename(filename, true), '__')[4]"
)


def _detect_parquet_format(sample_file: str) -> dict:
    """Detect data format and column types from a single parquet file."""
    con = duckdb.connect()
    try:
        desc = con.sql(f"SELECT * FROM read_parquet('{sample_file}') LIMIT 1").description
        columns = [col[0] for col in desc]
        dtypes = {col[0]: col[1] for col in desc}

        has_long = all(c in columns for c in ("branch", "metric", "object"))

        if not has_long:
            feature_cols = [
                c for c in columns
                if c not in ("tile", "label", "filename", "tp", "site")
            ]
            return {"format": "wide", "feature_cols": feature_cols}

        # Determine value column and whether UNNEST is needed
        unnest_values = "values" in columns and dtypes.get("values") == "list"
        value_col = "value"
        if "values" in columns and "value" not in columns and not unnest_values:
            value_col = "values"

        # Detect CellProfiler vs DL model by object values
        objects = set(
            r[0] for r in con.sql(
                f"SELECT DISTINCT object FROM read_parquet('{sample_file}')"
            ).fetchall()
        )
        is_cp = objects.issubset({"cell", "nuclei"})

        return {
            "format": "cp_long" if is_cp else "dl_long",
            "value_col": value_col,
            "unnest_values": unnest_values,
            "objects": objects,
            "has_label": "label" in columns,
        }
    finally:
        con.close()


def _process_plate(
    plate_key: str,
    file_list: list[str],
    fmt: dict,
    n_threads: int,
) -> pl.DataFrame | None:
    """Process one plate's parquet files into a well-level DataFrame.

    Designed to be called standalone or via joblib (must be picklable).
    """
    con = duckdb.connect(config={"threads": n_threads})
    try:
        flist = "[" + ",".join(f"'{f}'" for f in file_list) + "]"
        data_format = fmt["format"]

        if data_format in ("dl_long", "cp_long"):
            vcol = fmt["value_col"]

            # Handle old datasets where "values" is a list column
            if fmt.get("unnest_values"):
                src = (
                    f"(SELECT *, UNNEST(values) AS _val, ({_WELL_ID_EXPR}) AS well_id "
                    f" FROM read_parquet({flist}, filename=true))"
                )
                vcol = "_val"
            else:
                src = (
                    f"(SELECT *, ({_WELL_ID_EXPR}) AS well_id "
                    f" FROM read_parquet({flist}, filename=true))"
                )

            if data_format == "dl_long":
                agg = con.sql(f"""
                    SELECT well_id,
                           branch || metric AS fname,
                           median({vcol}) AS fval
                    FROM {src}
                    GROUP BY well_id, fname
                """)
                return con.sql("PIVOT agg ON fname USING first(fval)").pl()

            else:  # cp_long
                agg = con.sql(f"""
                    SELECT well_id,
                           branch || metric AS mname,
                           object,
                           median({vcol}) AS cval
                    FROM {src}
                    GROUP BY well_id, mname, object
                """)
                result = con.sql("PIVOT agg ON object, mname USING first(cval)").pl()

                if fmt.get("has_label"):
                    counts = con.sql(f"""
                        SELECT well_id,
                            COUNT(DISTINCT CASE WHEN object='cell' THEN label END)
                                AS Metadata_n_cells,
                            COUNT(DISTINCT CASE WHEN object='nuclei' THEN label END)
                                AS Metadata_n_nuclei
                        FROM {src}
                        GROUP BY well_id
                    """).pl()
                    result = result.join(counts, on="well_id", how="left")

                return result

        else:  # wide format
            cols = fmt["feature_cols"]
            agg = ", ".join(f'mean("{c}") AS "{c}"' for c in cols)
            if not agg:
                return None
            return con.sql(f"""
                SELECT ({_WELL_ID_EXPR}) AS well_id, {agg}
                FROM read_parquet({flist}, filename=true)
                GROUP BY well_id
            """).pl()

    except Exception as e:
        print(f"    WARNING: plate {plate_key} failed: {e}")
        return None
    finally:
        con.close()


def get_features_chunked(
    profiles_dir: Path,
    n_threads: int = 8,
    n_workers: int = 32,
) -> pl.DataFrame:
    """Feature extraction for directories with many small parquet files.

    Groups files by plate (source__batch__plate) and processes each plate
    independently, avoiding DuckDB's per-file overhead when globbing
    hundreds of thousands of files.

    Args:
        profiles_dir: Directory containing parquet files.
        n_threads: Total DuckDB threads budget.
        n_workers: Number of parallel plate workers (via joblib).
    """
    # 1. Group files by plate
    print(f"  Scanning files...")
    t0 = time.perf_counter()
    plate_groups: dict[str, list[str]] = defaultdict(list)
    for entry in os.scandir(str(profiles_dir)):
        if entry.name.endswith(".parquet") and entry.is_file(follow_symlinks=False):
            parts = entry.name[:-8].split("__")  # strip .parquet
            key = "__".join(parts[:3]) if len(parts) >= 3 else "_default"
            plate_groups[key].append(entry.path)

    n_files = sum(len(v) for v in plate_groups.values())
    print(f"  {n_files:,} files across {len(plate_groups)} plates "
          f"({time.perf_counter() - t0:.1f}s)")

    # 2. Detect format from a sample file
    sample = next(iter(plate_groups.values()))[0]
    fmt = _detect_parquet_format(sample)
    print(f"  Format: {fmt['format']}", end="")
    if fmt["format"] != "wide":
        print(f"  objects={fmt.get('objects', '?')}")
    else:
        print()

    # 3. Process plates
    plates = sorted(plate_groups.items())

    if n_workers > 1 and len(plates) > 1:
        thr_per = max(1, n_threads // n_workers)
        print(f"  Parallel: {n_workers} workers x {thr_per} threads/worker")
        t1 = time.perf_counter()
        results = Parallel(n_jobs=n_workers, prefer="processes", verbose=10)(
            delayed(_process_plate)(k, fs, fmt, thr_per)
            for k, fs in plates
        )
        print(f"  All plates done in {time.perf_counter() - t1:.1f}s")
    else:
        print(f"  Sequential: {len(plates)} plates")
        results = []
        t1 = time.perf_counter()
        for i, (k, fs) in enumerate(plates):
            if i % 50 == 0:
                el = time.perf_counter() - t1
                rate = (i + 1) / el if el > 0 else 0
                eta = (len(plates) - i - 1) / rate if rate > 0 else 0
                print(f"    [{i + 1}/{len(plates)}] {k} ({len(fs)} files) "
                      f"~{eta:.0f}s remaining")
            results.append(_process_plate(k, fs, fmt, n_threads))
        print(f"  Done in {time.perf_counter() - t1:.1f}s")

    # 4. Concat results
    valid = [r for r in results if r is not None]
    if not valid:
        raise RuntimeError("All plates failed")
    if len(valid) < len(results):
        print(f"  WARNING: {len(results) - len(valid)} plates failed")

    return pl.concat(valid, how="diagonal")


def get_features(profiles_dir: Path, cache_dir: Path | None = None, filter_border_cells: bool = False) -> pl.DataFrame:
    """
    Extract and process features from parquet files.

    Optimized version: combines SQL queries to reduce materializations.
    Automatically switches to plate-chunked processing for large directories.

    Args:
        profiles_dir: Path to profiles directory containing parquet files
        cache_dir: Optional path to cache directory for intermediate databases
        filter_border_cells: If True, exclude cells touching image borders (default: False)

    Returns:
        Polars DataFrame with pivoted well-level features and cell counts
    """
    # Fast check: switch to chunked mode for large directories
    n_sampled = 0
    for entry in os.scandir(str(profiles_dir)):
        if entry.name.endswith(".parquet"):
            n_sampled += 1
            if n_sampled > _CHUNK_FILE_THRESHOLD:
                print(f"  Large directory (>{_CHUNK_FILE_THRESHOLD} files), "
                      f"using chunked plate-by-plate processing")
                return get_features_chunked(profiles_dir)

    parquet_files = profiles_dir / "*.parquet"

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
        con = duckdb.connect(str(db_file), config={'threads': 50})
    else:
        con = duckdb.connect(config={'threads': 50})

    try:
        # Combined query: parse filename AND extract well_id in one pass
        raw = con.sql(f"""
            SELECT *,
                parse_filename(filename, true) AS {site_col},
                string_split(parse_filename(filename, true), '__')[1] || '__' ||
                string_split(parse_filename(filename, true), '__')[2] || '__' ||
                string_split(parse_filename(filename, true), '__')[3] || '__' ||
                string_split(parse_filename(filename, true), '__')[4] AS well_id
            FROM read_parquet('{parquet_files}', filename=true)
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
                # Single CTE chain for border cell filtering
                con.sql("""
                    CREATE OR REPLACE TABLE raw_filtered AS (
                        WITH bbox_info AS (
                            SELECT
                                well_id, filename, label, object,
                                MAX(CASE WHEN metric = 'BoundingBoxMinimum_X' THEN value END) AS min_x,
                                MAX(CASE WHEN metric = 'BoundingBoxMinimum_Y' THEN value END) AS min_y,
                                MAX(CASE WHEN metric = 'BoundingBoxMaximum_X' THEN value END) AS max_x,
                                MAX(CASE WHEN metric = 'BoundingBoxMaximum_Y' THEN value END) AS max_y
                            FROM raw
                            WHERE metric LIKE 'BoundingBox%'
                            GROUP BY well_id, filename, label, object
                        ),
                        image_dims AS (
                            SELECT filename,
                                MAX(max_x) AS img_width,
                                MAX(max_y) AS img_height
                            FROM bbox_info
                            GROUP BY filename
                        ),
                        interior_labels AS (
                            SELECT DISTINCT b.well_id, b.filename, b.label, b.object
                            FROM bbox_info b
                            JOIN image_dims d ON b.filename = d.filename
                            WHERE b.min_x > 0
                              AND b.min_y > 0
                              AND b.max_x < d.img_width
                              AND b.max_y < d.img_height
                        )
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

            # Single-query aggregation + pivot (no intermediate CREATE TABLE)
            pivoted = con.sql(f"""
                PIVOT (
                    SELECT
                        well_id,
                        {branch_name} || {metric_name} AS full_metric_name,
                        object,
                        median(value) AS cvalue
                    FROM {raw_table}
                    GROUP BY {tp_name}, well_id, {branch_name}, {metric_name}, object
                ) ON object, full_metric_name USING any_value(cvalue)
            """)
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

            # Single-query aggregation + pivot
            pivoted = con.sql(f"""
                PIVOT (
                    SELECT
                        well_id,
                        {branch_name} || {metric_name} AS full_metric_name,
                        object,
                        median(value) AS cvalue
                    FROM raw
                    GROUP BY {tp_name}, well_id, {branch_name}, {metric_name}, object
                ) ON object, full_metric_name USING any_value(cvalue)
            """)
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
        t0 = time.perf_counter()

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
        elapsed = time.perf_counter() - t0
        print(f"  Output: {output_path}")
        print(f"  Shape: {df.shape} ({elapsed:.1f}s)")

        return output_path

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    """Main entry point for feature extraction (parallel)."""
    parser = argparse.ArgumentParser(
        description="Extract features from aliby_output profiles (fast/parallel version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover and process all combinations (uses all CPU cores)
  python src/extract_features.py --input data/aliby_output --output ./output

  # Process specific model with 4 workers
  python src/extract_features.py --input data/aliby_output --model cp_measure --output ./output --n-jobs 4

  # Sequential mode (like original extract_features.py)
  python src/extract_features.py --input data/aliby_output --output ./output --n-jobs 1
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
        help="Cache directory for intermediate database files (default: .cache/features under output dir)",
    )
    parser.add_argument(
        "--filter-border-cells",
        action="store_true",
        help="Exclude cells/objects touching image borders (CellProfiler FilterObjects behavior)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers (-1 = all cores, 1 = sequential). Default: -1",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input.exists():
        print(f"Error: Input directory does not exist: {args.input}")
        return 1

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Default cache dir under output directory
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = args.output / ".cache" / "features"

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

    # Process profile directories in parallel
    warnings.filterwarnings("ignore")
    n_jobs = args.n_jobs
    if n_jobs == 1 or len(profiles_list) == 1:
        # Sequential fallback
        print(f"\nProcessing {len(profiles_list)} profiles sequentially...")
        results = [
            process_profiles(info, args.output, cache_dir, args.filter_border_cells)
            for info in profiles_list
        ]
    else:
        effective_jobs = n_jobs if n_jobs > 0 else "all cores"
        print(f"\nProcessing {len(profiles_list)} profiles in parallel (n_jobs={effective_jobs})...")
        t_start = time.perf_counter()
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(process_profiles)(info, args.output, cache_dir, args.filter_border_cells)
            for info in profiles_list
        )
        t_elapsed = time.perf_counter() - t_start
        print(f"\nParallel processing took {t_elapsed:.1f}s")

    successful = sum(1 for r in results if r is not None)
    failed = sum(1 for r in results if r is None)

    # Summary
    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {args.output}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
