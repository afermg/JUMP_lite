"""
Feature Correlation Analysis for CellProfiler Measures
Converted from Marimo notebook to standalone Python script with functions

Analyzes correlation between features extracted from compressed vs uncompressed images.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import islice
from math import pow
from pathlib import Path, PosixPath

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from polars import selectors as cs
from trommel.core import basic_cleanup
from umap import UMAP
import pandas as pd
import seaborn as sns

import os


def get_features(profiles_dir, workspace_dir, cache_dir):
    """
    Extract and process features from parquet files.

    Args:
        profiles_dir: Path to profiles directory containing parquet files
        workspace_dir: Path to workspace root
        cache_dir: Path to cache directory for intermediate databases

    Returns:
        tuple: (clean DataFrame, pivoted DataFrame, number dropped)
    """
    parquet_files = profiles_dir / "*.parquet"
    masks_dir = profiles_dir / ".." / "steps"
    distance = "cosine"

    site_col = "site"
    db_file = Path(cache_dir) / (
        "_".join(profiles_dir.relative_to(workspace_dir).parts) + f"_{distance}" + ".db"
    )

    cache_dir.mkdir(exist_ok=True, parents=True)

    objects = tuple([
        x.name.split("_")[-1]
        for x in next(masks_dir.glob("*")).glob("*")
        if x.name.startswith("segment")
    ])

    metric_name = "metric"
    branch_name = "branch"
    value_name = "values"
    cc_metric = "Area"  # Feature to be used for cell count
    tp_name = "tp"

    overwrite_str = "OR REPLACE TABLE"

    with duckdb.connect(db_file) as con:
        raw = con.sql(
            f"""
            SELECT *, parse_filename(filename,true) AS {site_col}, from read_parquet('{parquet_files}', filename=true)
            """
        )
        # Cover old datasets with column name "values" and datatype lists
        value_dtype = [x[1] for x in raw.description if x[0] == value_name]
        if len(value_dtype) and value_dtype[0] == "list":
            raw = con.sql(f"SELECT *,UNNEST({value_name}) AS value FROM raw")

        con.sql(  # Create well-level dataset
            f"""
            CREATE {overwrite_str} well_level AS (SELECT {site_col},{branch_name} || {metric_name} AS full_metric_name,object,mean(value) AS cvalue FROM raw GROUP BY {tp_name},{site_col},{branch_name},{metric_name},object)
            """
        )

        oc_df = con.sql(
            f"""
            SELECT {site_col},object,count({site_col}) AS oc FROM raw WHERE {metric_name} = '{cc_metric}' GROUP BY {site_col},{branch_name},{metric_name},object ORDER BY SITE,object
            """,
        )
        oc_piv = con.sql("PIVOT oc_df ON object USING any_value(oc)").pl()
        pivoted = con.sql(
            f"PIVOT well_level ON object,full_metric_name USING any_value(cvalue)"
        )
        pivoted_pl = pivoted.pl()

    warnings.filterwarnings("ignore")

    # For some reason this is necessary for viability assay
    clean, ndropped = basic_cleanup(pivoted_pl, meta_selector=cs.by_dtype(pl.String))

    return clean, pivoted_pl, ndropped


def find_profiles_dirs(top_dir):
    """
    Recursively find all 'profiles' directories.

    Args:
        top_dir: Top-level directory to search

    Returns:
        list: List of Path objects to profiles directories
    """
    top_level_dirs = list(top_dir.glob("*"))

    
    
    profiles_dirs = []
    for x in top_level_dirs:
        found = next(x.rglob("profiles"), None)
        if found:
            profiles_dirs.append(Path(found))

    if not profiles_dirs:
        raise FileNotFoundError(f"No 'profiles' directories found in {top_dir}")

    print(f"Using profile directories: {profiles_dirs}")
    return profiles_dirs


def extract_metadata_from_site(df, path, output_dir, df_name="clean"):
    """
    Extract metadata columns from site string and rename columns.
    
    Args:
        df: Polars DataFrame with 'site' column
        path: Path object to extract compression name from
        
    Returns:
        pd.DataFrame: Pandas DataFrame with metadata columns added
    """
    df = df.to_pandas()
    df[["Metadata_Source",
        "Metadata_Batch",
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Site"]] = df.site.str.split("__", expand=True)
    
    name = str(path).split("/")[-2]
    df = df.rename(columns={"site": "Metadata_id"})
    
    # Save to parquet
    output_path = Path(output_dir) / f"{name.split('.')[0]}_{df_name}.parquet"
    df.to_parquet(output_path)
    print(f"Saved features to: {output_path}")
    
    return df, name

def process_all_compressions(profiles_dirs, workspace_dir, cache_dir, output_dir):
    """
    Process features for all compression methods.

    Args:
        profiles_dirs: List of profile directories
        workspace_dir: Path to workspace root
        cache_dir: Path to cache directory
        output_dir: Path to save output parquet files

    Returns:
        dict: Dictionary mapping compression name to features
    """
    features_per_compression = {}

    for path in profiles_dirs:
        _clean, _pivoted_pl, _ndropped = get_features(path, workspace_dir, cache_dir)

        # Extract and save 
        _clean, name = extract_metadata_from_site(_clean, path, output_dir)
        
        # In process_all_compressions function:
        _, name = extract_metadata_from_site(_pivoted_pl, path, output_dir, df_name="raw_features")
        
        
        features_per_compression[name] = {
            "clean": _clean,
            "pivoted_pl": _pivoted_pl.to_pandas(),
            "ndropped": _ndropped
        }


    return features_per_compression


def calculate_feature_correlations(features_per_compression, reference_key="zstd.zarr", nan_threshold=0.30):
    """
    Calculate correlations between compressed and uncompressed features.

    Args:
        features_per_compression: Dict of features per compression method
        reference_key: Key for reference (uncompressed) dataset
        nan_threshold: Maximum fraction of NaN values to tolerate

    Returns:
        pd.DataFrame: DataFrame with correlation results
    """
    uncompressed_df = features_per_compression[reference_key]["pivoted_pl"].copy()
    results = []

    for key in features_per_compression.keys():
        df_compression = features_per_compression[key]["pivoted_pl"].copy()
        print(f"Original: {key}, shape: {df_compression.shape}")

        merged_df_with_non_compressed = uncompressed_df.merge(
            df_compression, on="site", how="inner", suffixes=("_x", "_y")
        )
        print(f"Merged: {merged_df_with_non_compressed.shape}")

        # Drop columns with more than nan_threshold fraction of NaN values
        merged_df_with_non_compressed_nan = merged_df_with_non_compressed.dropna(
            axis=1, thresh=int((1.0 - nan_threshold) * merged_df_with_non_compressed.shape[0])
        )
        print(f"After dropping columns: {merged_df_with_non_compressed_nan.shape}")

        merged_df_with_non_compressed_nan = merged_df_with_non_compressed_nan.dropna(axis=0)
        print(f"After dropping rows: {merged_df_with_non_compressed_nan.shape}")

        size_ = merged_df_with_non_compressed_nan.shape[0]

        count = 0
        for feature in [x for x in df_compression.columns
                       if x != "site"
                       and ((x + "_x" in merged_df_with_non_compressed_nan.columns)
                            and (x + "_y" in merged_df_with_non_compressed_nan.columns))]:

            corr = np.corrcoef(
                merged_df_with_non_compressed_nan[feature + "_x"],
                merged_df_with_non_compressed_nan[feature + "_y"]
            )

            results.append([key, feature, corr[0, 1], size_])
            count += 1
        print(f"Calculated {count} correlations")

    df = pd.DataFrame(results, columns=["key", "feature", "corr", "size"])
    return df


def parse_feature_names(df):
    """
    Parse feature names into compartment, mode, and feature components.

    Args:
        df: DataFrame with 'feature' column

    Returns:
        pd.DataFrame: DataFrame with added compartment, mode, feature columns
    """
    df[["compartment", "mode", "feature"]] = df.feature.str.split("/", expand=True)
    return df


def plot_correlation_boxplot(df, output_path="../output/correlation_boxplot.png"):
    """
    Create boxenplot of feature correlations across compression methods.

    Args:
        df: DataFrame with correlation results
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))

    order = ["jpegxl_lossy_lq.zarr",
             "jpegxl_lossy_mq.zarr",
             "jpegxl_lossy_effort_3.zarr",
             "jpegxl_lossy_hq.zarr",
             "zstd.zarr"]

    sns.boxenplot(data=df, x="key", y="corr", order=order)
    plt.title("Distribution of Feature Correlations Across Compression Methods")
    plt.xlabel("Compression strategy")
    plt.ylabel("Feature Correlation Coefficient")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved boxplot to: {output_path}")
    plt.show()


def plot_feature_heatmap(df, compression_key="jpegxl_lossy_effort_3.zarr",
                        output_path="../output/feature_heatmap.png"):
    """
    Create heatmap of correlations by feature and compartment.

    Args:
        df: DataFrame with correlation results (must have compartment column)
        compression_key: Which compression method to visualize
        output_path: Path to save plot
    """
    perf = df[df.key == compression_key].groupby(["feature", "compartment"], as_index=False)["corr"].mean()
    pivot_df = perf.pivot(index="feature", columns="compartment", values="corr")

    plt.figure(figsize=(30, 60))
    sns.heatmap(pivot_df, cmap="viridis", cbar_kws={'label': 'Correlation Coefficient'}, vmin=0, vmax=1)
    plt.title("Feature Correlation Across Compression Methods")
    plt.xlabel("Compartment")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved feature heatmap to: {output_path}")
    plt.show()


def plot_compression_heatmap(df, output_path="../output/compression_heatmap.png"):
    """
    Create heatmap of median correlations by feature and compression method.

    Args:
        df: DataFrame with correlation results
        output_path: Path to save plot
    """
    perf = df.groupby(["feature", "key"], as_index=False)["corr"].median()

    order = ["jpegxl_lossy_lq.zarr",
             "jpegxl_lossy_mq.zarr",
             "jpegxl_lossy_effort_3.zarr",
             "jpegxl_lossy_hq.zarr",
             "zstd.zarr"]

    pivot_df = perf.pivot(index="feature", columns="key", values="corr").reindex(columns=order)

    plt.figure(figsize=(30, 60))
    sns.heatmap(pivot_df, cmap="viridis", cbar_kws={'label': 'Correlation Coefficient'}, vmin=0, vmax=1)
    plt.title("Feature Correlation Across Compression Methods")
    plt.xlabel("Compression strategy")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved compression heatmap to: {output_path}")
    plt.show()


def plot_compartment_heatmap(df, output_path="../output/compartment_heatmap.png"):
    """
    Create heatmap of median correlations by compartment and compression method.

    Args:
        df: DataFrame with correlation results (must have compartment column)
        output_path: Path to save plot
    """
    perf = df.groupby(["compartment", "key"], as_index=False)["corr"].median()

    order = ["jpegxl_lossy_lq.zarr",
             "jpegxl_lossy_mq.zarr",
             "jpegxl_lossy_effort_3.zarr",
             "jpegxl_lossy_hq.zarr",
             "zstd.zarr"]

    pivot_df = perf.pivot(index="compartment", columns="key", values="corr").reindex(columns=order)

    plt.figure(figsize=(5, 10))
    sns.heatmap(pivot_df, cmap="viridis", cbar_kws={'label': 'Correlation Coefficient'},
                vmin=0, vmax=1, annot=True)
    plt.title("Feature Correlation Across Compression Methods")
    plt.xlabel("Compression strategy")
    plt.ylabel("Compartment")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved compartment heatmap to: {output_path}")
    plt.show()


def main():
    """Main pipeline for feature correlation analysis."""
    # Configuration
    workspace_dir = Path("/work") / "datasets" / "aliby_output" / "jump_target2_4plate"
    cache_dir = Path(workspace_dir) / "db_cache"
    output_dir = Path("../output")

    # For local testing, you can override this:
    # workspace_dir = Path("path/to/your/workspace")

    print(f"Workspace directory: {workspace_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Output directory: {output_dir}")

    # Find profile directories
    print("\nFinding profile directories...")
    profiles_dirs = find_profiles_dirs(workspace_dir)

    # Process all compressions
    print("\nProcessing all compression methods...")
    features_per_compression = process_all_compressions(
        profiles_dirs, workspace_dir, cache_dir, output_dir
    )

    print(f"\nFound {len(features_per_compression)} compression methods:")
    for key in features_per_compression.keys():
        print(f"  - {key}")

    # Calculate correlations
    print("\nCalculating feature correlations...")
    df = calculate_feature_correlations(features_per_compression, reference_key="zstd.zarr")

    print("\nCorrelation statistics by compression method:")
    print(df.groupby("key")["corr"].agg(['mean', 'median', 'std']))

    # Parse feature names
    print("\nParsing feature names...")
    df = parse_feature_names(df)

    # Create visualizations
    print("\nCreating visualizations...")
    plot_correlation_boxplot(df)
    plot_feature_heatmap(df)
    plot_compression_heatmap(df)
    plot_compartment_heatmap(df)

    print("\nAnalysis complete!")
    return df, features_per_compression


if __name__ == "__main__":
    df, features_per_compression = main()
