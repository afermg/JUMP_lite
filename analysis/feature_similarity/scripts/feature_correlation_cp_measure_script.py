"""
Feature Correlation Analysis for CellProfiler Measures
Converted from Marimo notebook to standalone Python script with functions

Analyzes correlation between features extracted from compressed vs uncompressed images.
"""

from utils_cp_measure_name_mapping import get_cpm_to_measurement_mapper as get_name_mapper

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
    df["full_feature_name"] = df["feature"]
    df[["compartment", "mode", "feature"]] = df.feature.str.split("/", expand=True)
    
    return df


def plot_correlation_violinplot(df, output_path="../output_figures/correlation_violinplot.png"):
    """
    Create violin + strip plot of feature correlations across compression methods.

    Args:
        df: DataFrame with correlation results
        output_path: Path to save plot
    """
    # Filter to lossy codecs only (exclude zstd reference)
    codec_order = [
        "jpegxl_lossy_lq.zarr",
        "jpegxl_lossy_mq.zarr",
        "jpegxl_lossy_effort_3.zarr",
        "jpegxl_lossy_hq.zarr",
    ]
    df_filtered = df[df['key'].isin(codec_order)].copy()
    df_filtered['codec'] = df_filtered['key'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')

    display_order = ['jxl_lq', 'jxl_mq', 'jxl_effort_3', 'jxl_hq']

    # Stats for tick labels
    codec_stats = df_filtered.groupby('codec').agg(
        n_total=('corr', 'count'),
    )
    codec_labels = {
        codec: f"{codec}\nn={int(codec_stats.loc[codec, 'n_total'])}"
        for codec in display_order if codec in codec_stats.index
    }
    label_order = [c for c in display_order if c in codec_stats.index]

    fig, ax = plt.subplots(figsize=(6, 6))

    sns.violinplot(
        data=df_filtered,
        x='codec',
        y='corr',
        order=label_order,
        palette='viridis',
        inner='box',
        cut=0,
        ax=ax
    )

    # sns.stripplot(
    #     data=df_filtered,
    #     x='codec',
    #     y='corr',
    #     order=label_order,
    #     color='black',
    #     alpha=0.1,
    #     size=3,
    #     jitter=True,
    #     ax=ax
    # )

    ax.set_xlabel('Codec', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Feature Correlations Across Compression Methods',
                 fontsize=14, fontweight='bold')

    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([codec_labels[c] for c in label_order])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved correlation plot to: {output_path}")
    plt.show()


def plot_feature_heatmap(df, compression_key="jpegxl_lossy_effort_3.zarr",
                        output_path="../output_figures/feature_heatmap.png"):
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


def plot_compression_heatmap(df, output_path="../output_figures/compression_heatmap.png"):
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


def plot_compartment_heatmap(df, output_path="../output_figures/compartment_heatmap.png"):
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


def map_features_to_groups(df):
    """
    Map feature names to their measurement groups using the CellProfiler name mapper.

    Args:
        df: DataFrame with 'feature' column containing feature names

    Returns:
        pd.DataFrame: DataFrame with added 'feature_group' column
    """
    
    # Get string before first upper case letter as key
    df['feature_group'] = df['feature'].str.extract(r'^([a-z]+)', expand=False)

    return df


def plot_feature_group_boxplot(df, output_path="../output_figures/feature_group_boxplot.png"):
    """
    Create boxplot of median feature correlations grouped by feature group.

    Shows feature groups on x-axis with separate boxes for each codec within each group.

    Args:
        df: DataFrame with correlation results (must have 'feature_group' column)
        output_path: Path to save plot
    """
    # Define codec order (excluding zstd since we're comparing against it)
    codec_order = [
        "jpegxl_lossy_lq.zarr",
        "jpegxl_lossy_mq.zarr",
        "jpegxl_lossy_effort_3.zarr",
        "jpegxl_lossy_hq.zarr",
    ]

    # Filter to only include codecs we want to compare (exclude zstd)
    df_filtered = df[df['key'].isin(codec_order)].copy()

    # Clean up codec names for display
    df_filtered['codec'] = df_filtered['key'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')

    # Get unique feature groups sorted by median correlation
    group_stats = df_filtered.groupby('feature_group').agg(
        median_corr=('corr', 'median'),
        n_total=('corr', 'count'),
        n_features=('feature', 'nunique')
    ).sort_values('median_corr', ascending=False)

    # Calculate per-codec counts
    n_codecs = df_filtered['codec'].nunique()
    group_stats['n_per_codec'] = (group_stats['n_total'] / n_codecs).astype(int)
    group_order = group_stats.index.tolist()

    # Nice display names for feature groups
    nice_names = {
        'sizeshape': 'Size & Shape',
        'radial': 'Radial Distribution',
        'zernike': 'Zernike',
        'intensity': 'Intensity',
        'texture': 'Texture',
        'correlation': 'Correlation',
        'granularity': 'Granularity',
        'location': 'Location',
    }

    # Create labels with condensed counts (total:per_codec:feat)
    group_labels = {
        grp: f"{nice_names.get(grp, grp)}\nn={group_stats.loc[grp, 'n_total']}:{group_stats.loc[grp, 'n_per_codec']}:{group_stats.loc[grp, 'n_features']}"
        for grp in group_order
    }

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.boxplot(
        data=df_filtered,
        x='feature_group',
        y='corr',
        hue='codec',
        order=group_order,
        hue_order=['jxl_lq', 'jxl_mq', 'jxl_effort_3', 'jxl_hq'],
        palette='viridis',
        ax=ax
    )

    ax.set_xlabel('Feature Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Correlation with ZSTD', fontsize=12, fontweight='bold')
    ax.set_title('Feature Correlation by Feature Group Across Compression Methods',
                 fontsize=14, fontweight='bold')

    # Update x-axis labels to include feature counts
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels([group_labels[grp] for grp in group_order], rotation=45, ha='right')

    # Add horizontal line at 1.0 for reference
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect correlation')

    # Adjust legend
    ax.legend(title='Codec', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved feature group boxplot to: {output_path}")
    plt.show()


def plot_feature_group_violinplot(df, output_path="../output_figures/feature_group_violinplot.png"):
    """
    Create violinplot of feature correlations grouped by feature group.

    Shows distribution of correlations for each feature group with
    separate violins for each codec.

    Args:
        df: DataFrame with correlation results (must have 'feature_group' column)
        output_path: Path to save plot
    """
    # Define codec order (excluding zstd since we're comparing against it)
    codec_order = [
        "jpegxl_lossy_lq.zarr",
        "jpegxl_lossy_mq.zarr",
        "jpegxl_lossy_effort_3.zarr",
        "jpegxl_lossy_hq.zarr",
    ]

    # Filter to only include codecs we want to compare (exclude zstd)
    df_filtered = df[df['key'].isin(codec_order)].copy()

    # Clean up codec names for display
    df_filtered['codec'] = df_filtered['key'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')

    # Get unique feature groups sorted by median correlation
    group_stats = df_filtered.groupby('feature_group').agg(
        median_corr=('corr', 'median'),
        n_total=('corr', 'count'),
        n_features=('feature', 'nunique')
    ).sort_values('median_corr', ascending=False)

    # Calculate per-codec counts
    n_codecs = df_filtered['codec'].nunique()
    group_stats['n_per_codec'] = (group_stats['n_total'] / n_codecs).astype(int)
    group_order = group_stats.index.tolist()

    # Create labels with condensed counts (total:per_codec:feat)
    group_labels = {
        grp: f"{grp}\nn={group_stats.loc[grp, 'n_total']}:{group_stats.loc[grp, 'n_per_codec']}:{group_stats.loc[grp, 'n_features']}"
        for grp in group_order
    }

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))

    sns.violinplot(
        data=df_filtered,
        x='feature_group',
        y='corr',
        hue='codec',
        order=group_order,
        hue_order=['jxl_lq', 'jxl_mq', 'jxl_effort_3', 'jxl_hq'],
        palette='viridis',
        inner='box',
        cut=0,
        ax=ax
    )

    ax.set_xlabel('Feature Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Correlation with ZSTD', fontsize=12, fontweight='bold')
    ax.set_title('Feature Correlation by Feature Group Across Compression Methods',
                 fontsize=14, fontweight='bold')

    # Update x-axis labels to include feature counts
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels([group_labels[grp] for grp in group_order], rotation=45, ha='right')

    # Add horizontal line at 1.0 for reference
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect correlation')

    # Adjust legend
    ax.legend(title='Codec', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved feature group violinplot to: {output_path}")
    plt.show()


def plot_feature_group_by_compartment(df, output_path="../output_figures/feature_group_by_compartment.png"):
    """
    Create boxplots for SizeShape, RadialDistribution, and Zernike (AreaShape) features,
    with separation by both codec and compartment.

    Args:
        df: DataFrame with correlation results (must have 'feature_group' and 'compartment' columns)
        output_path: Path to save plot
    """
    # Define codec order (excluding zstd since we're comparing against it)
    codec_order = [
        "jpegxl_lossy_lq.zarr",
        "jpegxl_lossy_mq.zarr",
        "jpegxl_lossy_effort_3.zarr",
        "jpegxl_lossy_hq.zarr",
    ]

    # Filter to only include codecs we want to compare (exclude zstd)
    df_filtered = df[df['key'].isin(codec_order)].copy()

    # Clean up codec names for display
    df_filtered['codec'] = df_filtered['key'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')

    # Define the three feature groups to plot
    feature_groups = ['sizeshape', 'radial', 'zernike']
    feature_group_titles = {
        'sizeshape': 'SizeShape Features',
        'radial': 'Radial Distribution Features',
        'zernike': 'Zernike Features'
    }

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)

    for idx, feature_group in enumerate(feature_groups):
        ax = axes[idx]

        # Filter data for this feature group
        df_group = df_filtered[df_filtered['feature_group'] == feature_group].copy()

        if len(df_group) == 0:
            ax.text(0.5, 0.5, f'No data for {feature_group}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature_group_titles.get(feature_group, feature_group))
            continue

        # Get compartment order sorted alphabetically
        compartment_order = sorted(df_group['compartment'].unique())

        # Calculate stats for labels
        compartment_stats = df_group.groupby('compartment').agg(
            n_total=('corr', 'count'),
            n_features=('feature', 'nunique')
        )
        n_codecs = df_group['codec'].nunique()
        compartment_stats['n_per_codec'] = (compartment_stats['n_total'] / n_codecs).astype(int)

        # Create labels with condensed counts (total:per_codec:feat)
        compartment_labels = {
            comp: f"{comp}\nn={compartment_stats.loc[comp, 'n_total']}:{compartment_stats.loc[comp, 'n_features']}"
            for comp in compartment_order
        }

        sns.boxplot(
            data=df_group,
            x='compartment',
            y='corr',
            hue='codec',
            order=compartment_order,
            hue_order=['jxl_lq', 'jxl_mq', 'jxl_effort_3', 'jxl_hq'],
            palette='viridis',
            ax=ax
        )

        ax.set_xlabel('Compartment', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Feature Correlation with ZSTD', fontsize=11, fontweight='bold')
        else:
            ax.set_ylabel('')

        ax.set_title(feature_group_titles.get(feature_group, feature_group), fontsize=12, fontweight='bold')

        # Update x-axis labels
        ax.set_xticks(range(len(compartment_order)))
        ax.set_xticklabels([compartment_labels[comp] for comp in compartment_order], rotation=45, ha='right', fontsize=9)

        # Add horizontal line at 1.0 for reference
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        # Only show legend on the last subplot
        if idx < 2:
            ax.get_legend().remove()
        else:
            ax.legend(title='Codec', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.suptitle('Feature Correlation by Compartment and Codec', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved feature group by compartment plot to: {output_path}")
    plt.show()


def extract_feature_subcategory(feature_name):
    """
    Extract subcategory from feature name.

    Examples:
        radial_zernikesRadialDistribution_ZernikePhase_7_3 -> zernikesRadialDistribution_ZernikePhase
        radial_distributionRadialDistribution_FracAtD_4of4 -> distributionRadialDistribution_FracAtD
        radial_distributionRadialDistribution_MeanFrac_1of4 -> distributionRadialDistribution_MeanFrac
        zernikeZernike_4_0 -> Zernike
        sizeshapeArea -> Area
        sizeshapeCentralMoment_0_0 -> CentralMoment
    """
    import re

    if pd.isna(feature_name):
        return 'Unknown'

    # Handle zernike features: zernikeZernike_4_0 -> 4_0
    if feature_name.startswith('zernike') and 'Zernike' in feature_name:
        # Extract the trailing X_Y part
        match = re.search(r'_(\d+_\d+)$', feature_name)
        if match:
            return match.group(1)
        return 'Zernike'

    # Handle radial features: strip 'radial_' prefix and trailing numeric params
    # radial_zernikesRadialDistribution_ZernikePhase_7_3 -> zernikesRadialDistribution_ZernikePhase
    if feature_name.startswith('radial_'):
        rest = feature_name[len('radial_'):]
        # Remove trailing numeric parameters like _7_3, _4of4, _1of4
        rest = re.sub(r'_\d+_\d+$', '', rest)  # Remove _X_Y suffix
        rest = re.sub(r'_\d+of\d+$', '', rest)  # Remove _Xof4 suffix
        return rest

    # Handle sizeshape features - use original logic
    # Strip lowercase prefix and get the part starting with uppercase
    match = re.match(r'^[a-z]+([A-Z].*)', feature_name)
    if match:
        rest = match.group(1)
        # Handle Moment features
        moment_prefixes = ['CentralMoment', 'HuMoment', 'NormalizedMoment', 'SpatialMoment',
                           'InertiaTensor', 'InertiaTensorEigenvalues']
        for prefix in moment_prefixes:
            if rest.startswith(prefix):
                return prefix
        # For other features, get the CamelCase word before any underscore or number
        match2 = re.match(r'^([A-Z][a-zA-Z]*)', rest)
        if match2:
            return match2.group(1)
        return rest

    return feature_name


def plot_feature_subgroups_by_compartment(df, output_path="../output_figures/feature_subgroups_by_compartment.png"):
    """
    Create boxplots with rows for cell/nuclei compartments and columns for feature groups,
    grouped by feature subcategory within each subplot.

    Args:
        df: DataFrame with correlation results
        output_path: Path to save plot
    """
    # Define codec order (excluding zstd since we're comparing against it)
    codec_order = [
        "jpegxl_lossy_lq.zarr",
        "jpegxl_lossy_mq.zarr",
        "jpegxl_lossy_effort_3.zarr",
        "jpegxl_lossy_hq.zarr",
    ]

    # Filter to only include codecs we want to compare (exclude zstd)
    df_filtered = df[df['key'].isin(codec_order)].copy()

    # Clean up codec names for display
    df_filtered['codec'] = df_filtered['key'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')

    # Clean up compartment names (e.g., 'nuclei_1' -> 'nuclei')
    df_filtered['compartment_clean'] = df_filtered['compartment'].str.extract(r'^([a-zA-Z]+)', expand=False)

    # Extract feature subcategory
    df_filtered['feature_subcategory'] = df_filtered['feature'].apply(extract_feature_subcategory)

    # Define the feature groups and compartments
    feature_groups = ['sizeshape', 'radial', 'zernike']
    feature_group_titles = {
        'sizeshape': 'SizeShape',
        'radial': 'Radial Distribution',
        'zernike': 'Zernike (AreaShape)'
    }
    compartments = ['cell', 'nuclei']

    # Create figure with 2 rows (compartments) × 3 columns (feature groups)
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), sharey=True)

    for row_idx, compartment in enumerate(compartments):
        for col_idx, feature_group in enumerate(feature_groups):
            ax = axes[row_idx, col_idx]

            # Filter data for this compartment and feature group
            df_subset = df_filtered[
                (df_filtered['compartment_clean'] == compartment) &
                (df_filtered['feature_group'] == feature_group)
            ].copy()

            if len(df_subset) == 0:
                ax.text(0.5, 0.5, f'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{compartment.capitalize()} - {feature_group_titles.get(feature_group, feature_group)}")
                continue

            # Calculate stats for labels
            subcat_stats = df_subset.groupby('feature_subcategory').agg(
                n_total=('corr', 'count'),
                n_features=('feature', 'nunique')
            )
            n_codecs = df_subset['codec'].nunique()
            subcat_stats['n_per_codec'] = (subcat_stats['n_total'] / n_codecs).astype(int)

            # Get subcategory order - sort by name for zernike, by median for others
            if feature_group == 'zernike':
                subcat_order = sorted(df_subset['feature_subcategory'].unique())
            else:
                subcat_order = df_subset.groupby('feature_subcategory')['corr'].median().sort_values(ascending=False).index.tolist()

            # Create condensed labels (single line for sizeshape/zernike, two lines for radial)
            if feature_group in ['sizeshape', 'zernike']:
                subcat_labels = {
                    subcat: f"{subcat} (n={subcat_stats.loc[subcat, 'n_total']}:{subcat_stats.loc[subcat, 'n_features']})"
                    for subcat in subcat_order
                }
            else:
                subcat_labels = {
                    subcat: f"{subcat}\nn={subcat_stats.loc[subcat, 'n_total']}:{subcat_stats.loc[subcat, 'n_features']}"
                    for subcat in subcat_order
                }

            sns.boxplot(
                data=df_subset,
                x='feature_subcategory',
                y='corr',
                hue='codec',
                order=subcat_order,
                hue_order=['jxl_lq', 'jxl_mq', 'jxl_effort_3', 'jxl_hq'],
                palette='viridis',
                ax=ax
            )

            # Set labels
            if row_idx == 1:
                ax.set_xlabel('Feature Subcategory', fontsize=10, fontweight='bold')
            else:
                ax.set_xlabel('')

            if col_idx == 0:
                ax.set_ylabel(f'{compartment.capitalize()}\nCorrelation with ZSTD', fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel('')

            # Set title only for top row
            if row_idx == 0:
                ax.set_title(feature_group_titles.get(feature_group, feature_group), fontsize=12, fontweight='bold')

            # Update x-axis labels
            ax.set_xticks(range(len(subcat_order)))
            ax.set_xticklabels([subcat_labels[s] for s in subcat_order], rotation=45, ha='right', fontsize=8)

            # Add horizontal line at 1.0 for reference
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

            # Remove legend except for top-right subplot
            if not (row_idx == 0 and col_idx == 2):
                if ax.get_legend():
                    ax.get_legend().remove()
            else:
                ax.legend(title='Codec', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

    plt.suptitle('Feature Correlation by Subcategory: Cell vs Nuclei', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved feature subgroups by compartment plot to: {output_path}")
    plt.show()


def main():
    """Main pipeline for feature correlation analysis."""
    # Configuration
    workspace_dir = Path("/work") / "datasets" / "aliby_output" / "cp_measure" /"jump_target2_4plate"
    cache_dir = Path(workspace_dir) / "db_cache"
    output_dir = Path("../output_figures")

    # If output_dir / "feature_correlations.parquet" exists, ask if to rerun or load and skip
    if (output_dir / "feature_correlations.parquet").exists():
        rerun = input(f"{output_dir / 'feature_correlations.parquet'} exists. Rerun analysis? (y/n): ")
        if rerun.lower() != 'y':
            df = pd.read_parquet(output_dir / "feature_correlations.parquet")
            print("Loaded existing correlation results.")
            rerun = False
        else:
            rerun = True
    
    # For local testing, you can override this:
    # workspace_dir = Path("path/to/your/workspace")
    if rerun:
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

        df.to_parquet(output_dir / "feature_correlations.parquet")
        print(f"Saved correlation results to: {output_dir / 'feature_correlations.parquet'}")
    
    # Map features to groups
    print("\nMapping features to groups...")
    df = map_features_to_groups(df)

    # Create visualizations
    print("\nCreating visualizations...")
    plot_correlation_violinplot(df)
    plot_feature_heatmap(df)
    plot_compression_heatmap(df)
    plot_compartment_heatmap(df)
    plot_feature_group_boxplot(df)
    plot_feature_group_violinplot(df)
    plot_feature_group_by_compartment(df)
    plot_feature_subgroups_by_compartment(df)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
