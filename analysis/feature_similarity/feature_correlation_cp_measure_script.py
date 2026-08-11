"""
Feature Correlation Analysis for CellProfiler Measures
Converted from Marimo notebook to standalone Python script with functions

Analyzes correlation between features extracted from compressed vs uncompressed images.
"""

from utils_cp_measure_name_mapping import get_cpm_to_measurement_mapper as get_name_mapper

import argparse
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

SCRIPT_DIR = Path(__file__).resolve().parent


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


# Fixed codec order: best to worst quality (left to right)
_CODEC_ORDER = [
    'jxl_hq', 'jxl_effort_3', 'jxl_d2_e8', 'jxl_mq', 'jxl_lq',
    'jxl_d10', 'jxl_d15', 'jxl_d20_e2', 'jxl_d30',
]
_KEY_ORDER = [f'jpegxl_lossy_{c.replace("jxl_", "")}.zarr' for c in _CODEC_ORDER]
_CODEC_LABELS = {
    'jxl_hq': 'HQ', 'jxl_effort_3': 'E3', 'jxl_d2_e8': 'D2-E8',
    'jxl_mq': 'MQ', 'jxl_lq': 'LQ', 'jxl_d10': 'D10',
    'jxl_d15': 'D15', 'jxl_d20_e2': 'D20', 'jxl_d30': 'D25',
}


def _prepare_codec_data(df):
    """Prepare codec-filtered data with labels and ordering.

    Returns (df_filtered, label_order, codec_labels).
    """
    df_filtered = df[df['key'] != 'zstd.zarr'].copy()
    df_filtered['codec'] = df_filtered['key'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')
    present_codecs = set(df_filtered['codec'].unique())
    label_order = [c for c in _CODEC_ORDER if c in present_codecs]
    # Append any unknown codecs at the end
    label_order += sorted(c for c in present_codecs if c not in _CODEC_ORDER)
    codec_labels = {c: _CODEC_LABELS.get(c, c.replace('jxl_', '').upper()) for c in label_order}
    return df_filtered, label_order, codec_labels


def plot_correlation_violinplot(df, output_path=SCRIPT_DIR / "output" / "correlation_violinplot.png"):
    """
    Create violin + strip plot of feature correlations across compression methods.

    Args:
        df: DataFrame with correlation results
        output_path: Path to save plot
    """
    df_filtered, label_order, codec_labels = _prepare_codec_data(df)

    fig, ax = plt.subplots(figsize=(7, 7))

    sns.violinplot(
        data=df_filtered,
        x='codec',
        y='corr',
        hue='codec',
        order=label_order,
        hue_order=label_order,
        palette='viridis',
        inner='box',
        cut=0,
        legend=False,
        ax=ax
    )

    ax.set_xlabel('', fontsize=24, fontweight='bold')
    ax.set_ylabel('Feature Correlation', fontsize=24, fontweight='bold')
    ax.set_title('', fontsize=24, fontweight='bold')

    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved correlation plot to: {output_path}")
    plt.close(fig)


def plot_correlation_boxenplot(df, output_path=SCRIPT_DIR / "output" / "correlation_boxenplot.png"):
    """
    Create boxenplot (letter-value plot) of feature correlations across compression methods.

    Args:
        df: DataFrame with correlation results
        output_path: Path to save plot
    """
    df_filtered, label_order, codec_labels = _prepare_codec_data(df)

    fig, ax = plt.subplots(figsize=(7, 7))

    sns.boxenplot(
        data=df_filtered,
        x='codec',
        y='corr',
        hue='codec',
        order=label_order,
        hue_order=label_order,
        palette='viridis',
        legend=False,
        ax=ax
    )

    ax.set_xlabel('', fontsize=24, fontweight='bold')
    ax.set_ylabel('Feature Correlation', fontsize=24, fontweight='bold')
    ax.set_title('', fontsize=24, fontweight='bold')

    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved correlation boxenplot to: {output_path}")
    plt.close(fig)


def plot_feature_heatmap(df, compression_key="jpegxl_lossy_effort_3.zarr",
                        output_path=SCRIPT_DIR / "output" / "feature_heatmap.png"):
    """
    Create heatmap of correlations by feature and compartment.

    Args:
        df: DataFrame with correlation results (must have compartment column)
        compression_key: Which compression method to visualize
        output_path: Path to save plot
    """
    perf = df[df.key == compression_key].groupby(["feature", "compartment"], as_index=False)["corr"].mean()
    pivot_df = perf.pivot(index="feature", columns="compartment", values="corr")

    fig = plt.figure(figsize=(30, 60))
    sns.heatmap(pivot_df, cmap="viridis", cbar_kws={'label': 'Correlation Coefficient'}, vmin=0, vmax=1)
    plt.title("Feature Correlation Across Compression Methods")
    plt.xlabel("Compartment")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved feature heatmap to: {output_path}")
    plt.close(fig)


def plot_compression_heatmap(df, output_path=SCRIPT_DIR / "output" / "compression_heatmap.png"):
    """
    Create heatmap of median correlations by feature and compression method.

    Args:
        df: DataFrame with correlation results
        output_path: Path to save plot
    """
    perf = df.groupby(["feature", "key"], as_index=False)["corr"].median()

    present_keys = set(df['key'].unique())
    order = [k for k in _KEY_ORDER if k in present_keys]
    order += sorted(k for k in present_keys if k not in _KEY_ORDER and k != 'zstd.zarr')
    if 'zstd.zarr' in present_keys:
        order.append('zstd.zarr')

    pivot_df = perf.pivot(index="feature", columns="key", values="corr").reindex(columns=order)

    fig = plt.figure(figsize=(30, 60))
    sns.heatmap(pivot_df, cmap="viridis", cbar_kws={'label': 'Correlation Coefficient'}, vmin=0, vmax=1)
    plt.title("Feature Correlation Across Compression Methods")
    plt.xlabel("Compression strategy")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved compression heatmap to: {output_path}")
    plt.close(fig)


def plot_compartment_heatmap(df, output_path=SCRIPT_DIR / "output" / "compartment_heatmap.png"):
    """
    Create heatmap of median correlations by compartment and compression method.

    Args:
        df: DataFrame with correlation results (must have compartment column)
        output_path: Path to save plot
    """
    perf = df.groupby(["compartment", "key"], as_index=False)["corr"].median()

    present_keys = set(df['key'].unique())
    order = [k for k in _KEY_ORDER if k in present_keys]
    order += sorted(k for k in present_keys if k not in _KEY_ORDER and k != 'zstd.zarr')
    if 'zstd.zarr' in present_keys:
        order.append('zstd.zarr')

    pivot_df = perf.pivot(index="compartment", columns="key", values="corr").reindex(columns=order)

    fig = plt.figure(figsize=(5, 10))
    sns.heatmap(pivot_df, cmap="viridis", cbar_kws={'label': 'Correlation Coefficient'},
                vmin=0, vmax=1, annot=True)
    plt.title("Feature Correlation Across Compression Methods")
    plt.xlabel("Compression strategy")
    plt.ylabel("Compartment")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved compartment heatmap to: {output_path}")
    plt.close(fig)


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


def plot_feature_group_boxplot(df, output_path=SCRIPT_DIR / "output" / "feature_group_boxplot.png"):
    """
    Create boxplot of median feature correlations grouped by feature group.

    Shows feature groups on x-axis with separate boxes for each codec within each group.

    Args:
        df: DataFrame with correlation results (must have 'feature_group' column)
        output_path: Path to save plot
    """
    # Dynamic codec discovery (excluding zstd since we're comparing against it)
    df_filtered = df[df['key'] != 'zstd.zarr'].copy()
    df_filtered['codec'] = df_filtered['key'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')
    codec_mean_corr = df_filtered.groupby('codec')['corr'].mean().sort_values(ascending=False)
    label_order = list(codec_mean_corr.index)
    _known_labels = {
        'jxl_lq': 'Low', 'jxl_mq': 'Medium', 'jxl_effort_3': 'Mid-High',
        'jxl_hq': 'High', 'jxl_d2_e8': 'D2 E8', 'jxl_d10': 'D10',
        'jxl_d15': 'D15', 'jxl_d20_e2': 'D20 E2', 'jxl_d30': 'D25',
    }
    codec_labels = {c: _known_labels.get(c, c.replace('jxl_', '').upper()) for c in label_order}

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
        hue_order=label_order,
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
    plt.close(fig)


def plot_feature_group_violinplot(df, output_path=SCRIPT_DIR / "output" / "feature_group_violinplot.png"):
    """
    Create violinplot of feature correlations grouped by feature group.

    Shows distribution of correlations for each feature group with
    separate violins for each codec.

    Args:
        df: DataFrame with correlation results (must have 'feature_group' column)
        output_path: Path to save plot
    """
    # Dynamic codec discovery (excluding zstd since we're comparing against it)
    df_filtered = df[df['key'] != 'zstd.zarr'].copy()
    df_filtered['codec'] = df_filtered['key'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')
    codec_mean_corr = df_filtered.groupby('codec')['corr'].mean().sort_values(ascending=False)
    label_order = list(codec_mean_corr.index)
    _known_labels = {
        'jxl_lq': 'Low', 'jxl_mq': 'Medium', 'jxl_effort_3': 'Mid-High',
        'jxl_hq': 'High', 'jxl_d2_e8': 'D2 E8', 'jxl_d10': 'D10',
        'jxl_d15': 'D15', 'jxl_d20_e2': 'D20 E2', 'jxl_d30': 'D25',
    }
    codec_labels = {c: _known_labels.get(c, c.replace('jxl_', '').upper()) for c in label_order}

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
        hue_order=label_order,
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
    plt.close(fig)


def plot_feature_group_by_compartment(df, output_path=SCRIPT_DIR / "output" / "feature_group_by_compartment.png"):
    """
    Create boxplots for SizeShape, RadialDistribution, and Zernike (AreaShape) features,
    with separation by both codec and compartment.

    Args:
        df: DataFrame with correlation results (must have 'feature_group' and 'compartment' columns)
        output_path: Path to save plot
    """
    # Dynamic codec discovery (excluding zstd since we're comparing against it)
    df_filtered = df[df['key'] != 'zstd.zarr'].copy()
    df_filtered['codec'] = df_filtered['key'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')
    codec_mean_corr = df_filtered.groupby('codec')['corr'].mean().sort_values(ascending=False)
    label_order = list(codec_mean_corr.index)
    _known_labels = {
        'jxl_lq': 'Low', 'jxl_mq': 'Medium', 'jxl_effort_3': 'Mid-High',
        'jxl_hq': 'High', 'jxl_d2_e8': 'D2 E8', 'jxl_d10': 'D10',
        'jxl_d15': 'D15', 'jxl_d20_e2': 'D20 E2', 'jxl_d30': 'D25',
    }
    codec_labels = {c: _known_labels.get(c, c.replace('jxl_', '').upper()) for c in label_order}

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
            hue_order=label_order,
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
    plt.close(fig)


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


def plot_feature_subgroups_by_compartment(df, output_path=SCRIPT_DIR / "output" / "feature_subgroups_by_compartment.png"):
    """
    Create boxplots with rows for cell/nuclei compartments and columns for feature groups,
    grouped by feature subcategory within each subplot.

    Args:
        df: DataFrame with correlation results
        output_path: Path to save plot
    """
    # Dynamic codec discovery (excluding zstd since we're comparing against it)
    df_filtered = df[df['key'] != 'zstd.zarr'].copy()
    df_filtered['codec'] = df_filtered['key'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')
    codec_mean_corr = df_filtered.groupby('codec')['corr'].mean().sort_values(ascending=False)
    label_order = list(codec_mean_corr.index)
    _known_labels = {
        'jxl_lq': 'Low', 'jxl_mq': 'Medium', 'jxl_effort_3': 'Mid-High',
        'jxl_hq': 'High', 'jxl_d2_e8': 'D2 E8', 'jxl_d10': 'D10',
        'jxl_d15': 'D15', 'jxl_d20_e2': 'D20 E2', 'jxl_d30': 'D25',
    }
    codec_labels = {c: _known_labels.get(c, c.replace('jxl_', '').upper()) for c in label_order}

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

    # Create figure with 2 rows (compartments) x 3 columns (feature groups)
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
                hue_order=label_order,
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
    plt.close(fig)


def plot_features_ordered_by_noise(df, noisy_features_path="data/aliby_output/archived_runs/tables/noisy_features.parquet",
                                    output_path=SCRIPT_DIR / "output" / "features_ordered_by_noise.png"):
    """
    Create 3 subplots showing correlation values with features ordered by noise metrics.

    Each subplot orders features on x-axis by a different metric from noisy_features.parquet:
    - avg (average noise)
    - std (standard deviation of noise)
    - var (variance of noise)

    Args:
        df: DataFrame with correlation results (must have 'feature' and 'corr' columns)
        noisy_features_path: Path to the noisy features parquet file
        output_path: Path to save plot
    """
    # Load noisy features data
    noisy_df = pd.read_parquet(noisy_features_path)

    # Dynamic codec discovery (including zstd)
    df_filtered = df.copy()
    df_filtered['codec'] = df_filtered['key'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')
    codec_mean_corr = df_filtered.groupby('codec')['corr'].mean().sort_values(ascending=False)
    codec_display_order = list(codec_mean_corr.index)
    palette = sns.color_palette('viridis', n_colors=len(codec_display_order))
    color_map = dict(zip(codec_display_order, palette))

    # Merge correlation data with noisy features
    # Use 'feature' column (not full_feature_name) as it matches the noisy features naming
    feature_col = 'feature'
    df_merged = df_filtered.merge(noisy_df, left_on=feature_col, right_on='feature', suffixes=('', '_noise'))

    if len(df_merged) == 0:
        print(f"Warning: No features matched between correlation data and noisy features data")
        print(f"Correlation features sample: {df_filtered[feature_col].head().tolist()}")
        print(f"Noisy features sample: {noisy_df['feature'].head().tolist()}")
        return

    print(f"Matched {df_merged[feature_col].nunique()} features out of {df_filtered[feature_col].nunique()} correlation features")

    # Define the four ordering metrics
    order_columns = ['avg', 'std', 'var', 'ratio']
    order_titles = {
        'avg': 'Features Ordered by Average Noise',
        'std': 'Features Ordered by Noise Std Dev',
        'var': 'Features Ordered by Noise Variance',
        'ratio': 'Features Ordered by Noise Ratio'
    }

    # Create figure with 4 stacked subplots
    fig, axes = plt.subplots(4, 1, figsize=(20, 24), sharex=True)

    for idx, order_col in enumerate(order_columns):
        ax = axes[idx]

        # Get feature order based on the noise metric (ascending)
        feature_order_df = noisy_df[['feature', order_col]].drop_duplicates().sort_values(order_col, ascending=True)
        feature_order = feature_order_df['feature'].tolist()

        # Filter to only features present in merged data
        features_in_data = df_merged[feature_col].unique()
        feature_order = [f for f in feature_order if f in features_in_data]

        # Create numeric positions for features
        feature_positions = {f: i for i, f in enumerate(feature_order)}
        df_plot = df_merged[df_merged[feature_col].isin(feature_order)].copy()
        df_plot['feature_pos'] = df_plot[feature_col].map(feature_positions)

        # Plot each codec
        for codec in codec_display_order:
            codec_data = df_plot[df_plot['codec'] == codec]
            ax.scatter(
                codec_data['feature_pos'],
                codec_data['corr'],
                c=[color_map[codec]],
                label=codec,
                alpha=0.6,
                s=10
            )

        # Only show x-axis label on bottom subplot
        if idx == 3:
            ax.set_xlabel('Features', fontsize=11, fontweight='bold')

        ax.set_ylabel('Correlation', fontsize=11, fontweight='bold')
        ax.set_title(order_titles[order_col], fontsize=12, fontweight='bold')

        # Add horizontal line at 1.0 for reference
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        # Only show legend on the last subplot
        if idx == 3:
            ax.legend(title='Codec', bbox_to_anchor=(1.02, 1), loc='upper left')

        # Set x limits
        ax.set_xlim(-5, len(feature_order) + 5)

        # Add feature count annotation
        ax.text(0.02, 0.02, f'n={len(feature_order)} features',
                transform=ax.transAxes, fontsize=9, verticalalignment='bottom')

    plt.suptitle('Feature Correlations Ordered by Noise Metrics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved features ordered by noise plot to: {output_path}")
    plt.close(fig)


def plot_feature_similarity_vs_correlation(df,
                                            raw_features_path="data/features/cp_measure_jump_target2_4plate_zstd_raw_features.parquet",
                                            output_path=SCRIPT_DIR / "output" / "feature_similarity_vs_correlation.png"):
    """
    Create scatter plots of feature cosine similarity stats vs correlation.

    Loads raw features, computes cosine similarity matrix between all features,
    then calculates mean, std, max, min for each feature's similarities.

    Args:
        df: DataFrame with correlation results (must have 'feature' and 'corr' columns)
        raw_features_path: Path to the raw features parquet file
        output_path: Path to save plot
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Load raw features data
    print(f"Loading raw features from: {raw_features_path}")
    raw_df = pd.read_parquet(raw_features_path)

    # Get numeric columns (features) - exclude metadata columns
    meta_cols = [c for c in raw_df.columns if c.startswith('Metadata_') or c == 'site']
    feature_cols = [c for c in raw_df.columns if c not in meta_cols]

    print(f"Found {len(feature_cols)} feature columns")

    # Extract feature matrix (samples x features) and transpose to (features x samples)
    feature_matrix = raw_df[feature_cols].values.T

    # Handle NaN values - fill with column mean
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

    # Calculate cosine similarity (features x features)
    print("Calculating cosine similarity matrix...")
    cos_sim_matrix = cosine_similarity(feature_matrix)

    # Set diagonal to NaN so we don't include self-similarity in stats
    np.fill_diagonal(cos_sim_matrix, np.nan)

    # Calculate stats for each feature (mean, std, max, min of similarities)
    with np.errstate(all='ignore'):
        feature_sim_stats = pd.DataFrame({
            'full_feature': feature_cols,
            'sim_mean': np.nanmean(cos_sim_matrix, axis=1),
            'sim_std': np.nanstd(cos_sim_matrix, axis=1),
            'sim_max': np.nanmax(cos_sim_matrix, axis=1),
            'sim_min': np.nanmin(cos_sim_matrix, axis=1)
        })

    # Extract feature name from full path (last part after '/')
    feature_sim_stats['feature'] = feature_sim_stats['full_feature'].str.split('/').str[-1]

    # Average similarity stats per feature group
    feature_sim_stats = feature_sim_stats.groupby('feature').agg({
        'sim_mean': 'mean',
        'sim_std': 'mean',
        'sim_max': 'mean',
        'sim_min': 'mean'
    }).reset_index()

    print(f"Calculated similarity stats for {len(feature_sim_stats)} unique features (after grouping)")

    # Dynamic codec discovery (including zstd)
    df_filtered = df.copy()
    df_filtered['codec'] = df_filtered['key'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')
    codec_mean_corr = df_filtered.groupby('codec')['corr'].mean().sort_values(ascending=False)
    codec_display_order = list(codec_mean_corr.index)
    palette = sns.color_palette('viridis', n_colors=len(codec_display_order))
    color_map = dict(zip(codec_display_order, palette))

    # Merge correlation data with similarity stats
    df_merged = df_filtered.merge(feature_sim_stats, on='feature', how='inner')

    if len(df_merged) == 0:
        print(f"Warning: No features matched between correlation data and similarity stats")
        print(f"Correlation features sample: {df_filtered['feature'].head().tolist()}")
        print(f"Similarity features sample: {feature_sim_stats['feature'].head().tolist()}")
        return

    print(f"Matched {df_merged['feature'].nunique()} features for similarity vs correlation plot")

    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    stat_columns = ['sim_mean', 'sim_std', 'sim_max', 'sim_min']
    stat_titles = {
        'sim_mean': 'Mean Cosine Similarity',
        'sim_std': 'Std Dev Cosine Similarity',
        'sim_max': 'Max Cosine Similarity',
        'sim_min': 'Min Cosine Similarity'
    }

    for idx, stat_col in enumerate(stat_columns):
        ax = axes[idx]

        for codec in codec_display_order:
            codec_data = df_merged[df_merged['codec'] == codec]
            ax.scatter(
                codec_data[stat_col],
                codec_data['corr'],
                c=[color_map[codec]],
                label=codec,
                alpha=0.6,
                s=20
            )

        ax.set_xlabel(stat_titles[stat_col], fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature Correlation with ZSTD', fontsize=11, fontweight='bold')
        ax.set_title(f'{stat_titles[stat_col]} vs Correlation', fontsize=12, fontweight='bold')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        # Only show legend on last subplot
        if idx == 3:
            ax.legend(title='Codec', bbox_to_anchor=(1.02, 1), loc='upper left')

        ax.text(0.02, 0.02, f'n={df_merged["feature"].nunique()} features',
                transform=ax.transAxes, fontsize=9, verticalalignment='bottom')

    plt.suptitle('Feature Cosine Similarity vs Correlation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved feature similarity vs correlation plot to: {output_path}")
    plt.close(fig)


def plot_noise_ratio_vs_correlation(df, noisy_features_path="data/aliby_output/tables/noisy_features.parquet",
                                     output_path=SCRIPT_DIR / "output" / "noise_ratio_vs_correlation.png",
                                     abs_ratio_threshold=1000):
    """
    Create scatter plots of noise ratio vs correlation (with and without absolute value).

    Args:
        df: DataFrame with correlation results (must have 'feature' and 'corr' columns)
        noisy_features_path: Path to the noisy features parquet file
        output_path: Path to save plot
        abs_ratio_threshold: Threshold for filtering outliers based on absolute ratio (default 1000)
    """
    # Load noisy features data
    noisy_df = pd.read_parquet(noisy_features_path)

    # Dynamic codec discovery (including zstd)
    df_filtered = df.copy()
    df_filtered['codec'] = df_filtered['key'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')
    codec_mean_corr = df_filtered.groupby('codec')['corr'].mean().sort_values(ascending=False)
    codec_display_order = list(codec_mean_corr.index)
    palette = sns.color_palette('viridis', n_colors=len(codec_display_order))
    color_map = dict(zip(codec_display_order, palette))

    # Merge correlation data with noisy features
    feature_col = 'feature'
    df_merged = df_filtered.merge(noisy_df, left_on=feature_col, right_on='feature', suffixes=('', '_noise'))

    if len(df_merged) == 0:
        print(f"Warning: No features matched between correlation data and noisy features data")
        return

    # Add absolute ratio column and filter outliers
    df_merged['abs_ratio'] = df_merged['ratio'].abs()
    df_filtered_outliers = df_merged[df_merged['abs_ratio'] <= abs_ratio_threshold].copy()

    print(f"Matched {df_merged[feature_col].nunique()} features for noise ratio vs correlation plot")
    print(f"After filtering (abs_ratio <= {abs_ratio_threshold}): {len(df_filtered_outliers)} / {len(df_merged)} points")

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Subplot 1: Ratio
    ax = axes[0]
    for codec in codec_display_order:
        codec_data = df_filtered_outliers[df_filtered_outliers['codec'] == codec]
        ax.scatter(
            codec_data['ratio'],
            codec_data['corr'],
            c=[color_map[codec]],
            label=codec,
            alpha=0.6,
            s=20
        )

    ax.set_xlabel('Noise Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Correlation with ZSTD', fontsize=12, fontweight='bold')
    ax.set_title('Noise Ratio vs Feature Correlation', fontsize=14, fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.02, 0.02, f'n={len(df_filtered_outliers)} points\n(|ratio| <= {abs_ratio_threshold})',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom')

    # Subplot 2: Absolute Ratio
    ax = axes[1]
    for codec in codec_display_order:
        codec_data = df_filtered_outliers[df_filtered_outliers['codec'] == codec]
        ax.scatter(
            codec_data['abs_ratio'],
            codec_data['corr'],
            c=[color_map[codec]],
            label=codec,
            alpha=0.6,
            s=20
        )

    ax.set_xlabel('Absolute Noise Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Correlation with ZSTD', fontsize=12, fontweight='bold')
    ax.set_title('Absolute Noise Ratio vs Feature Correlation', fontsize=14, fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(title='Codec', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.text(0.02, 0.02, f'n={len(df_filtered_outliers)} points\n(|ratio| <= {abs_ratio_threshold})',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom')

    plt.suptitle('Noise Ratio vs Feature Correlation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved noise ratio vs correlation plot to: {output_path}")
    plt.close(fig)


def filter_by_greenlist(df, greenlist_dir=None):
    """Filter correlation DataFrame to only include green-listed features.

    Loads cell and nuclei greenlist CSVs, maps their feature names to the
    correlation data's naming convention, and returns only matching rows.

    The greenlist features use the format ``{channel}_{Measurement}``
    (e.g. ``0_Intensity_IntegratedIntensity``) while the correlation data
    uses ``{object}_{channel}/{mode}/{branch}{Measurement}`` for
    ``full_feature_name``.  The mapping strips the channel prefix from the
    greenlist and the lowercase branch prefix from the correlation feature,
    then matches on (object_type, channel, core_measurement).

    Args:
        df: Correlation DataFrame with ``full_feature_name`` and
            ``compartment`` columns.
        greenlist_dir: Directory containing ``cell/greenlist_features_cell.csv``
            and ``nuclei/greenlist_features_nuclei.csv``.  Defaults to
            ``SCRIPT_DIR / "output"``.

    Returns:
        Filtered DataFrame containing only green-listed features.
    """
    import re

    if greenlist_dir is None:
        greenlist_dir = SCRIPT_DIR / "output"
    greenlist_dir = Path(greenlist_dir)

    # channel-specific greenlist: (obj_type, channel, core_measurement)
    green_set = set()
    # channel-agnostic greenlist (SizeShape features without channel prefix):
    # (obj_type, core_measurement) — matches any channel for that object type
    green_any_channel = set()

    for obj_type, subdir, filename in [
        ("cell", "cell", "greenlist_features_cell.csv"),
        ("nuclei", "nuclei", "greenlist_features_nuclei.csv"),
    ]:
        path = greenlist_dir / subdir / filename
        if not path.exists():
            print(f"  WARNING: greenlist not found at {path}, skipping {obj_type}")
            continue
        gdf = pd.read_csv(path)
        for feat in gdf["feature"]:
            m = re.match(r"^(\d+)_(.*)", feat)
            if m:
                green_set.add((obj_type, m.group(1), m.group(2)))
            else:
                # SizeShape features (Area, Compactness, etc.) have no channel
                green_any_channel.add((obj_type, feat))

    if not green_set and not green_any_channel:
        print("  WARNING: no greenlist features loaded, returning empty DataFrame")
        return df.iloc[:0]

    # Advanced moment/tensor features to exclude (noise-sensitive, not
    # meaningful for downstream profiling).
    _EXCLUDED_PREFIXES = (
        "InertiaTensor_",
        "InertiaTensorEigenvalues_",
        "SpatialMoment_",
        "HuMoment_",
        "CentralMoment_",
        "NormalizedMoment_",
    )

    def _is_excluded(core_name):
        return any(core_name.startswith(p) for p in _EXCLUDED_PREFIXES)

    # Filter greenlist sets
    n_before_channel = len(green_set)
    n_before_any = len(green_any_channel)
    green_set = {t for t in green_set if not _is_excluded(t[2])}
    green_any_channel = {t for t in green_any_channel if not _is_excluded(t[1])}
    n_excluded = (n_before_channel - len(green_set)) + (n_before_any - len(green_any_channel))
    print(f"  Excluded {n_excluded} advanced moment/tensor features from greenlist")

    def _strip_branch(name):
        m = re.match(r"^[a-z][a-z_]*([A-Z].*)", name)
        return m.group(1) if m else name

    mask = []
    for ffn in df["full_feature_name"]:
        parts = ffn.split("/")
        if len(parts) < 3:
            mask.append(False)
            continue
        compartment, _mode, feature = parts[0], parts[1], parts[2]
        m_comp = re.match(r"(cell|nuclei)_(\d+)", compartment)
        if not m_comp:
            mask.append(False)
            continue
        obj_type = m_comp.group(1)
        channel = m_comp.group(2)
        core = _strip_branch(feature)
        mask.append(
            (obj_type, channel, core) in green_set
            or (obj_type, core) in green_any_channel
        )

    df_filtered = df[mask].copy()
    n_total = df["full_feature_name"].nunique()
    n_kept = df_filtered["full_feature_name"].nunique()
    print(f"  Greenlist filter: {n_kept}/{n_total} unique features kept "
          f"({len(green_set) + len(green_any_channel)} greenlist entries after exclusion)")
    return df_filtered


def main():
    """Main pipeline for feature correlation analysis."""
    parser = argparse.ArgumentParser(description="Feature Correlation Analysis for CellProfiler Measures")
    parser.add_argument('--workspace-dir', type=str,
                        default='data/aliby_output/cp_measure/jump_target2_4plate',
                        help='Path to workspace directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: analysis/feature_similarity/output/)')
    parser.add_argument('--raw-features-path', type=str, default=None,
                        help='Path to cp_measure raw features parquet for the noise/correlation plot')
    parser.add_argument('--greenlist-dir', type=str, default=None,
                        help='Directory containing {cell,nuclei}/greenlist_features_*.csv '
                             '(default: analysis/feature_similarity/output/)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Rerun analysis even if cached results exist')
    args = parser.parse_args()

    # Configuration
    workspace_dir = Path(args.workspace_dir)
    cache_dir = Path(workspace_dir) / "db_cache"
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "output"

    output_dir.mkdir(exist_ok=True, parents=True)

    # If output_dir / "feature_correlations.parquet" exists, check --overwrite flag
    if (output_dir / "feature_correlations.parquet").exists() and not args.overwrite:
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
    plot_correlation_violinplot(df, output_path=output_dir / "correlation_violinplot.png")
    plot_correlation_boxenplot(df, output_path=output_dir / "correlation_boxenplot.png")
    plot_feature_heatmap(df, output_path=output_dir / "feature_heatmap.png")
    plot_compression_heatmap(df, output_path=output_dir / "compression_heatmap.png")
    plot_compartment_heatmap(df, output_path=output_dir / "compartment_heatmap.png")
    plot_feature_group_boxplot(df, output_path=output_dir / "feature_group_boxplot.png")
    plot_feature_group_violinplot(df, output_path=output_dir / "feature_group_violinplot.png")
    plot_feature_group_by_compartment(df, output_path=output_dir / "feature_group_by_compartment.png")
    plot_feature_subgroups_by_compartment(df, output_path=output_dir / "feature_subgroups_by_compartment.png")
    _sim_kwargs = {"output_path": output_dir / "feature_similarity_vs_correlation.png"}
    if args.raw_features_path:
        _sim_kwargs["raw_features_path"] = args.raw_features_path
    plot_feature_similarity_vs_correlation(df, **_sim_kwargs)

    # Greenlist-filtered violin plot
    print("\nFiltering to greenlist features...")
    df_green = filter_by_greenlist(df, greenlist_dir=args.greenlist_dir)
    if len(df_green) > 0:
        plot_correlation_violinplot(
            df_green,
            output_path=output_dir / "correlation_violinplot_greenlist.png",
        )
        plot_correlation_boxenplot(
            df_green,
            output_path=output_dir / "correlation_boxenplot_greenlist.png",
        )
    else:
        print("  No greenlist features matched, skipping greenlist violin plot.")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
