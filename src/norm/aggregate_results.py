#!/usr/bin/env python3
"""Aggregate metrics directly from data/features output directories."""

import json
from pathlib import Path
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import re


def parse_config_name(config_dir_name: str) -> dict:
    """
    Parse parameter values from directory name.

    Example: clean_meta_filter_varthresh0.05_outlier50_agg_prune0.8_std
    Returns: {var_thresh: 0.05, outlier_cut: 50, corr_thresh: 0.8, norm_method: 'std'}
    """
    params = {}

    # Extract variance threshold
    match = re.search(r'varthresh([\d.]+)', config_dir_name)
    if match:
        params['var_thresh'] = float(match.group(1))

    # Extract outlier cutoff
    match = re.search(r'outlier([\d]+)', config_dir_name)
    if match:
        params['outlier_cut'] = int(match.group(1))

    # Extract correlation threshold (prune)
    match = re.search(r'prune([\d.]+)', config_dir_name)
    if match:
        params['corr_thresh'] = float(match.group(1))

    # Extract normalization method (last token)
    if config_dir_name.endswith('_std'):
        params['norm_method'] = 'standardize'
    elif config_dir_name.endswith('_robustmad'):
        params['norm_method'] = 'robustmad'
    elif config_dir_name.endswith('_robustize'):
        params['norm_method'] = 'robustize'

    return params


def aggregate_from_data_dir(data_dir: Path) -> pd.DataFrame:
    """
    Aggregate metrics from data/features output directories.

    Args:
        data_dir: Path to data/features/input_name directory

    Returns:
        DataFrame with parameters and metrics
    """
    results = []

    # Find all metrics.json files
    metrics_files = list(data_dir.rglob("results/metrics.json"))
    print(f"Found {len(metrics_files)} metrics files")

    for metrics_path in metrics_files:
        # Get config directory name (parent of parent of metrics.json)
        config_dir = metrics_path.parent.parent
        config_name = config_dir.name

        # Parse parameters from directory name
        params = parse_config_name(config_name)
        params['config'] = config_name

        # Load metrics
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            params.update(metrics)
        except Exception as e:
            print(f"Warning: Could not load {metrics_path}: {e}")
            continue

        results.append(params)

    df = pd.DataFrame(results)

    # Sort by PA if available
    if "PA" in df.columns:
        df = df.sort_values("PA", ascending=False)

    return df


def create_heatmaps(df: pd.DataFrame, output_dir: Path):
    """Create a single figure with all heatmaps as subplots."""
    metrics = [col for col in ["PA", "PC", "Silhouette", "kBET"] if col in df.columns]

    if not metrics:
        print("No metrics found to plot")
        return

    # Identify parameter columns
    param_cols = [col for col in ["var_thresh", "outlier_cut", "corr_thresh", "norm_method"]
                  if col in df.columns and df[col].nunique() > 1]

    if len(param_cols) < 2:
        print(f"Need at least 2 varying parameters, found: {param_cols}")
        return

    print(f"\nCreating heatmaps for parameters: {param_cols}")

    # Create heatmaps for key parameter pairs
    param_pairs = [
        ("corr_thresh", "var_thresh"),
        ("var_thresh", "outlier_cut"),
        ("corr_thresh", "outlier_cut"),
    ]

    # Filter to pairs that exist in data
    param_pairs = [(p1, p2) for p1, p2 in param_pairs if p1 in param_cols and p2 in param_cols]

    if not param_pairs:
        print("No valid parameter pairs found")
        return

    output_dir.mkdir(exist_ok=True, parents=True)

    # Organize subplot data by parameter pairs (grouped logically)
    subplot_groups = {}  # {(param1, param2): [(group_df, suffix), ...]}

    for param1, param2 in param_pairs:
        # Check if we need to facet by other parameters
        other_params = [p for p in param_cols if p not in [param1, param2]]

        key = (param1, param2)
        if key not in subplot_groups:
            subplot_groups[key] = []

        if other_params:
            # Create separate heatmaps for each combination of other params
            for name, group in df.groupby(other_params):
                if len(group) < 2:
                    continue

                # Create title suffix
                if isinstance(name, tuple):
                    suffix = ", ".join([f"{p}={v}" for p, v in zip(other_params, name)])
                else:
                    suffix = f"{other_params[0]}={name}"

                subplot_groups[key].append((group, suffix))
        else:
            subplot_groups[key].append((df, ""))

    if not subplot_groups:
        print("No subplot data generated")
        return

    # Pre-calculate color scale ranges for each metric (consistent across all subplots)
    metric_ranges = {}
    for metric in metrics:
        all_values = []
        for (param1, param2), groups in subplot_groups.items():
            for group_df, suffix in groups:
                pivot = group_df.pivot_table(
                    values=metric,
                    index=param2,
                    columns=param1,
                    aggfunc='mean'
                )
                all_values.extend(pivot.values.flatten())

        # Remove NaN values
        all_values = [v for v in all_values if not pd.isna(v)]

        if all_values:
            metric_ranges[metric] = (min(all_values), max(all_values))
        else:
            metric_ranges[metric] = (0, 1)

    # Calculate total rows needed (with gaps between parameter pair sections)
    total_rows = sum(len(groups) for groups in subplot_groups.values())
    total_rows += len(subplot_groups) - 1  # Add gaps between sections

    n_metrics = len(metrics)
    fig = plt.figure(figsize=(6*n_metrics, 4.5*total_rows))

    current_row = 0

    for section_idx, ((param1, param2), groups) in enumerate(subplot_groups.items()):
        # Add section header
        if section_idx > 0:
            current_row += 1  # Gap between sections

        for group_idx, (group_df, suffix) in enumerate(groups):
            for col_idx, metric in enumerate(metrics):
                ax = plt.subplot(total_rows, n_metrics, current_row * n_metrics + col_idx + 1)

                # Create pivot table
                pivot = group_df.pivot_table(
                    values=metric,
                    index=param2,
                    columns=param1,
                    aggfunc='mean'
                )

                # Determine colormap direction and get consistent color range
                if metric in ['PA', 'PC']:
                    cmap = 'RdYlGn'  # Green = good (high values)
                else:
                    cmap = 'RdYlGn_r'  # Green = good (low values)

                vmin, vmax = metric_ranges[metric]

                # Create heatmap with consistent color scale
                sns.heatmap(
                    pivot,
                    annot=True,
                    fmt='.2f',
                    cmap=cmap,
                    ax=ax,
                    cbar_kws={'label': metric},
                    vmin=vmin,
                    vmax=vmax,
                )

                # Section title on first row of each section
                if group_idx == 0 and col_idx == n_metrics // 2:
                    ax.text(0.5, 1.15, f'═══ {param1.replace("_", " ").title()} vs {param2.replace("_", " ").title()} ═══',
                           transform=ax.transAxes, fontsize=13, fontweight='bold',
                           ha='center', va='bottom')

                # Y-axis label shows fixed params
                if col_idx == 0 and suffix:
                    ax.set_ylabel(f'[{suffix}]', fontsize=9, fontweight='bold')
                else:
                    if col_idx == 0:
                        ax.set_ylabel(param2.replace('_', ' ').title(), fontsize=9)

                # Metric name as column title (only on very first row)
                if current_row == 0:
                    ax.set_title(f'{metric}\n(range: {vmin:.2f}-{vmax:.2f})',
                               fontsize=11, fontweight='bold')

                # X-label only on last row of each section
                is_last_in_section = group_idx == len(groups) - 1
                is_last_section = section_idx == len(subplot_groups) - 1

                if is_last_in_section:
                    ax.set_xlabel(param1.replace('_', ' ').title(), fontsize=9)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                else:
                    ax.set_xlabel('')
                    ax.set_xticklabels([])

                # Y-tick labels
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

            current_row += 1

    plt.tight_layout()

    # Save single comprehensive figure
    output_path = output_dir / "parameter_sweep_heatmaps.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    n_sections = len(subplot_groups)
    n_total_subplots = sum(len(groups) for groups in subplot_groups.values())
    n_total_heatmaps = n_total_subplots * n_metrics

    print(f"✓ Saved comprehensive heatmap to: {output_path}")
    print(f"  {n_sections} parameter pair sections")
    print(f"  {n_total_subplots} parameter combinations × {n_metrics} metrics = {n_total_heatmaps} heatmaps")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aggregate_results_from_data.py <data_dir>")
        print("Example: python aggregate_results_from_data.py data/features/zstd_raw_features")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        sys.exit(1)

    # Aggregate results
    df = aggregate_from_data_dir(data_dir)

    if df.empty:
        print("No results found")
        sys.exit(1)

    # Save CSV
    output_csv = data_dir / "aggregated_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved aggregated results to: {output_csv}")

    # Print summary
    print("\n" + "="*80)
    print("SWEEP RESULTS SUMMARY")
    print("="*80)

    param_cols = [c for c in df.columns if c in ["var_thresh", "outlier_cut", "corr_thresh", "norm_method"]]
    metric_cols = [c for c in df.columns if c in ["PA", "PC", "Silhouette", "kBET"]]

    if metric_cols:
        print(f"\nFound {len(df)} configurations")
        print("\nTop 10 configurations by PA:")
        display_cols = param_cols + metric_cols
        print(df[display_cols].head(10).to_string(index=False))

        print("\n" + "="*80)
        print("BEST PARAMETERS:")
        print("="*80)

        if "PA" in df.columns:
            best_idx = df["PA"].idxmax()
            best_row = df.loc[best_idx]
            print(f"\nHighest PA: {best_row['PA']:.2f}%")
            print(f"Configuration: {best_row['config']}")
            for col in param_cols:
                if col in best_row:
                    print(f"  {col}: {best_row[col]}")

        # Create heatmaps
        print("\n" + "="*80)
        print("GENERATING HEATMAPS")
        print("="*80)
        heatmap_dir = data_dir / "heatmaps"
        create_heatmaps(df, heatmap_dir)

    print("\n✓ Done!")
