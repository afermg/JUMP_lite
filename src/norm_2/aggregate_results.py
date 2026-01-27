#!/usr/bin/env python3
"""Aggregate metrics directly from data/features output directories."""

import json
from pathlib import Path
import pandas as pd
import polars as pl
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import re
import yaml

# Suppress seaborn heatmap warnings for edge cases
warnings.filterwarnings('ignore', message='.*identical.*xlims.*')
warnings.filterwarnings('ignore', message='.*identical.*ylims.*')

# Import from norm_2 (relative imports)
from .io import infer_columns
from .visualization import plot_dimensionality_reduction_extended


def generate_visualization_for_result(result_dir: Path, skip_umap: bool = False) -> bool:
    """
    Generate PCA/UMAP visualization for a sweep result directory.

    Args:
        result_dir: Path to the sweep result directory (contains processed parquet and results/)
        skip_umap: If True, skip UMAP computation (faster, PCA only)

    Returns:
        True if visualization was generated successfully, False otherwise
    """
    # Find the processed parquet file
    parquet_candidates = [
        result_dir / "embeddings_processed.parquet",
        result_dir / "processed.parquet",
    ]
    parquet_path = None
    for candidate in parquet_candidates:
        if candidate.exists():
            parquet_path = candidate
            break

    if parquet_path is None:
        print(f"  Warning: No processed parquet found in {result_dir}")
        return False

    try:
        # Load the processed data
        df = pl.read_parquet(parquet_path)

        # Infer feature columns
        features, metadata = infer_columns(df, ["Metadata_"])
        numeric_features = [
            f for f in features
            if df[f].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                               pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        ]

        if len(numeric_features) == 0:
            print(f"  Warning: No numeric features found in {parquet_path}")
            return False

        # Load phenotypic activity results for highlighting top compounds
        results_dir = result_dir / "results"
        evaluation_results = {}

        pa_csv = results_dir / "phenotypic_activity_per_compound.csv"
        if pa_csv.exists():
            activity_ap = pd.read_csv(pa_csv)
            evaluation_results["phenotypic_activity"] = {"activity_ap": activity_ap}
        else:
            evaluation_results["phenotypic_activity"] = {}

        # Set output path
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / "dimreduction.png"

        # Generate visualization
        plot_dimensionality_reduction_extended(
            df=df,
            features=numeric_features,
            evaluation_results=evaluation_results,
            output_path=output_path,
            n_top_compounds=20,
            skip_umap=skip_umap,
        )

        print(f"  ✓ Saved visualization to: {output_path}")
        return True

    except Exception as e:
        print(f"  Error generating visualization: {e}")
        return False


def parse_config_name(config_dir_name: str) -> dict:
    """
    Parse parameter values from directory name.

    Example: clean__meta__filter_varthresh0.05__outlier50__agg__prune0.8__std
    Returns: {var_thresh: 0.05, outlier_cut: 50, corr_thresh: 0.8, norm_method: 'std'}

    Format uses double underscores (__) to separate parameters,
    single underscores (_) within parameter values.
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
    elif 'prune__' in config_dir_name or 'prune_' in config_dir_name or config_dir_name.endswith('_prune') or config_dir_name.endswith('__prune'):
        # Default value when no number specified
        params['corr_thresh'] = 0.9

    # Extract normalization method (handles both single and double underscore formats)
    if '__std__' in config_dir_name or '__std' in config_dir_name or '_std_' in config_dir_name or config_dir_name.endswith('_std'):
        params['norm_method'] = 'standardize'
    elif '__robustmad__' in config_dir_name or '__robustmad' in config_dir_name or '_robustmad_' in config_dir_name or config_dir_name.endswith('_robustmad'):
        params['norm_method'] = 'robustmad'
    elif '__robustize__' in config_dir_name or '__robustize' in config_dir_name or '_robustize_' in config_dir_name or config_dir_name.endswith('_robustize'):
        params['norm_method'] = 'robustize'
    elif '__tvn__' in config_dir_name or '__tvn' in config_dir_name or '_tvn_' in config_dir_name or config_dir_name.endswith('_tvn'):
        params['norm_method'] = 'tvn'
    elif '__spherize__' in config_dir_name or '__spherize' in config_dir_name or 'spherize' in config_dir_name:
        params['norm_method'] = 'spherize'

    return params


def extract_params_from_config(config: dict) -> dict:
    """
    Extract ALL sweep parameters from pipeline config.

    Args:
        config: Loaded pipeline configuration

    Returns:
        Dictionary with extracted parameters
    """
    params = {}

    for step in config.get("steps", []):
        step_name = step.get("name")
        step_params = step.get("params", {})
        step_enabled = step.get("enabled", True)

        # Extract sample normalization parameters
        if step_name == "sample_norm":
            params["sample_norm"] = step_enabled
            if step_enabled and "norm" in step_params:
                params["sample_norm_type"] = step_params["norm"]

        # Extract filter parameters
        elif step_name == "filter_features":
            params["filter_enabled"] = step_enabled
            if step_enabled:
                for op in step_params.get("operations", []):
                    if op.get("name") == "variance_threshold":
                        if "var_threshold" in op:
                            params["var_thresh"] = op["var_threshold"]
                    elif op.get("name") == "drop_outliers":
                        if "outlier_cutoff" in op:
                            params["outlier_cut"] = op["outlier_cutoff"]

        # Extract correlation threshold
        elif step_name == "prune_correlated":
            params["prune_enabled"] = step_enabled
            if step_enabled and "threshold" in step_params:
                params["corr_thresh"] = step_params["threshold"]

        # Extract standard normalization parameters
        elif step_name == "normalize_standard":
            if step_enabled:
                if "method" in step_params:
                    params["norm_method"] = step_params["method"]
                if "fit_on_controls" in step_params:
                    params["norm_fit_ctrl"] = step_params["fit_on_controls"]
                if "batch_col" in step_params:
                    batch = step_params["batch_col"]
                    params["norm_batch"] = "plate" if batch else "global"

        # Extract TVN parameters
        elif step_name == "normalize_tvn":
            params["tvn"] = step_enabled
            if step_enabled:
                if "alpha" in step_params:
                    params["tvn_alpha"] = step_params["alpha"]
                if "epsilon" in step_params:
                    params["tvn_eps"] = step_params["epsilon"]
                if "fit_on_controls" in step_params:
                    params["tvn_fit_ctrl"] = step_params["fit_on_controls"]
                if "batch_col" in step_params:
                    batch = step_params["batch_col"]
                    params["tvn_batch"] = "plate" if batch else "global"

        # Extract Spherize parameters
        elif step_name == "normalize_spherize":
            params["spherize"] = step_enabled
            if step_enabled:
                if "method" in step_params:
                    params["sph_method"] = step_params["method"]
                if "fit_on_controls" in step_params:
                    params["sph_fit_ctrl"] = step_params["fit_on_controls"]
                if "batch_col" in step_params:
                    batch = step_params["batch_col"]
                    params["sph_batch"] = "plate" if batch else "global"

        # Extract aggregation parameters
        elif step_name == "aggregate_wells":
            if "method" in step_params:
                params["agg"] = step_params["method"]

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

        # Try to load parameters from saved config file (preferred)
        config_file = config_dir / "pipeline_config.yaml"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                params = extract_params_from_config(config)
            except Exception as e:
                print(f"Warning: Could not parse config {config_file}: {e}")
                # Fallback to directory name parsing
                params = parse_config_name(config_name)
        else:
            # Fallback to directory name parsing for older runs
            params = parse_config_name(config_name)

        params['config'] = config_name
        params['run_path'] = str(config_dir)

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

    # Identify parameter columns that vary (for heatmap axes)
    potential_param_cols = [
        "sample_norm", "var_thresh", "outlier_cut", "corr_thresh",
        "norm_method", "norm_fit_ctrl", "tvn", "tvn_alpha", "tvn_fit_ctrl",
        "spherize", "sph_method", "sph_fit_ctrl", "agg"
    ]
    all_param_cols = [col for col in potential_param_cols
                      if col in df.columns and df[col].nunique() > 1]

    # norm_method is always used for faceting, not as heatmap axis
    has_norm_method = "norm_method" in all_param_cols
    param_cols = [col for col in all_param_cols if col != "norm_method"]

    if len(param_cols) < 2:
        print(f"Need at least 2 varying parameters (excluding norm_method), found: {param_cols}")
        return

    print(f"\nCreating heatmaps for parameters: {param_cols}")
    if has_norm_method:
        print(f"  Faceting by: norm_method")

    # Create heatmaps for key parameter pairs
    param_pairs = [
        # Classic CellProfiler params
        ("corr_thresh", "var_thresh"),
        ("var_thresh", "outlier_cut"),
        ("corr_thresh", "outlier_cut"),
        # TVN params
        ("tvn", "tvn_alpha"),
        ("tvn", "tvn_fit_ctrl"),
        ("tvn_alpha", "tvn_fit_ctrl"),
        # Spherize params
        ("spherize", "sph_fit_ctrl"),
        ("spherize", "sph_method"),
        # Cross-method
        ("tvn", "spherize"),
        ("sample_norm", "tvn"),
        ("sample_norm", "spherize"),
        ("norm_fit_ctrl", "tvn_fit_ctrl"),
        ("agg", "tvn"),
        ("agg", "spherize"),
    ]

    # Filter to pairs that exist in data and have enough variation for a meaningful heatmap
    valid_pairs = []
    for p1, p2 in param_pairs:
        if p1 in param_cols and p2 in param_cols:
            # Check if both params have at least 2 unique values
            if df[p1].nunique() >= 2 and df[p2].nunique() >= 2:
                valid_pairs.append((p1, p2))
    param_pairs = valid_pairs

    if not param_pairs:
        print("No valid parameter pairs found (need at least 2x2 variation)")
        return

    output_dir.mkdir(exist_ok=True, parents=True)

    # Organize subplot data by parameter pairs (grouped logically)
    subplot_groups = {}  # {(param1, param2): [(group_df, suffix), ...]}

    for param1, param2 in param_pairs:
        # Always facet by norm_method first, then other parameters
        other_params = [p for p in param_cols if p not in [param1, param2]]

        # Add norm_method to faceting if it varies
        facet_params = []
        if has_norm_method:
            facet_params.append("norm_method")
        facet_params.extend(other_params)

        key = (param1, param2)
        if key not in subplot_groups:
            subplot_groups[key] = []

        if facet_params:
            # Create separate heatmaps for each combination of facet params
            for name, group in df.groupby(facet_params):
                if len(group) < 2:
                    continue

                # Check if this group has enough variation for a heatmap
                if group[param1].nunique() < 2 or group[param2].nunique() < 2:
                    continue

                # Create title suffix
                if isinstance(name, tuple):
                    suffix = ", ".join([f"{p}={v}" for p, v in zip(facet_params, name)])
                else:
                    suffix = f"{facet_params[0]}={name}"

                subplot_groups[key].append((group, suffix))
        else:
            # Check if df has enough variation
            if df[param1].nunique() >= 2 and df[param2].nunique() >= 2:
                subplot_groups[key].append((df, ""))

    # Remove empty subplot groups
    subplot_groups = {k: v for k, v in subplot_groups.items() if v}

    if not subplot_groups:
        print("No subplot data generated (not enough variation in parameters)")
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
        print("Usage: python aggregate_results.py <data_dir>")
        print("Example: python aggregate_results.py data/features/zstd_raw_features")
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

    # All possible parameter columns (in logical order)
    all_param_cols = [
        # Sample norm
        "sample_norm", "sample_norm_type",
        # Filter
        "filter_enabled", "var_thresh", "outlier_cut",
        # Prune
        "prune_enabled", "corr_thresh",
        # Standard normalization
        "norm_method", "norm_fit_ctrl", "norm_batch",
        # TVN
        "tvn", "tvn_alpha", "tvn_eps", "tvn_fit_ctrl", "tvn_batch",
        # Spherize
        "spherize", "sph_method", "sph_fit_ctrl", "sph_batch",
        # Aggregation
        "agg",
    ]
    # Only include columns that exist and have variation
    param_cols = [c for c in all_param_cols if c in df.columns]
    metric_cols = [c for c in df.columns if c in ["PA", "PC", "Silhouette", "kBET"]]

    if metric_cols:
        print(f"\nFound {len(df)} configurations")
        print(f"Parameters tracked: {param_cols}")

        top_n = min(50, len(df))
        # Top n by PA
        print(f"\nTop {top_n} configurations by PA:")
        display_cols = param_cols + metric_cols + ['config']
        print(df[display_cols].head(top_n).to_string(index=False))
        # Top n by PC
        if "PC" in df.columns:
            print(f"\nTop {top_n} configurations by PC:")
            df_sorted_pc = df.sort_values("PC", ascending=False)
            print(df_sorted_pc[display_cols].head(top_n).to_string(index=False))

        # Make a third table based on the product of PC and PA
        if "PA" in df.columns and "PC" in df.columns:
            df['Balance'] = df['PA'] * df['PC']
            print(f"\nTop {top_n} configurations by Balance (PA × PC):")
            df_sorted_product = df.sort_values("Balance", ascending=False)
            print(df_sorted_product[display_cols + ["Balance"]].head(top_n).to_string(index=False))

        print("\n" + "="*80)
        print("BEST CONFIGURATIONS")
        print("="*80)

        if "PA" in df.columns:
            best_idx = df["PA"].idxmax()
            best_row = df.loc[best_idx]
            print(f"\n[Best PA: {best_row['PA']:.2f}%]")
            for col in param_cols:
                if col in best_row and pd.notna(best_row[col]):
                    print(f"  {col}: {best_row[col]}")
            print(f"  path: {best_row['run_path']}")

        if "PC" in df.columns:
            best_idx = df["PC"].idxmax()
            best_row = df.loc[best_idx]
            print(f"\n[Best PC: {best_row['PC']}]")
            for col in param_cols:
                if col in best_row and pd.notna(best_row[col]):
                    print(f"  {col}: {best_row[col]}")
            print(f"  path: {best_row['run_path']}")

        if "Balance" in df.columns:
            best_idx = df["Balance"].idxmax()
            best_row = df.loc[best_idx]
            print(f"\n[Best Balance: {best_row['Balance']:.2f}]")
            for col in param_cols:
                if col in best_row and pd.notna(best_row[col]):
                    print(f"  {col}: {best_row[col]}")
            print(f"  path: {best_row['run_path']}")

        # Generate visualizations for best configs
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS FOR BEST CONFIGS")
        print("="*80)

        # Collect unique best configs (avoid duplicates if same config wins multiple metrics)
        best_configs = {}
        best_balance_path = None
        if "PA" in df.columns:
            best_pa_path = df.loc[df["PA"].idxmax(), 'run_path']
            best_configs[best_pa_path] = "Best PA"
        if "PC" in df.columns:
            best_pc_path = df.loc[df["PC"].idxmax(), 'run_path']
            if best_pc_path not in best_configs:
                best_configs[best_pc_path] = "Best PC"
            else:
                best_configs[best_pc_path] += " + Best PC"
        if "Balance" in df.columns:
            best_balance_path = df.loc[df["Balance"].idxmax(), 'run_path']
            if best_balance_path not in best_configs:
                best_configs[best_balance_path] = "Best Balance"
            else:
                best_configs[best_balance_path] += " + Best Balance"
            df.drop(columns=['Balance'], inplace=True)

        for run_path, label in best_configs.items():
            print(f"\n[{label}]")
            print(f"  Generating visualization for: {Path(run_path).name}")
            generate_visualization_for_result(Path(run_path), skip_umap=False)

        # Copy best Balance config to best_settings directory
        if best_balance_path:
            print("\n" + "="*80)
            print("SAVING BEST BALANCE CONFIG")
            print("="*80)

            best_settings_dir = Path("/home/jfredinh/projects/JUMP_core/src/norm/conf/preset")
            best_settings_dir.mkdir(parents=True, exist_ok=True)

            # Extract feature type from data_dir name
            dir_name = data_dir.name  # e.g., "openphenom_8bit_jump_target2_4plate_zstd_raw_features"
            if "_jump_" in dir_name:
                feature_type = dir_name.split("_jump_")[0]
            else:
                feature_type = dir_name.split("_")[0]

            # Load, clean, and save pipeline_config.yaml
            src_config = Path(best_balance_path) / "pipeline_config.yaml"
            dst_config = best_settings_dir / f"{feature_type}.yaml"

            if src_config.exists():
                with open(src_config) as f:
                    config = yaml.safe_load(f)

                # Keep only essential keys (remove sweep variables)
                clean_config = {
                    "input": {"path": "INPUT_FILE.parquet"},
                    "output": {"path": "data/features/OUTPUT.parquet", "compression": "zstd"},
                    "steps": config.get("steps", []),
                    "hydra": {
                        "run": {"dir": "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"},
                        "sweep": {
                            "dir": "outputs/multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}",
                            "subdir": "${hydra.job.num}"
                        }
                    }
                }

                # Enable visualization for general use
                for step in clean_config["steps"]:
                    if step.get("name") == "evaluate_metrics":
                        step.setdefault("params", {})
                        step["params"]["skip_visualization"] = False

                # Save cleaned config with Hydra package directive
                with open(dst_config, 'w') as f:
                    f.write("# @package _global_\n\n")
                    f.write(f"# {feature_type.capitalize()}-optimized Configuration\n")
                    f.write("# Best settings from TVN/Spherize sweep\n\n")
                    yaml.dump(clean_config, f, default_flow_style=False, sort_keys=False)

                print(f"\n  Feature type: {feature_type}")
                print(f"  Source: {src_config}")
                print(f"  Saved to: {dst_config}")
                print("  (Hydra format with @package _global_)")
            else:
                print(f"\n  Warning: Config not found: {src_config}")

        # # Create heatmaps (disabled - uncomment to enable)
        # print("\n" + "="*80)
        # print("GENERATING HEATMAPS")
        # print("="*80)
        # heatmap_dir = data_dir / "heatmaps"
        # create_heatmaps(df, heatmap_dir)

    print("\n✓ Done!")
