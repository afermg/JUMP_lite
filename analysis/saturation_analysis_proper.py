#!/usr/bin/env python3
"""Proper saturation analysis: subsample BEFORE normalization.

For each (subsample_size, seed, model):
1. Subsample the raw feature parquet (nested: smaller is subset of larger)
   - optionally filter to a specific compound group (group_low, group_orf, etc.)
2. Save subsampled parquet to temp file
3. Call run_pipeline() with the existing config YAML pointing to the temp file
4. Read metrics.json from the pipeline output
5. Record results incrementally

Usage:
    # Run for group_low compounds, best config per model
    pixi run python ../../analysis/saturation_analysis_proper.py \
        --group group_low --n-seeds 5

    # Run for group_orf compounds
    pixi run python ../../analysis/saturation_analysis_proper.py \
        --group group_orf --n-seeds 5

    # Replot from existing CSV
    pixi run python ../../analysis/saturation_analysis_proper.py \
        --group group_low --plot-only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

N_SEEDS = 5
RXRX3_CORE_COMPOUNDS = 1674
RXRX3_CORE_PERTURBATIONS = 1674 + 735  # compounds + CRISPR

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = _PROJECT_ROOT / "metadata/metadata_dataset_filtered_4reps.parquet"

# Raw feature parquets (pre-normalization), using "raw" codec
RAW_FEATURES = {
    "morphem": _PROJECT_ROOT / "data/features/jump_lite_cl_3/morphem_jump_lite_updated_jpegxl_lossy_raw_raw_features.parquet",
    "cellprofiler": _PROJECT_ROOT / "data/features/jump_lite/cellprofiler_raw_jump_lite_raw_features.parquet",
    "dinov2": _PROJECT_ROOT / "data/features/jump_lite_cl_3/dinov2_jump_lite_updated_jpegxl_lossy_raw_raw_features.parquet",
    "openphenom": _PROJECT_ROOT / "data/features/jump_lite_cl_3/openphenom_jump_lite_updated_jpegxl_lossy_raw_raw_features.parquet",
    "subcell": _PROJECT_ROOT / "data/features/jump_lite_cl_3/subcell__clip01_jump_lite_updated_jpegxl_lossy_raw_raw_features.parquet",
}

# Pipeline config templates per model
TEMPLATE_CONFIGS = {
    "morphem": _PROJECT_ROOT / "analysis/saturation_pipeline_template.yaml",
    "dinov2": _PROJECT_ROOT / "analysis/saturation_pipeline_template.yaml",
    "openphenom": _PROJECT_ROOT / "analysis/saturation_pipeline_template.yaml",
    "subcell": _PROJECT_ROOT / "analysis/saturation_pipeline_template.yaml",
    "cellprofiler": _PROJECT_ROOT / "analysis/saturation_pipeline_template_cp.yaml",
}

# Unique base configs derived from best-per-model in sweep (raw codec)
# Each model's best: morphem=c170, dinov2=c96, openphenom=std_all+c96+noprune, subcell=c96+noprune
# CP's best (std_ctrl+c96) overlaps with dinov2 → 4 unique base configs
BASE_CONFIGS = [
    "std_ctrl__tvn_efaar_e1.0_c170",          # morphem's best
    "std_ctrl__tvn_efaar_e1.0_c96",            # dinov2's best (= CP's base)
    "std_all__tvn_efaar_e1.0_c96__noprune",    # openphenom's best
    "std_ctrl__tvn_efaar_e1.0_c96__noprune",   # subcell's best
]


def get_configs_for_model(model: str) -> list[str]:
    """Return config list for a model. CP gets outlier100/INT/prune0.95 inserted."""
    if model == "cellprofiler":
        configs = []
        for c in BASE_CONFIGS:
            # Insert __outlier100__INT__prune0.95__ after std_ctrl/std_all
            parts = c.split("__", 1)
            cp_config = f"{parts[0]}__outlier100__INT__prune0.95__{parts[1]}"
            configs.append(cp_config)
        return configs
    return list(BASE_CONFIGS)

# Default subsample sizes per group (adjusted to group size)
GROUP_SIZES = {
    "group_low": [50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000],
    "group_orf": [100, 500, 1000, 2500, 5000, 7500, 10000, 12500],
    None: [100, 200, 500, 750, 1000, 1500, 2500, 3500, 5000],  # all groups
}

COMPOUND_COL = "Metadata_JCP2022"
NEGCON_COL = "Metadata_negcon"
BATCH_COL = "Metadata_Plate"
GROUP_COL = "Metadata_Group"


def subsample_parquet(
    raw_path: Path, metadata_path: Path, n: int, seed: int, output_path: Path,
    group: str | None = None,
):
    """Subsample raw features: pick n treatments, keep plate-matched negcons, save.

    If group is specified, only subsample from compounds in that group.
    """
    df = pl.read_parquet(raw_path)
    meta = pl.read_parquet(metadata_path)

    # Merge to get JCP2022, negcon, plate, group
    join_cols = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"]
    available = [c for c in join_cols if c in df.columns and c in meta.columns]

    if not available and "Metadata_id" in df.columns:
        df = df.with_columns([
            pl.col("Metadata_id").str.split("__").list.get(0).alias("Metadata_Source"),
            pl.col("Metadata_id").str.split("__").list.get(1).alias("Metadata_Batch"),
            pl.col("Metadata_id").str.split("__").list.get(2).alias("Metadata_Plate"),
            pl.col("Metadata_id").str.split("__").list.get(3).alias("Metadata_Well"),
        ])
        available = [c for c in join_cols if c in df.columns and c in meta.columns]

    # Merge only columns we need for subsampling
    needed = [COMPOUND_COL, NEGCON_COL, BATCH_COL, GROUP_COL]
    cols_to_add = [c for c in needed if c not in df.columns]
    if cols_to_add:
        meta_subset = meta.select(
            available + [c for c in meta.columns if c not in df.columns]
        ).unique(subset=available)
        df = df.join(meta_subset, on=available, how="left")

    # Get treatments, optionally filtered by group
    treatments = df.filter(pl.col(NEGCON_COL) == False)
    if group is not None:
        treatments = treatments.filter(pl.col(GROUP_COL) == group)

    unique_ids = treatments[COMPOUND_COL].unique().to_list()

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_ids).tolist()

    if n >= len(unique_ids):
        sampled_ids = unique_ids
        n_actual = len(unique_ids)
    else:
        sampled_ids = shuffled[:n]
        n_actual = n

    sampled_treatments = treatments.filter(pl.col(COMPOUND_COL).is_in(sampled_ids))

    # Only keep negcons from plates with sampled compounds
    sampled_plates = sampled_treatments[BATCH_COL].unique().to_list()
    negcons = df.filter(
        (pl.col(NEGCON_COL) == True) & pl.col(BATCH_COL).is_in(sampled_plates)
    )
    combined = pl.concat([negcons, sampled_treatments])

    # Drop metadata columns that were added for subsampling — keep original columns only
    orig_cols = pl.read_parquet(raw_path, n_rows=0).columns
    keep_cols = [c for c in combined.columns if c in orig_cols]
    combined.select(keep_cols).write_parquet(output_path)
    return n_actual


def apply_config_overrides(config: dict, norm_config_name: str) -> dict:
    """Apply normalization config overrides to the pipeline config."""
    import copy
    config = copy.deepcopy(config)

    parts = norm_config_name.split("__")

    # Determine norm method
    norm_method = "none"
    fit_controls = False
    use_prune = True
    use_int = False
    outlier_cutoff = None

    for part in parts:
        if part.startswith("robustmad"):
            norm_method = "robustmad"
            fit_controls = "ctrl" in part
        elif part.startswith("std"):
            norm_method = "standardize"
            fit_controls = "ctrl" in part
        elif part == "noprune":
            use_prune = False
        elif part == "INT":
            use_int = True
        elif part.startswith("outlier"):
            outlier_cutoff = int(part[len("outlier"):])
        elif part.startswith("prune"):
            # e.g. prune0.95 — corr_thresh
            config["corr_thresh"] = float(part[len("prune"):])

    # Determine TVN-EFAAR params
    tvn_epsilon = None
    tvn_n_components = 128
    for part in parts:
        if part.startswith("tvn_efaar_e"):
            rest = part[len("tvn_efaar_e"):]
            if "_c" in rest:
                eps_str, comp_str = rest.split("_c")
                tvn_epsilon = float(eps_str)
                tvn_n_components = int(comp_str)
            else:
                tvn_epsilon = float(rest)

    config["norm_method"] = norm_method
    config["norm_fit_controls"] = fit_controls
    config["use_prune_correlated"] = use_prune
    config["use_int"] = use_int
    if outlier_cutoff is not None:
        config["outlier_cutoff"] = outlier_cutoff
    if tvn_epsilon is not None:
        config["batch_method"] = "tvn_efaar"
        config["tvn_efaar_epsilon"] = tvn_epsilon
        config["tvn_efaar_n_components"] = tvn_n_components
    else:
        config["batch_method"] = "none"

    # Disable skip_existing so pipeline always runs
    config["skip_existing"] = False
    config["use_pipeline_cache"] = False
    config["skip_visualization"] = True
    config["skip_batch_effects"] = True

    return config


def run_single(
    raw_path: Path, metadata_path: Path,
    template_config: dict, norm_config_name: str,
    n: int, seed: int, output_base: Path,
    group: str | None = None,
) -> dict:
    """Subsample, run pipeline, read metrics."""
    from norm_3.pipeline import run_pipeline

    # Create temp dir for this run
    run_dir = output_base / f"runs/n{n}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Subsample and save
    subsample_path = run_dir / "input_subsampled.parquet"
    n_actual = subsample_parquet(
        raw_path, metadata_path, n, seed, subsample_path, group=group,
    )

    # Build config
    config = apply_config_overrides(template_config, norm_config_name)
    config["output"] = {"path": str(run_dir / "output.parquet"), "compression": "zstd"}

    # Save config
    config_path = run_dir / "pipeline_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Run pipeline
    run_pipeline(config_path=config_path, input_override=str(subsample_path))

    # Find metrics.json (pipeline nests it under generated output name)
    metrics_files = list(run_dir.rglob("metrics.json"))
    metrics_path = metrics_files[0] if metrics_files else run_dir / "results" / "metrics.json"
    result = {
        "n_requested": n,
        "n_actual": n_actual,
        "seed": seed,
        "PA_mean_nap": np.nan,
        "PA_mean_nap_clipped": np.nan,
        "PA_pct": np.nan,
        "PC_mean_nap": np.nan,
        "PC_mean_nap_clipped": np.nan,
        "n_compounds": 0,
    }

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        if group is not None:
            # Read group-specific PA from metrics
            pa_groups = metrics.get("PA_group_summary", {})
            if group in pa_groups:
                g = pa_groups[group]
                result["PA_mean_nap"] = g.get("mean_normalized_average_precision", np.nan)
                result["PA_mean_nap_clipped"] = g.get(
                    "mean_normalized_average_precision_clipped", np.nan
                )
                result["PA_pct"] = g.get("pct_active", np.nan)
                result["n_compounds"] = g.get("n_unique_compounds", 0)

            # PC is only computed for group_high and group_low
            pc_groups = metrics.get("PC_group_summary", {})
            if group in pc_groups:
                result["PC_mean_nap"] = pc_groups[group].get(
                    "mean_normalized_average_precision", np.nan
                )
                result["PC_mean_nap_clipped"] = pc_groups[group].get(
                    "mean_normalized_average_precision_clipped", np.nan
                )
        else:
            result["PA_mean_nap"] = metrics.get("PA_mean_nap", np.nan)
            result["PA_mean_nap_clipped"] = metrics.get("PA_mean_nap_clipped", np.nan)
            result["PA_pct"] = metrics.get("PA", np.nan)
            result["PC_mean_nap"] = metrics.get("PC_mean_nap", np.nan)
            result["PC_mean_nap_clipped"] = metrics.get("PC_mean_nap_clipped", np.nan)
            result["n_compounds"] = metrics.get("n_compounds", 0)

    # Clean up temp files (keep metrics.json)
    subsample_path.unlink(missing_ok=True)
    output_parquet = run_dir / "output.parquet"
    if output_parquet.exists():
        output_parquet.unlink()

    return result


def _plot_metric(results: pd.DataFrame, metric: str, ylabel: str, title: str,
                 output_path: Path):
    """Plot a single metric convergence curve."""
    models = results["model"].unique()
    has_configs = "config_id" in results.columns and results["config_id"].nunique() > 1

    cmap = plt.cm.tab10
    model_colors = {m: cmap(i) for i, m in enumerate(sorted(models))}

    fig, ax = plt.subplots(figsize=(10, 6))

    for model in sorted(models):
        color = model_colors[model]
        model_data = results[results["model"] == model]

        if has_configs:
            config_ids = sorted(model_data["config_id"].unique())
            linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]
            for idx, cid in enumerate(config_ids):
                config_data = model_data[model_data["config_id"] == cid]
                summary = config_data.groupby("n_actual")[metric].agg(["mean", "std"]).reset_index()
                summary = summary.sort_values("n_actual")
                ls = linestyles[idx % len(linestyles)]
                lbl = model if idx == 0 else None
                ax.plot(summary["n_actual"], summary["mean"], marker="o",
                        linestyle=ls, color=color, label=lbl,
                        markersize=3, linewidth=1, alpha=0.7)
                ax.fill_between(summary["n_actual"],
                                summary["mean"] - summary["std"],
                                summary["mean"] + summary["std"],
                                color=color, alpha=0.05)
        else:
            summary = model_data.groupby("n_actual")[metric].agg(["mean", "std"]).reset_index()
            summary = summary.sort_values("n_actual")
            ax.plot(summary["n_actual"], summary["mean"], "o-",
                    color=color, label=model, markersize=4)
            ax.fill_between(summary["n_actual"],
                            summary["mean"] - summary["std"],
                            summary["mean"] + summary["std"],
                            color=color, alpha=0.15)

    ax.axvline(x=RXRX3_CORE_COMPOUNDS, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.axvline(x=RXRX3_CORE_PERTURBATIONS, color="darkred", linestyle=":", alpha=0.7, linewidth=1.5)
    ylim = ax.get_ylim()
    ax.text(RXRX3_CORE_COMPOUNDS * 0.85, ylim[1] * 0.95,
            f"RxRx3-core\ncompounds\n({RXRX3_CORE_COMPOUNDS})",
            color="red", fontsize=8, va="top", ha="right")
    ax.text(RXRX3_CORE_PERTURBATIONS * 1.05, ylim[1] * 0.95,
            f"RxRx3-core\nperturbations\n({RXRX3_CORE_PERTURBATIONS})",
            color="darkred", fontsize=8, va="top", ha="left")

    ax.set_xlabel("Number of Treatments")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    fig.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_saturation(results: pd.DataFrame, output_dir: Path, group: str | None = None):
    """Plot PA and PC convergence curves."""
    group_label = f" ({group})" if group else ""
    suffix = f"_{group}" if group else ""

    _plot_metric(
        results, "PA_mean_nap", "PA (NAP)",
        f"PA (NAP) vs. Treatment Count{group_label}\n(Proper: normalize after subsampling)",
        output_dir / f"saturation_proper_PA_mean_nap{suffix}.png",
    )

    # Only plot PC if we have non-NaN values
    if results["PC_mean_nap"].notna().any():
        _plot_metric(
            results, "PC_mean_nap", "PC (NAP)",
            f"PC (NAP) vs. Treatment Count{group_label}\n(Proper: normalize after subsampling)",
            output_dir / f"saturation_proper_PC_mean_nap{suffix}.png",
        )
        # Zoomed-in version: only n >= 300
        zoomed = results[results["n_actual"] >= 300]
        if len(zoomed) > 0:
            _plot_metric(
                zoomed, "PC_mean_nap", "PC (NAP)",
                f"PC (NAP) vs. Treatment Count{group_label}\n(n >= 300, proper: normalize after subsampling)",
                output_dir / f"saturation_proper_PC_mean_nap{suffix}_zoomed.png",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to run (default: all)")
    parser.add_argument("--group", type=str, default=None,
                        help="Compound group to subsample from (group_low, group_orf, etc.)")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="Subsample sizes (default: auto based on group)")
    parser.add_argument("--output-dir", type=Path,
                        default=_PROJECT_ROOT / "analysis/output/saturation_proper")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    # Auto-select sizes based on group
    if args.sizes is None:
        args.sizes = GROUP_SIZES.get(args.group, GROUP_SIZES[None])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    group_suffix = f"_{args.group}" if args.group else ""
    results_csv = args.output_dir / f"saturation_results{group_suffix}.csv"

    if args.plot_only:
        if not results_csv.exists():
            print(f"No results at {results_csv}")
            return
        combined = pd.read_csv(results_csv)
        plot_saturation(combined, args.output_dir, group=args.group)
        return

    # Select models
    models = args.models or list(RAW_FEATURES.keys())
    models = [m for m in models if m in RAW_FEATURES]

    # Load template configs
    templates = {}
    for model in models:
        tpath = TEMPLATE_CONFIGS.get(model)
        if tpath is None or not tpath.exists():
            print(f"WARNING: No template config for {model}, skipping")
            continue
        with open(tpath) as f:
            templates[model] = yaml.safe_load(f)
    models = [m for m in models if m in templates]

    # Show configs per model
    print("Configs per model:")
    for model in models:
        configs = get_configs_for_model(model)
        print(f"  {model}:")
        for i, c in enumerate(configs):
            print(f"    {i}: {c}")

    # Load existing results for resume
    existing = set()
    if results_csv.exists():
        prev = pd.read_csv(results_csv)
        for _, row in prev.iterrows():
            existing.add((row["model"], int(row["config_id"]),
                          int(row["n_requested"]), int(row["seed"])))
        print(f"Resuming: {len(existing)} runs already completed")

    for model in models:
        raw_path = RAW_FEATURES[model]
        if not raw_path.exists():
            print(f"Skipping {model}: {raw_path} not found")
            continue

        configs = get_configs_for_model(model)
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        if args.group:
            print(f"Group: {args.group}")
        print(f"Raw features: {raw_path}")

        for config_id, config_name in enumerate(configs):
            print(f"\n  Config {config_id}: {config_name}")

            for seed in range(args.n_seeds):
                for n in args.sizes:
                    if (model, config_id, n, seed) in existing:
                        continue

                    t0 = time.time()
                    print(f"    n={n:>6}, seed={seed}", end=" → ", flush=True)

                    run_output = args.output_dir / f"{model}/c{config_id}_{config_name}/n{n}_s{seed}"
                    result = run_single(
                        raw_path, METADATA_PATH,
                        templates[model], config_name,
                        n, seed, run_output,
                        group=args.group,
                    )
                    result["model"] = model
                    result["config"] = config_name
                    result["config_id"] = config_id
                    result["group"] = args.group or "all"
                    elapsed = time.time() - t0

                    pa = result["PA_mean_nap"]
                    print(f"PA={pa:.4f} ({elapsed:.1f}s)" if not np.isnan(pa) else f"PA=nan ({elapsed:.1f}s)")

                    # Save incrementally
                    row_df = pd.DataFrame([result])
                    row_df.to_csv(results_csv, mode="a",
                                  header=not results_csv.exists() or results_csv.stat().st_size == 0,
                                  index=False)

    # Read back and plot
    if results_csv.exists():
        combined = pd.read_csv(results_csv)
        print(f"\nTotal: {len(combined)} rows")
        plot_saturation(combined, args.output_dir, group=args.group)


if __name__ == "__main__":
    main()
