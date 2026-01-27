# norm_2 Pipeline Usage Guide

This guide shows how to run the norm_2 pipeline for batch correction sweeps on different embedding types.

## Prerequisites

- UV package manager installed
- Dependencies installed via `uv sync`
- Input parquet files in `output/` directory

## Running Batch Correction Sweeps

### Using batch_correction_pca_sweep

This sweep compares batch correction methods (TVN, Spherize) across different feature types.

```bash
# DINOv2 embeddings
uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=64 hydra.sweeper.max_batch_size=64 +sweep=batch_correction_pca_sweep input_override=./output/dinov2_jump_target2_4plate_zstd_raw_features.parquet

# CellProfiler measurements (with custom filtering)
uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=64 hydra.sweeper.max_batch_size=64 +sweep=batch_correction_pca_sweep input_override=./output/cp_measure_jump_target2_4plate_zstd_raw_features.parquet 'outlier_cut=15,50,500' 'corr_thresh=0.50,0.90,0.95' use_sample_norm=false

# Subcellular features
uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=64 hydra.sweeper.max_batch_size=64 +sweep=batch_correction_pca_sweep input_override=./output/subcell_jump_target2_4plate_zstd_raw_features.parquet

# DINOv2 random baseline
uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=64 hydra.sweeper.max_batch_size=64 +sweep=batch_correction_pca_sweep input_override=./output/dinov2_random_jump_target2_4plate_zstd_raw_features.parquet

# OpenPhenom 8-bit
uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=64 hydra.sweeper.max_batch_size=64 +sweep=batch_correction_pca_sweep input_override=./output/openphenom_8bit_jump_target2_4plate_zstd_raw_features.parquet

# DINOv2 490 variant
uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=64 hydra.sweeper.max_batch_size=64 +sweep=batch_correction_pca_sweep input_override=./output/dinov2_490_jump_target2_4plate_zstd_raw_features.parquet

# Morphem embeddings
uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=64 hydra.sweeper.max_batch_size=64 +sweep=batch_correction_pca_sweep input_override=./output/morphem_jump_target2_4plate_zstd_raw_features.parquet

# DINOv2 tilesize 224
uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=64 hydra.sweeper.max_batch_size=64 +sweep=batch_correction_pca_sweep input_override=./output/dinov2_tilesize_224_jump_target2_4plate_zstd_raw_features.parquet
```


## Aggregating Results

After sweeps complete, aggregate results to find best configurations:

```bash
# Run aggregation for each embedding type (can run in parallel with &)
uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_pca_sweep/dinov2_jump_target2_4plate_zstd_raw_features/ &

uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_pca_sweep/cp_measure_jump_target2_4plate_zstd_raw_features/ &

uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_pca_sweep/subcell_jump_target2_4plate_zstd_raw_features/ &

uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_pca_sweep/dinov2_random_jump_target2_4plate_zstd_raw_features/ &

uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_pca_sweep/openphenom_8bit_jump_target2_4plate_zstd_raw_features/ &

uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_pca_sweep/dinov2_490_jump_target2_4plate_zstd_raw_features/ &

uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_pca_sweep/dinov2_tilesize_224_jump_target2_4plate_zstd_raw_features/ &

uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_pca_sweep/morphem_jump_target2_4plate_zstd_raw_features/ &
```

## Running All Normalizations with Best Config

Apply the best normalization configuration to all data:

```bash
uv run python src/norm_2/run_all_normalizations.py --input output --output data/batch_correction_pca_sweep/dinov2 --preset dinov2 --filter dinov2 &

uv run python src/norm_2/run_all_normalizations.py --input output --output data/batch_correction_pca_sweep/cp_measure --preset cp_measure --filter cp_measure &

uv run python src/norm_2/run_all_normalizations.py --input output --output data/batch_correction_pca_sweep/subcell --preset subcell --filter subcell &

uv run python src/norm_2/run_all_normalizations.py --input output --output data/batch_correction_pca_sweep/dinov2_random --preset dinov2_random --filter dinov2_random &

uv run python src/norm_2/run_all_normalizations.py --input output --output data/batch_correction_pca_sweep/openphenom_8bit --preset openphenom_8bit --filter openphenom_8bit &

uv run python src/norm_2/run_all_normalizations.py --input output --output data/batch_correction_pca_sweep/dinov2_490 --preset dinov2_490 --filter dinov2_490 &

uv run python src/norm_2/run_all_normalizations.py --input output --output data/batch_correction_pca_sweep/dinov2_tilesize_224 --preset dinov2_tilesize_224 --filter dinov2_tilesize_224 &

uv run python src/norm_2/run_all_normalizations.py --input output --output data/batch_correction_pca_sweep/morphem --preset morphem --filter morphem &
```

## Generating Plots and Summary

Generate comparison plots for each model type:

```bash
python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_pca_sweep/dinov2 --output ./plots_summary/batch_correction_pca_sweep/ --prefix dinov2 &

python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_pca_sweep/cp_measure --output ./plots_summary/batch_correction_pca_sweep/ --prefix cp_measure &

python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_pca_sweep/subcell --output ./plots_summary/batch_correction_pca_sweep/ --prefix subcell &

python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_pca_sweep/dinov2_random --output ./plots_summary/batch_correction_pca_sweep/ --prefix dinov2_random &

python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_pca_sweep/openphenom_8bit --output ./plots_summary/batch_correction_pca_sweep/ --prefix openphenom_8bit &

python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_pca_sweep/dinov2_490 --output ./plots_summary/batch_correction_pca_sweep/ --prefix dinov2_490 &

python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_pca_sweep/dinov2_tilesize_224 --output ./plots_summary/batch_correction_pca_sweep/ --prefix dinov2_tilesize_224 &

python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_pca_sweep/morphem --output ./plots_summary/batch_correction_pca_sweep/ --prefix morphem &
```

## Combining All Results

Combine results from all model types into a single summary:

```bash
python analysis/codec_plots/combine_results.py plots_summary/batch_correction_pca_sweep/
```

## Running a Single Preset

To run a single configuration (not a sweep):

```bash
# Run morphem preset
uv run python src/norm_2/pipeline.py +preset=morphem input_override=./output/morphem_jump_target2_4plate_zstd_raw_features.parquet

# Run morphem with PCA and z-score normalization
uv run python src/norm_2/pipeline.py +preset=morphem_pca_zscore input_override=./output/morphem_jump_target2_4plate_zstd_raw_features.parquet

# Run dinov2 preset
uv run python src/norm_2/pipeline.py +preset=dinov2_490 input_override=./output/dinov2_490_jump_target2_4plate_zstd_raw_features.parquet
```

## Available Presets

- `cp_measure` - CellProfiler measurements
- `dinov2_490` - DINOv2 490 embeddings
- `dinov2_random` - DINOv2 random baseline
- `dinov2_tilesize_224` - DINOv2 with 224 tile size
- `morphem` - Morphem embeddings
- `morphem_pca` - Morphem with PCA (128 components)
- `morphem_pca_zscore` - Morphem with z-score + PCA (EFAAR-style)
- `morphem_high_eps` - Morphem with high epsilon for stability
- `openphenom_8bit` - OpenPhenom 8-bit
- `subcell` - Subcellular features

## Available Sweeps

- `batch_correction_pca_sweep` - Standard batch correction sweep
- `batch_correction_pca_sweep` - Sweep with PCA and epsilon tuning
- `tvn_spherize_sweep` - TVN and Spherize parameter sweep

## Metrics Output

Each run produces a `metrics.json` file with:

- `PA` - Phenotypic Activity (% compounds active)
- `PC` - Phenotypic Consistency (% targets active)
- `Silhouette` - Batch effect silhouette score
- `kBET` - Batch effect k-nearest neighbor score
- `tvn_ill_conditioned` - Whether TVN encountered numerical instability
- `tvn_max_condition_number` - Maximum covariance matrix condition number

## Tips

1. **Parallel execution**: Use `&` at the end of commands to run in background
2. **Memory management**: Adjust `hydra.launcher.n_jobs` based on available RAM
3. **Ill-conditioned matrices**: Check `tvn_ill_conditioned` in metrics.json for numerical issues
4. **Morphem stability**: Use `morphem_pca_zscore` preset or enable PCA in sweep for better stability
