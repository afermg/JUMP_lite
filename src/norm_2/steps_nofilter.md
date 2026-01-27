# norm_2 Pipeline - No Filter Sweep

This guide shows how to run the `batch_correction_pca_sweep_nofilter` sweep which disables all feature filtering.

## Differences from standard sweep

- **No correlation pruning** (`prune_correlated` disabled)
- **No feature filtering** (`filter_features` disabled)
- **No PC1 variance check** (`max_pc1_variance` set to 1.0)

## Running the No-Filter Sweep

```bash
# DINOv2 embeddings
uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=16 hydra.sweeper.max_batch_size=16 +sweep=batch_correction_pca_sweep_nofilter input_override=./output/dinov2_jump_target2_4plate_zstd_raw_features.parquet

# CellProfiler measurements
uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=16 hydra.sweeper.max_batch_size=16 +sweep=batch_correction_pca_sweep_nofilter input_override=./output/cp_measure_jump_target2_4plate_zstd_raw_features.parquet

# Subcellular features
uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=16 hydra.sweeper.max_batch_size=16 +sweep=batch_correction_pca_sweep_nofilter input_override=./output/subcell_jump_target2_4plate_zstd_raw_features.parquet

# DINOv2 random baseline
uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=16 hydra.sweeper.max_batch_size=16 +sweep=batch_correction_pca_sweep_nofilter input_override=./output/dinov2_random_jump_target2_4plate_zstd_raw_features.parquet

# OpenPhenom 8-bit
uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=16 hydra.sweeper.max_batch_size=16 +sweep=batch_correction_pca_sweep_nofilter input_override=./output/openphenom_8bit_jump_target2_4plate_zstd_raw_features.parquet

# DINOv2 490 variant
uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=16 hydra.sweeper.max_batch_size=16 +sweep=batch_correction_pca_sweep_nofilter input_override=./output/dinov2_490_jump_target2_4plate_zstd_raw_features.parquet

# Morphem embeddings
uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=16 hydra.sweeper.max_batch_size=16 +sweep=batch_correction_pca_sweep_nofilter input_override=./output/morphem_jump_target2_4plate_zstd_raw_features.parquet

# DINOv2 tilesize 224
uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=16 hydra.sweeper.max_batch_size=16 +sweep=batch_correction_pca_sweep_nofilter input_override=./output/dinov2_tilesize_224_jump_target2_4plate_zstd_raw_features.parquet
```

## Aggregating Results

```bash
uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_pca_sweep_nofilter/dinov2_jump_target2_4plate_zstd_raw_features/ &

uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_pca_sweep_nofilter/cp_measure_jump_target2_4plate_zstd_raw_features/ &

uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_pca_sweep_nofilter/subcell_jump_target2_4plate_zstd_raw_features/ &

uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_pca_sweep_nofilter/dinov2_random_jump_target2_4plate_zstd_raw_features/ &

uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_pca_sweep_nofilter/openphenom_8bit_jump_target2_4plate_zstd_raw_features/ &

uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_pca_sweep_nofilter/dinov2_490_jump_target2_4plate_zstd_raw_features/ &

uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_pca_sweep_nofilter/dinov2_tilesize_224_jump_target2_4plate_zstd_raw_features/ &

uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_pca_sweep_nofilter/morphem_jump_target2_4plate_zstd_raw_features/ &
```

## Running All Normalizations with Best Config

```bash
uv run python src/norm_2/run_all_normalizations.py --input output --output data/batch_correction_pca_sweep_nofilter/dinov2 --preset dinov2 --filter dinov2 &

uv run python src/norm_2/run_all_normalizations.py --input output --output data/batch_correction_pca_sweep_nofilter/cp_measure --preset cp_measure --filter cp_measure &

uv run python src/norm_2/run_all_normalizations.py --input output --output data/batch_correction_pca_sweep_nofilter/subcell --preset subcell --filter subcell &

uv run python src/norm_2/run_all_normalizations.py --input output --output data/batch_correction_pca_sweep_nofilter/dinov2_random --preset dinov2_random --filter dinov2_random &

uv run python src/norm_2/run_all_normalizations.py --input output --output data/batch_correction_pca_sweep_nofilter/openphenom_8bit --preset openphenom_8bit --filter openphenom_8bit &

uv run python src/norm_2/run_all_normalizations.py --input output --output data/batch_correction_pca_sweep_nofilter/dinov2_490 --preset dinov2_490 --filter dinov2_490 &

uv run python src/norm_2/run_all_normalizations.py --input output --output data/batch_correction_pca_sweep_nofilter/dinov2_tilesize_224 --preset dinov2_tilesize_224 --filter dinov2_tilesize_224 &

uv run python src/norm_2/run_all_normalizations.py --input output --output data/batch_correction_pca_sweep_nofilter/morphem --preset morphem --filter morphem &
```

## Generating Plots and Summary

```bash
python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_pca_sweep_nofilter/dinov2 --output ./plots_summary/batch_correction_pca_sweep_nofilter/ --prefix dinov2 &

python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_pca_sweep_nofilter/cp_measure --output ./plots_summary/batch_correction_pca_sweep_nofilter/ --prefix cp_measure &

python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_pca_sweep_nofilter/subcell --output ./plots_summary/batch_correction_pca_sweep_nofilter/ --prefix subcell &

python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_pca_sweep_nofilter/dinov2_random --output ./plots_summary/batch_correction_pca_sweep_nofilter/ --prefix dinov2_random &

python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_pca_sweep_nofilter/openphenom_8bit --output ./plots_summary/batch_correction_pca_sweep_nofilter/ --prefix openphenom_8bit &

python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_pca_sweep_nofilter/dinov2_490 --output ./plots_summary/batch_correction_pca_sweep_nofilter/ --prefix dinov2_490 &

python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_pca_sweep_nofilter/dinov2_tilesize_224 --output ./plots_summary/batch_correction_pca_sweep_nofilter/ --prefix dinov2_tilesize_224 &

python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_pca_sweep_nofilter/morphem --output ./plots_summary/batch_correction_pca_sweep_nofilter/ --prefix morphem &
```

## Combining All Results

```bash
python analysis/codec_plots/combine_results.py plots_summary/batch_correction_pca_sweep_nofilter/
```

## Notes

- Uses `n_jobs=16` instead of 64 to avoid memory issues
- All feature filtering is disabled - raw features go directly to normalization
- PC1 variance check is disabled - allows data with high PC1 variance
- Useful for comparing whether filtering affects biological metrics
