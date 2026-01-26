


## Get the single cell profiles into site level data per codec-model combo

``` bash
python src/extract_features.py --input /work/datasets/aliby_output --output ./output
```
## Run sweep for each of the models (Only using the non-lossy codec ZSTD)

``` bash
python src/norm/run_pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=32 +sweep=tvn_spherize_sweep input_override=./output/dinov2_jump_target2_4plate_zstd_raw_features.parquet &
python src/norm/run_pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=32 +sweep=tvn_spherize_sweep input_override=./output/subcell_jump_target2_4plate_zstd_raw_features.parquet &
python src/norm/run_pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=32 +sweep=tvn_spherize_sweep input_override=./output/dinov2_random_jump_target2_4plate_zstd_raw_features.parquet &
python src/norm/run_pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=32 +sweep=tvn_spherize_sweep input_override=./output/cp_measure_jump_target2_4plate_zstd_raw_features.parquet &
python src/norm/run_pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=32 +sweep=tvn_spherize_sweep input_override=./output/openphenom_8bit_jump_target2_4plate_zstd_raw_features.parquet &
python src/norm/run_pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=32 +sweep=tvn_spherize_sweep input_override=./output/dinov2_490_jump_target2_4plate_zstd_raw_features.parquet &
python src/norm/run_pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=32 +sweep=tvn_spherize_sweep input_override=./output/dinov2_tilesize_224_jump_target2_4plate_zstd_raw_features.parquet &
```

## Summarize the results of each model sweep and generate the config to use for all codecs of the model
``` bash
python src/norm/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/dinov2_jump_target2_4plate_zstd_raw_features/tvn_spherize_sweep/ &
python src/norm/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/subcell_jump_target2_4plate_zstd_raw_features/tvn_spherize_sweep/ &
python src/norm/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/dinov2_random_jump_target2_4plate_zstd_raw_features/tvn_spherize_sweep/ &
python src/norm/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/cp_measure_jump_target2_4plate_zstd_raw_features/tvn_spherize_sweep/ &
python src/norm/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/openphenom_8bit_jump_target2_4plate_zstd_raw_features/tvn_spherize_sweep/ &
python src/norm/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/dinov2_490_jump_target2_4plate_zstd_raw_features/tvn_spherize_sweep/ &
python src/norm/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/dinov2_tilesize_224_jump_target2_4plate_zstd_raw_features/tvn_spherize_sweep/ &
```

## Run the best setting per model on all codecs for that model
``` bash
python src/run_all_normalizations.py --input output --output data/dinov2 --preset dinov2 --filter dinov2  &
python src/run_all_normalizations.py --input output --output data/cp_measure --preset cp_measure --filter cp_measure  &
python src/run_all_normalizations.py --input output --output data/subcell --preset subcell --filter subcell &
python src/run_all_normalizations.py --input output --output data/dinov2_random --preset dinov2_random --filter dinov2_random  &
python src/run_all_normalizations.py --input output --output data/openphenom_8bit --preset openphenom_8bit --filter openphenom_8bit  &
python src/run_all_normalizations.py --input output --output data/dinov2_490 --preset dinov2_490 --filter dinov2_490  &
python src/run_all_normalizations.py --input output --output data/dinov2_tilesize_224 --preset dinov2_tilesize_224 --filter dinov2_tilesize_224  &
``` 

## Generate plot and summary of the performance for each model type
``` bash
python analysis/codec_plots/plot_codec_comparison.py data/dinov2 --output ./plots_summary --prefix dinov2 &
python analysis/codec_plots/plot_codec_comparison.py data/cp_measure --output ./plots_summary --prefix cp_measure &
python analysis/codec_plots/plot_codec_comparison.py data/subcell --output ./plots_summary --prefix subcell &
python analysis/codec_plots/plot_codec_comparison.py data/dinov2_random --output ./plots_summary --prefix dinov2_random &
python analysis/codec_plots/plot_codec_comparison.py data/openphenom_8bit --output ./plots_summary --prefix openphenom_8bit &
python analysis/codec_plots/plot_codec_comparison.py data/dinov2_490 --output ./plots_summary --prefix dinov2_490 &
python analysis/codec_plots/plot_codec_comparison.py data/dinov2_tilesize_224 --output ./plots_summary --prefix dinov2_tilesize_224 &
```

## Combine all the results into a single output
``` bash
python analysis/codec_plots/combine_results.py plots_summary/
```









# Alternative LARGER sweep:


batch_correction_sweep.yaml

``` bash
pixi run python src/norm/run_pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=64 hydra.sweeper.max_batch_size=64  +sweep=batch_correction_sweep_larger_new_target input_override=./output/dinov2_jump_target2_4plate_zstd_raw_features.parquet


pixi run python src/norm/run_pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=64 hydra.sweeper.max_batch_size=64  +sweep=batch_correction_sweep_larger_new_target input_override=./output/cp_measure_jump_target2_4plate_zstd_raw_features.parquet 'outlier_cut=15,50,500' 'corr_thresh=0.50,0.90,0.95' use_sample_norm=false 

pixi run python src/norm/run_pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=64 hydra.sweeper.max_batch_size=64  +sweep=batch_correction_sweep_larger_new_target input_override=./output/subcell_jump_target2_4plate_zstd_raw_features.parquet 
pixi run python src/norm/run_pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=64 hydra.sweeper.max_batch_size=64  +sweep=batch_correction_sweep_larger_new_target input_override=./output/dinov2_random_jump_target2_4plate_zstd_raw_features.parquet
pixi run python src/norm/run_pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=64 hydra.sweeper.max_batch_size=64  +sweep=batch_correction_sweep_larger_new_target input_override=./output/openphenom_8bit_jump_target2_4plate_zstd_raw_features.parquet
pixi run python src/norm/run_pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=64 hydra.sweeper.max_batch_size=64  +sweep=batch_correction_sweep_larger_new_target input_override=./output/dinov2_490_jump_target2_4plate_zstd_raw_features.parquet
pixi run python src/norm/run_pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=64 hydra.sweeper.max_batch_size=64  +sweep=batch_correction_sweep_larger_new_target input_override=./output/morphem_jump_target2_4plate_zstd_raw_features.parquet 

pixi run python src/norm/run_pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=64 hydra.sweeper.max_batch_size=64  +sweep=batch_correction_sweep_larger_new_target input_override=./output/dinov2_tilesize_224_jump_target2_4plate_zstd_raw_features.parquet 

```


``` bash
pixi run python src/norm/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_sweep_larger_new_target/dinov2_jump_target2_4plate_zstd_raw_features/ &

pixi run python src/norm/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_sweep_larger_new_target/cp_measure_jump_target2_4plate_zstd_raw_features/ &
pixi run python src/norm/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_sweep_larger_new_target/subcell_jump_target2_4plate_zstd_raw_features/ &
pixi run python src/norm/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_sweep_larger_new_target/dinov2_random_jump_target2_4plate_zstd_raw_features/ &
pixi run python src/norm/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_sweep_larger_new_target/openphenom_8bit_jump_target2_4plate_zstd_raw_features/ &
pixi run python src/norm/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_sweep_larger_new_target/dinov2_490_jump_target2_4plate_zstd_raw_features/ &
pixi run python src/norm/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_sweep_larger_new_target/dinov2_tilesize_224_jump_target2_4plate_zstd_raw_features/ &
pixi run python src/norm/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/batch_correction_sweep_larger_new_target/morphem_jump_target2_4plate_zstd_raw_features/ &
```


``` bash
pixi run python src/run_all_normalizations.py --input output --output data/batch_correction_sweep_larger_new_target/dinov2 --preset dinov2 --filter dinov2  &

pixi run python src/run_all_normalizations.py --input output --output data/batch_correction_sweep_larger_new_target/cp_measure --preset cp_measure --filter cp_measure  &
pixi run python src/run_all_normalizations.py --input output --output data/batch_correction_sweep_larger_new_target/subcell --preset subcell --filter subcell &
pixi run python src/run_all_normalizations.py --input output --output data/batch_correction_sweep_larger_new_target/dinov2_random --preset dinov2_random --filter dinov2_random  &
pixi run python src/run_all_normalizations.py --input output --output data/batch_correction_sweep_larger_new_target/openphenom_8bit --preset openphenom_8bit --filter openphenom_8bit  &
pixi run python src/run_all_normalizations.py --input output --output data/batch_correction_sweep_larger_new_target/dinov2_490 --preset dinov2_490 --filter dinov2_490  &
pixi run python src/run_all_normalizations.py --input output --output data/batch_correction_sweep_larger_new_target/dinov2_tilesize_224 --preset dinov2_tilesize_224 --filter dinov2_tilesize_224  &
pixi run python src/run_all_normalizations.py --input output --output data/batch_correction_sweep_larger_new_target/morphem --preset morphem --filter morphem  &
```



## Generate plot and summary of the performance for each model type
``` bash
python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_sweep_larger_new_target/dinov2 --output ./plots_summary/batch_correction_sweep_larger_new_target/ --prefix dinov2 &

python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_sweep_larger_new_target/cp_measure --output ./plots_summary/batch_correction_sweep_larger_new_target/ --prefix cp_measure &
python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_sweep_larger_new_target/subcell --output ./plots_summary/batch_correction_sweep_larger_new_target/ --prefix subcell &
python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_sweep_larger_new_target/dinov2_random --output ./plots_summary/batch_correction_sweep_larger_new_target/ --prefix dinov2_random &
python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_sweep_larger_new_target/openphenom_8bit --output ./plots_summary/batch_correction_sweep_larger_new_target/ --prefix openphenom_8bit &
python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_sweep_larger_new_target/dinov2_490 --output ./plots_summary/batch_correction_sweep_larger_new_target/ --prefix dinov2_490 &
python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_sweep_larger_new_target/dinov2_tilesize_224 --output ./plots_summary/batch_correction_sweep_larger_new_target/ --prefix dinov2_tilesize_224 &
python analysis/codec_plots/plot_codec_comparison.py data/batch_correction_sweep_larger_new_target/morphem --output ./plots_summary/batch_correction_sweep_larger_new_target/ --prefix morphem &
```

## Combine all the results into a single output
``` bash
python analysis/codec_plots/combine_results.py plots_summary/batch_correction_sweep_larger_new_target/
```







Downloading negcons of data:

``` bash
python src/download_images.py --metadata /home/jfredinh/projects/JUMP_core/metadata/metadata_negative_controls.parquet --output /work/datasets/jump_core/raw_negcon  
```


Moving data from raw to jump_core_subset

``` bash
python src/move_images_by_metadata.py --metadata metadata/metadata_filtered.parquet --input /work/datasets/jump_core_annotated/raw --output /work/datasets/jump_core_annotated/raw_1_4
```

``` bash
python src/move_images_by_metadata.py --metadata metadata/metadata_negative_controls.parquet --input /work/datasets/jump_core/raw_negcon --output /work/datasets/jump_core_annotated/raw
```

Check that all files are there:

``` bash
python src/check_images_by_metadata.py --metadata /home/jfredinh/projects/JUMP_core/metadata/metadata_filtered.parquet /home/jfredinh/projects/JUMP_core/metadata/metadata_negative_controls.parquet --folder /work/datasets/jump_core_annotated/raw  
```

Compress the data:

``` bash
python src/compress_tif_single.py --input /work/datasets/jump_core_annotated/raw --output /work/datasets/jump_core_annotated/ --codec jpegxl_lossy_mq --n-jobs 32  
```