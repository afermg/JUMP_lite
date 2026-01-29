## Run sweep:
#!/bin/bash

num_jobs=32

sweep_name="larger_comprehensive_optuna"

# models=(subcell) # subcell dinov2_random openphenom_8bit morphem)
# compressions=(zstd) # jpegxl_lossy_lq)

models=(cp_measure subcell dinov2_random openphenom_8bit morphem)
compressions=(zstd jpegxl_lossy_effort_3 jpegxl_lossy_hq jpegxl_lossy_mq jpegxl_lossy_lq)

# SWEEP
# Run normalization pipeline for each model and compression
for model in "${models[@]}"; do
    for compression in "${compressions[@]}"; do

        # # Sweep possible settings
        # uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=${num_jobs} hydra.launcher.prefer=threads +sweep=${sweep_name} input_override=./output/${model}_jump_target2_4plate_${compression}_raw_features.parquet hydra/job_logging=disabled hydra/hydra_logging=disabled hydra.verbose=False

        uv run python src/norm_2/pipeline.py --multirun hydra/sweeper=optuna hydra/launcher=joblib hydra.launcher.n_jobs=${num_jobs} +sweep=${sweep_name} input_override=./output/${model}_jump_target2_4plate_${compression}_raw_features.parquet      

        # Aggregate results
        # summarize results for each model and compression, get the best balanced setting 
        uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/${sweep_name}/${model}_jump_target2_4plate_${compression}_raw_features/ --suffix $compression &
    done
done
wait

uv run python src/norm_2/summarize_sweep.py /home/jfredinh/projects/JUMP_core/data/features/${sweep_name}/



# uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=32 +sweep=basic_test_sweep input_override=./output/cp_measure_jump_target2_4plate_zstd_raw_features.parquet 


# uv run python src/norm_2/pipeline.py +sweep=basic_test_sweep input_override=./output/cp_measure_jump_target2_4plate_zstd_raw_features.parquet --cfg job > /dev/null 2>&1            