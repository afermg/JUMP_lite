## Run sweep:
#!/bin/bash

num_jobs=4

sweep_name="basic_test_sweep_minimum"
sweep_name_output="basic_test_sweep_minimum_output_second"

models=(cp_measure subcell dinov2_random openphenom_8bit morphem)
compressions=(zstd jpegxl_lossy_effort_3 jpegxl_lossy_hq jpegxl_lossy_mq jpegxl_lossy_lq)

# SWEEP
# Run normalization pipeline for each model and compression
for model in "${models[@]}"; do
    for compression in "${compressions[@]}"; do
        uv run python src/norm_2/pipeline.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=${num_jobs} +sweep=${sweep_name} input_override=./output/${model}_jump_target2_4plate_${compression}_raw_features.parquet 
    done
done

# Aggregate results
# summarize results for each model and compression, get the best balanced setting 
for model in "${models[@]}"; do
    for compression in "${compressions[@]}"; do
        uv run python src/norm_2/aggregate_results.py /home/jfredinh/projects/JUMP_core/data/features/${sweep_name}/${model}_jump_target2_4plate_${compression}_raw_features/ --suffix $compression &
    done
done
wait

# Process best settings
# Run normalization pipeline for each model and compression with best settings
for model in "${models[@]}"; do
    for compression in "${compressions[@]}"; do
        uv run python src/norm_2/pipeline.py +preset=${model}_${compression} +input_override=./output/${model}_jump_target2_4plate_${compression}_raw_features.parquet output.path=./data/${sweep_name_output}/${model}/${model}_${compression}/${model}_${compression}/processed.parquet &
    done
done
wait

# Generate plots
# Generate plots for each model and compression
for model in "${models[@]}"; do
    for compression in "${compressions[@]}"; do
        uv run python analysis/codec_plots/plot_codec_comparison.py ./data/${sweep_name_output}/${model}/ --output ./plots_summary/${sweep_name_output}/ --prefix ${model} &
    done
done
wait

# Combine results
# Combine all plots into a summary
python analysis/codec_plots/combine_results.py plots_summary/${sweep_name_output}/