#!/usr/bin/env bash

# Run from JUMP_core root directory
cd "$(dirname "$0")"

echo "==================================================="
echo "Focused Sweep v6 — CP (54 configs) + DL (12 configs)"
echo "==================================================="
echo ""
echo "Both sweeps output to variance_first_v6/ for unified summary."
echo ""

# Function to cleanup GPU memory
cleanup_gpu() {
    echo "  Cleaning up GPU memory..."
    cd src/norm_3
    pixi run python -c "import cupy as cp; cp.get_default_memory_pool().free_all_blocks(); cp.cuda.Stream.null.synchronize()" 2>&1 || echo "  GPU cleanup skipped"
    cd ../..
    sleep 2
    echo "  Cleanup complete"
}

# Initial GPU cleanup before starting
echo "Performing initial GPU cleanup..."
cleanup_gpu
echo ""

# ============================================================
# PART 1: CellProfiler Models — focused_cp_v6 (54 configs each)
# ============================================================

CP_FEATURE_FILES=(
    # --- CellProfiler reformatted (original JUMP CP profiles) ---
    "../../output/raw_jump_cp_profiles_reformatted_filtered.parquet"
    # --- cp_measure raw (7 codecs, from jump_target2_4plate/) ---
    "../../data/features/jump_target2_4plate/cp_measure_jump_target2_4plate_zstd_raw_features.parquet"
    "../../data/features/jump_target2_4plate/cp_measure_jump_target2_4plate_jpegxl_lossy_hq_raw_features.parquet"
    "../../data/features/jump_target2_4plate/cp_measure_jump_target2_4plate_jpegxl_lossy_effort_3_raw_features.parquet"
    "../../data/features/jump_target2_4plate/cp_measure_jump_target2_4plate_jpegxl_lossy_mq_raw_features.parquet"
    "../../data/features/jump_target2_4plate/cp_measure_jump_target2_4plate_jpegxl_lossy_lq_raw_features.parquet"
    "../../data/features/jump_target2_4plate/cp_measure_jump_target2_4plate_jpegxl_lossy_d2_e8_raw_features.parquet"
    "../../data/features/jump_target2_4plate/cp_measure_jump_target2_4plate_jpegxl_lossy_d10_raw_features.parquet"
    # --- cp_measure filtered_border_size (7 codecs, from jump_target2_4plate_filtered/) ---
    "../../data/features/jump_target2_4plate_filtered/cp_measure_jump_target2_4plate_zstd_filtered_border_size_raw_features.parquet"
    "../../data/features/jump_target2_4plate_filtered/cp_measure_jump_target2_4plate_jpegxl_lossy_hq_filtered_border_size_raw_features.parquet"
    "../../data/features/jump_target2_4plate_filtered/cp_measure_jump_target2_4plate_jpegxl_lossy_effort_3_filtered_border_size_raw_features.parquet"
    "../../data/features/jump_target2_4plate_filtered/cp_measure_jump_target2_4plate_jpegxl_lossy_mq_filtered_border_size_raw_features.parquet"
    "../../data/features/jump_target2_4plate_filtered/cp_measure_jump_target2_4plate_jpegxl_lossy_lq_filtered_border_size_raw_features.parquet"
    "../../data/features/jump_target2_4plate_filtered/cp_measure_jump_target2_4plate_jpegxl_lossy_d2_e8_filtered_border_size_raw_features.parquet"
    "../../data/features/jump_target2_4plate_filtered/cp_measure_jump_target2_4plate_jpegxl_lossy_d10_filtered_border_size_raw_features.parquet"
)

CP_COMPRESSION_NAMES=(
    "reformatted"
    "cp_raw_zstd" "cp_raw_hq" "cp_raw_effort_3" "cp_raw_mq" "cp_raw_lq" "cp_raw_d2_e8" "cp_raw_d10"
    "cp_fbs_zstd" "cp_fbs_hq" "cp_fbs_effort_3" "cp_fbs_mq" "cp_fbs_lq" "cp_fbs_d2_e8" "cp_fbs_d10"
)

echo "==================================================="
echo "PART 1: CellProfiler Models (${#CP_FEATURE_FILES[@]} datasets × 54 configs)"
echo "==================================================="
echo ""

# for i in "${!CP_FEATURE_FILES[@]}"; do
#     feature_file="${CP_FEATURE_FILES[$i]}"
#     compression="${CP_COMPRESSION_NAMES[$i]}"

#     echo "==================================================="
#     echo "[CP $((i+1))/${#CP_FEATURE_FILES[@]}] Running sweep: ${compression}"
#     echo "==================================================="
#     echo "  Input: ${feature_file}"
#     echo ""

#     cd src/norm_3
#     if [ ! -f "${feature_file}" ]; then
#         echo "  SKIPPING: Input file not found"
#         cd ../..
#         continue
#     fi

#     pixi run python -m norm_3.pipeline --multirun +sweep=focused_cp_v6 input.path="${feature_file}" hydra/launcher=joblib hydra.launcher.n_jobs=32 || {
#         echo "  Warning: Sweep encountered errors (some configs may have failed)"
#     }
#     cd ../..

#     echo ""
#     echo "Sweep complete for ${compression}"
#     echo ""
#     cleanup_gpu
#     echo ""
# done

# ============================================================
# PART 2: Deep Learning Models — focused_dl_v6 (256 configs each)
# Auto-discovers all parquet files in DL feature directories
# ============================================================

# Auto-discover all DL feature parquet files
# Paths are relative to project root for discovery, then converted to ../../ for src/norm_3
DL_FEATURE_FILES=()
DL_DIRS=(
    "data/features/jump_target2_4plate_cl"
)
for dl_dir in "${DL_DIRS[@]}"; do
    if [ -d "$dl_dir" ]; then
        while IFS= read -r f; do
            # Convert to relative path from src/norm_3/
            DL_FEATURE_FILES+=("../../${f}")
        done < <(find "$dl_dir" -maxdepth 1 -name "*.parquet" | sort)
    fi
done

echo "==================================================="
echo "PART 2: Deep Learning Models (${#DL_FEATURE_FILES[@]} datasets × 256 configs)"
echo "==================================================="
echo ""

for i in "${!DL_FEATURE_FILES[@]}"; do
    feature_file="${DL_FEATURE_FILES[$i]}"
    # Derive name from filename (strip path and .parquet extension)
    compression="$(basename "${feature_file}" .parquet)"

    echo "==================================================="
    echo "[DL $((i+1))/${#DL_FEATURE_FILES[@]}] Running sweep: ${compression}"
    echo "==================================================="
    echo "  Input: ${feature_file}"
    echo ""

    cd src/norm_3
    if [ ! -f "${feature_file}" ]; then
        echo "  SKIPPING: Input file not found"
        cd ../..
        continue
    fi

    pixi run python -m norm_3.pipeline --multirun +sweep=focused_dl_v6 input.path="${feature_file}" hydra/launcher=joblib hydra.launcher.n_jobs=32 || {
        echo "  Warning: Sweep encountered errors (some configs may have failed)"
    }
    cd ../..

    echo ""
    echo "Sweep complete for ${compression}"
    echo ""
    cleanup_gpu
    echo ""
done

echo "==================================================="
echo "All Focused v6 Sweeps Complete!"
echo "==================================================="
echo ""
echo "Total: ${#CP_FEATURE_FILES[@]} CP datasets (54 configs) + ${#DL_FEATURE_FILES[@]} DL datasets (12 configs)"
echo ""

# Count results
echo "Final counts:"
if [ -d "src/norm_3/data/features/variance_first_v6" ]; then
    cd src/norm_3/data/features/variance_first_v6
    ls -1 | xargs -I {} sh -c 'echo "  {}: $(find {} -name metrics.json 2>/dev/null | wc -l) configs"'
    cd ../../../../..
else
    echo "  Output directory not found"
fi
echo ""
echo "Run summary with:"
echo "  cd src/norm_3 && pixi run python gather_sweep_results.py --sweep-dir data/features/variance_first_v6 --plot --filter-degenerate"
