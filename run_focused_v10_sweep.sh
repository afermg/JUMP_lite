#!/usr/bin/env bash
set -euo pipefail

# Run from JUMP_core root directory
cd "$(dirname "$0")"

LOG_DIR="logs/sweep_v10"
mkdir -p "$LOG_DIR"

echo "==================================================="
echo "Focused Sweep v10 — target2 (9 DL + CP, 4 GPUs)"
echo "  tvn_efaar only (none/tvn_original dropped — consistently worse)"
echo "  DL: 48 configs/codec, CP: 48 configs/codec"
echo "==================================================="
echo ""

# Limit numpy/scipy threading to avoid oversubscription
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

# DL codecs (10): zstd + 9 JPEG XL variants
DL_CODECS="zstd jpegxl_lossy_hq jpegxl_lossy_mq jpegxl_lossy_mq_new jpegxl_lossy_lq jpegxl_lossy_d2_e8 jpegxl_lossy_d10 jpegxl_lossy_d15 jpegxl_lossy_d30 jpegxl_lossy_effort_3"

# CP codecs (7): zstd + 6 JPEG XL variants
CP_CODECS="zstd jpegxl_lossy_hq jpegxl_lossy_mq jpegxl_lossy_lq jpegxl_lossy_d2_e8 jpegxl_lossy_d10 jpegxl_lossy_effort_3"

# Sweep configs (tvn_efaar only — none and tvn_original consistently underperform)
DL_CONFIGS="focused_dl_v10_tvn_efaar"
CP_CONFIGS="focused_cp_v10_tvn_efaar"

# Feature path templates
DL_FEATURE_DIR="../../data/features/jump_target2_4plate_cl_2"
CP_FEATURE_DIR="../../data/features/jump_target2_4plate"

# Helper: run all codecs × configs for a DL model on a given GPU
run_dl_model() {
    local model="$1"
    local gpu="$2"
    echo "=== ${model} on GPU ${gpu} === $(date)"
    for codec in $DL_CODECS; do
        feature_file="${DL_FEATURE_DIR}/${model}_jump_target2_4plate_${codec}_raw_features.parquet"
        echo "--- ${model} codec: ${codec} --- $(date)"
        if [ ! -f "${feature_file}" ]; then
            echo "  SKIPPING: ${feature_file} not found"
            continue
        fi
        for cfg in $DL_CONFIGS; do
            echo "  config: ${cfg} $(date)"
            CUDA_VISIBLE_DEVICES="${gpu}" pixi run python -m norm_3.pipeline --multirun \
                +sweep="${cfg}" \
                input.path="${feature_file}" \
                hydra/launcher=joblib hydra.launcher.n_jobs=4 || {
                echo "  Warning: ${cfg} encountered errors"
            }
        done
    done
    echo "=== ${model} DONE === $(date)"
    echo ""
}

echo "  GPU 0: cellprofiler (7 codecs × 1 config) + cell_count baseline (7 codecs × 1 config)"
echo "  GPU 1: morphem + subcell + subcell__nonstd (10 codecs × 1 config each)"
echo "  GPU 2: openphenom + openphenom_nonclip + dinov2_random (10 codecs × 1 config each)"
echo "  GPU 3: dinov2 + openphenom_stdscale + openphenom_stdscale_false (10 codecs × 1 config each)"
echo ""
echo "Logs: ${LOG_DIR}/"
echo ""

# ============================================================
# GPU 0: CellProfiler only (7 codecs — slowest, dedicated GPU)
# ============================================================
(
    cd src/norm_3
    # echo "=== cellprofiler on GPU 0 === $(date)"
    # for codec in $CP_CODECS; do
    #     feature_file="${CP_FEATURE_DIR}/cp_measure_jump_target2_4plate_${codec}_raw_features.parquet"
    #     echo "--- codec: ${codec} --- $(date)"
    #     if [ ! -f "${feature_file}" ]; then
    #         echo "  SKIPPING: ${feature_file} not found"
    #         continue
    #     fi
    #     for cfg in $CP_CONFIGS; do
    #         echo "  config: ${cfg} $(date)"
    #         CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
    #             +sweep="${cfg}" \
    #             input.path="${feature_file}" \
    #             hydra/launcher=joblib hydra.launcher.n_jobs=4 || {
    #             echo "  Warning: ${cfg} encountered errors"
    #         }
    #     done
    # done
    # echo "=== cellprofiler DONE === $(date)"

    # --- Cell Count baseline (same 7 CP codecs, 4 configs each) ---
    echo "=== cell_count baseline on GPU 0 === $(date)"
    CC_CONFIG="focused_cell_count_v10"
    for codec in $CP_CODECS; do
        feature_file="${CP_FEATURE_DIR}/cell_count_jump_target2_4plate_${codec}_raw_features.parquet"
        echo "--- cell_count codec: ${codec} --- $(date)"
        if [ ! -f "${feature_file}" ]; then
            echo "  SKIPPING: ${feature_file} not found"
            continue
        fi
        echo "  config: ${CC_CONFIG} $(date)"
        CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
            +sweep="${CC_CONFIG}" \
            input.path="${feature_file}" \
            hydra/launcher=joblib hydra.launcher.n_jobs=4 || {
            echo "  Warning: ${CC_CONFIG} encountered errors"
        }
    done
    echo "=== cell_count baseline DONE === $(date)"
) > "${LOG_DIR}/gpu0_cellprofiler.log" 2>&1 &
echo "  Started cellprofiler + cell_count on GPU 0 (PID $!)"

# ============================================================
# GPU 1: morphem + subcell + subcell__nonstd
# ============================================================
(
    cd src/norm_3
    run_dl_model subcell__clip01 1
    run_dl_model morphem 1
    run_dl_model subcell 1
    run_dl_model subcell__nonstd 1
) > "${LOG_DIR}/gpu1_morphem_subcell.log" 2>&1 &
echo "  Started morphem+subcell+subcell__nonstd on GPU 1 (PID $!)"

# ============================================================
# GPU 2: openphenom + openphenom_nonclip + dinov2_random
# ============================================================
(
    cd src/norm_3
    run_dl_model openphenom 2
    run_dl_model openphenom_nonclip 2
    run_dl_model dinov2_random 2
) > "${LOG_DIR}/gpu2_openphenom.log" 2>&1 &
echo "  Started openphenom+openphenom_nonclip+dinov2_random on GPU 2 (PID $!)"

# ============================================================
# GPU 3: dinov2 + openphenom_stdscale + openphenom_stdscale_false
# ============================================================
(
    cd src/norm_3
    run_dl_model dinov2 3
    run_dl_model openphenom_stdscale 3
    run_dl_model openphenom_stdscale_false 3
) > "${LOG_DIR}/gpu3_dinov2.log" 2>&1 &
echo "  Started dinov2+openphenom_stdscale+openphenom_stdscale_false on GPU 3 (PID $!)"

echo ""
echo "All models launched. Waiting for completion..."
echo "  Monitor with: tail -f ${LOG_DIR}/*.log"
echo ""

wait

echo ""
echo "==================================================="
echo "All Focused v10 Sweeps Complete! $(date)"
echo "==================================================="

# Count results
echo ""
echo "Final counts:"
for d in src/norm_3/data/features/variance_first_v10/*/; do
    name=$(basename "$d")
    count=$(find "$d" -name metrics.json 2>/dev/null | wc -l)
    echo "  ${name}: ${count} configs"
done

echo ""
echo "Run summary with:"
echo "  cd src/norm_3 && pixi run python gather_sweep_results.py --sweep-dir data/features/variance_first_v10 --plot --filter-degenerate --best-metric nap_balanced --exclude-codecs mq_new d20_e2 d50"
