#!/usr/bin/env bash
set -euo pipefail

# Run from JUMP_core root directory
cd "$(dirname "$0")"

LOG_DIR="logs/sweep_v11"
mkdir -p "$LOG_DIR"

echo "==================================================="
echo "Focused Sweep v11 — target2 (5 DL + CP + cell_count, 3 GPUs)"
echo "  DL features re-extracted with median aggregation (cl_3)"
echo "  tvn_efaar only (none/tvn_original dropped — consistently worse)"
echo "  DL: 48 configs/codec, CP: 48 configs/codec, cell_count: 4 configs/codec"
echo "==================================================="
echo ""

# Limit numpy/scipy threading to avoid oversubscription
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

# DL codecs (10): zstd + 9 JPEG XL variants — from cl_3 (median aggregation)
DL_CODECS="zstd jpegxl_lossy_hq jpegxl_lossy_mq jpegxl_lossy_mq_new jpegxl_lossy_lq jpegxl_lossy_d2_e8 jpegxl_lossy_d10 jpegxl_lossy_d15 jpegxl_lossy_d30 jpegxl_lossy_effort_3"

# CP codecs (7): zstd + 6 JPEG XL variants — from existing cp features (already median)
CP_CODECS="zstd jpegxl_lossy_hq jpegxl_lossy_mq jpegxl_lossy_lq jpegxl_lossy_d2_e8 jpegxl_lossy_d10 jpegxl_lossy_effort_3"

# Sweep configs (v11 — output to variance_first_v11)
DL_CONFIGS="focused_dl_v11_tvn_efaar"
CP_CONFIGS="focused_cp_v11_tvn_efaar"
CC_CONFIG="focused_cell_count_v11"

# Feature path templates
DL_FEATURE_DIR="../../data/features/jump_target2_4plate_cl_3"
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

echo "  GPU 0: cp_measure (7 codecs x 48 configs) + cell_count (7 codecs x 4 configs)"
echo "  GPU 1: morphem + subcell__clip01 (10 codecs x 48 configs each)"
echo "  GPU 2: openphenom + dinov2 + dinov2_random (10 codecs x 48 configs each)"
echo ""
echo "Logs: ${LOG_DIR}/"
echo ""

# ============================================================
# GPU 0: CellProfiler + cell_count (7 codecs each)
# ============================================================
(
    cd src/norm_3
    echo "=== cp_measure on GPU 0 === $(date)"
    for codec in $CP_CODECS; do
        feature_file="${CP_FEATURE_DIR}/cp_measure_jump_target2_4plate_${codec}_raw_features.parquet"
        echo "--- cp_measure codec: ${codec} --- $(date)"
        if [ ! -f "${feature_file}" ]; then
            echo "  SKIPPING: ${feature_file} not found"
            continue
        fi
        for cfg in $CP_CONFIGS; do
            echo "  config: ${cfg} $(date)"
            CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
                +sweep="${cfg}" \
                input.path="${feature_file}" \
                hydra/launcher=joblib hydra.launcher.n_jobs=4 || {
                echo "  Warning: ${cfg} encountered errors"
            }
        done
    done
    echo "=== cp_measure DONE === $(date)"

    # --- Cell Count baseline (same 7 CP codecs, 4 configs each) ---
    echo "=== cell_count baseline on GPU 0 === $(date)"
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
) > "${LOG_DIR}/gpu0_cp_cellcount.log" 2>&1 &
echo "  Started cp_measure + cell_count on GPU 0 (PID $!)"

# ============================================================
# GPU 1: morphem + subcell__clip01
# ============================================================
(
    cd src/norm_3
    run_dl_model morphem 1
    run_dl_model subcell__clip01 1
) > "${LOG_DIR}/gpu1_morphem_subcell.log" 2>&1 &
echo "  Started morphem + subcell__clip01 on GPU 1 (PID $!)"

# ============================================================
# GPU 2: openphenom + dinov2 + dinov2_random
# ============================================================
(
    cd src/norm_3
    run_dl_model openphenom 2
    run_dl_model dinov2 2
    run_dl_model dinov2_random 2
) > "${LOG_DIR}/gpu2_openphenom_dinov2.log" 2>&1 &
echo "  Started openphenom + dinov2 + dinov2_random on GPU 2 (PID $!)"

echo ""
echo "All models launched. Waiting for completion..."
echo "  Monitor with: tail -f ${LOG_DIR}/*.log"
echo ""

wait

echo ""
echo "==================================================="
echo "All Focused v11 Sweeps Complete! $(date)"
echo "==================================================="

# Count results
echo ""
echo "Final counts:"
for d in src/norm_3/data/features/variance_first_v11/*/; do
    name=$(basename "$d")
    count=$(find "$d" -name metrics.json 2>/dev/null | wc -l)
    echo "  ${name}: ${count} configs"
done

echo ""
echo "Run summary with:"
echo "  cd src/norm_3 && pixi run python gather_sweep_results.py --sweep-dir data/features/variance_first_v11 --plot --filter-degenerate --best-metric nap_balanced --exclude-codecs mq_new d20_e2 d50"
