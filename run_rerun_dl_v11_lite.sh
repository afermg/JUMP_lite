#!/usr/bin/env bash
set -euo pipefail

# Rerun ALL DL models for v11_lite with fixed prune/noprune naming
# Split across GPU 3, GPU 2, and GPU 1

cd "$(dirname "$0")"

LOG_DIR="logs/sweep_v11_lite_rerun"
mkdir -p "$LOG_DIR"

echo "==================================================="
echo "DL Rerun — v11_lite (fixed prune/noprune naming)"
echo "  GPU 3: morphem (4 codecs)"
echo "  GPU 2: dinov2 (4 codecs) + dinov2_random (4 codecs)"
echo "  GPU 1: openphenom (4 codecs)"
echo "  GPU 0: subcell_clip01 (4 codecs)"
echo "==================================================="
echo ""

# Limit numpy/scipy threading
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

NJOBS_DL=16
DL_CONFIGS="focused_dl_v11_lite_tvn_efaar"
DL_FEATURE_DIR="../../data/features/jump_lite_cl_3"

# ============================================================
# GPU 3: morphem (mq+d20+hq+raw)
# ============================================================
(
    cd src/norm_3

    # morphem: mq, d20, hq, raw
    for codec in jpegxl_lossy_mq jpegxl_lossy_d20 jpegxl_lossy_hq jpegxl_lossy_raw; do
        feature_file="${DL_FEATURE_DIR}/morphem_jump_lite_updated_${codec}_raw_features.parquet"
        echo "--- morphem codec: ${codec} --- $(date)"
        if [ ! -f "${feature_file}" ]; then
            echo "  SKIPPING: ${feature_file} not found"
            continue
        fi
        for cfg in $DL_CONFIGS; do
            echo "  config: ${cfg} $(date)"
            CUDA_VISIBLE_DEVICES=3 pixi run python -m norm_3.pipeline --multirun \
                +sweep="${cfg}" \
                input.path="${feature_file}" \
                hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_DL} || {
                echo "  Warning: ${cfg} encountered errors"
            }
        done
    done
    echo "=== morphem DONE === $(date)"
) > "${LOG_DIR}/gpu3_morphem.log" 2>&1 &
echo "  Started morphem on GPU 3 (PID $!)"

# ============================================================
# GPU 0: subcell_clip01 (mq+d20+hq+raw)
# ============================================================
(
    cd src/norm_3

    # subcell_clip01: mq, d20, hq, raw
    for codec in jpegxl_lossy_raw jpegxl_lossy_mq jpegxl_lossy_d20 jpegxl_lossy_hq; do
        feature_file="${DL_FEATURE_DIR}/subcell__clip01_jump_lite_updated_${codec}_raw_features.parquet"
        echo "--- subcell_clip01 codec: ${codec} --- $(date)"
        if [ ! -f "${feature_file}" ]; then
            echo "  SKIPPING: ${feature_file} not found"
            continue
        fi
        for cfg in $DL_CONFIGS; do
            echo "  config: ${cfg} $(date)"
            CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
                +sweep="${cfg}" \
                input.path="${feature_file}" \
                hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_DL} || {
                echo "  Warning: ${cfg} encountered errors"
            }
        done
    done
    echo "=== subcell_clip01 DONE === $(date)"

    # # subcell: mq only
    # feature_file="${DL_FEATURE_DIR}/subcell_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet"
    # echo "--- subcell codec: jpegxl_lossy_mq --- $(date)"
    # if [ -f "${feature_file}" ]; then
    #     for cfg in $DL_CONFIGS; do
    #         echo "  config: ${cfg} $(date)"
    #         CUDA_VISIBLE_DEVICES=2 pixi run python -m norm_3.pipeline --multirun \
    #             +sweep="${cfg}" \
    #             input.path="${feature_file}" \
    #             hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_DL} || {
    #             echo "  Warning: ${cfg} encountered errors"
    #         }
    #     done
    # else
    #     echo "  SKIPPING: ${feature_file} not found"
    # fi
    # echo "=== subcell DONE === $(date)"
) > "${LOG_DIR}/gpu0_subcell.log" 2>&1 &
echo "  Started subcell_clip01 on GPU 0 (PID $!)"

# ============================================================
# GPU 1: openphenom (mq+d20+hq+raw)
# ============================================================
(
    cd src/norm_3

    # openphenom: mq, d20, hq, raw
    for codec in jpegxl_lossy_raw jpegxl_lossy_mq jpegxl_lossy_d20 jpegxl_lossy_hq; do
        feature_file="${DL_FEATURE_DIR}/openphenom_jump_lite_updated_${codec}_raw_features.parquet"
        echo "--- openphenom codec: ${codec} --- $(date)"
        if [ ! -f "${feature_file}" ]; then
            echo "  SKIPPING: ${feature_file} not found"
            continue
        fi
        for cfg in $DL_CONFIGS; do
            echo "  config: ${cfg} $(date)"
            CUDA_VISIBLE_DEVICES=1 pixi run python -m norm_3.pipeline --multirun \
                +sweep="${cfg}" \
                input.path="${feature_file}" \
                hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_DL} || {
                echo "  Warning: ${cfg} encountered errors"
            }
        done
    done
    echo "=== openphenom DONE === $(date)"
) > "${LOG_DIR}/gpu1_openphenom.log" 2>&1 &
echo "  Started openphenom on GPU 1 (PID $!)"

# ============================================================
# GPU 2: dinov2 (mq+d20+hq+raw) + dinov2_random (mq+d20+hq+raw)
# ============================================================
(
    cd src/norm_3

    # dinov2: mq, d20, hq, raw
    for codec in jpegxl_lossy_raw jpegxl_lossy_mq jpegxl_lossy_d20 jpegxl_lossy_hq; do
        feature_file="${DL_FEATURE_DIR}/dinov2_jump_lite_updated_${codec}_raw_features.parquet"
        echo "--- dinov2 codec: ${codec} --- $(date)"
        if [ ! -f "${feature_file}" ]; then
            echo "  SKIPPING: ${feature_file} not found"
            continue
        fi
        for cfg in $DL_CONFIGS; do
            echo "  config: ${cfg} $(date)"
            CUDA_VISIBLE_DEVICES=2 pixi run python -m norm_3.pipeline --multirun \
                +sweep="${cfg}" \
                input.path="${feature_file}" \
                hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_DL} || {
                echo "  Warning: ${cfg} encountered errors"
            }
        done
    done
    echo "=== dinov2 DONE === $(date)"

    # dinov2_random: mq, d20, hq, raw
    for codec in jpegxl_lossy_mq jpegxl_lossy_d20 jpegxl_lossy_hq jpegxl_lossy_raw; do
        feature_file="${DL_FEATURE_DIR}/dinov2_random_jump_lite_updated_${codec}_raw_features.parquet"
        echo "--- dinov2_random codec: ${codec} --- $(date)"
        if [ ! -f "${feature_file}" ]; then
            echo "  SKIPPING: ${feature_file} not found"
            continue
        fi
        for cfg in $DL_CONFIGS; do
            echo "  config: ${cfg} $(date)"
            CUDA_VISIBLE_DEVICES=2 pixi run python -m norm_3.pipeline --multirun \
                +sweep="${cfg}" \
                input.path="${feature_file}" \
                hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_DL} || {
                echo "  Warning: ${cfg} encountered errors"
            }
        done
    done
    echo "=== dinov2_random DONE === $(date)"
) > "${LOG_DIR}/gpu2_dinov2.log" 2>&1 &
echo "  Started dinov2 + dinov2_random on GPU 2 (PID $!)"

echo ""
echo "All DL models launched. Waiting for completion..."
echo "  Monitor with: tail -f ${LOG_DIR}/*.log"
echo ""

wait

echo ""
echo "==================================================="
echo "All DL Reruns Complete! $(date)"
echo "==================================================="

# Count results
echo ""
echo "Final counts:"
for d in src/norm_3/data/features/variance_first_v11_lite/*/; do
    name=$(basename "$d")
    count=$(find "$d" -name metrics.json 2>/dev/null | wc -l)
    echo "  ${name}: ${count} configs"
done
