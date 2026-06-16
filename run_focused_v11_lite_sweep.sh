#!/usr/bin/env bash
set -euo pipefail

# Run from JUMP_core root directory
cd "$(dirname "$0")"

LOG_DIR="logs/sweep_v11_lite"
mkdir -p "$LOG_DIR"

echo "==================================================="
echo "Focused Sweep v11 lite — jump_lite (5 DL + CP + cell_count, 3 GPUs)"
echo "  DL features re-extracted with median aggregation (cl_3)"
echo "  tvn_efaar only"
echo "  DL: 48 configs/codec, CP: 48 configs/codec, cell_count: 4 configs"
echo "==================================================="
echo ""

# Limit numpy/scipy threading to avoid oversubscription
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

# Parallelism per model type (CP uses ~35GB/worker, DL ~8GB, cell_count ~2GB)
NJOBS_CP=8
NJOBS_DL=16
NJOBS_CELL_COUNT=4

# Sweep configs (v11 lite — output to variance_first_v11_lite)
DL_CONFIGS="focused_dl_v11_lite_tvn_efaar"
CP_CONFIGS="focused_cp_v11_lite_tvn_efaar"

# Feature paths
DL_FEATURE_DIR="../../data/features/jump_lite_cl_3"
CP_FEATURE_DIR="../../data/features/jump_lite"

echo "  GPU 0: cellprofiler (1 dataset x 48 configs) + cell_count (4 configs) + subcell mq (48 configs)"
echo "  GPU 1: morphem (mq + d20 + hq = 3 x 48) + subcell_clip01 (mq + d20 + hq = 3 x 48)"
echo "  GPU 2: openphenom (mq + d20 + hq) + dinov2 (mq + d20) + dinov2_random (mq)"
echo ""
echo "Logs: ${LOG_DIR}/"
echo ""

# ============================================================
# GPU 0: CellProfiler + subcell
# ============================================================
(
    cd src/norm_3
    echo "=== cellprofiler on GPU 0 === $(date)"
    for cfg in $CP_CONFIGS; do
        echo "--- ${cfg} --- $(date)"
        CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
            +sweep="${cfg}" \
            input.path="${CP_FEATURE_DIR}/cellprofiler_raw_jump_lite_raw_features.parquet" \
            hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_CP} || {
            echo "  Warning: ${cfg} encountered errors"
        }
    done
    echo "=== cellprofiler DONE === $(date)"

    echo "=== cell_count baseline on GPU 0 === $(date)"
    CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
        +sweep=focused_cell_count_v11_lite \
        input.path="${CP_FEATURE_DIR}/cell_count_jump_lite_raw_features.parquet" \
        hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_CELL_COUNT} || {
        echo "  Warning: cell_count baseline encountered errors"
    }
    echo "=== cell_count baseline DONE === $(date)"

    echo "=== subcell on GPU 0 === $(date)"
    feature_file="${DL_FEATURE_DIR}/subcell_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet"
    if [ -f "${feature_file}" ]; then
        for cfg in $DL_CONFIGS; do
            echo "--- ${cfg} --- $(date)"
            CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
                +sweep="${cfg}" \
                input.path="${feature_file}" \
                hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_DL} || {
                echo "  Warning: ${cfg} encountered errors"
            }
        done
    else
        echo "  SKIPPING: ${feature_file} not found"
    fi
    echo "=== subcell DONE === $(date)"
) > "${LOG_DIR}/gpu0_cp_subcell.log" 2>&1 &
echo "  Started cellprofiler + subcell on GPU 0 (PID $!)"

# ============================================================
# GPU 1: morphem (mq + d20 + hq)
# ============================================================
(
    cd src/norm_3
    for codec in jpegxl_lossy_mq jpegxl_lossy_d20 jpegxl_lossy_hq; do
        feature_file="${DL_FEATURE_DIR}/morphem_jump_lite_updated_${codec}_raw_features.parquet"
        echo "--- morphem codec: ${codec} --- $(date)"
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
    echo "=== morphem DONE === $(date)"

    # subcell_clip01: mq, d20
    for codec in jpegxl_lossy_mq jpegxl_lossy_d20 jpegxl_lossy_hq; do
        feature_file="${DL_FEATURE_DIR}/subcell__clip01_jump_lite_updated_${codec}_raw_features.parquet"
        echo "--- subcell_clip01 codec: ${codec} --- $(date)"
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
    echo "=== subcell_clip01 DONE === $(date)"
) > "${LOG_DIR}/gpu1_morphem_subcell_clip01.log" 2>&1 &
echo "  Started morphem + subcell_clip01 on GPU 1 (PID $!)"

# ============================================================
# GPU 2: openphenom (mq+d20+hq) + dinov2 (mq+d20) + dinov2_random (mq)
# ============================================================
(
    cd src/norm_3
    # openphenom: mq, d20, hq
    for codec in jpegxl_lossy_mq jpegxl_lossy_d20 jpegxl_lossy_hq; do
        feature_file="${DL_FEATURE_DIR}/openphenom_jump_lite_updated_${codec}_raw_features.parquet"
        echo "--- openphenom codec: ${codec} --- $(date)"
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
    echo "=== openphenom DONE === $(date)"

    # dinov2: mq, d20, hq
    for codec in jpegxl_lossy_mq jpegxl_lossy_d20 jpegxl_lossy_hq; do
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

    # dinov2_random: mq only
    feature_file="${DL_FEATURE_DIR}/dinov2_random_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet"
    echo "--- dinov2_random codec: jpegxl_lossy_mq --- $(date)"
    if [ -f "${feature_file}" ]; then
        for cfg in $DL_CONFIGS; do
            echo "  config: ${cfg} $(date)"
            CUDA_VISIBLE_DEVICES=2 pixi run python -m norm_3.pipeline --multirun \
                +sweep="${cfg}" \
                input.path="${feature_file}" \
                hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_DL} || {
                echo "  Warning: ${cfg} encountered errors"
            }
        done
    else
        echo "  SKIPPING: ${feature_file} not found"
    fi
    echo "=== dinov2_random DONE === $(date)"
) > "${LOG_DIR}/gpu2_openphenom_dinov2.log" 2>&1 &
echo "  Started openphenom + dinov2 + dinov2_random on GPU 2 (PID $!)"

echo ""
echo "All models launched. Waiting for completion..."
echo "  Monitor with: tail -f ${LOG_DIR}/*.log"
echo ""

wait

echo ""
echo "==================================================="
echo "All Focused v11 Lite Sweeps Complete! $(date)"
echo "==================================================="

# Count results
echo ""
echo "Final counts:"
for d in src/norm_3/data/features/variance_first_v11_lite/*/; do
    name=$(basename "$d")
    count=$(find "$d" -name metrics.json 2>/dev/null | wc -l)
    echo "  ${name}: ${count} configs"
done

echo ""
echo "Run summary with:"
echo "  cd src/norm_3 && pixi run python gather_sweep_results.py --sweep-dir data/features/variance_first_v11_lite --plot --filter-degenerate --best-metric nap_balanced"
