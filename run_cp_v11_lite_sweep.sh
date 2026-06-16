#!/usr/bin/env bash
set -euo pipefail

# Run CellProfiler sweep for v11_lite on GPU 2
# Single feature file, expanded sweep grid (norm_method + n_components)

cd "$(dirname "$0")"

LOG_DIR="logs/sweep_v11_lite_rerun"
mkdir -p "$LOG_DIR"

echo "==================================================="
echo "CellProfiler Sweep — v11_lite (GPU 2)"
echo "==================================================="
echo ""

# Limit numpy/scipy threading
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

NJOBS_CP=4
CP_CONFIGS="focused_cp_v11_lite_tvn_efaar"
CP_FEATURE_DIR="../../data/features/jump_lite"

(
    cd src/norm_3

    echo "=== cellprofiler on GPU 2 === $(date)"
    feature_file="${CP_FEATURE_DIR}/cellprofiler_raw_jump_lite_raw_features.parquet"
    if [ ! -f "${feature_file}" ]; then
        echo "  ERROR: ${feature_file} not found"
        exit 1
    fi
    for cfg in $CP_CONFIGS; do
        echo "--- ${cfg} --- $(date)"
        CUDA_VISIBLE_DEVICES=2 pixi run python -m norm_3.pipeline --multirun \
            +sweep="${cfg}" \
            input.path="${feature_file}" \
            hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_CP} || {
            echo "  Warning: ${cfg} encountered errors"
        }
    done
    echo "=== cellprofiler DONE === $(date)"
) > "${LOG_DIR}/gpu2_cp.log" 2>&1 &
echo "  Started cellprofiler on GPU 2 (PID $!)"

echo ""
echo "Waiting for completion..."
echo "  Monitor with: tail -f ${LOG_DIR}/gpu2_cp.log"
echo ""

wait

echo ""
echo "==================================================="
echo "CellProfiler v11 Lite Sweep Complete! $(date)"
echo "==================================================="

# Count results
echo ""
echo "Final counts:"
for d in src/norm_3/data/features/variance_first_v11_lite/*/; do
    name=$(basename "$d")
    count=$(find "$d" -name metrics.json 2>/dev/null | wc -l)
    echo "  ${name}: ${count} configs"
done
