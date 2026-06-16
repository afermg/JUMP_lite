#!/usr/bin/env bash
set -euo pipefail

# Run from JUMP_core root directory
cd "$(dirname "$0")"

LOG_DIR="logs/sweep_v9"
mkdir -p "$LOG_DIR"

echo "==================================================="
echo "Focused Sweep v9 — jump_lite (5 datasets, 4 GPUs)"
echo "  DL: no outlier removal, no correlation pruning, no INT"
echo "  CP: fixed outlier=100, corr=0.9, no INT"
echo "==================================================="
echo ""
# Limit numpy/scipy threading to avoid oversubscription
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

# DL: ~12 + 48 + 16 + ~24 = ~100 configs per dataset
# CP: ~12 + 48 + 16 + ~24 = ~100 configs
echo "  GPU 0: cellprofiler  (~100 configs) -> subcell (~100 configs)"
echo "  GPU 1: morphem       (~100 configs)"
echo "  GPU 2: openphenom    (~100 configs)"
echo "  GPU 3: dinov2        (~100 configs)"
echo ""
echo "Logs: ${LOG_DIR}/"
echo ""

# Sweep configs (run sequentially per model)
DL_CONFIGS="focused_dl_v9_none focused_dl_v9_tvn_efaar focused_dl_v9_tvn_original focused_dl_v9_spherize"
CP_CONFIGS="focused_cp_v9_none focused_cp_v9_tvn_efaar focused_cp_v9_tvn_original focused_cp_v9_spherize"

# ============================================================
# GPU 0: CellProfiler then subcell (background)
# ============================================================
(
    cd src/norm_3
    echo "=== cellprofiler on GPU 0 === $(date)"
    for cfg in $CP_CONFIGS; do
        echo "--- ${cfg} --- $(date)"
        CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
            +sweep="${cfg}" \
            input.path="../../data/features/jump_lite/cellprofiler_raw_jump_lite_raw_features.parquet" \
            hydra/launcher=joblib hydra.launcher.n_jobs=4
    done
    echo "=== cellprofiler DONE === $(date)"
    echo ""
    echo "=== subcell on GPU 0 === $(date)"
    for cfg in $DL_CONFIGS; do
        echo "--- ${cfg} --- $(date)"
        CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
            +sweep="${cfg}" \
            input.path="../../data/features/jump_lite/subcell_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet" \
            hydra/launcher=joblib hydra.launcher.n_jobs=4
    done
    echo "=== subcell DONE === $(date)"
) > "${LOG_DIR}/gpu0_cellprofiler_subcell.log" 2>&1 &
echo "  Started cellprofiler+subcell on GPU 0 (PID $!)"

# ============================================================
# GPU 1: morphem mq + d20 (background)
# ============================================================
(
    cd src/norm_3
    for input_file in \
        "../../data/features/jump_lite/morphem_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet" \
        "../../data/features/jump_lite/morphem_jump_lite_updated_jpegxl_lossy_d20_raw_features.parquet"; do
        echo "=== morphem $(basename "$input_file") === $(date)"
        for cfg in $DL_CONFIGS; do
            echo "--- ${cfg} --- $(date)"
            CUDA_VISIBLE_DEVICES=1 pixi run python -m norm_3.pipeline --multirun \
                +sweep="${cfg}" \
                input.path="${input_file}" \
                hydra/launcher=joblib hydra.launcher.n_jobs=4
        done
    done
    echo "=== morphem DONE === $(date)"
) > "${LOG_DIR}/morphem.log" 2>&1 &
echo "  Started morphem (mq + d20) on GPU 1 (PID $!)"

# ============================================================
# GPU 2: openphenom (background)
# ============================================================
(
    cd src/norm_3
    for cfg in $DL_CONFIGS; do
        echo "--- ${cfg} --- $(date)"
        CUDA_VISIBLE_DEVICES=2 pixi run python -m norm_3.pipeline --multirun \
            +sweep="${cfg}" \
            input.path="../../data/features/jump_lite/openphenom_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet" \
            hydra/launcher=joblib hydra.launcher.n_jobs=4
    done
    echo "=== openphenom DONE === $(date)"
) > "${LOG_DIR}/openphenom.log" 2>&1 &
echo "  Started openphenom on GPU 2 (PID $!)"

# ============================================================
# GPU 3: dinov2 + dinov2_random (background)
# ============================================================
(
    cd src/norm_3
    for input_file in \
        "../../data/features/jump_lite/dinov2_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet" \
        "../../data/features/jump_lite/dinov2_random_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet"; do
        echo "=== $(basename "$input_file") === $(date)"
        for cfg in $DL_CONFIGS; do
            echo "--- ${cfg} --- $(date)"
            CUDA_VISIBLE_DEVICES=3 pixi run python -m norm_3.pipeline --multirun \
                +sweep="${cfg}" \
                input.path="${input_file}" \
                hydra/launcher=joblib hydra.launcher.n_jobs=4
        done
    done
    echo "=== dinov2 + dinov2_random DONE === $(date)"
) > "${LOG_DIR}/dinov2.log" 2>&1 &
echo "  Started dinov2 + dinov2_random on GPU 3 (PID $!)"

echo ""
echo "All models launched. Waiting for completion..."
echo "  Monitor with: tail -f ${LOG_DIR}/*.log"
echo ""

wait

echo ""
echo "==================================================="
echo "All Focused v9 Sweeps Complete! $(date)"
echo "==================================================="

# Count results
echo ""
echo "Final counts:"
for d in src/norm_3/data/features/variance_first_v9/*/; do
    name=$(basename "$d")
    count=$(find "$d" -name metrics.json 2>/dev/null | wc -l)
    echo "  ${name}: ${count} configs"
done

echo ""
echo "Run summary with:"
echo "  cd src/norm_3 && pixi run python gather_sweep_results.py --sweep-dir data/features/variance_first_v9 --plot --filter-degenerate"
