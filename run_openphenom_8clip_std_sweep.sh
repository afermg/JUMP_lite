#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/src/norm_3"

FEATURE_DIR="../../data/features/jump_target2_4plate_cl_2"
MODEL="openphenom_8clip_std"
CODECS="zstd jpegxl_lossy_hq jpegxl_lossy_mq jpegxl_lossy_lq jpegxl_lossy_d2_e8 jpegxl_lossy_d10 jpegxl_lossy_d15 jpegxl_lossy_d30 jpegxl_lossy_effort_3"
SWEEP="focused_dl_v10_tvn_efaar"

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

for codec in $CODECS; do
    feature_file="${FEATURE_DIR}/${MODEL}_jump_target2_4plate_${codec}_raw_features.parquet"
    if [ ! -f "${feature_file}" ]; then
        echo "SKIPPING: ${feature_file} not found"
        continue
    fi
    echo "=== ${MODEL} codec: ${codec} === $(date)"
    CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
        +sweep="${SWEEP}" \
        input.path="${feature_file}" \
        hydra/launcher=joblib hydra.launcher.n_jobs=4 || {
        echo "  Warning: ${codec} encountered errors"
    }
done

echo "=== ${MODEL} sweep DONE === $(date)"
