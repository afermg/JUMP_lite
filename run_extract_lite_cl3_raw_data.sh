#!/usr/bin/env bash
set -euo pipefail

INPUT="/work/datasets/aliby_output/jump_lite_raw/"
OUTPUT="data/features/jump_lite_cl_3"

# Limit threading for all numeric libraries
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8

for model in dinov2_random; do
    echo "=== Extracting $model === $(date)"
    nix develop . --command uv run python src/extract_features_fast.py \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --model "$model" --n-jobs 1
done

echo ""
echo "Extraction done! $(date)"
echo "Parquet count: $(ls "$OUTPUT"/*.parquet 2>/dev/null | wc -l) (expected 10)"

echo ""
echo "=== Starting v11 lite sweep ==="
#bash run_focused_v11_lite_sweep.sh
