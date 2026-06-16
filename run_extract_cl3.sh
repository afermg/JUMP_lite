#!/usr/bin/env bash
set -euo pipefail

INPUT="/work/datasets/aliby_output/plate4_rerun_scale_std"
DATASET="jump_target2_4plate"
OUTPUT="data/features/jump_target2_4plate_cl_3"

for model in morphem dinov2 dinov2_random openphenom subcell__clip01; do
    echo "=== Extracting $model === $(date)"
    nix develop . --command uv run python src/extract_features_fast.py \
        --input "$INPUT" \
        --dataset "$DATASET" \
        --output "$OUTPUT" \
        --model "$model" --n-jobs 4
done

echo ""
echo "Extraction done! $(date)"
echo "Parquet count: $(ls "$OUTPUT"/*.parquet 2>/dev/null | wc -l) (expected 60)"

echo ""
echo "=== Starting v11 sweep ==="
bash run_focused_v11_sweep.sh
