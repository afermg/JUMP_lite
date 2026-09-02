#!/usr/bin/env bash
# Run MOTIVE eval on EVERY config under a sweep dir (not just the top-N).
# Mirrors run_motive_top.sh, but uses motive-eval-sweep (walks all parquets)
# instead of motive-eval-list (reads a filtered list).
#
# Output dirs default to the same locations as run_motive_top.sh — already-
# completed configs from the top-N run are skipped via idempotency, so this
# fills in the remaining configs without redoing work.

set -euo pipefail

SWEEP_DIR=${SWEEP_DIR:-data/intermediate/sweep_v11_lite}
OUT_FULL=${OUT_FULL:-data/intermediate/motive_eval/large_full}
OUT_STRICT=${OUT_STRICT:-data/intermediate/motive_eval/large_strict}

JOBS=${JOBS:-32}

ANN_FULL=${ANN_FULL:-metadata/motive_annotations.parquet}
ANN_STRICT=${ANN_STRICT:-metadata/motive_annotations_strict.parquet}
SPLITS=${SPLITS:-metadata/motive_eval_compounds.parquet}

PLOT_FULL=${PLOT_FULL:-data/results/figures/motive_large_full}
PLOT_STRICT=${PLOT_STRICT:-data/results/figures/motive_large_strict}

n_total=$(find "${SWEEP_DIR}" -name output.parquet 2>/dev/null | wc -l)
echo "[run_motive_all] sweep has ${n_total} output.parquet files"

echo
echo "=== 1/3: motive-eval-sweep (full annotations) → ${OUT_FULL} ==="
just motive-eval-sweep \
    "${SWEEP_DIR}" "${OUT_FULL}" "${JOBS}" "${ANN_FULL}" "${SPLITS}"

echo
echo "=== 2/3: motive-eval-sweep (strict annotations) → ${OUT_STRICT} ==="
just motive-eval-sweep \
    "${SWEEP_DIR}" "${OUT_STRICT}" "${JOBS}" "${ANN_STRICT}" "${SPLITS}"

echo
echo "=== 3/3: regenerate plots + delta plots ==="
just motive-plot       "${OUT_FULL}"   "${PLOT_FULL}"
just motive-plot-delta "${PLOT_FULL}"
just motive-plot       "${OUT_STRICT}" "${PLOT_STRICT}"
just motive-plot-delta "${PLOT_STRICT}"

echo
echo "DONE."
echo "  full results:   ${OUT_FULL}"
echo "  strict results: ${OUT_STRICT}"
echo "  full plots:     ${PLOT_FULL}"
echo "  strict plots:   ${PLOT_STRICT}"
