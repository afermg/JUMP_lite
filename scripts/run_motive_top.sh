#!/usr/bin/env bash
# End-to-end MOTIVE eval on the top-N configs of a norm_3 sweep.
#
#   1. filter sweep_results.csv → top-N configs per (family, codec)
#   2. evaluate against full annotations → motive_large_full/
#   3. evaluate against strict annotations → motive_large_strict/
#   4. plot both
#
# Override any path by editing the variables at the top, or pass them as env
# vars (e.g. `JOBS=16 ./scripts/run_motive_top.sh`).

set -euo pipefail

SWEEP_DIR=${SWEEP_DIR:-data/intermediate/sweep_v11_lite}
SWEEP_RESULTS=${SWEEP_RESULTS:-${SWEEP_DIR}/sweep_results.csv}
OUT_FULL=${OUT_FULL:-data/intermediate/motive_eval/large_full}
OUT_STRICT=${OUT_STRICT:-data/intermediate/motive_eval/large_strict}

LIST=${LIST:-metadata/motive_top_configs.txt}
TOP_N=${TOP_N:-50}
METRIC=${METRIC:-PA_mean_nap}
JOBS=${JOBS:-32}

ANN_FULL=${ANN_FULL:-metadata/motive_annotations.parquet}
ANN_STRICT=${ANN_STRICT:-metadata/motive_annotations_strict.parquet}
SPLITS=${SPLITS:-metadata/motive_eval_compounds.parquet}

PLOT_FULL=${PLOT_FULL:-data/results/figures/motive_large_full}
PLOT_STRICT=${PLOT_STRICT:-data/results/figures/motive_large_strict}

echo "=== 1/4: filter top-${TOP_N} per (family, codec) by ${METRIC} ==="
nix develop . --command just motive-filter-top \
    "${SWEEP_RESULTS}" "${SWEEP_DIR}" "${TOP_N}" "${METRIC}" "${LIST}"

echo
echo "=== 2/4: motive-eval-list (full annotations) → ${OUT_FULL} ==="
nix develop . --command just motive-eval-list \
    "${SWEEP_DIR}" "${OUT_FULL}" "${LIST}" "${JOBS}" "${ANN_FULL}" "${SPLITS}"

echo
echo "=== 3/4: motive-eval-list (strict annotations) → ${OUT_STRICT} ==="
nix develop . --command just motive-eval-list \
    "${SWEEP_DIR}" "${OUT_STRICT}" "${LIST}" "${JOBS}" "${ANN_STRICT}" "${SPLITS}"

echo
echo "=== 4/4: plot both ==="
nix develop . --command just motive-plot "${OUT_FULL}"   "${PLOT_FULL}"
nix develop . --command just motive-plot "${OUT_STRICT}" "${PLOT_STRICT}"

echo
echo "DONE."
echo "  full results:   ${OUT_FULL}"
echo "  strict results: ${OUT_STRICT}"
echo "  full plots:     ${PLOT_FULL}"
echo "  strict plots:   ${PLOT_STRICT}"
