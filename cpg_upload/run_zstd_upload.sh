#!/usr/bin/env bash
# Stream checkpoint-confirmed Zstd arrays to CPG staging with durable logs.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
LOG_PARENT="/work/datasets/jump_lite/zstd_upload_logs"
RUN_ID=${ZSTD_UPLOAD_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT="$LOG_PARENT/$RUN_ID"
PYTHON=${ZSTD_UPLOAD_PYTHON:-"$PROJECT_ROOT/.venv/bin/python"}

if [[ ${1:-} != "--apply" || $# -ne 1 ]]; then
  echo "Usage: run_zstd_upload.sh --apply" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python environment is unavailable: $PYTHON" >&2
  exit 1
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "ERROR: run directory already exists: $RUN_ROOT" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"
printf '%s\n' "$$" >"$RUN_ROOT/pid"
ln -sfn "$RUN_ROOT" "$LOG_PARENT/latest"

set +e
"$PYTHON" "$SCRIPT_DIR/upload_zstd_to_staging.py" \
  --apply \
  --workers "${ZSTD_UPLOAD_WORKERS:-24}" \
  --batch-size "${ZSTD_UPLOAD_BATCH_SIZE:-192}" \
  --poll-seconds "${ZSTD_UPLOAD_POLL_SECONDS:-30}" \
  >"$RUN_ROOT/upload.log" 2>&1
status=$?
set -e

if [[ $status -eq 0 ]]; then
  printf '%s\n' "$(date -u +%FT%TZ)" >"$RUN_ROOT/COMPLETE"
else
  printf '%s\n' "$(date -u +%FT%TZ)" >"$RUN_ROOT/FAILED"
fi
exit "$status"
