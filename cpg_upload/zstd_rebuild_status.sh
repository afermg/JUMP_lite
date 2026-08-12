#!/usr/bin/env bash
# Report progress for the active lossless Zstd rebuild.
set -euo pipefail

LOG_PARENT="/work/datasets/jump_lite/zstd_rebuild_logs"
STATE_ROOT="/work/datasets/jump_lite/zstd_rebuild_state/v1.0"
RUN_ROOT=${1:-"$LOG_PARENT/latest"}

if [[ ! -e "$RUN_ROOT" ]]; then
  echo "ERROR: Zstd rebuild run does not exist: $RUN_ROOT" >&2
  exit 1
fi
RUN_ROOT=$(readlink -f "$RUN_ROOT")
echo "Run: $RUN_ROOT"

if [[ -f "$RUN_ROOT/pid" ]]; then
  pid=$(<"$RUN_ROOT/pid")
  if kill -0 "$pid" 2>/dev/null; then
    echo "Process: running (PID $pid)"
  else
    echo "Process: not running (last PID $pid)"
  fi
fi
[[ -f "$RUN_ROOT/COMPLETE" ]] && echo "State: COMPLETE"
[[ -f "$RUN_ROOT/FAILED" ]] && echo "State: FAILED"

if [[ -f "$STATE_ROOT/checkpoint.json" ]]; then
  echo
echo "Checkpoint:"
  python - "$STATE_ROOT/checkpoint.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
for key in (
    "processed_manifest_sites", "target_sites", "created_sites",
    "skipped_complete_sites", "sites_per_second", "last_site_key",
    "downloaded_bytes_this_run", "compressed_bytes_created_this_run",
    "updated_at", "complete", "final_path", "legacy_quarantine_path",
):
    if key in p:
        value=p[key]
        if isinstance(value, int): value=f"{value:,}"
        print(f"  {key}: {value}")
PY
fi

echo
echo "Recent log lines:"
tail -n 20 "$RUN_ROOT/rebuild.log" 2>/dev/null || true
