#!/usr/bin/env bash
# Report local progress for the active JUMP-Lite CPG staging upload.
set -euo pipefail

LOG_PARENT="/work/datasets/jump_lite/cpg_upload_logs"
STATE_ROOT="/work/datasets/jump_lite/cpg_upload_state/v1.0/profiles"
RUN_ROOT=${1:-"$LOG_PARENT/latest"}

if [[ ! -e "$RUN_ROOT" ]]; then
  echo "ERROR: upload run does not exist: $RUN_ROOT" >&2
  exit 1
fi
RUN_ROOT=$(readlink -f "$RUN_ROOT")

echo "Run: $RUN_ROOT"
if [[ -f "$RUN_ROOT/supervisor.pid" ]]; then
  pid=$(<"$RUN_ROOT/supervisor.pid")
  if kill -0 "$pid" 2>/dev/null; then
    echo "Supervisor: running (PID $pid)"
  else
    echo "Supervisor: not running (last PID $pid)"
  fi
fi
[[ -f "$RUN_ROOT/COMPLETE" ]] && echo "Overall state: COMPLETE"
[[ -f "$RUN_ROOT/FAILED" ]] && echo "Overall state: FAILED"

echo
echo "Components:"
if [[ -f "$RUN_ROOT/component-pids.tsv" ]]; then
  while IFS=$'\t' read -r pid name; do
    if [[ -f "$RUN_ROOT/components/$name.complete" ]]; then
      state=complete
    elif kill -0 "$pid" 2>/dev/null; then
      state=running
    else
      state=stopped
    fi
    printf '  %-28s %-8s PID=%s\n' "$name" "$state" "$pid"
  done <"$RUN_ROOT/component-pids.tsv"
else
  echo "  component PID inventory not written yet"
fi

echo
echo "Profile checkpoints:"
python - "$STATE_ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
paths = sorted(root.glob("*/*.json")) if root.exists() else []
if not paths:
    print("  no completed checkpoint intervals yet")
else:
    uploaded = total = 0
    for path in paths:
        data = json.loads(path.read_text())
        done = int(data.get("next_index", 0))
        count = int(data.get("file_count", 0))
        uploaded += done
        total += count
        print(f"  {data['model']}/{data['codec']}: {done:,}/{count:,}")
    print(f"  checkpointed total: {uploaded:,}/{total:,}")
PY

echo
echo "Recent log lines:"
for log in "$RUN_ROOT"/components/*.log; do
  [[ -e "$log" ]] || continue
  echo "--- $(basename "$log")"
  tail -n 3 "$log"
done
