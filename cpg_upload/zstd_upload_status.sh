#!/usr/bin/env bash
# Report streaming Zstd-to-CPG upload progress.
set -euo pipefail

LOG_PARENT="/work/datasets/jump_lite/zstd_upload_logs"
STATE="/work/datasets/jump_lite/cpg_upload_state/v1.0/zstd/checkpoint.json"
REBUILD_STATE="/work/datasets/jump_lite/zstd_rebuild_state/v1.0/checkpoint.json"
EXPECTED=655101

latest="$LOG_PARENT/latest"
if [[ ! -L "$latest" && ! -d "$latest" ]]; then
  echo "No Zstd upload run found under $LOG_PARENT"
  exit 0
fi
run_root=$(readlink -f "$latest")
echo "Run: $run_root"

pid=""
if [[ -f "$run_root/pid" ]]; then
  pid=$(<"$run_root/pid")
fi
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  echo "Process: running (PID $pid)"
else
  echo "Process: not running${pid:+ (last PID $pid)}"
fi
if [[ -f "$run_root/COMPLETE" ]]; then
  echo "Run state: COMPLETE"
elif [[ -f "$run_root/FAILED" ]]; then
  echo "Run state: FAILED"
fi

if [[ -f "$STATE" ]]; then
  echo
  echo "Upload checkpoint:"
  jq -r --argjson expected "$EXPECTED" '
    "  uploaded_sites: \(.next_index // 0) / \($expected)",
    "  percent: \((((.next_index // 0) * 10000 / $expected) | floor) / 100)%",
    "  uploaded_objects: \(.uploaded_objects // 0)",
    "  uploaded_bytes: \(.uploaded_bytes // 0)",
    "  last_site_key: \(.last_site_key // "none")",
    "  root_metadata_published: \(.root_metadata_published // false)",
    "  remote_verified_objects: \(.remote_verification.object_count // 0)",
    "  remote_verified_bytes: \(.remote_verification.total_bytes // 0)",
    "  remote_verified_at: \(.remote_verification.verified_at // "not yet")",
    "  updated_at: \(.updated_at // "unknown")",
    "  complete: \(.complete // false)"
  ' "$STATE"
else
  echo
  echo "Upload checkpoint: not written yet"
fi

if [[ -f "$REBUILD_STATE" ]]; then
  echo
  echo "Rebuild checkpoint:"
  jq -r --argjson expected "$EXPECTED" '
    "  safe_sites: \(.processed_manifest_sites // 0) / \($expected)",
    "  rebuild_complete: \(.complete // false)",
    "  updated_at: \(.updated_at // "unknown")"
  ' "$REBUILD_STATE"
fi

if [[ -f "$run_root/upload.log" ]]; then
  echo
  echo "Recent log lines:"
  tail -20 "$run_root/upload.log"
fi
