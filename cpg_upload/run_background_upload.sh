#!/usr/bin/env bash
# Supervise the complete JUMP-Lite v1.0 upload with credential renewal/resume.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
RELEASE_VERSION="v1.0"
RELEASE_BATCH="2026_jump_lite_${RELEASE_VERSION}"
PUBLICATION_ID="2026_jump_lite"
IMAGE_ROOT="/work/datasets/jump_lite/images/compressed/compressed_test/jump_lite_updated"
METADATA_ROOT="/work/datasets/jump_lite/cpg_release/metadata"
RELEASE_README="/work/datasets/jump_lite/cpg_release/README.md"
STATE_ROOT=${CPG_UPLOAD_STATE_ROOT:-"/work/datasets/jump_lite/cpg_upload_state/$RELEASE_VERSION"}
LOG_PARENT="/work/datasets/jump_lite/cpg_upload_logs"
STAGING_ROOT="s3://staging-cellpainting-gallery/cpg0016-jump/source_all"
METADATA_DESTINATION="$STAGING_ROOT/workspace/publication_data/$PUBLICATION_ID/metadata/$RELEASE_VERSION"
IMAGE_DESTINATION_ROOT="$STAGING_ROOT/images/$RELEASE_BATCH/images_compressed"
REGION="us-east-1"
RUN_ID=${CPG_UPLOAD_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT="$LOG_PARENT/$RUN_ID"
VALIDATION_REPORT="$RUN_ROOT/validation_report.json"
AWS_CONFIG_FILE="$RUN_ROOT/aws-config"
PYTHON=${CPG_UPLOAD_PYTHON:-"$PROJECT_ROOT/.venv/bin/python"}

if [[ ${1:-} != "--apply" || $# -ne 1 ]]; then
  echo "Usage: run_background_upload.sh --apply" >&2
  echo "This command performs the complete JUMP-Lite v1.0 staging upload." >&2
  exit 2
fi
if ! command -v aws >/dev/null; then
  echo "ERROR: AWS CLI is unavailable; run this script through nix shell nixpkgs#awscli2" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python environment is unavailable: $PYTHON" >&2
  exit 1
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "ERROR: upload run directory already exists: $RUN_ROOT" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/components" "$RUN_ROOT/metadata-layout" "$STATE_ROOT"
printf '%s\n' "$$" >"$RUN_ROOT/supervisor.pid"
ln -sfn "$RUN_ROOT" "$LOG_PARENT/latest"

cat >"$AWS_CONFIG_FILE" <<'EOF'
[default]
region = us-east-1
s3 =
    max_concurrent_requests = 24
    max_queue_size = 10000
    multipart_threshold = 64MB
    multipart_chunksize = 16MB
EOF
export AWS_CONFIG_FILE

# Build a clean metadata view so README.md and metadata artifacts share one
# public version directory without mutating the frozen release directory.
cp -a "$METADATA_ROOT"/. "$RUN_ROOT/metadata-layout/"
cp -a "$RELEASE_README" "$RUN_ROOT/metadata-layout/README.md"

# One fail-closed validation gates every component in this supervised run.
"$PYTHON" "$SCRIPT_DIR/validate_release.py" --json-output "$VALIDATION_REPORT"

sync_with_renewal() {
  local label=$1
  local source=$2
  local destination=$3
  local attempt=0
  local status

  while true; do
    attempt=$((attempt + 1))
    printf '[%s] %s attempt=%d source=%s destination=%s\n' \
      "$(date -u +%FT%TZ)" "$label" "$attempt" "$source" "$destination"

    # Each aws process receives fresh 12-hour credentials. Stop after 11 hours
    # so a long sync can renew before expiry and resume safely.
    source "$SCRIPT_DIR/activate_cpg_credentials.sh"
    set +e
    timeout --signal=TERM --kill-after=120s 39600 \
      aws s3 sync "$source" "$destination/" \
        --region "$REGION" \
        --no-follow-symlinks \
        --only-show-errors
    status=$?
    set -e
    unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN

    if [[ $status -eq 0 ]]; then
      printf '[%s] %s complete\n' "$(date -u +%FT%TZ)" "$label"
      printf '%s\n' "$(date -u +%FT%TZ)" >"$RUN_ROOT/components/$label.complete"
      return 0
    fi
    printf '[%s] %s interrupted status=%d; renewing and resuming in 30 seconds\n' \
      "$(date -u +%FT%TZ)" "$label" "$status" >&2
    sleep 30
  done
}

start_sync() {
  local label=$1
  local source=$2
  local destination=$3
  (
    sync_with_renewal "$label" "$source" "$destination"
  ) >"$RUN_ROOT/components/$label.log" 2>&1 &
  COMPONENT_PIDS+=("$!")
  COMPONENT_NAMES+=("$label")
}

COMPONENT_PIDS=()
COMPONENT_NAMES=()

start_sync \
  metadata \
  "$RUN_ROOT/metadata-layout" \
  "$METADATA_DESTINATION"

for codec in jpegxl_lossy_mq jpegxl_lossy_hq jpegxl_lossy_d20; do
  start_sync \
    "images_$codec" \
    "$IMAGE_ROOT/$codec.zarr" \
    "$IMAGE_DESTINATION_ROOT/$codec.zarr"
done

(
  "$PYTHON" "$SCRIPT_DIR/upload_profiles_to_staging.py" \
    --apply \
    --validation-report "$VALIDATION_REPORT" \
    --checkpoint-root "$STATE_ROOT/profiles" \
    --workers 64
) >"$RUN_ROOT/components/profiles.log" 2>&1 &
COMPONENT_PIDS+=("$!")
COMPONENT_NAMES+=("profiles")

{
  for index in "${!COMPONENT_PIDS[@]}"; do
    printf '%s\t%s\n' "${COMPONENT_PIDS[$index]}" "${COMPONENT_NAMES[$index]}"
  done
} >"$RUN_ROOT/component-pids.tsv"

printf '[%s] upload supervisor started run_id=%s\n' "$(date -u +%FT%TZ)" "$RUN_ID"
printf 'Run directory: %s\n' "$RUN_ROOT"
printf 'Components:\n'
cat "$RUN_ROOT/component-pids.tsv"

failed=0
for index in "${!COMPONENT_PIDS[@]}"; do
  if ! wait "${COMPONENT_PIDS[$index]}"; then
    printf '[%s] component failed: %s\n' \
      "$(date -u +%FT%TZ)" "${COMPONENT_NAMES[$index]}" >&2
    failed=1
  fi
done

if [[ $failed -ne 0 ]]; then
  printf '%s\n' "$(date -u +%FT%TZ)" >"$RUN_ROOT/FAILED"
  exit 1
fi

"$PYTHON" "$SCRIPT_DIR/validate_release.py" --json-output "$RUN_ROOT/final_validation_report.json"
printf '%s\n' "$(date -u +%FT%TZ)" >"$RUN_ROOT/COMPLETE"
echo "All upload components completed; staging object-count verification is still required."
