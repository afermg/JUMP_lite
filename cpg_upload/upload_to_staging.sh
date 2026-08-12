#!/usr/bin/env bash
# Validate the complete JUMP-Lite release, then sync one component to CPG staging.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGING_ROOT="s3://staging-cellpainting-gallery/cpg0016-jump/source_all"
REGION="us-east-1"
VALIDATION_REPORT="/work/datasets/jump_lite/cpg_release/validation_report.json"
APPLY=false

usage() {
  cat <<'EOF'
Usage:
  upload_to_staging.sh [--apply] LOCAL_PATH DESTINATION_RELATIVE_PATH

Without --apply, AWS CLI runs with --dryrun. With --apply, the command first
writes a successful validation report and then uploads. The destination is
always constrained beneath:

  s3://staging-cellpainting-gallery/cpg0016-jump/source_all/

Temporary CPG credentials, including AWS_SESSION_TOKEN, must already be active.
Use: source cpg_upload/activate_cpg_credentials.sh
EOF
}

if [[ ${1:-} == "--apply" ]]; then
  APPLY=true
  shift
fi
if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

SOURCE=$1
RELATIVE_DESTINATION=${2#/}

if [[ ! -d "$SOURCE" ]]; then
  echo "ERROR: local source must be a directory: $SOURCE" >&2
  exit 1
fi
if [[ -z "$RELATIVE_DESTINATION" || "$RELATIVE_DESTINATION" == *".."* ]]; then
  echo "ERROR: unsafe or empty destination: $RELATIVE_DESTINATION" >&2
  exit 1
fi
if [[ "$RELATIVE_DESTINATION" == /*
   || "$RELATIVE_DESTINATION" == */
   || "$RELATIVE_DESTINATION" == *"//"* ]]; then
  echo "ERROR: destination must not contain empty path components: $RELATIVE_DESTINATION" >&2
  exit 1
fi
if [[ ! "$RELATIVE_DESTINATION" =~ ^[A-Za-z0-9_./-]+$ ]]; then
  echo "ERROR: destination contains characters outside [A-Za-z0-9_./-]" >&2
  exit 1
fi
if ! command -v aws >/dev/null; then
  echo "ERROR: AWS CLI is not available" >&2
  exit 1
fi
if [[ -z ${AWS_ACCESS_KEY_ID:-} || -z ${AWS_SECRET_ACCESS_KEY:-} || -z ${AWS_SESSION_TOKEN:-} ]]; then
  echo "ERROR: temporary CPG credentials are not active (AWS_SESSION_TOKEN is required)" >&2
  exit 1
fi

REPORT_PATH="$VALIDATION_REPORT"
if ! $APPLY; then
  REPORT_PATH=$(mktemp)
  trap 'rm -f "$REPORT_PATH"' EXIT
fi
VALIDATION_ARGS=(--json-output "$REPORT_PATH")
SOURCE_CODEC=""
if [[ "$RELATIVE_DESTINATION" == images ]]; then
  echo "ERROR: refusing image-root sync; provide one exact codec destination" >&2
  exit 1
elif [[ "$RELATIVE_DESTINATION" == images/* ]]; then
  SOURCE_CODEC=$(basename -- "${SOURCE%/}")
  VALIDATION_ARGS+=(--image-root "$(dirname -- "$SOURCE")")
fi
python "$SCRIPT_DIR/validate_release.py" "${VALIDATION_ARGS[@]}"

# This generic wrapper uses aws s3 sync and therefore cannot filter individual
# site directories. Bind image uploads to the exact tree validated above and
# require that tree to contain no non-release sites.
if [[ -n "$SOURCE_CODEC" ]]; then
  python - "$REPORT_PATH" "$SOURCE_CODEC" "$SOURCE" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
codec = sys.argv[2]
source = Path(sys.argv[3]).resolve()
validated_source = (Path(report["image_root"]) / codec).resolve()
if source != validated_source:
    raise SystemExit(
        f"ERROR: upload source {source} differs from validated tree {validated_source}"
    )
details = report.get("images", {}).get(codec)
if details is None:
    raise SystemExit(
        "ERROR: refusing image sync: source must be one validated codec tree"
    )
extra = int(details.get("extra_sites", 0))
if extra:
    raise SystemExit(
        f"ERROR: refusing unfiltered {codec} sync: {extra:,} non-release sites"
    )
PY
fi

DESTINATION="$STAGING_ROOT/$RELATIVE_DESTINATION/"
ARGS=(
  s3 sync "$SOURCE" "$DESTINATION"
  --region "$REGION"
  --no-follow-symlinks
  --only-show-errors
)
if ! $APPLY; then
  ARGS+=(--dryrun)
fi

printf 'Validated source: %s\nDestination: %s\nMode: %s\n' \
  "$SOURCE" "$DESTINATION" "$($APPLY && echo APPLY || echo DRY-RUN)"
aws "${ARGS[@]}"
