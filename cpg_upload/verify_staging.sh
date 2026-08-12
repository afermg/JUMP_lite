#!/usr/bin/env bash
# Compare local file count with the corresponding CPG staging object count.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUCKET="staging-cellpainting-gallery"
PREFIX_ROOT="cpg0016-jump/source_all"
REGION="us-east-1"

usage() {
  cat <<'EOF'
Usage:
  verify_staging.sh LOCAL_DIRECTORY DESTINATION_RELATIVE_PATH

The destination is interpreted beneath:
  s3://staging-cellpainting-gallery/cpg0016-jump/source_all/
EOF
}

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
if [[ -z ${AWS_SESSION_TOKEN:-} ]]; then
  echo "ERROR: temporary CPG credentials are not active" >&2
  exit 1
fi
command -v aws >/dev/null || { echo "ERROR: AWS CLI is not available" >&2; exit 1; }

python "$SCRIPT_DIR/validate_release.py"

PREFIX="$PREFIX_ROOT/${RELATIVE_DESTINATION%/}/"
echo "Counting local files under $SOURCE ..."
LOCAL_COUNT=$(find "$SOURCE" -type f -printf . | wc -c)
echo "Counting staging objects under s3://$BUCKET/$PREFIX ..."
# AWS CLI paginates the recursive listing; only object lines contain a date.
REMOTE_COUNT=$(aws s3 ls "s3://$BUCKET/$PREFIX" \
  --recursive --region "$REGION" | wc -l)

printf 'Local files:    %s\nStaging objects: %s\n' "$LOCAL_COUNT" "$REMOTE_COUNT"
if [[ "$LOCAL_COUNT" -ne "$REMOTE_COUNT" ]]; then
  echo "ERROR: object-count verification failed" >&2
  exit 1
fi

echo "Object-count verification passed."
