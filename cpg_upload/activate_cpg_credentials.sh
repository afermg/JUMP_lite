#!/usr/bin/env bash
# Source this file to exchange the long-lived CPG grant keys for temporary,
# prefix-scoped AWS credentials. Nothing is printed except status/expiration.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: source this script so credentials remain in your current shell:" >&2
  echo "  source cpg_upload/activate_cpg_credentials.sh" >&2
  exit 2
fi

_cpg_fail() {
  echo "ERROR: $*" >&2
  unset _cpg_response _cpg_key_id_file _cpg_secret_file
  return 1
}

_cpg_key_id_file=${CPG_KEY_ID_FILE:-"$HOME/.cpg_key_id"}
_cpg_secret_file=${CPG_SECRET_FILE:-"$HOME/.cpg_access_key"}

command -v aws >/dev/null || _cpg_fail "AWS CLI is not available" || return 1
command -v jq >/dev/null || _cpg_fail "jq is not available" || return 1
[[ -r "$_cpg_key_id_file" ]] || _cpg_fail "cannot read $_cpg_key_id_file" || return 1
[[ -r "$_cpg_secret_file" ]] || _cpg_fail "cannot read $_cpg_secret_file" || return 1

# Initial grant credentials supplied by the CPG maintainer.
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
AWS_ACCESS_KEY_ID=$(<"$_cpg_key_id_file")
AWS_SECRET_ACCESS_KEY=$(<"$_cpg_secret_file")
unset AWS_SESSION_TOKEN

_cpg_response=$(aws s3control get-data-access \
  --account-id 309624411020 \
  --target 's3://staging-cellpainting-gallery/cpg0016-jump/*' \
  --permission READWRITE \
  --duration-seconds 43200 \
  --region us-east-1) || {
    _cpg_fail "failed to obtain temporary CPG credentials"
    return 1
  }

export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
export AWS_SESSION_TOKEN
AWS_ACCESS_KEY_ID=$(jq -er '.Credentials.AccessKeyId // .AccessKeyId' <<<"$_cpg_response") || {
  _cpg_fail "temporary response did not contain AccessKeyId"
  return 1
}
AWS_SECRET_ACCESS_KEY=$(jq -er '.Credentials.SecretAccessKey // .SecretAccessKey' <<<"$_cpg_response") || {
  _cpg_fail "temporary response did not contain SecretAccessKey"
  return 1
}
AWS_SESSION_TOKEN=$(jq -er '.Credentials.SessionToken // .SessionToken' <<<"$_cpg_response") || {
  _cpg_fail "temporary response did not contain SessionToken"
  return 1
}
_cpg_expiration=$(jq -r '.Credentials.Expiration // .Expiration // "unknown"' <<<"$_cpg_response")

unset _cpg_response _cpg_key_id_file _cpg_secret_file
echo "Temporary READWRITE credentials active for cpg0016-jump (expires: $_cpg_expiration)"
unset _cpg_expiration
