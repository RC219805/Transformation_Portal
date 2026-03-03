#!/usr/bin/env bash
set -euo pipefail

PREFIX="preserve/branch-cleanup-"
RETENTION_DAYS=30
REMOTE="origin"
APPLY=false
DELETE_LOCAL=true

usage() {
  cat <<'EOF'
Usage:
  prune_preservation_tags.sh [options]

Options:
  --prefix <value>        Tag prefix to prune (default: preserve/branch-cleanup-)
  --days <n>              Retention window in days (default: 30)
  --remote <name>         Remote used for remote tag deletion (default: origin)
  --apply                 Execute deletions (default: dry-run)
  --no-delete-local       Keep local tags; delete remote tags only when --apply is set
  --help                  Show this help text

Notes:
  - Candidate tags are selected by tag creator date (%(creatordate:unix)).
  - Without --apply, this script only prints candidates.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="${2:-}"
      shift 2
      ;;
    --days)
      RETENTION_DAYS="${2:-}"
      shift 2
      ;;
    --remote)
      REMOTE="${2:-}"
      shift 2
      ;;
    --apply)
      APPLY=true
      shift
      ;;
    --no-delete-local)
      DELETE_LOCAL=false
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "${RETENTION_DAYS}" =~ ^[0-9]+$ ]] || [[ "${RETENTION_DAYS}" -lt 1 ]]; then
  echo "Invalid --days value: ${RETENTION_DAYS}" >&2
  exit 2
fi

if [[ -z "${PREFIX}" ]]; then
  echo "--prefix cannot be empty" >&2
  exit 2
fi

now_epoch="$(date +%s)"
cutoff_epoch="$((now_epoch - RETENTION_DAYS * 86400))"

echo "prefix=${PREFIX}"
echo "retention_days=${RETENTION_DAYS}"
echo "cutoff_epoch=${cutoff_epoch}"
echo "mode=$([[ "${APPLY}" == "true" ]] && echo apply || echo dry-run)"

CANDIDATES=()
while IFS= read -r tag_name; do
  [[ -n "${tag_name}" ]] || continue
  CANDIDATES+=("${tag_name}")
done < <(
  git for-each-ref \
    --format='%(refname:short)|%(creatordate:unix)' \
    "refs/tags/${PREFIX}*" \
  | awk -F'|' -v cutoff="${cutoff_epoch}" '
      $2 ~ /^[0-9]+$/ && $2 <= cutoff { print $1 }
    '
)

if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
  echo "candidates=0"
  exit 0
fi

echo "candidates=${#CANDIDATES[@]}"
printf '%s\n' "${CANDIDATES[@]}"

if [[ "${APPLY}" != "true" ]]; then
  exit 0
fi

if [[ "${DELETE_LOCAL}" == "true" ]]; then
  git tag -d "${CANDIDATES[@]}" >/dev/null
  echo "deleted_local_tags=${#CANDIDATES[@]}"
fi

git push "${REMOTE}" --delete "${CANDIDATES[@]}"
echo "deleted_remote_tags=${#CANDIDATES[@]}"
