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
  - Candidate tags are selected by annotated tagger date (%(taggerdate:unix)).
  - Non-annotated tags under the managed prefix are rejected (fail-fast).
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
NON_ANNOTATED=()
while IFS='|' read -r tag_name object_type tagger_unix; do
  [[ -n "${tag_name}" ]] || continue

  if [[ "${object_type}" != "tag" ]]; then
    NON_ANNOTATED+=("${tag_name}")
    continue
  fi

  if ! [[ "${tagger_unix}" =~ ^[0-9]+$ ]]; then
    NON_ANNOTATED+=("${tag_name}")
    continue
  fi

  if [[ "${tagger_unix}" -le "${cutoff_epoch}" ]]; then
    CANDIDATES+=("${tag_name}")
  fi
done < <(
  git for-each-ref \
    --format='%(refname:short)|%(objecttype)|%(taggerdate:unix)' \
    "refs/tags/${PREFIX}*"
)

if [[ "${#NON_ANNOTATED[@]}" -gt 0 ]]; then
  echo "error=non_annotated_tags_detected"
  printf '%s\n' "${NON_ANNOTATED[@]}"
  echo "Refusing to prune prefix '${PREFIX}' because non-annotated tags were found." >&2
  echo "Convert these tags to annotated tags before pruning." >&2
  exit 3
fi

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
