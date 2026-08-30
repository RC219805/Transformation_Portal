#!/usr/bin/env bash
set -euo pipefail

# Python startup variables can execute caller-controlled code before any
# manifest, source, or lock verification. Scrub the entire namespace first.
while IFS= read -r fastvlm_env_name; do
  case "$fastvlm_env_name" in
    PYTHON*) unset "$fastvlm_env_name" ;;
  esac
done < <(compgen -e)
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1

ORIGINAL_ARGS=("$@")
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST_PATH="$REPO_ROOT/config/fastvlm_runtime_manifest.json"
RUNTIME_REQUIREMENTS_FILE="$REPO_ROOT/config/fastvlm_runtime_requirements.txt"
RUNTIME_ROOT="$REPO_ROOT/.runtime/fastvlm"
MODELS="smoke,default"
ALL_MODELS=0
DRY_RUN=0
VERIFY_ONLY=0
SKIP_MODEL_DOWNLOAD=0
SKIP_VERIFY=0
BASE_PYTHON="${TP_FASTVLM_BASE_PYTHON:-}"

usage() {
  cat <<'EOF'
Usage: scripts/setup/install_fastvlm_runtime.sh [options]

Options:
  --models ROLE[,ROLE...]     Model roles to install (default: smoke,default)
  --all-models                Install all manifest model roles
  --base-python PATH          Trusted interpreter used to build or audit the runtime venv
  --verify-only               Verify the existing runtime and exit
  --skip-model-download       Install runtime sources/venv but do not download models
  --skip-verify               Skip Python/model verification; source trust still runs last
  --dry-run                   Print planned actions without writing files or using network
  --help                      Show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --models)
      MODELS="${2:-}"
      shift 2
      ;;
    --models=*)
      MODELS="${1#*=}"
      shift
      ;;
    --all-models)
      ALL_MODELS=1
      shift
      ;;
    --base-python)
      BASE_PYTHON="${2:-}"
      if [ -z "$BASE_PYTHON" ]; then
        echo "--base-python requires a path" >&2
        exit 2
      fi
      shift 2
      ;;
    --base-python=*)
      BASE_PYTHON="${1#*=}"
      if [ -z "$BASE_PYTHON" ]; then
        echo "--base-python requires a path" >&2
        exit 2
      fi
      shift
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    --skip-model-download)
      SKIP_MODEL_DOWNLOAD=1
      shift
      ;;
    --skip-verify)
      SKIP_VERIFY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ "$ALL_MODELS" -eq 1 ]; then
  MODEL_ARGS=(--all-models)
else
  MODEL_ARGS=(--models "$MODELS")
fi

REPO_PY="$("$REPO_ROOT/scripts/setup/resolve_python_311.sh")"
RUNTIME_BASE_PY="${BASE_PYTHON:-$REPO_PY}"

SOURCE_INSTALLER="$REPO_ROOT/scripts/setup/install_fastvlm_sources.py"
VENV_INSTALLER="$REPO_ROOT/scripts/setup/install_fastvlm_venv.py"
LOCK_RUNNER="$REPO_ROOT/scripts/setup/run_fastvlm_install_locked.py"
LOCK_FILE="$REPO_ROOT/.runtime/.fastvlm-install.lock"
VENV_DIR="$RUNTIME_ROOT/.venv-fastvlm"
VENV_PY="$VENV_DIR/bin/python"

if [ "$DRY_RUN" -eq 0 ]; then
  if [ -z "${TP_FASTVLM_INSTALL_LOCK_FD:-}" ] && [ -z "${TP_FASTVLM_INSTALL_LOCK_TOKEN:-}" ]; then
    exec "$REPO_PY" -I -S "$LOCK_RUNNER" run \
      --lock-file "$LOCK_FILE" \
      -- "$0" "${ORIGINAL_ARGS[@]}"
  fi
  LOCK_FD="${TP_FASTVLM_INSTALL_LOCK_FD:-}"
  LOCK_TOKEN="${TP_FASTVLM_INSTALL_LOCK_TOKEN:-}"
  if [ -z "$LOCK_FD" ] || [ -z "$LOCK_TOKEN" ]; then
    echo "FastVLM installer inherited an incomplete transaction lock handoff" >&2
    exit 1
  fi
  case "$LOCK_FD" in
    ''|*[!0-9]*)
      echo "FastVLM installer inherited an invalid transaction lock descriptor" >&2
      exit 1
      ;;
  esac
  case "$LOCK_TOKEN" in
    ''|*[!0-9]*)
      echo "FastVLM installer inherited an invalid transaction lock token" >&2
      exit 1
      ;;
  esac
  "$REPO_PY" -I -S "$LOCK_RUNNER" assert-held \
    --lock-file "$LOCK_FILE" \
    --fd "$LOCK_FD" \
    --token "$LOCK_TOKEN"
  # Retain the inherited descriptor through the final exec so the transaction
  # remains serialized even if the parent lock-runner process is terminated.
  unset TP_FASTVLM_INSTALL_LOCK_FD TP_FASTVLM_INSTALL_LOCK_TOKEN
  if ! "$RUNTIME_BASE_PY" -I -S -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)'; then
    echo "FastVLM base Python must be Python 3.11 or newer: $RUNTIME_BASE_PY" >&2
    exit 1
  fi
fi

run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry-run]'
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    printf '\n'
  else
    "$@"
  fi
}

verify_runtime() {
  run "$REPO_PY" -I -S "$REPO_ROOT/scripts/validation/validate_fastvlm_runtime.py" \
    --manifest "$MANIFEST_PATH" \
    --runtime-root "$RUNTIME_ROOT" \
    --base-python "$RUNTIME_BASE_PY" \
    "${MODEL_ARGS[@]}" \
    --verify-only
}

install_sources() {
  local source_args=(
    --manifest "$MANIFEST_PATH"
    --runtime-root "$RUNTIME_ROOT"
  )
  if [ "$DRY_RUN" -eq 1 ]; then
    source_args+=(--dry-run)
  fi
  "$REPO_PY" -I -S "$SOURCE_INSTALLER" "${source_args[@]}"
}

verify_sources_last() {
  local source_args=(
    --manifest "$MANIFEST_PATH"
    --runtime-root "$RUNTIME_ROOT"
  )
  if [ "$DRY_RUN" -eq 1 ]; then
    source_args+=(--dry-run)
  else
    source_args+=(--verify-only)
  fi
  exec "$REPO_PY" -I -S "$SOURCE_INSTALLER" "${source_args[@]}"
}

audit_venv() {
  run "$REPO_PY" -I -S "$VENV_INSTALLER" \
    --manifest "$MANIFEST_PATH" \
    --runtime-root "$RUNTIME_ROOT" \
    --base-python "$RUNTIME_BASE_PY" \
    --audit-only
}

if [ "$VERIFY_ONLY" -eq 1 ]; then
  verify_runtime
  audit_venv
  verify_sources_last
fi

install_sources

run "$REPO_PY" -I -S "$VENV_INSTALLER" \
  --manifest "$MANIFEST_PATH" \
  --runtime-root "$RUNTIME_ROOT" \
  --base-python "$RUNTIME_BASE_PY" \
  --requirements "$RUNTIME_REQUIREMENTS_FILE"

if [ "$SKIP_MODEL_DOWNLOAD" -eq 0 ]; then
  run "$VENV_PY" -I "$REPO_ROOT/scripts/setup/download_fastvlm_models.py" \
    --manifest "$MANIFEST_PATH" \
    --runtime-root "$RUNTIME_ROOT" \
    "${MODEL_ARGS[@]}"
fi

if [ "$SKIP_VERIFY" -eq 0 ]; then
  verify_runtime
fi

audit_venv
verify_sources_last
