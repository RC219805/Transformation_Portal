#!/usr/bin/env bash
#
# install_raw_runtime.sh
# Bootstrap a repo-local RAW ingest runtime for the subprocess-backed RAW adapter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_RESOLVER="${REPO_ROOT}/scripts/setup/resolve_python_311.sh"

VENV_DIR="${REPO_ROOT}/.venv-raw"
RUNTIME_METADATA_DIR="${REPO_ROOT}/.runtime"
RUNTIME_FREEZE_FILE="${RUNTIME_METADATA_DIR}/raw-pip-freeze.txt"
DRY_RUN=false
SKIP_VERIFY=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install a stable repo-local RAW ingest runtime for:
  --raw-python ./.venv-raw/bin/python

Default paths:
  venv: ${VENV_DIR}

OPTIONS:
  --venv-dir PATH       Override the isolated RAW venv path
  --dry-run             Print commands without executing them
  --skip-verify         Skip the RAW worker readiness check
  --help                Show this help text
EOF
}

log() {
    printf '[INFO] %s\n' "$*"
}

run() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        printf '+'
        for arg in "$@"; do
            printf ' %q' "${arg}"
        done
        printf '\n'
        return 0
    fi
    "$@"
}

run_in_repo() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        printf '+ (cd %q &&' "${REPO_ROOT}"
        for arg in "$@"; do
            printf ' %q' "${arg}"
        done
        printf ' )\n'
        return 0
    fi
    (
        cd "${REPO_ROOT}"
        "$@"
    )
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv-dir)
            VENV_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-verify)
            SKIP_VERIFY=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            printf '[ERROR] Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -x "${PYTHON_RESOLVER}" ]]; then
    printf '[ERROR] Python resolver missing: %s\n' "${PYTHON_RESOLVER}" >&2
    exit 1
fi

BOOTSTRAP_PYTHON="$("${PYTHON_RESOLVER}")"

if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating isolated RAW venv at ${VENV_DIR}"
    log "Using bootstrap interpreter: ${BOOTSTRAP_PYTHON}"
    run "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"

log "Upgrading pip in ${VENV_DIR}"
run "${PYTHON_BIN}" -m pip install --upgrade pip

log "Installing Transformation Portal with RAW support into ${VENV_DIR}"
run_in_repo "${PYTHON_BIN}" -m pip install -e ".[raw]"

run mkdir -p "${RUNTIME_METADATA_DIR}"
if [[ "${DRY_RUN}" == "true" ]]; then
    log "Would capture RAW runtime package snapshot at ${RUNTIME_FREEZE_FILE}"
else
    log "Capturing RAW runtime package snapshot at ${RUNTIME_FREEZE_FILE}"
    "${PYTHON_BIN}" -m pip freeze > "${RUNTIME_FREEZE_FILE}"
fi

if [[ "${SKIP_VERIFY}" != "true" ]]; then
    log "Running RAW worker readiness check"
    run env \
        PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_BIN}" \
        -m transformation_portal.spatial_ai.ingest.raw_worker \
        --check
fi

cat <<EOF
[INFO] RAW runtime ready.
[INFO] Stable executable: ./.venv-raw/bin/python
[INFO] Example:
lux-depth-v3 --input-dir ./input_images --output-dir ./output --raw-python ./.venv-raw/bin/python
EOF
