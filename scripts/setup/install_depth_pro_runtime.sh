#!/usr/bin/env bash
#
# install_depth_pro_runtime.sh
# Bootstrap a repo-local Depth Pro runtime for the subprocess-backed adapter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_RESOLVER="${REPO_ROOT}/scripts/setup/resolve_python_311.sh"

VENV_DIR="${REPO_ROOT}/.venv-depth-pro"
RUNTIME_METADATA_DIR="${REPO_ROOT}/.runtime"
RUNTIME_FREEZE_FILE="${RUNTIME_METADATA_DIR}/depth-pro-pip-freeze.txt"
CHECKPOINT_PATH="${REPO_ROOT}/checkpoints/depth_pro.pt"
DEPTH_PRO_REPO_URL="https://github.com/apple/ml-depth-pro.git"
DEFAULT_REF="9efe5c1def37a26c5367a71df664b18e1306c708"
REF="${DEPTH_PRO_RUNTIME_REF:-${DEFAULT_REF}}"
VERIFY_DEVICE="auto"
DRY_RUN=false
SKIP_VERIFY=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install a pinned repo-local Depth Pro runtime for:
  --depth-pro-python ./.venv-depth-pro/bin/python

Default paths:
  venv:       ${VENV_DIR}
  checkpoint: ${CHECKPOINT_PATH}

OPTIONS:
  --venv-dir PATH         Override the isolated Depth Pro venv path
  --checkpoint PATH       Override the checkpoint path used for readiness checks
  --ref REF               Apple ml-depth-pro git ref to install (default: ${DEFAULT_REF})
  --verify-device DEVICE  Device to validate after install: auto|cpu|mps|cuda
  --dry-run               Print commands without executing them
  --skip-verify           Skip the Depth Pro worker readiness check
  --help                  Show this help text
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

resolve_verify_device() {
    if [[ "${VERIFY_DEVICE}" != "auto" ]]; then
        printf '%s\n' "${VERIFY_DEVICE}"
        return 0
    fi

    case "$(uname -s)-$(uname -m)" in
        Darwin-arm64)
            printf 'mps\n'
            ;;
        *)
            printf 'cpu\n'
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv-dir)
            VENV_DIR="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --ref)
            REF="$2"
            shift 2
            ;;
        --verify-device)
            VERIFY_DEVICE="$2"
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

if ! command -v git >/dev/null 2>&1; then
    printf '[ERROR] git is required but not available on PATH.\n' >&2
    exit 1
fi

if [[ ! -x "${PYTHON_RESOLVER}" ]]; then
    printf '[ERROR] Python resolver missing: %s\n' "${PYTHON_RESOLVER}" >&2
    exit 1
fi

case "${VERIFY_DEVICE}" in
    auto|cpu|mps|cuda)
        ;;
    *)
        printf '[ERROR] Unsupported --verify-device value: %s\n' "${VERIFY_DEVICE}" >&2
        exit 1
        ;;
esac

BOOTSTRAP_PYTHON="$("${PYTHON_RESOLVER}")"

if [[ -d "${VENV_DIR}" ]]; then
    log "Refreshing isolated Depth Pro venv at ${VENV_DIR}"
else
    log "Creating isolated Depth Pro venv at ${VENV_DIR}"
fi
log "Using bootstrap interpreter: ${BOOTSTRAP_PYTHON}"
run "${BOOTSTRAP_PYTHON}" -m venv --clear "${VENV_DIR}"

PYTHON_BIN="${VENV_DIR}/bin/python"

log "Upgrading pip in ${VENV_DIR}"
run "${PYTHON_BIN}" -m pip install --upgrade pip

log "Installing pinned Depth Pro dependencies"
run "${PYTHON_BIN}" -m pip install \
    "numpy==1.26.4" \
    "torch==2.13.0" \
    "torchvision==0.28.0" \
    "matplotlib==3.10.8" \
    "pillow==12.3.0" \
    "pillow_heif==1.3.0" \
    "timm==1.0.26"

log "Installing Depth Pro from pinned git ref ${REF}"
run "${PYTHON_BIN}" -m pip install --force-reinstall --no-deps \
    "depth_pro @ git+${DEPTH_PRO_REPO_URL}@${REF}"

run mkdir -p "${RUNTIME_METADATA_DIR}"
if [[ "${DRY_RUN}" == "true" ]]; then
    log "Would capture Depth Pro runtime package snapshot at ${RUNTIME_FREEZE_FILE}"
else
    log "Capturing Depth Pro runtime package snapshot at ${RUNTIME_FREEZE_FILE}"
    "${PYTHON_BIN}" -m pip freeze > "${RUNTIME_FREEZE_FILE}"
fi

log "Running pip check in ${VENV_DIR}"
run "${PYTHON_BIN}" -m pip check

if [[ "${SKIP_VERIFY}" != "true" ]]; then
    if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
        printf '[ERROR] Depth Pro checkpoint not found for readiness check: %s\n' "${CHECKPOINT_PATH}" >&2
        printf '[ERROR] Download it first or rerun with --skip-verify.\n' >&2
        exit 1
    fi
    RESOLVED_VERIFY_DEVICE="$(resolve_verify_device)"
    log "Running Depth Pro worker readiness check on device=${RESOLVED_VERIFY_DEVICE}"
    run env \
        PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_BIN}" \
        -m transformation_portal.depth.backends.depth_pro_worker \
        --check \
        --checkpoint "${CHECKPOINT_PATH}" \
        --device "${RESOLVED_VERIFY_DEVICE}"
fi

cat <<EOF
[INFO] Depth Pro runtime ready.
[INFO] Stable executable: ./.venv-depth-pro/bin/python
[INFO] Pinned refs: torch==2.13.0 torchvision==0.28.0 numpy==1.26.4 depth_pro@${REF}
[INFO] Example:
lux-depth-v3 --input-dir ./input_images --output-dir ./output --depth-pro-python ./.venv-depth-pro/bin/python
EOF
