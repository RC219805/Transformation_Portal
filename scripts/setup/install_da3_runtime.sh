#!/usr/bin/env bash
#
# install_da3_runtime.sh
# Bootstrap a repo-local Depth Anything 3 runtime for the subprocess-backed DA3 adapter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_RESOLVER="${REPO_ROOT}/scripts/setup/resolve_python_311.sh"

CHECKOUT_DIR="${REPO_ROOT}/.runtime/Depth-Anything-3"
VENV_DIR="${REPO_ROOT}/.venv-da3"
REPO_URL="https://github.com/ByteDance-Seed/Depth-Anything-3"
DEFAULT_REF="41736238f5bced4debf3f2a12375d2466874866d"
REF="${DA3_RUNTIME_REF:-${DEFAULT_REF}}"
DRY_RUN=false
SKIP_VERIFY=false
RUNTIME_METADATA_DIR="${REPO_ROOT}/.runtime"
RUNTIME_FREEZE_FILE="${RUNTIME_METADATA_DIR}/da3-pip-freeze.txt"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install a stable repo-local Depth Anything 3 runtime for:
  --da3-python ./.venv-da3/bin/python

Default paths:
  checkout: ${CHECKOUT_DIR}
  venv:     ${VENV_DIR}

OPTIONS:
  --checkout-dir PATH   Override the Depth Anything 3 checkout path
  --venv-dir PATH       Override the isolated DA3 venv path
  --ref REF             Git ref to checkout after clone/update (default: ${DEFAULT_REF})
  --dry-run             Print commands without executing them
  --skip-verify         Skip the DA3 worker readiness check
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout-dir)
            CHECKOUT_DIR="$2"
            shift 2
            ;;
        --venv-dir)
            VENV_DIR="$2"
            shift 2
            ;;
        --ref)
            REF="$2"
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

BOOTSTRAP_PYTHON="$("${PYTHON_RESOLVER}")"

mkdir -p "$(dirname "${CHECKOUT_DIR}")"

if [[ ! -d "${CHECKOUT_DIR}/.git" ]]; then
    log "Cloning Depth Anything 3 into ${CHECKOUT_DIR}"
    run git clone "${REPO_URL}" "${CHECKOUT_DIR}"
else
    log "Using existing Depth Anything 3 checkout at ${CHECKOUT_DIR}"
fi

log "Synchronizing Depth Anything 3 checkout to ${REF}"
run git -C "${CHECKOUT_DIR}" fetch --tags origin
run git -C "${CHECKOUT_DIR}" checkout "${REF}"
run git -C "${CHECKOUT_DIR}" reset --hard "${REF}"
run git -C "${CHECKOUT_DIR}" clean -fd

if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating isolated DA3 venv at ${VENV_DIR}"
    log "Using bootstrap interpreter: ${BOOTSTRAP_PYTHON}"
    run "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"

log "Upgrading pip in ${VENV_DIR}"
run "${PYTHON_BIN}" -m pip install --upgrade pip

log "Installing pinned DA3-compatible dependencies (without xformers)"
run "${PYTHON_BIN}" -m pip install \
    "torch==2.11.0" \
    "torchvision==0.26.0" \
    "transformers==5.5.0" \
    "cryptography==46.0.6" \
    "moviepy==1.0.3" \
    "einops==0.8.2" \
    "huggingface_hub==1.9.0" \
    "imageio==2.37.3" \
    "numpy==1.26.4" \
    "opencv-python==4.11.0.86" \
    "open3d==0.19.0" \
    "fastapi==0.135.3" \
    "uvicorn==0.43.0" \
    "requests==2.33.1" \
    "typer==0.24.1" \
    "pillow==12.2.0" \
    "omegaconf==2.3.0" \
    "evo==1.34.3" \
    "e3nn==0.6.0" \
    "plyfile==1.1.3" \
    "pillow_heif==1.3.0" \
    "safetensors==0.7.0" \
    "pycolmap==4.0.2" \
    "trimesh==4.11.5" \
    "addict==2.4.0" \
    "pre-commit==4.5.1"

log "Installing Depth Anything 3 in editable mode without upstream xformers dependency"
run "${PYTHON_BIN}" -m pip install -e "${CHECKOUT_DIR}" --no-deps

run mkdir -p "${RUNTIME_METADATA_DIR}"
if [[ "${DRY_RUN}" == "true" ]]; then
    log "Would capture DA3 runtime package snapshot at ${RUNTIME_FREEZE_FILE}"
else
    log "Capturing DA3 runtime package snapshot at ${RUNTIME_FREEZE_FILE}"
    "${PYTHON_BIN}" -m pip freeze > "${RUNTIME_FREEZE_FILE}"
fi

if [[ "${SKIP_VERIFY}" != "true" ]]; then
    MPLCONFIGDIR="${RUNTIME_METADATA_DIR}/mplconfig"
    log "Running DA3 worker readiness check"
    run mkdir -p "${MPLCONFIGDIR}"
    run env \
        KMP_DUPLICATE_LIB_OK=TRUE \
        MPLCONFIGDIR="${MPLCONFIGDIR}" \
        PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_BIN}" \
        -m transformation_portal.depth.backends.da3_worker \
        --check \
        --model-variant METRIC_LARGE \
        --device cpu
fi

cat <<EOF
[INFO] DA3 runtime ready.
[INFO] Stable executable: ./.venv-da3/bin/python
[INFO] Example:
lux-depth-v3 --input-dir ./input --output-dir ./output --da3-python ./.venv-da3/bin/python
EOF
