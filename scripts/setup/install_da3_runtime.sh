#!/usr/bin/env bash
#
# install_da3_runtime.sh
# Bootstrap a repo-local Depth Anything 3 runtime for the subprocess-backed DA3 adapter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CHECKOUT_DIR="${REPO_ROOT}/.runtime/Depth-Anything-3"
VENV_DIR="${REPO_ROOT}/.venv-da3"
REPO_URL="https://github.com/ByteDance-Seed/Depth-Anything-3"
REF=""
DRY_RUN=false
SKIP_VERIFY=false

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
  --ref REF             Optional git ref to checkout after clone/update
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

if ! command -v python3 >/dev/null 2>&1; then
    printf '[ERROR] python3 is required but not available on PATH.\n' >&2
    exit 1
fi

mkdir -p "$(dirname "${CHECKOUT_DIR}")"

if [[ ! -d "${CHECKOUT_DIR}/.git" ]]; then
    log "Cloning Depth Anything 3 into ${CHECKOUT_DIR}"
    run git clone "${REPO_URL}" "${CHECKOUT_DIR}"
else
    log "Using existing Depth Anything 3 checkout at ${CHECKOUT_DIR}"
fi

if [[ -n "${REF}" ]]; then
    log "Checking out ${REF}"
    run git -C "${CHECKOUT_DIR}" fetch --tags origin
    run git -C "${CHECKOUT_DIR}" checkout "${REF}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating isolated DA3 venv at ${VENV_DIR}"
    run python3 -m venv "${VENV_DIR}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"

log "Upgrading pip in ${VENV_DIR}"
run "${PYTHON_BIN}" -m pip install --upgrade pip

log "Installing DA3-compatible dependencies (without xformers)"
run "${PYTHON_BIN}" -m pip install \
    "torch>=2" \
    torchvision \
    transformers \
    cryptography \
    "moviepy==1.0.3" \
    einops \
    huggingface_hub \
    imageio \
    "numpy<2" \
    opencv-python \
    open3d \
    fastapi \
    uvicorn \
    requests \
    typer \
    pillow \
    omegaconf \
    evo \
    e3nn \
    plyfile \
    pillow_heif \
    safetensors \
    pycolmap \
    trimesh \
    addict \
    pre-commit

log "Installing Depth Anything 3 in editable mode without upstream xformers dependency"
run "${PYTHON_BIN}" -m pip install -e "${CHECKOUT_DIR}" --no-deps

if [[ "${SKIP_VERIFY}" != "true" ]]; then
    MPLCONFIGDIR="${REPO_ROOT}/.runtime/mplconfig"
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
