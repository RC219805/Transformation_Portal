#!/usr/bin/env bash
#
# install_da3_runtime.sh
# Bootstrap a repo-local Depth Anything 3 runtime for the subprocess-backed DA3 adapter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_RESOLVER="${REPO_ROOT}/scripts/setup/resolve_python_311.sh"

CHECKOUT_DIR="${REPO_ROOT}/.runtime/Depth-Anything-3"
VENV_DIR=""
REPO_URL="https://github.com/ByteDance-Seed/Depth-Anything-3"
# Authoritative DA3 runtime contract pin. This revision carries the PR #110
# dependency shape: NumPy 2, optional pycolmap, and optional xformers.
DEFAULT_REF="95a2adea1a8180104bf51937409034bdec70a244"
REF="${DA3_RUNTIME_REF:-${DEFAULT_REF}}"
DEFAULT_FETCH_REF="refs/pull/110/head"
FETCH_REF="${DA3_RUNTIME_FETCH_REF:-${DEFAULT_FETCH_REF}}"
FETCH_LOCAL_REF="refs/remotes/da3-runtime/fetch-ref"
DA3_RUNTIME_CONTRACT="${DA3_RUNTIME_CONTRACT:-pr110-numpy2-optional-colmap-xformers}"
DA3_NUMPY_SPEC="${DA3_NUMPY_SPEC:-numpy>=2.0,<3}"
DA3_XFORMERS_SPEC="${DA3_XFORMERS_SPEC:-xformers}"
DA3_PROFILE="${DA3_PROFILE:-baseline}"
DRY_RUN=false
SKIP_VERIFY=false
RUNTIME_METADATA_DIR="${REPO_ROOT}/.runtime"
RUNTIME_FREEZE_FILE="${RUNTIME_METADATA_DIR}/da3-pip-freeze.txt"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install a stable repo-local Depth Anything 3 runtime for:
  --da3-python ./.runtime/Depth-Anything-3/.venv-da3/bin/python

Default paths:
  checkout: ${CHECKOUT_DIR}
  venv:     ${CHECKOUT_DIR}/.venv-da3

OPTIONS:
  --checkout-dir PATH   Override the Depth Anything 3 checkout path
  --venv-dir PATH       Override the isolated DA3 venv path
  --profile PROFILE     Dependency profile: baseline, colmap, xformers, or comma-combined
  --ref REF             Commit SHA/tag/local ref to checkout after clone/update (default: ${DEFAULT_REF})
  --fetch-ref REF        Remote ref to fetch before checkout (default: ${DEFAULT_FETCH_REF})
  --dry-run             Print commands without executing them
  --skip-verify         Skip the DA3 worker readiness check
  --help                Show this help text

Environment:
  DA3_XFORMERS_SPEC     Optional xformers pip spec. The default "xformers" is
                        intentionally unpinned and operator-managed because
                        compatible wheels vary by torch/platform.
EOF
}

log() {
    printf '[INFO] %s\n' "$*"
}

git_clean_checkout() {
    local checkout_dir="$1"
    local venv_dir="$2"
    local checkout_prefix="${checkout_dir%/}"
    local venv_path="${venv_dir%/}"
    local clean_args=("-fd")

    if [[ "${venv_path}" == "${checkout_prefix}/"* ]]; then
        local relative_venv="${venv_path#"${checkout_prefix}/"}"
        clean_args+=("-e" "${relative_venv}")
        log "Preserving DA3 venv during checkout clean: ${relative_venv}"
    fi

    run git -C "${checkout_dir}" clean "${clean_args[@]}"
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
        --profile)
            DA3_PROFILE="$2"
            shift 2
            ;;
        --ref)
            REF="$2"
            shift 2
            ;;
        --fetch-ref)
            FETCH_REF="$2"
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

if [[ -z "${VENV_DIR}" ]]; then
    VENV_DIR="${CHECKOUT_DIR}/.venv-da3"
fi

PROFILE_TOKENS=()
case "${DA3_PROFILE}" in
    baseline|"")
        DA3_PROFILE="baseline"
        ;;
    colmap|xformers|colmap,xformers|xformers,colmap)
        IFS=',' read -r -a PROFILE_TOKENS <<< "${DA3_PROFILE}"
        ;;
    *)
        printf '[ERROR] Unsupported DA3 profile: %s\n' "${DA3_PROFILE}" >&2
        printf '[ERROR] Expected baseline, colmap, xformers, colmap,xformers, or xformers,colmap.\n' >&2
        exit 1
        ;;
esac

DA3_INSTALL_PYCOLMAP=0
DA3_INSTALL_XFORMERS=0
if [[ "${DA3_PROFILE}" != "baseline" ]]; then
    for profile_token in "${PROFILE_TOKENS[@]}"; do
        case "${profile_token}" in
            colmap)
                DA3_INSTALL_PYCOLMAP=1
                ;;
            xformers)
                DA3_INSTALL_XFORMERS=1
                ;;
        esac
    done
fi

if ! command -v git >/dev/null 2>&1; then
    printf '[ERROR] git is required but not available on PATH.\n' >&2
    exit 1
fi

if [[ ! -x "${PYTHON_RESOLVER}" ]]; then
    printf '[ERROR] Python resolver missing: %s\n' "${PYTHON_RESOLVER}" >&2
    exit 1
fi

BOOTSTRAP_PYTHON="$("${PYTHON_RESOLVER}")"

log "DA3 runtime contract: ${DA3_RUNTIME_CONTRACT}"
log "DA3 runtime ref: ${REF}"
log "DA3 runtime fetch ref: ${FETCH_REF}"
log "DA3 dependency profile: ${DA3_PROFILE}"
log "DA3 NumPy spec: ${DA3_NUMPY_SPEC}"

mkdir -p "$(dirname "${CHECKOUT_DIR}")"

if [[ ! -d "${CHECKOUT_DIR}/.git" ]]; then
    log "Cloning Depth Anything 3 into ${CHECKOUT_DIR}"
    run git clone "${REPO_URL}" "${CHECKOUT_DIR}"
else
    log "Using existing Depth Anything 3 checkout at ${CHECKOUT_DIR}"
fi

log "Synchronizing Depth Anything 3 checkout to ${REF}"
run git -C "${CHECKOUT_DIR}" fetch --tags origin
CHECKOUT_REF="${REF}"
if [[ -n "${FETCH_REF}" ]]; then
    run git -C "${CHECKOUT_DIR}" fetch origin "+${FETCH_REF}:${FETCH_LOCAL_REF}"
    if [[ "${REF}" == "${FETCH_REF}" ]]; then
        CHECKOUT_REF="${FETCH_LOCAL_REF}"
        log "Using fetched remote ref for checkout: ${FETCH_REF} -> ${FETCH_LOCAL_REF}"
    fi
fi
run git -C "${CHECKOUT_DIR}" checkout "${CHECKOUT_REF}"
run git -C "${CHECKOUT_DIR}" reset --hard "${CHECKOUT_REF}"
git_clean_checkout "${CHECKOUT_DIR}" "${VENV_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating isolated DA3 venv at ${VENV_DIR}"
    log "Using bootstrap interpreter: ${BOOTSTRAP_PYTHON}"
    run "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"

log "Upgrading pip in ${VENV_DIR}"
run "${PYTHON_BIN}" -m pip install --upgrade pip

BASE_DEPS=(
    "torch==2.12.0"
    "torchvision==0.27.0"
    "transformers==5.5.0"
    "cryptography==47.0.0"
    "moviepy==1.0.3"
    "einops==0.8.2"
    "huggingface_hub==1.9.0"
    "imageio==2.37.3"
    "${DA3_NUMPY_SPEC}"
    "opencv-python==4.11.0.86"
    "open3d==0.19.0"
    "fastapi==0.135.3"
    "uvicorn==0.43.0"
    "requests==2.33.1"
    "typer==0.24.1"
    "pillow==12.2.0"
    "omegaconf==2.3.0"
    "evo==1.34.3"
    "e3nn==0.6.0"
    "plyfile==1.1.3"
    "pillow_heif==1.3.0"
    "safetensors==0.7.0"
    "trimesh==4.11.5"
    "addict==2.4.0"
    "pre-commit==4.5.1"
)
OPTIONAL_DEPS=()
if [[ "${DA3_INSTALL_PYCOLMAP}" == "1" ]]; then
    OPTIONAL_DEPS+=("pycolmap==4.0.2")
fi
if [[ "${DA3_INSTALL_XFORMERS}" == "1" ]]; then
    OPTIONAL_DEPS+=("${DA3_XFORMERS_SPEC}")
    if [[ "${DA3_XFORMERS_SPEC}" == "xformers" ]]; then
        log "DA3 optional xformers spec: xformers (operator-managed; intentionally unpinned by default for platform wheel resolution)"
    else
        log "DA3 optional xformers spec: ${DA3_XFORMERS_SPEC} (operator-provided override)"
    fi
fi

log "Installing DA3 ${DA3_PROFILE} dependency profile (baseline stack pinned; optional profile specs shown above)"
if [[ ${#OPTIONAL_DEPS[@]} -gt 0 ]]; then
    run "${PYTHON_BIN}" -m pip install "${BASE_DEPS[@]}" "${OPTIONAL_DEPS[@]}"
else
    run "${PYTHON_BIN}" -m pip install "${BASE_DEPS[@]}"
fi

log "Installing Depth Anything 3 in editable mode without upstream dependency expansion"
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
[INFO] Stable executable: ./.runtime/Depth-Anything-3/.venv-da3/bin/python
[INFO] Example:
lux-depth-v3 --input-dir ./input_images --output-dir ./output --da3-python ./.runtime/Depth-Anything-3/.venv-da3/bin/python
EOF
