#!/usr/bin/env bash
#
# install_da3_runtime.sh
# Bootstrap a repo-local Depth Anything 3 runtime for the subprocess-backed DA3 adapter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_RESOLVER="${REPO_ROOT}/scripts/setup/resolve_python_311.sh"
DA3_LOCK_PATH="${REPO_ROOT}/requirements/da3-runtime-darwin-arm64.txt"
DA3_LOCK_SHA256="2520dfc2c4b0c2b4a0f5405175dd7db1dce8f76a5cc31d451bc62f70238f7c2c"
PIP_VERSION="26.2.1"
SETUPTOOLS_VERSION="82.0.0"

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
DA3_RUNTIME_CONTRACT_OVERRIDE=false
DA3_NUMPY_SPEC_OVERRIDE=false
if [[ "${DA3_RUNTIME_CONTRACT+x}" == "x" ]]; then
    DA3_RUNTIME_CONTRACT_OVERRIDE=true
fi
if [[ "${DA3_NUMPY_SPEC+x}" == "x" ]]; then
    DA3_NUMPY_SPEC_OVERRIDE=true
fi
DA3_RUNTIME_CONTRACT="${DA3_RUNTIME_CONTRACT:-pr110-numpy2-optional-colmap-xformers}"
DA3_NUMPY_SPEC="${DA3_NUMPY_SPEC:-numpy>=2.0,<3}"
DA3_XFORMERS_SPEC="${DA3_XFORMERS_SPEC:-xformers}"
DA3_PROFILE="${DA3_PROFILE:-baseline}"
DA3_BOOTSTRAP_PYTHON="${DA3_BOOTSTRAP_PYTHON:-}"
VALIDATED_VENV_IDENTITY=""
DRY_RUN=false
SKIP_VERIFY=false
RUNTIME_METADATA_DIR="${REPO_ROOT}/.runtime"
RUNTIME_FREEZE_FILE="${RUNTIME_METADATA_DIR}/da3-pip-freeze.txt"
RUNTIME_AUTHORITY_MARKER_NAME=".tp-da3-runtime-authority.json"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install a stable repo-local Depth Anything 3 runtime for:
  --da3-python ./.runtime/Depth-Anything-3/.venv-da3/bin/python

Default paths:
  checkout: ${CHECKOUT_DIR}
  venv:     ${CHECKOUT_DIR}/.venv-da3

Inference installation remains cross-platform. Cache authority is limited to
--profile baseline on native Darwin arm64 with Python 3.11, the checked-in exact
DA3 lock, and the default source revision. Optional profiles and legacy
DA3_NUMPY_SPEC/DA3_RUNTIME_CONTRACT overrides remain inference-only.

OPTIONS:
  --checkout-dir PATH   Override the Depth Anything 3 checkout path
  --venv-dir PATH       Override the isolated DA3 venv path
  --profile PROFILE     Dependency profile: baseline, colmap, xformers, or comma-combined
  --bootstrap-python PATH
                        Bootstrap interpreter override (use Python 3.11 for cache authority)
  --ref REF             Commit SHA/tag/local ref to checkout after clone/update (default: ${DEFAULT_REF})
  --fetch-ref REF        Remote ref to fetch before checkout (default: ${DEFAULT_FETCH_REF})
  --dry-run             Print commands without executing them
  --skip-verify         Skip the DA3 worker readiness check
  --help                Show this help text

Environment:
  DA3_BOOTSTRAP_PYTHON  Bootstrap interpreter override; the CLI option wins
  DA3_XFORMERS_SPEC     Optional xformers pip spec. The default "xformers" is
                        intentionally unpinned and operator-managed because
                        compatible wheels vary by torch/platform.
EOF
}

log() {
    printf '[INFO] %s\n' "$*"
}

sha256_file() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        printf '[ERROR] shasum or sha256sum is required to verify the DA3 lock.\n' >&2
        return 1
    fi
}

write_runtime_authority_marker() {
    local enabled="$1"
    local source_revision="$2"
    local marker_path="${VENV_DIR}/${RUNTIME_AUTHORITY_MARKER_NAME}"
    local payload
    payload="{\"cache_authority_enabled\":${enabled},\"dependency_lock_sha256\":\"${DA3_LOCK_SHA256}\",\"platform_machine\":\"${HOST_MACHINE}\",\"platform_system\":\"${HOST_SYSTEM}\",\"profile\":\"${DA3_PROFILE}\",\"python_version\":\"${RUNTIME_MAJOR_MINOR}\",\"schema\":\"tp.da3.runtime-authority.v1\",\"source_revision\":\"${source_revision}\"}"
    if [[ "${DRY_RUN}" == "true" ]]; then
        log "Would write DA3 runtime cache-authority marker (${enabled}) at ${marker_path}"
        return 0
    fi
    printf '%s' "${payload}" > "${marker_path}"
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

resolve_da3_bootstrap_python() {
    local candidate

    if [[ -n "${DA3_BOOTSTRAP_PYTHON}" ]]; then
        candidate="${DA3_BOOTSTRAP_PYTHON}"
        if [[ "${candidate}" != */* ]]; then
            candidate="$(command -v "${candidate}" 2>/dev/null || true)"
        fi
        if [[ -z "${candidate}" || ! -x "${candidate}" ]] \
            || ! "${candidate}" -I -S -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' \
                >/dev/null 2>&1; then
            printf '[ERROR] DA3 bootstrap interpreter must be executable Python 3.11+: %s\n' \
                "${DA3_BOOTSTRAP_PYTHON}" >&2
            return 1
        fi
        printf '%s\n' "${candidate}"
        return 0
    fi

    candidate="$(command -v python3.11 2>/dev/null || true)"
    if [[ -n "${candidate}" ]] \
        && "${candidate}" -I -S -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' \
            >/dev/null 2>&1; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    "${PYTHON_RESOLVER}"
}

validate_existing_da3_venv_for_clear() {
    local canonical_venv lexical_venv runtime_executable

    if [[ "${VENV_DIR}" == "/" || -L "${VENV_DIR}" || ! -d "${VENV_DIR}" \
        || ! -f "${VENV_DIR}/pyvenv.cfg" || -L "${VENV_DIR}/pyvenv.cfg" \
        || ! -x "${VENV_DIR}/bin/python" ]]; then
        printf '[ERROR] Refusing to clear an unverified DA3 venv: %s\n' "${VENV_DIR}" >&2
        return 1
    fi
    lexical_venv="$("${BOOTSTRAP_PYTHON}" -I -S -c \
        'import os, sys; print(os.path.abspath(sys.argv[1]))' "${VENV_DIR}")"
    canonical_venv="$(cd "${VENV_DIR}" && pwd -P)"
    if [[ "${lexical_venv}" != "${canonical_venv}" ]]; then
        printf '[ERROR] Refusing to clear DA3 venv reached through a symlinked ancestor: %s\n' "${VENV_DIR}" >&2
        return 1
    fi
    # ``-S`` intentionally avoids executing attacker-controlled .pth/site hooks
    # from the old environment. On Python <=3.13 it also resets sys.prefix to
    # the base interpreter, so validate the no-site invocation path instead.
    runtime_executable="$("${VENV_DIR}/bin/python" -I -S -c \
        'import os, sys; print(os.path.abspath(sys.executable))')"
    if [[ "${runtime_executable}" != "${canonical_venv}/bin/python" ]]; then
        printf '[ERROR] Refusing to clear DA3 venv with mismatched interpreter path: expected %s, observed %s\n' \
            "${canonical_venv}/bin/python" "${runtime_executable}" >&2
        return 1
    fi
    VALIDATED_VENV_IDENTITY="$("${BOOTSTRAP_PYTHON}" -I -S -c \
        'import os, stat, sys; value = os.lstat(sys.argv[1]); print(f"{value.st_dev}:{value.st_ino}") if stat.S_ISDIR(value.st_mode) else sys.exit("invalid venv")' \
        "${VENV_DIR}")"
}

safe_clear_existing_da3_venv() {
    "${BOOTSTRAP_PYTHON}" -I -S -c '
import os
import shutil
import stat
import sys

target = os.path.abspath(sys.argv[1])
expected = tuple(int(value) for value in sys.argv[2].split(":"))
flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
root_fd = os.open(target, flags)
try:
    opened = os.fstat(root_fd)
    if not stat.S_ISDIR(opened.st_mode) or (opened.st_dev, opened.st_ino) != expected:
        raise SystemExit("DA3 venv changed before safe clear")
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise SystemExit("platform lacks symlink-safe directory removal")
    for name in os.listdir(root_fd):
        observed = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if stat.S_ISDIR(observed.st_mode):
            shutil.rmtree(name, dir_fd=root_fd)
        else:
            os.unlink(name, dir_fd=root_fd)
    current = os.lstat(target)
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != expected:
        raise SystemExit("DA3 venv changed during safe clear")
finally:
    os.close(root_fd)
' "${VENV_DIR}" "${VALIDATED_VENV_IDENTITY}"
}

create_da3_venv() {
    # Keep ambient Python environment state out of venv/ensurepip startup.
    # This is required on Python 3.11 runners where an inherited PYTHONPATH can
    # otherwise make ensurepip fail before the governed lock is installed.
    run env \
        -u PYTHONHOME \
        -u PYTHONPATH \
        -u VIRTUAL_ENV \
        -u __PYVENV_LAUNCHER__ \
        "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"
}

require_option_value() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2}" == --* ]]; then
        printf '[ERROR] %s requires a value.\n' "$1" >&2
        exit 2
    fi
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
            require_option_value "$@"
            CHECKOUT_DIR="$2"
            shift 2
            ;;
        --venv-dir)
            require_option_value "$@"
            VENV_DIR="$2"
            shift 2
            ;;
        --profile)
            require_option_value "$@"
            DA3_PROFILE="$2"
            shift 2
            ;;
        --bootstrap-python)
            require_option_value "$@"
            DA3_BOOTSTRAP_PYTHON="$2"
            shift 2
            ;;
        --ref)
            require_option_value "$@"
            REF="$2"
            shift 2
            ;;
        --fetch-ref)
            require_option_value "$@"
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

BOOTSTRAP_PYTHON="$(resolve_da3_bootstrap_python)"

HOST_SYSTEM="$(uname -s)"
HOST_MACHINE="$(uname -m)"
if [[ "${HOST_MACHINE}" == "aarch64" ]]; then
    HOST_MACHINE="arm64"
fi
BOOTSTRAP_VERSION="$("${BOOTSTRAP_PYTHON}" -V 2>&1)"
if [[ ! -f "${DA3_LOCK_PATH}" ]]; then
    printf '[ERROR] Governed DA3 runtime lock missing: %s\n' "${DA3_LOCK_PATH}" >&2
    exit 1
fi
OBSERVED_LOCK_SHA256="$(sha256_file "${DA3_LOCK_PATH}")"
if [[ "${OBSERVED_LOCK_SHA256}" != "${DA3_LOCK_SHA256}" ]]; then
    printf '[ERROR] Governed DA3 runtime lock digest mismatch: expected %s, observed %s.\n' \
        "${DA3_LOCK_SHA256}" "${OBSERVED_LOCK_SHA256}" >&2
    exit 1
fi

log "DA3 runtime authority: Darwin arm64 / Python 3.11 / baseline profile only"
log "DA3 runtime ref: ${REF}"
log "DA3 runtime fetch ref: ${FETCH_REF}"
log "DA3 dependency profile: ${DA3_PROFILE}"
log "DA3 exact runtime lock: ${DA3_LOCK_PATH} (${DA3_LOCK_SHA256})"
if [[ "${DA3_PROFILE}" != "baseline" ]]; then
    log "DA3 optional profile disables cache authority; inference remains available"
fi
if [[ "${DA3_RUNTIME_CONTRACT_OVERRIDE}" == "true" || "${DA3_NUMPY_SPEC_OVERRIDE}" == "true" ]]; then
    log "Deprecated DA3 runtime/NumPy compatibility override detected; cache authority is disabled"
fi

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

if [[ "${DA3_PROFILE}" == "baseline" && -d "${VENV_DIR}" ]]; then
    log "Recreating existing DA3 baseline venv to remove non-governed distributions: ${VENV_DIR}"
    if [[ "${DRY_RUN}" != "true" ]]; then
        validate_existing_da3_venv_for_clear
        safe_clear_existing_da3_venv
    fi
    log "Using bootstrap interpreter: ${BOOTSTRAP_PYTHON}"
    create_da3_venv
elif [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating isolated DA3 venv at ${VENV_DIR}"
    log "Using bootstrap interpreter: ${BOOTSTRAP_PYTHON}"
    create_da3_venv
fi

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ -x "${PYTHON_BIN}" ]]; then
    RUNTIME_VERSION="$("${PYTHON_BIN}" -V 2>&1)"
elif [[ "${DRY_RUN}" == "true" ]]; then
    RUNTIME_VERSION="${BOOTSTRAP_VERSION}"
else
    printf '[ERROR] DA3 runtime Python missing after venv creation: %s\n' "${PYTHON_BIN}" >&2
    exit 1
fi
RUNTIME_FULL_VERSION="${RUNTIME_VERSION#Python }"
RUNTIME_MAJOR_MINOR="${RUNTIME_FULL_VERSION%.*}"
RUNTIME_VERSION_FINAL=false
if [[ "${RUNTIME_FULL_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    RUNTIME_VERSION_FINAL=true
fi
if [[
    "${HOST_SYSTEM}" != "Darwin" \
    || "${HOST_MACHINE}" != "arm64" \
    || "${RUNTIME_MAJOR_MINOR}" != "3.11" \
    || "${RUNTIME_VERSION_FINAL}" != "true" \
]]; then
    log "DA3 host ${HOST_SYSTEM}/${HOST_MACHINE} with runtime Python ${RUNTIME_FULL_VERSION} is inference-only and cannot authorize cache reuse"
fi

# Invalidate any previous authority marker before mutating the environment.
write_runtime_authority_marker false "${DEFAULT_REF}"

log "Installing governed DA3 bootstrap tools"
run "${PYTHON_BIN}" -m pip install --upgrade "pip==${PIP_VERSION}" "setuptools==${SETUPTOOLS_VERSION}"

OPTIONAL_DEPS=()
if [[ "${DA3_NUMPY_SPEC_OVERRIDE}" == "true" ]]; then
    OPTIONAL_DEPS+=("${DA3_NUMPY_SPEC}")
fi
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

log "Installing DA3 baseline dependency closure from the checked-in exact lock"
run "${PYTHON_BIN}" -m pip install --requirement "${DA3_LOCK_PATH}"

log "Installing DA3 ${DA3_PROFILE} optional dependency additions"
if [[ ${#OPTIONAL_DEPS[@]} -gt 0 ]]; then
    run "${PYTHON_BIN}" -m pip install "${OPTIONAL_DEPS[@]}"
fi

log "Installing Depth Anything 3 in editable mode without upstream dependency expansion"
run "${PYTHON_BIN}" -m pip install -e "${CHECKOUT_DIR}" --no-deps
run "${PYTHON_BIN}" -m pip check

if [[ "${DRY_RUN}" == "true" ]]; then
    OBSERVED_SOURCE_REVISION="${REF}"
else
    OBSERVED_SOURCE_REVISION="$(git -C "${CHECKOUT_DIR}" rev-parse HEAD)"
fi
if [[
    "${DA3_PROFILE}" == "baseline" \
    && "${OBSERVED_SOURCE_REVISION}" == "${DEFAULT_REF}" \
    && "${HOST_SYSTEM}" == "Darwin" \
    && "${HOST_MACHINE}" == "arm64" \
    && "${RUNTIME_MAJOR_MINOR}" == "3.11" \
    && "${RUNTIME_VERSION_FINAL}" == "true" \
    && "${DA3_RUNTIME_CONTRACT_OVERRIDE}" == "false" \
    && "${DA3_NUMPY_SPEC_OVERRIDE}" == "false" \
]]; then
    write_runtime_authority_marker true "${DEFAULT_REF}"
    log "DA3 cache authority marker enabled for the governed baseline"
else
    write_runtime_authority_marker false "${OBSERVED_SOURCE_REVISION}"
    log "DA3 runtime is inference-capable but non-authorizing for cache reuse"
fi

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
        PYTHONNOUSERSITE=1 \
        PYTHONSAFEPATH=1 \
        PYTHONPATH="${REPO_ROOT}/src" \
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
