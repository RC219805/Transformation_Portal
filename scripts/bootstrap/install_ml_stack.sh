#!/usr/bin/env bash
# scripts/bootstrap/install_ml_stack.sh
#
# Profile-based ML stack installation script (ADR-032).
# Installs ML dependencies using the layered requirements system.
#
# PLATFORM MATRIX:
#   Axis       Values                    Detection
#   OS         Darwin / Linux            platform_system
#   ISA        arm64 / x86_64            platform_machine
#   Accel      cpu / mps / cuda          explicit profile (NEVER inferred)
#
# Requirements:
#   - Bash 4.3+ (uses nameref for array passing)
#   - Python 3.11+
#   - pip
#
# Usage:
#   ./scripts/bootstrap/install_ml_stack.sh --profile core-cpu
#   ./scripts/bootstrap/install_ml_stack.sh --profile core-mps
#   ./scripts/bootstrap/install_ml_stack.sh --profile core-cpu,raw
#   ./scripts/bootstrap/install_ml_stack.sh --profile core-mps,sam2
#   ./scripts/bootstrap/install_ml_stack.sh --profile full
#   ./scripts/bootstrap/install_ml_stack.sh --help
#
# Core Profiles (mutually exclusive):
#   core-cpu    Apple Silicon CPU baseline (darwin-arm64 CPU fallback)
#   core-mps    Apple Silicon MPS (darwin-arm64-mps)
#   core-cuda   Retired unsupported lane; fails closed
#
# Capability Layers (stack on core):
#   raw         RAW camera file ingest (rawpy)
#   sam2        SAM2 segmentation backend
#   coreml      Apple CoreML conversion (macOS only)
#   research    Research/experimental extras (reserved)
#
# Convenience:
#   full        Reserved until a trusted umbrella contract exists again
#
# Environment Variables:
#   PYTORCH_INDEX   Custom PyTorch index URL (default: https://download.pytorch.org/whl/cpu)
#   PIP_OPTS        Additional pip options (e.g., --no-cache-dir)
#
# Exit Codes:
#   0   Success
#   1   Invalid arguments or missing prerequisites
#   2   Installation failure

# --- Bash version auto-upgrade ---
# macOS ships with Bash 3.2 by default. This script requires Bash 4.3+
# for nameref support. On macOS, install newer bash with: brew install bash
# The script will auto-exec to the Homebrew-installed bash if available.
if [[ "${BASH_VERSINFO[0]}" -lt 4 ]] || { [[ "${BASH_VERSINFO[0]}" -eq 4 ]] && [[ "${BASH_VERSINFO[1]}" -lt 3 ]]; }; then
    # Try to find a newer bash
    if [[ -x /usr/local/bin/bash ]]; then
        exec /usr/local/bin/bash "$0" "$@"
    elif [[ -x /opt/homebrew/bin/bash ]]; then
        exec /opt/homebrew/bin/bash "$0" "$@"
    else
        echo "[ERROR] Bash 4.3+ is required (found ${BASH_VERSION})" >&2
        echo "[ERROR] On macOS, install with: brew install bash" >&2
        echo "[ERROR] Then run: /usr/local/bin/bash $0 $@" >&2
        exit 1
    fi
fi

set -euo pipefail

# Script directory for relative path resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REQUIREMENTS_DIR="${REPO_ROOT}/requirements"
PYTHON_RESOLVER="${REPO_ROOT}/scripts/setup/resolve_python_311.sh"
PYTHON_BIN=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
PROFILE=""
VERBOSE=false
DRY_RUN=false
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] --profile PROFILES

Install ML dependencies using the layered requirements system.

OPTIONS:
    --profile PROFILES    Comma-separated list of profiles to install
    --verbose             Enable verbose output (adds -v to pip commands)
    --dry-run             Show what would be installed without installing
    --help                Show this help message

PROFILES (Platform Matrix - ADR-032):
    Core profiles (mutually exclusive - choose one):
      core-cpu    Apple Silicon CPU baseline (darwin-arm64 CPU fallback)
      core-mps    Apple Silicon MPS (darwin-arm64-mps)
      core-cuda   Retired unsupported Linux CUDA lane (fails closed)

    Capability layers (stack on top of a supported core profile):
      raw         RAW camera file ingest (rawpy)
      sam2        SAM2 segmentation backend (requires core-*)
      coreml      Apple CoreML conversion (macOS only)
      research    Research/experimental extras (reserved)

    Convenience profiles:
      full        Reserved until a trusted umbrella contract exists again

PLATFORM TARGETS:
    darwin-arm64-cpu    macOS Apple Silicon, CPU-only (core-cpu)
    darwin-arm64-mps    macOS Apple Silicon, Metal (core-mps)
    darwin-x86_64-cpu   retired unsupported lane (fails closed)
    linux-x86_64-*      retired unsupported lane (fails closed)

ENVIRONMENT VARIABLES:
    PYTORCH_INDEX   Custom PyTorch index URL (default: https://download.pytorch.org/whl/cpu)
    PIP_OPTS        Additional pip options (e.g., --no-cache-dir)

EXAMPLES:
    # Install Apple Silicon CPU baseline
    $(basename "$0") --profile core-cpu

    # Install Apple Silicon MPS acceleration
    $(basename "$0") --profile core-mps

    # Retired Linux CUDA lane fails closed
    PYTORCH_INDEX=https://download.pytorch.org/whl/cu121 $(basename "$0") --profile core-cuda

    # Install SAM2 on top of a trusted core profile
    $(basename "$0") --profile core-mps,sam2

    # full/raw/coreml/research are disabled until trusted target-correct
    # checked-in contracts exist again
    $(basename "$0") --profile full

    # Dry run to see what a trusted profile would install
    $(basename "$0") --profile core-mps,sam2 --dry-run

EOF
    exit 0
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_verbose() {
    if [[ "${VERBOSE}" == "true" ]]; then
        echo -e "${GREEN}[VERBOSE]${NC} $*"
    fi
}

check_prerequisites() {
    # Note: Bash version check is handled by auto-exec at script start.
    # If we reach here, bash 4.3+ is guaranteed.

    if [[ ! -x "${PYTHON_RESOLVER}" ]]; then
        log_error "Python resolver missing: ${PYTHON_RESOLVER}"
        exit 1
    fi

    PYTHON_BIN="$("${PYTHON_RESOLVER}")"

    # Check pip is available
    if ! "${PYTHON_BIN}" -m pip --version &> /dev/null; then
        log_error "pip is required but not found."
        exit 1
    fi

    # Check requirements directory exists
    if [[ ! -d "${REQUIREMENTS_DIR}" ]]; then
        log_error "Requirements directory not found: ${REQUIREMENTS_DIR}"
        exit 1
    fi
}

check_lockfile() {
    local lockfile="$1"
    if [[ ! -f "${REQUIREMENTS_DIR}/${lockfile}" ]]; then
        log_error "Lockfile not found: ${REQUIREMENTS_DIR}/${lockfile}"
        log_error "Run 'make -C requirements compile-ml-darwin-arm64' on native Darwin arm64 to generate the target-owned ML lockfile."
        exit 1
    fi
}

# Build pip command as array for safe quoting
build_pip_cmd() {
    local -n cmd_array=$1
    cmd_array=("${PYTHON_BIN}" -m pip install)

    # Add verbose flag if enabled
    if [[ "${VERBOSE}" == "true" ]]; then
        cmd_array+=(-v)
    fi

    # Add any extra pip options safely using read -ra
    if [[ -n "${PIP_OPTS:-}" ]]; then
        local opts=()
        read -ra opts <<< "${PIP_OPTS}"
        cmd_array+=("${opts[@]}")
    fi
}

normalize_arch() {
    local arch="$1"
    case "${arch}" in
        aarch64) echo "arm64" ;;
        amd64) echo "x86_64" ;;
        *) echo "${arch}" ;;
    esac
}

python_platform_os() {
    "${PYTHON_BIN}" -c 'import platform; print(platform.system())'
}

python_platform_arch() {
    local arch
    arch="$("${PYTHON_BIN}" -c 'import platform; print(platform.machine())')"
    normalize_arch "${arch}"
}

# Detect current platform and return the supported platform-specific lockfile.
detect_platform_lockfile() {
    local os_type arch
    os_type="$(python_platform_os)"
    arch="$(python_platform_arch)"

    case "${os_type}" in
        Darwin)
            case "${arch}" in
                x86_64) echo "__unsupported_darwin_x86_64__" ;;
                arm64) echo "ml-core-darwin-arm64.txt" ;;
                *)
                    log_error "Unsupported Darwin architecture: ${arch}"
                    exit 1
                    ;;
            esac
            ;;
        Linux)
            echo "__unsupported_linux__"
            ;;
        *)
            log_error "Unsupported platform: ${os_type}"
            exit 1
            ;;
    esac
}

# Get platform identity string for CAS fingerprinting
get_platform_id() {
    local os_type arch accel
    os_type="$(python_platform_os | tr '[:upper:]' '[:lower:]')"
    arch="$(python_platform_arch)"
    accel="${1:-cpu}"  # Default to CPU if not specified

    echo "${os_type}-${arch}-${accel}"
}

install_profile() {
    local profile="$1"
    local pip_cmd=()
    build_pip_cmd pip_cmd

    # Get platform-specific lockfile for ml-core
    local platform_lockfile
    platform_lockfile="$(detect_platform_lockfile)"
    local platform_id

    # SECURITY BLOCK: macOS Intel (x86_64) is NOT SUPPORTED for ML workloads
    # CVE-2025-32434 cannot be remediated on this platform due to lack of
    # supported secure PyTorch wheels. This is a HARD FAIL, not a warning.
    if [[ "$(python_platform_os)" == "Darwin" ]] && [[ "$(python_platform_arch)" == "x86_64" ]]; then
        log_error "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_error "SECURITY BLOCK: macOS Intel (x86_64) ML Stack NOT SUPPORTED"
        log_error "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_error ""
        log_error "PyTorch does not provide the repo-supported torch>=2.13.0 baseline for macOS Intel."
        log_error "The historical macOS Intel PyTorch 2.2.x lockfile has been retired"
        log_error "from installable requirements because it is vulnerable to CVE-2025-32434."
        log_error ""
        log_error "While runtime hardening (weights_only=True) remains mandatory,"
        log_error "macOS Intel cannot receive a supported PyTorch version upgrade path."
        log_error ""
        log_error "REQUIRED ACTION:"
        log_error "  Migrate to macOS Apple Silicon (arm64)."
        log_error ""
        log_error "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        exit 1
    fi

    if [[ "$(python_platform_os)" == "Linux" ]]; then
        log_error "Linux ML lockfiles are retired unsupported manifests."
        log_error "No checked-in Linux ML stack may be installed from this bootstrap script."
        log_error "Use the supported macOS Apple Silicon lane, or add a new governed Linux lane in a separate change."
        exit 1
    fi

    case "${profile}" in
        core-cpu)
            # CPU baseline: platform-specific lockfile selection
            platform_id="$(get_platform_id cpu)"

            check_lockfile "${platform_lockfile}"
            log_info "Installing ML core layer + CPU baseline (${platform_id})..."
            log_info "Using platform-specific lockfile: ${platform_lockfile}"
            log_verbose "Using PyTorch index: ${PYTORCH_INDEX}"
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY-RUN] Would install: requirements/${platform_lockfile}"
                log_info "[DRY-RUN] With extra-index-url: ${PYTORCH_INDEX}"
            else
                "${pip_cmd[@]}" --extra-index-url "${PYTORCH_INDEX}" -r "${REQUIREMENTS_DIR}/${platform_lockfile}"
            fi
            ;;
        core-mps)
            # Apple Silicon MPS acceleration (darwin-arm64-mps)
            local python_os python_arch
            python_os="$(python_platform_os)"
            python_arch="$(python_platform_arch)"
            if [[ "${python_os}" != "Darwin" ]]; then
                log_error "core-mps profile requires macOS. Current platform: ${python_os}"
                exit 1
            fi
            if [[ "${python_arch}" != "arm64" ]]; then
                log_error "core-mps profile requires an arm64 Python interpreter. Current interpreter arch: ${python_arch}"
                exit 1
            fi
            platform_id="$(get_platform_id mps)"

            check_lockfile "${platform_lockfile}"
            log_info "Installing ML core layer + MPS acceleration (${platform_id})..."
            log_info "Using platform-specific lockfile: ${platform_lockfile}"
            log_verbose "Using PyTorch index: ${PYTORCH_INDEX}"
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY-RUN] Would install: requirements/${platform_lockfile}"
                log_info "[DRY-RUN] With extra-index-url: ${PYTORCH_INDEX}"
            else
                "${pip_cmd[@]}" --extra-index-url "${PYTORCH_INDEX}" -r "${REQUIREMENTS_DIR}/${platform_lockfile}"
            fi
            ;;
        core-cuda)
            log_error "core-cuda is retired with the unsupported Linux ML lock lane."
            log_error "Do not reinstall retired vulnerable torch baselines."
            exit 1
            ;;
        raw)
            log_error "raw profile no longer has a trusted checked-in lockfile contract."
            log_error "Generate a target-correct raw lockfile in the appropriate environment before using this profile."
            exit 1
            ;;
        sam2)
            # SAM2 is a SCRIPTED-ONLY capability - not a standard lockfile contract.
            # It requires non-standard install semantics on some platforms.
            log_info "Installing ML SAM2 segmentation layer (SCRIPTED-ONLY)..."
            log_warn "SAM2 requires ml-core dependencies. Ensure core-cpu/core-mps is installed first."

            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY-RUN] Would install sam2==1.1.0 with fallback to --no-build-isolation"
            else
                # Create secure temporary file for error logging
                local error_log
                local cleanup_trap
                error_log="$(mktemp)"
                cleanup_trap="$(printf 'rm -f %q' "${error_log}")"
                trap "${cleanup_trap}" RETURN

                # Try standard install first
                log_info "Attempting standard SAM2 install..."
                log_verbose "Using PyTorch index: ${PYTORCH_INDEX}"
                if "${pip_cmd[@]}" --extra-index-url "${PYTORCH_INDEX}" sam2==1.1.0 2>"${error_log}"; then
                    rm -f "${error_log}"
                    trap - RETURN
                    log_info "SAM2 installed successfully via standard path."
                else
                    log_warn "Standard install failed. Error log:"
                    cat "${error_log}" >&2
                    rm -f "${error_log}"
                    trap - RETURN
                    log_warn "Retrying with --no-build-isolation (torch must be pre-installed)..."
                    if "${pip_cmd[@]}" --extra-index-url "${PYTORCH_INDEX}" --no-build-isolation sam2==1.1.0; then
                        log_info "SAM2 installed successfully with --no-build-isolation."
                    else
                        log_error "SAM2 installation failed. Platform may not be supported."
                        log_error "Ensure PyTorch is installed first on Apple Silicon: ./scripts/bootstrap/install_ml_stack.sh --profile core-cpu"
                        exit 2
                    fi
                fi
            fi
            ;;
        coreml)
            # Check platform
            if [[ "$(uname -s)" != "Darwin" ]]; then
                log_warn "CoreML layer is only available on macOS. Skipping."
                return 0
            fi
            log_error "coreml profile no longer has a trusted checked-in lockfile contract."
            log_error "Generate a target-correct CoreML lockfile in the appropriate environment before using this profile."
            exit 1
            ;;
        research)
            log_error "research profile no longer has a trusted checked-in lockfile contract."
            log_error "Generate a target-correct research lockfile in the appropriate environment before using this profile."
            exit 1
            ;;
        full)
            log_error "full profile is disabled until a trusted umbrella lockfile contract exists again."
            log_error "Use target-specific core profiles explicitly instead."
            exit 1
            ;;
        *)
            log_error "Unknown profile: ${profile}"
            log_error "Valid profiles: core-cpu, core-mps, core-cuda, raw, sam2, coreml, research, full"
            exit 1
            ;;
    esac
}

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile)
                PROFILE="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
                usage
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done

    # Validate profile
    if [[ -z "${PROFILE}" ]]; then
        log_error "No profile specified. Use --profile PROFILES"
        usage
    fi

    # Check prerequisites
    check_prerequisites

    # Enable verbose mode if requested
    if [[ "${VERBOSE}" == "true" ]]; then
        log_info "Verbose mode enabled"
    fi

    log_verbose "Using bootstrap/runtime interpreter: ${PYTHON_BIN}"

    # Parse and install profiles
    IFS=',' read -ra PROFILES <<< "${PROFILE}"

    log_info "Installing ML stack with profiles: ${PROFILE}"
    log_verbose "PyTorch index URL: ${PYTORCH_INDEX}"
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY-RUN] No packages will be installed."
    fi

    for profile in "${PROFILES[@]}"; do
        # Trim whitespace using parameter expansion (safer than echo -e)
        profile="${profile//[[:space:]]/}"
        install_profile "${profile}"
    done

    log_info "ML stack installation complete!"
    if [[ "${DRY_RUN}" != "true" ]]; then
        log_info "Verify with: ${PYTHON_BIN} -c 'import torch; print(torch.__version__)'"
    fi
}

main "$@"
