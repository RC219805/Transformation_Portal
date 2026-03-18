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
#   ./scripts/bootstrap/install_ml_stack.sh --profile core-cuda
#   ./scripts/bootstrap/install_ml_stack.sh --profile core-cpu,raw
#   ./scripts/bootstrap/install_ml_stack.sh --profile core-mps,sam2
#   ./scripts/bootstrap/install_ml_stack.sh --profile full
#   ./scripts/bootstrap/install_ml_stack.sh --help
#
# Core Profiles (mutually exclusive):
#   core-cpu    CPU baseline (darwin-*/linux-*, CPU fallback)
#   core-mps    Apple Silicon MPS (darwin-arm64-mps)
#   core-cuda   NVIDIA CUDA (linux-x86_64-cuda)
#
# Capability Layers (stack on core):
#   raw         RAW camera file ingest (rawpy)
#   sam2        SAM2 segmentation backend
#   coreml      Apple CoreML conversion (macOS only)
#   research    Research/experimental extras (reserved)
#
# Convenience:
#   full        All ML capabilities (equivalent to ml.txt umbrella)
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
        echo "[ERROR] Then run: /usr/local/bin/bash $0 $*" >&2
        exit 1
    fi
fi

set -euo pipefail

# Script directory for relative path resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REQUIREMENTS_DIR="${REPO_ROOT}/requirements"

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
      core-cpu    CPU baseline (darwin-*/linux-*, CPU fallback)
      core-mps    Apple Silicon MPS (darwin-arm64-mps)
      core-cuda   NVIDIA CUDA (linux-x86_64-cuda)

    Capability layers (stack on top of core profile):
      raw         RAW camera file ingest (rawpy)
      sam2        SAM2 segmentation backend (requires core-*)
      coreml      Apple CoreML conversion (macOS only)
      research    Research/experimental extras (reserved)

    Convenience profiles:
      full        All ML capabilities (equivalent to ml.txt umbrella)

PLATFORM TARGETS:
    darwin-x86_64-cpu   macOS Intel (core-cpu)
    darwin-arm64-cpu    macOS Apple Silicon, CPU-only (core-cpu)
    darwin-arm64-mps    macOS Apple Silicon, Metal (core-mps)
    linux-x86_64-cpu    Linux Intel/AMD, CPU (core-cpu)
    linux-x86_64-cuda   Linux Intel/AMD, NVIDIA GPU (core-cuda)

ENVIRONMENT VARIABLES:
    PYTORCH_INDEX   Custom PyTorch index URL (default: https://download.pytorch.org/whl/cpu)
    PIP_OPTS        Additional pip options (e.g., --no-cache-dir)

EXAMPLES:
    # Install cross-platform CPU baseline
    $(basename "$0") --profile core-cpu

    # Install Apple Silicon MPS acceleration
    $(basename "$0") --profile core-mps

    # Install NVIDIA CUDA acceleration (Linux only)
    PYTORCH_INDEX=https://download.pytorch.org/whl/cu121 $(basename "$0") --profile core-cuda

    # Install ML baseline with RAW ingest capability
    $(basename "$0") --profile core-cpu,raw

    # Install full ML stack
    $(basename "$0") --profile full

    # Dry run to see what would be installed
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
    # Check bash version (need 4.3+ for nameref)
    # Note: This check is redundant due to auto-exec at script start,
    # but kept for clarity and explicit error messages.
    if [[ "${BASH_VERSINFO[0]}" -lt 4 ]] || { [[ "${BASH_VERSINFO[0]}" -eq 4 ]] && [[ "${BASH_VERSINFO[1]}" -lt 3 ]]; }; then
        log_error "Bash 4.3+ is required (found ${BASH_VERSION})"
        log_error "On macOS, install with: brew install bash"
        exit 1
    fi

    # Check Python is available
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not found."
        exit 1
    fi

    # Check pip is available
    if ! python3 -m pip --version &> /dev/null; then
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
        log_error "Run 'cd requirements && make compile' to generate lockfiles."
        exit 1
    fi
}

# Build pip command as array for safe quoting
build_pip_cmd() {
    local -n cmd_array=$1
    cmd_array=(python3 -m pip install)
    
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

# Detect current platform and return platform-specific lockfile
# This is critical for deterministic pip-compile resolution (Issue 1 fix)
detect_platform_lockfile() {
    local os_type
    os_type="$(uname -s)"
    
    case "${os_type}" in
        Darwin)
            echo "ml-core-darwin.txt"
            ;;
        Linux)
            echo "ml-core-linux.txt"
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
    os_type="$(uname -s | tr '[:upper:]' '[:lower:]')"
    arch="$(uname -m)"
    accel="${1:-cpu}"  # Default to CPU if not specified
    
    # Normalize architecture names
    case "${arch}" in
        aarch64) arch="arm64" ;;
        amd64) arch="x86_64" ;;
    esac
    
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
    
    case "${profile}" in
        core-cpu)
            # CPU baseline: platform-specific lockfile selection
            platform_id="$(get_platform_id cpu)"
            
            # Use platform-specific lockfile if available, fallback to generic
            if [[ -f "${REQUIREMENTS_DIR}/${platform_lockfile}" ]]; then
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
            else
                # Fallback to ml-cpu.txt
                check_lockfile "ml-cpu.txt"
                log_warn "Platform-specific lockfile not found: ${platform_lockfile}"
                log_info "Falling back to ml-cpu.txt..."
                log_info "Installing ML core layer + CPU baseline (${platform_id})..."
                log_verbose "Using PyTorch index: ${PYTORCH_INDEX}"
                if [[ "${DRY_RUN}" == "true" ]]; then
                    log_info "[DRY-RUN] Would install: requirements/ml-cpu.txt"
                    log_info "[DRY-RUN] With extra-index-url: ${PYTORCH_INDEX}"
                else
                    "${pip_cmd[@]}" --extra-index-url "${PYTORCH_INDEX}" -r "${REQUIREMENTS_DIR}/ml-cpu.txt"
                fi
            fi
            ;;
        core-mps)
            # Apple Silicon MPS acceleration (darwin-arm64-mps)
            if [[ "$(uname -s)" != "Darwin" ]]; then
                log_error "core-mps profile requires macOS. Current platform: $(uname -s)"
                exit 1
            fi
            if [[ "$(uname -m)" != "arm64" ]]; then
                log_error "core-mps profile requires Apple Silicon (arm64). Current arch: $(uname -m)"
                exit 1
            fi
            platform_id="$(get_platform_id mps)"
            
            # MPS always uses darwin lockfile
            if [[ -f "${REQUIREMENTS_DIR}/ml-core-darwin.txt" ]]; then
                check_lockfile "ml-core-darwin.txt"
                log_info "Installing ML core layer + MPS acceleration (${platform_id})..."
                log_info "Using platform-specific lockfile: ml-core-darwin.txt"
                log_verbose "Using PyTorch index: ${PYTORCH_INDEX}"
                if [[ "${DRY_RUN}" == "true" ]]; then
                    log_info "[DRY-RUN] Would install: requirements/ml-core-darwin.txt"
                    log_info "[DRY-RUN] With extra-index-url: ${PYTORCH_INDEX}"
                else
                    "${pip_cmd[@]}" --extra-index-url "${PYTORCH_INDEX}" -r "${REQUIREMENTS_DIR}/ml-core-darwin.txt"
                fi
            else
                # Fallback to ml-mps.txt
                check_lockfile "ml-mps.txt"
                log_warn "Platform-specific lockfile not found: ml-core-darwin.txt"
                log_info "Falling back to ml-mps.txt..."
                log_info "Installing ML core layer + MPS acceleration (${platform_id})..."
                log_verbose "Using PyTorch index: ${PYTORCH_INDEX}"
                if [[ "${DRY_RUN}" == "true" ]]; then
                    log_info "[DRY-RUN] Would install: requirements/ml-mps.txt"
                    log_info "[DRY-RUN] With extra-index-url: ${PYTORCH_INDEX}"
                else
                    "${pip_cmd[@]}" --extra-index-url "${PYTORCH_INDEX}" -r "${REQUIREMENTS_DIR}/ml-mps.txt"
                fi
            fi
            ;;
        core-cuda)
            # NVIDIA CUDA acceleration (linux-x86_64-cuda)
            if [[ "$(uname -s)" != "Linux" ]]; then
                log_error "core-cuda profile requires Linux. Current platform: $(uname -s)"
                exit 1
            fi
            platform_id="$(get_platform_id cuda)"
            
            # CUDA always uses linux lockfile + cuda packages
            if [[ -f "${REQUIREMENTS_DIR}/ml-core-linux.txt" ]]; then
                check_lockfile "ml-core-linux.txt"
                log_info "Installing ML core layer + CUDA acceleration (${platform_id})..."
                log_info "Using platform-specific lockfile: ml-core-linux.txt"
                log_info "Using PyTorch CUDA index: ${PYTORCH_INDEX}"
                log_warn "Ensure NVIDIA drivers (compatible with CUDA 12.x) are installed on the host system."
                if [[ "${DRY_RUN}" == "true" ]]; then
                    log_info "[DRY-RUN] Would install: requirements/ml-core-linux.txt"
                    log_info "[DRY-RUN] With extra-index-url: ${PYTORCH_INDEX}"
                else
                    "${pip_cmd[@]}" --extra-index-url "${PYTORCH_INDEX}" -r "${REQUIREMENTS_DIR}/ml-core-linux.txt"
                fi
            else
                # Fallback to ml-cuda.txt
                check_lockfile "ml-cuda.txt"
                log_warn "Platform-specific lockfile not found: ml-core-linux.txt"
                log_info "Falling back to ml-cuda.txt..."
                log_info "Installing ML core layer + CUDA acceleration (${platform_id})..."
                log_info "Using PyTorch CUDA index: ${PYTORCH_INDEX}"
                log_warn "Ensure NVIDIA drivers (compatible with CUDA 12.x) are installed on the host system."
                if [[ "${DRY_RUN}" == "true" ]]; then
                    log_info "[DRY-RUN] Would install: requirements/ml-cuda.txt"
                    log_info "[DRY-RUN] With extra-index-url: ${PYTORCH_INDEX}"
                else
                    "${pip_cmd[@]}" --extra-index-url "${PYTORCH_INDEX}" -r "${REQUIREMENTS_DIR}/ml-cuda.txt"
                fi
            fi
            ;;
        raw)
            check_lockfile "ml-raw.txt"
            log_info "Installing ML RAW ingest layer..."
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY-RUN] Would install: requirements/ml-raw.txt"
            else
                "${pip_cmd[@]}" -r "${REQUIREMENTS_DIR}/ml-raw.txt"
            fi
            ;;
        sam2)
            # SAM2 is a SCRIPTED-ONLY capability - not a standard lockfile contract.
            # It requires non-standard install semantics on some platforms.
            log_info "Installing ML SAM2 segmentation layer (SCRIPTED-ONLY)..."
            log_warn "SAM2 requires ml-core dependencies. Ensure core-cpu/core-mps/core-cuda is installed first."
            
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY-RUN] Would install sam2==1.1.0 with fallback to --no-build-isolation"
            else
                # Create secure temporary file for error logging
                local error_log
                error_log="$(mktemp)"
                trap 'rm -f "${error_log}"' RETURN
                
                # Try standard install first
                log_info "Attempting standard SAM2 install..."
                log_verbose "Using PyTorch index: ${PYTORCH_INDEX}"
                if "${pip_cmd[@]}" --extra-index-url "${PYTORCH_INDEX}" sam2==1.1.0 2>"${error_log}"; then
                    log_info "SAM2 installed successfully via standard path."
                else
                    log_warn "Standard install failed. Error log:"
                    cat "${error_log}" >&2
                    log_warn "Retrying with --no-build-isolation (torch must be pre-installed)..."
                    if "${pip_cmd[@]}" --extra-index-url "${PYTORCH_INDEX}" --no-build-isolation sam2==1.1.0; then
                        log_info "SAM2 installed successfully with --no-build-isolation."
                    else
                        log_error "SAM2 installation failed. Platform may not be supported."
                        log_error "Ensure PyTorch is installed first: ./scripts/bootstrap/install_ml_stack.sh --profile core-cpu"
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
            check_lockfile "ml-coreml.txt"
            log_info "Installing ML CoreML layer (macOS)..."
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY-RUN] Would install: requirements/ml-coreml.txt"
            else
                "${pip_cmd[@]}" -r "${REQUIREMENTS_DIR}/ml-coreml.txt"
            fi
            ;;
        research)
            check_lockfile "ml-research.txt"
            log_info "Installing ML research/experimental layer..."
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY-RUN] Would install: requirements/ml-research.txt"
            else
                "${pip_cmd[@]}" -r "${REQUIREMENTS_DIR}/ml-research.txt"
            fi
            ;;
        full)
            check_lockfile "ml.txt"
            log_info "Installing full ML stack (umbrella)..."
            log_verbose "Using PyTorch index: ${PYTORCH_INDEX}"
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY-RUN] Would install: requirements/ml.txt"
                log_info "[DRY-RUN] With extra-index-url: ${PYTORCH_INDEX}"
            else
                "${pip_cmd[@]}" --extra-index-url "${PYTORCH_INDEX}" -r "${REQUIREMENTS_DIR}/ml.txt"
            fi
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
        log_info "Verify with: python3 -c 'import torch; print(torch.__version__)'"
    fi
}

main "$@"
