#!/usr/bin/env bash
# scripts/bootstrap/install_ml_stack.sh
#
# Profile-based ML stack installation script.
# Installs ML dependencies using the layered requirements system.
#
# Usage:
#   ./scripts/bootstrap/install_ml_stack.sh --profile core-cpu
#   ./scripts/bootstrap/install_ml_stack.sh --profile core-cpu,raw
#   ./scripts/bootstrap/install_ml_stack.sh --profile core-cpu,sam2
#   ./scripts/bootstrap/install_ml_stack.sh --profile full
#   ./scripts/bootstrap/install_ml_stack.sh --help
#
# Profiles:
#   core-cpu    Cross-platform ML baseline (torch, diffusers, transformers)
#   raw         RAW camera file ingest (rawpy)
#   sam2        SAM2 segmentation backend
#   coreml      Apple CoreML acceleration (macOS only)
#   research    Research/experimental extras (reserved)
#   full        All ML capabilities (equivalent to ml.txt umbrella)
#
# Environment Variables:
#   PYTORCH_INDEX   Custom PyTorch index URL (default: CPU index for core/sam2)
#   PIP_OPTS        Additional pip options (e.g., --no-cache-dir)
#
# Exit Codes:
#   0   Success
#   1   Invalid arguments or missing prerequisites
#   2   Installation failure

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

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] --profile PROFILES

Install ML dependencies using the layered requirements system.

OPTIONS:
    --profile PROFILES    Comma-separated list of profiles to install
    --verbose             Enable verbose output
    --dry-run             Show what would be installed without installing
    --help                Show this help message

PROFILES:
    core-cpu    Cross-platform ML baseline (torch, diffusers, transformers)
    raw         RAW camera file ingest (rawpy)
    sam2        SAM2 segmentation backend (requires core-cpu)
    coreml      Apple CoreML acceleration (macOS only)
    research    Research/experimental extras (reserved)
    full        All ML capabilities (equivalent to ml.txt umbrella)

EXAMPLES:
    # Install just the cross-platform ML baseline
    $(basename "$0") --profile core-cpu

    # Install ML baseline with RAW ingest capability
    $(basename "$0") --profile core-cpu,raw

    # Install full ML stack
    $(basename "$0") --profile full

    # Dry run to see what would be installed
    $(basename "$0") --profile core-cpu,sam2 --dry-run

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

check_prerequisites() {
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

install_profile() {
    local profile="$1"
    local pip_cmd="python3 -m pip install ${PIP_OPTS:-}"

    case "${profile}" in
        core-cpu)
            check_lockfile "ml-core.txt"
            log_info "Installing ML core layer (cross-platform baseline)..."
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY-RUN] Would install: requirements/ml-core.txt"
            else
                ${pip_cmd} -r "${REQUIREMENTS_DIR}/ml-core.txt"
            fi
            ;;
        raw)
            check_lockfile "ml-raw.txt"
            log_info "Installing ML RAW ingest layer..."
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY-RUN] Would install: requirements/ml-raw.txt"
            else
                ${pip_cmd} -r "${REQUIREMENTS_DIR}/ml-raw.txt"
            fi
            ;;
        sam2)
            check_lockfile "ml-sam2.txt"
            log_info "Installing ML SAM2 segmentation layer..."
            log_warn "SAM2 may require --no-build-isolation on some platforms."
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY-RUN] Would install: requirements/ml-sam2.txt"
            else
                if ! ${pip_cmd} -r "${REQUIREMENTS_DIR}/ml-sam2.txt" 2>/tmp/sam2_install_error.log; then
                    log_warn "Standard install failed with error:"
                    cat /tmp/sam2_install_error.log >&2
                    log_warn "Trying with --no-build-isolation..."
                    ${pip_cmd} --no-build-isolation -r "${REQUIREMENTS_DIR}/ml-sam2.txt"
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
                ${pip_cmd} -r "${REQUIREMENTS_DIR}/ml-coreml.txt"
            fi
            ;;
        research)
            check_lockfile "ml-research.txt"
            log_info "Installing ML research/experimental layer..."
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY-RUN] Would install: requirements/ml-research.txt"
            else
                ${pip_cmd} -r "${REQUIREMENTS_DIR}/ml-research.txt"
            fi
            ;;
        full)
            check_lockfile "ml.txt"
            log_info "Installing full ML stack (umbrella)..."
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_info "[DRY-RUN] Would install: requirements/ml.txt"
            else
                ${pip_cmd} -r "${REQUIREMENTS_DIR}/ml.txt"
            fi
            ;;
        *)
            log_error "Unknown profile: ${profile}"
            log_error "Valid profiles: core-cpu, raw, sam2, coreml, research, full"
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

    # Parse and install profiles
    IFS=',' read -ra PROFILES <<< "${PROFILE}"
    
    log_info "Installing ML stack with profiles: ${PROFILE}"
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
