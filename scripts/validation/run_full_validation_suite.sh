#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_full_validation_suite.sh
#
# All-in-one validation orchestrator for local development.
# Runs the complete validation sequence with proper ordering and error handling.
#
# Usage:
#   ./scripts/validation/run_full_validation_suite.sh
#   ./scripts/validation/run_full_validation_suite.sh --quick
#   ./scripts/validation/run_full_validation_suite.sh --skip-browser
#
# Exit codes:
#   0       - All validations passed
#   non-zero - Validation or pre-flight failure; propagates the underlying
#              failing command's exit code
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_RESOLVER="${REPO_ROOT}/scripts/setup/resolve_python_311.sh"

if [[ ! -x "${PYTHON_RESOLVER}" ]]; then
    echo "[ERROR] Python resolver missing: ${PYTHON_RESOLVER}" >&2
    exit 1
fi

PYTHON_BIN="$("${PYTHON_RESOLVER}")"

# Options
QUICK_MODE=false
SKIP_BROWSER=false
SKIP_FRONTDOOR=false
VERBOSE=false

# Colors
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

log_step() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}" >&2
}

log_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

log_info() {
    echo "  $1"
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

All-in-one validation orchestrator for local development.

Options:
    --quick           Run quick validation (skip browser smokes)
    --skip-browser    Skip browser smoke tests
    --skip-frontdoor  Skip frontdoor validation (Node/npm)
    --verbose         Show verbose output
    -h, --help        Show this help message

Validation sequence:
    1. Environment pre-flight checks
    2. Fast Python tests (make test-fast)
    3. Orchestrator contract tests (make test-orchestrator-contract)
    4. Frontdoor contract tests (make test-frontdoor-contract)
    5. Portal browser smoke (make validate-portal-browser)
    6. Frontdoor browser smoke (make validate-frontdoor-browser)

Examples:
    $(basename "$0")                  # Full validation suite
    $(basename "$0") --quick          # Skip browser smokes
    $(basename "$0") --skip-frontdoor # Python-only validation
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK_MODE=true
            SKIP_BROWSER=true
            shift
            ;;
        --skip-browser)
            SKIP_BROWSER=true
            shift
            ;;
        --skip-frontdoor)
            SKIP_FRONTDOOR=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

cd "$REPO_ROOT"

# Track timing
START_TIME=$(date +%s)
STEP_TIMES=()

run_step() {
    local step_name="$1"
    shift
    local step_start=$(date +%s)

    log_step "$step_name"

    if "$@"; then
        local step_end=$(date +%s)
        local duration=$((step_end - step_start))
        log_success "$step_name completed (${duration}s)"
        STEP_TIMES+=("$step_name: ${duration}s")
        return 0
    else
        local exit_code=$?
        log_error "$step_name failed (exit code: $exit_code)"
        return $exit_code
    fi
}

# Header
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Transformation Portal — Full Validation Suite               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if [[ "$QUICK_MODE" == "true" ]]; then
    log_warning "Quick mode enabled — browser smokes will be skipped"
fi

if [[ "$SKIP_FRONTDOOR" == "true" ]]; then
    log_warning "Frontdoor validation skipped"
fi

# Step 1: Environment pre-flight
log_step "Step 1/6: Environment Pre-flight Checks"
if [[ "$SKIP_FRONTDOOR" == "true" ]]; then
    if "${PYTHON_BIN}" "${SCRIPT_DIR}/check_local_environment.py" --check python --check venv --check dependency-health; then
        log_success "Environment checks passed"
    else
        PREFLIGHT_STATUS=$?
        case "$PREFLIGHT_STATUS" in
            1)
                log_warning "Environment checks reported optional issues; continuing validation suite"
                ;;
            2)
                log_error "Environment checks failed"
                exit 2
                ;;
            *)
                log_error "Environment checks exited unexpectedly with status ${PREFLIGHT_STATUS}"
                exit "$PREFLIGHT_STATUS"
                ;;
        esac
    fi
else
    if "${PYTHON_BIN}" "${SCRIPT_DIR}/check_local_environment.py"; then
        log_success "Environment checks passed"
    else
        PREFLIGHT_STATUS=$?
        case "$PREFLIGHT_STATUS" in
            1)
                log_warning "Environment checks reported optional issues; continuing validation suite"
                ;;
            2)
                log_error "Environment checks failed"
                exit 2
                ;;
            *)
                log_error "Environment checks exited unexpectedly with status ${PREFLIGHT_STATUS}"
                exit "$PREFLIGHT_STATUS"
                ;;
        esac
    fi
fi

# Step 2: Fast Python tests
run_step "Step 2/6: Fast Python Tests" make test-fast

# Step 3: Orchestrator contract tests
run_step "Step 3/6: Orchestrator Contract Tests" make test-orchestrator-contract

# Step 4: Frontdoor contract tests
if [[ "$SKIP_FRONTDOOR" == "true" ]]; then
    log_step "Step 4/6: Frontdoor Contract Tests"
    log_warning "Skipped (--skip-frontdoor)"
else
    # Ensure Node version first
    if ! "${SCRIPT_DIR}/../setup/ensure_node_version.sh"; then
        log_error "Node version check failed — frontdoor tests cannot run"
        exit 1
    fi
    run_step "Step 4/6: Frontdoor Contract Tests" make test-frontdoor-contract
fi

# Step 5: Portal browser smoke
if [[ "$SKIP_BROWSER" == "true" ]]; then
    log_step "Step 5/6: Portal Browser Smoke"
    log_warning "Skipped (--skip-browser or --quick)"
else
    # Set default API key if not provided
    export TP_API_KEY="${TP_API_KEY:-contract-secret}"
    run_step "Step 5/6: Portal Browser Smoke" make validate-portal-browser
fi

# Step 6: Frontdoor browser smoke
if [[ "$SKIP_BROWSER" == "true" ]] || [[ "$SKIP_FRONTDOOR" == "true" ]]; then
    log_step "Step 6/6: Frontdoor Browser Smoke"
    log_warning "Skipped (--skip-browser, --quick, or --skip-frontdoor)"
else
    run_step "Step 6/6: Frontdoor Browser Smoke" make validate-frontdoor-browser
fi

# Summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Validation Summary                                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

for timing in "${STEP_TIMES[@]}"; do
    log_info "$timing"
done

echo ""
log_success "All validations passed (total: ${TOTAL_DURATION}s)"
echo ""

# Recommendations
if [[ "$QUICK_MODE" == "true" ]]; then
    log_info "Run without --quick to include browser smokes before merge"
fi
