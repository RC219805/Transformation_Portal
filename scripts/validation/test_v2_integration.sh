#!/bin/bash
# Test V2 Enhancement Integration
#
# Tests the full lux-depth-v3 pipeline with V2 enhancement enabled
# using the hardened placeholder enhance_image.py script.
#
# This script validates:
# - V2 script invocation and execution
# - V2 output generation (enhanced images + JSON reports)
# - V2 timing metadata in manifests
# - Pipeline orchestration with V2 stage enabled
# - Error handling and graceful degradation
#
# Usage:
#   ./scripts/test_v2_integration.sh
#   ./scripts/test_v2_integration.sh --verbose
#   ./scripts/test_v2_integration.sh --clean   # Remove test outputs first
#
# Exit codes:
#   0 - All tests passed
#   1 - Test failed or validation error
#   2 - Setup/prerequisite error

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

TEST_NAME="V2 Integration Test"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INPUT_DIR="${PROJECT_ROOT}/input_images"
OUTPUT_DIR="${PROJECT_ROOT}/output/v2_integration_test"
LOG_FILE="${OUTPUT_DIR}/test.log"
V2_SCRIPT="${PROJECT_ROOT}/scripts/enhance_image.py"
VERBOSE=false
CLEAN=false

# Test image selection (mix of JPEG and TIFF)
declare -a TEST_IMAGES=(
    "${INPUT_DIR}/750_picacho/source_jpegs/750Picacho_Kitchen.jpg"
    "${INPUT_DIR}/750_picacho/source_jpegs/750Picacho_Pool.jpg"
    "${INPUT_DIR}/750Picacho_PrimaryBedroom_Ultimate.tif"
)

# Pipeline configuration
QUALITY="standard"
DEPTH_BACKEND="da3"
V2_PRESET="default"

# ============================================================================
# Color Output
# ============================================================================

if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    NC=''
fi

# ============================================================================
# Utility Functions
# ============================================================================

print_header() {
    echo -e "\n${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_section() {
    echo -e "\n${CYAN}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_verbose() {
    if [[ "${VERBOSE}" == "true" ]]; then
        echo -e "${CYAN}[VERBOSE]${NC} $1"
    fi
}

# ============================================================================
# Parse Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --clean|-c)
            CLEAN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --verbose, -v    Show detailed output"
            echo "  --clean, -c      Remove test outputs before running"
            echo "  --help, -h       Show this help message"
            echo ""
            echo "Test images:"
            for img in "${TEST_IMAGES[@]}"; do
                echo "  - $(basename "$img")"
            done
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 2
            ;;
    esac
done

# ============================================================================
# Prerequisite Checks
# ============================================================================

check_prerequisites() {
    print_section "Checking Prerequisites"

    local all_ok=true

    # Check lux-depth-v3 command
    if ! command -v lux-depth-v3 &> /dev/null; then
        print_error "lux-depth-v3 command not found"
        echo "  Install: pip install -e '${PROJECT_ROOT}[ml]'"
        all_ok=false
    else
        print_success "lux-depth-v3 command found"
        log_verbose "Location: $(which lux-depth-v3)"
    fi

    # Check V2 enhancement script
    if [[ ! -f "${V2_SCRIPT}" ]]; then
        print_error "V2 enhancement script not found: ${V2_SCRIPT}"
        all_ok=false
    else
        print_success "V2 enhancement script found"
        log_verbose "Location: ${V2_SCRIPT}"
    fi

    # Check test images
    local missing_images=()
    for img in "${TEST_IMAGES[@]}"; do
        if [[ ! -f "$img" ]]; then
            missing_images+=("$(basename "$img")")
        fi
    done

    if [[ ${#missing_images[@]} -gt 0 ]]; then
        print_error "Missing test images: ${missing_images[*]}"
        all_ok=false
    else
        print_success "All ${#TEST_IMAGES[@]} test images found"
    fi

    if [[ "${all_ok}" != "true" ]]; then
        echo ""
        print_error "Prerequisites check failed"
        return 1
    fi

    echo ""
    print_success "All prerequisites satisfied"
    return 0
}

# ============================================================================
# Setup Functions
# ============================================================================

setup_test_environment() {
    print_section "Setting Up Test Environment"

    # Clean previous outputs if requested
    if [[ "${CLEAN}" == "true" ]] && [[ -d "${OUTPUT_DIR}" ]]; then
        print_info "Cleaning previous test outputs..."
        rm -rf "${OUTPUT_DIR}"
        print_success "Previous outputs removed"
    fi

    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    print_success "Output directory: ${OUTPUT_DIR}"

    # Create log directory
    mkdir -p "$(dirname "${LOG_FILE}")"
    print_success "Log file: ${LOG_FILE}"

    echo ""
    print_info "Test Configuration:"
    echo "  Quality: ${QUALITY}"
    echo "  Depth Backend: ${DEPTH_BACKEND}"
    echo "  V2 Preset: ${V2_PRESET}"
    echo "  Test Images: ${#TEST_IMAGES[@]}"
}

# ============================================================================
# Test Execution
# ============================================================================

run_pipeline_test() {
    print_section "Running Pipeline Test"

    local start_time=$(date +%s)

    print_info "Executing lux-depth-v3 with V2 enhancement enabled..."
    echo ""

    # Create temporary input directory with test images in /tmp to avoid /output/ path exclusion
    local temp_input_dir="/tmp/lux_v2_test_inputs_$$"
    mkdir -p "${temp_input_dir}"

    print_info "Creating test input directory with ${#TEST_IMAGES[@]} images..."
    for img in "${TEST_IMAGES[@]}"; do
        cp "$img" "${temp_input_dir}/$(basename "$img")"
    done

    # Build command arguments
    local cmd_args=(
        "--input-dir" "${temp_input_dir}"
        "--quality-tier" "${QUALITY}"
        "--enable-v2" "on"
        "--v2-preset" "${V2_PRESET}"
        "--depth-backend" "${DEPTH_BACKEND}"
        "--output-dir" "${OUTPUT_DIR}"
        "--overwrite"
    )

    # Show command if verbose
    if [[ "${VERBOSE}" == "true" ]]; then
        echo -e "${CYAN}Command:${NC}"
        echo "  lux-depth-v3 ${cmd_args[*]}"
        echo ""
    fi

    # Execute pipeline
    local exit_code=0
    if [[ "${VERBOSE}" == "true" ]]; then
        lux-depth-v3 "${cmd_args[@]}" 2>&1 | tee "${LOG_FILE}" || exit_code=$?
    else
        lux-depth-v3 "${cmd_args[@]}" > "${LOG_FILE}" 2>&1 || exit_code=$?
    fi

    # Clean up temp directory
    rm -rf "${temp_input_dir}"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo ""
    if [[ ${exit_code} -eq 0 ]]; then
        print_success "Pipeline completed successfully in ${duration}s"
        return 0
    else
        print_error "Pipeline failed with exit code ${exit_code}"
        echo ""
        print_info "Last 20 lines of log:"
        tail -20 "${LOG_FILE}" | sed 's/^/  /'
        return 1
    fi
}

# ============================================================================
# Validation Functions
# ============================================================================

validate_v2_execution() {
    print_section "Validating V2 Script Execution"

    local all_ok=true

    # Check if V2 script was invoked
    if grep -q "Running V2 script" "${LOG_FILE}" 2>/dev/null || \
       grep -q "enhance_image.py" "${LOG_FILE}" 2>/dev/null; then
        print_success "V2 script was invoked"

        # Count invocations
        local invocations=$(grep -c "enhance_image.py" "${LOG_FILE}" 2>/dev/null || echo "0")
        log_verbose "V2 script invocations: ${invocations}"
    else
        print_error "V2 script was not invoked (check logs)"
        all_ok=false
    fi

    # Check for V2 errors
    if grep -qi "V2 failed\|V2 error\|enhance_image.py.*failed" "${LOG_FILE}" 2>/dev/null; then
        print_warning "V2 errors detected in logs"
        log_verbose "$(grep -i "V2 failed\|V2 error" "${LOG_FILE}" 2>/dev/null | head -5)"
    else
        print_success "No V2 errors detected"
    fi

    [[ "${all_ok}" == "true" ]] && return 0 || return 1
}

validate_v2_outputs() {
    print_section "Validating V2 Outputs"

    local all_ok=true
    local v2_dir="${OUTPUT_DIR}/v2"

    # Check V2 directory exists
    if [[ ! -d "${v2_dir}" ]]; then
        print_error "V2 output directory not found: ${v2_dir}"
        return 1
    fi
    print_success "V2 output directory exists"

    # Count V2 outputs
    local image_count=$(find "${v2_dir}" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.tif" -o -name "*.tiff" \) 2>/dev/null | wc -l | tr -d ' ')
    local report_count=$(find "${v2_dir}" -type f -name "*_report.json" 2>/dev/null | wc -l | tr -d ' ')

    if [[ ${image_count} -eq 0 ]]; then
        print_error "No V2 enhanced images found"
        all_ok=false
    else
        print_success "V2 enhanced images: ${image_count}"

        if [[ "${VERBOSE}" == "true" ]]; then
            find "${v2_dir}" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.tif" -o -name "*.tiff" \) 2>/dev/null | while read -r img; do
                local size=$(du -h "$img" | cut -f1)
                log_verbose "  $(basename "$img") - ${size}"
            done
        fi
    fi

    if [[ ${report_count} -eq 0 ]]; then
        print_warning "No V2 report files found"
    else
        print_success "V2 report files: ${report_count}"
    fi

    [[ "${all_ok}" == "true" ]] && return 0 || return 1
}

validate_v2_reports() {
    print_section "Validating V2 Reports"

    local all_ok=true
    local v2_dir="${OUTPUT_DIR}/v2"

    # Find all report files
    local reports=()
    while IFS= read -r -d '' report; do
        reports+=("$report")
    done < <(find "${v2_dir}" -type f -name "*_report.json" -print0 2>/dev/null)

    if [[ ${#reports[@]} -eq 0 ]]; then
        print_warning "No report files to validate"
        return 0
    fi

    local valid_count=0
    local invalid_count=0

    for report in "${reports[@]}"; do
        if python3 -m json.tool "$report" > /dev/null 2>&1; then
            ((valid_count++))
            log_verbose "Valid JSON: $(basename "$report")"
        else
            ((invalid_count++))
            print_error "Invalid JSON: $(basename "$report")"
            all_ok=false
        fi
    done

    if [[ ${invalid_count} -eq 0 ]]; then
        print_success "All ${valid_count} report files are valid JSON"
    else
        print_error "${invalid_count} report files have invalid JSON"
    fi

    [[ "${all_ok}" == "true" ]] && return 0 || return 1
}

validate_v2_timing() {
    print_section "Validating V2 Timing Metadata"

    local manifest_dir="${OUTPUT_DIR}/manifests"

    if [[ ! -d "${manifest_dir}" ]]; then
        print_warning "Manifests directory not found"
        return 0
    fi

    # Check for V2 timing in manifests
    local manifests_with_v2=0
    while IFS= read -r -d '' manifest; do
        if grep -q "v2_seconds\|v2_duration\|v2_time" "$manifest" 2>/dev/null; then
            ((manifests_with_v2++))
            log_verbose "V2 timing found in: $(basename "$manifest")"
        fi
    done < <(find "${manifest_dir}" -type f -name "*.json" -print0 2>/dev/null)

    if [[ ${manifests_with_v2} -eq 0 ]]; then
        print_warning "No V2 timing metadata found in manifests"
        return 0
    fi

    print_success "V2 timing metadata found in ${manifests_with_v2} manifest(s)"
    return 0
}

validate_output_structure() {
    print_section "Validating Output Structure"

    local expected_dirs=("depth" "pbr" "v2" "manifests")
    local all_ok=true

    for dir in "${expected_dirs[@]}"; do
        if [[ -d "${OUTPUT_DIR}/${dir}" ]]; then
            local count=$(find "${OUTPUT_DIR}/${dir}" -type f 2>/dev/null | wc -l | tr -d ' ')
            print_success "${dir}/ directory exists (${count} files)"
        else
            print_warning "${dir}/ directory not found"
            [[ "${dir}" == "v2" ]] && all_ok=false
        fi
    done

    [[ "${all_ok}" == "true" ]] && return 0 || return 1
}

# ============================================================================
# Summary Report
# ============================================================================

print_summary() {
    local test_passed=$1

    print_header "Test Summary"

    echo -e "${BOLD}Test:${NC} ${TEST_NAME}"
    echo -e "${BOLD}Output Directory:${NC} ${OUTPUT_DIR}"
    echo -e "${BOLD}Log File:${NC} ${LOG_FILE}"
    echo ""

    # Output statistics
    if [[ -d "${OUTPUT_DIR}" ]]; then
        echo -e "${BOLD}Output Statistics:${NC}"

        local v2_images=$(find "${OUTPUT_DIR}/v2" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.tif" -o -name "*.tiff" \) 2>/dev/null | wc -l | tr -d ' ')
        local v2_reports=$(find "${OUTPUT_DIR}/v2" -type f -name "*_report.json" 2>/dev/null | wc -l | tr -d ' ')
        local depth_maps=$(find "${OUTPUT_DIR}/depth" -type f 2>/dev/null | wc -l | tr -d ' ')
        local pbr_maps=$(find "${OUTPUT_DIR}/pbr" -type f 2>/dev/null | wc -l | tr -d ' ')

        echo "  V2 Enhanced Images: ${v2_images}"
        echo "  V2 Report Files:    ${v2_reports}"
        echo "  Depth Maps:         ${depth_maps}"
        echo "  PBR Maps:           ${pbr_maps}"
        echo ""
    fi

    # V2 timing from logs
    if [[ -f "${LOG_FILE}" ]]; then
        echo -e "${BOLD}V2 Stage Timing:${NC}"
        if grep -q "V2.*completed\|V2.*seconds\|V2.*duration" "${LOG_FILE}" 2>/dev/null; then
            grep -i "V2.*completed\|V2.*seconds\|V2.*duration" "${LOG_FILE}" 2>/dev/null | head -5 | sed 's/^/  /'
        else
            echo "  (No timing data found in logs)"
        fi
        echo ""
    fi

    # Final status
    if [[ ${test_passed} -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}${BOLD}  ✓ ALL TESTS PASSED${NC}"
        echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    else
        echo -e "${RED}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${RED}${BOLD}  ✗ TEST FAILED${NC}"
        echo -e "${RED}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        print_info "Review logs: ${LOG_FILE}"
    fi
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    print_header "${TEST_NAME}"

    echo "Testing full lux-depth-v3 pipeline with V2 enhancement enabled"
    echo "This validates the V2 integration using the hardened enhance_image.py script"
    echo ""

    # Prerequisites
    if ! check_prerequisites; then
        return 2
    fi

    # Setup
    setup_test_environment

    # Run pipeline
    if ! run_pipeline_test; then
        print_summary 1
        return 1
    fi

    # Validation checks
    local validation_failed=false

    validate_v2_execution || validation_failed=true
    validate_v2_outputs || validation_failed=true
    validate_v2_reports || validation_failed=true
    validate_v2_timing || validation_failed=true
    validate_output_structure || validation_failed=true

    # Summary
    if [[ "${validation_failed}" == "true" ]]; then
        print_summary 1
        return 1
    else
        print_summary 0
        return 0
    fi
}

# ============================================================================
# Entry Point
# ============================================================================

main "$@"
exit_code=$?

# Cleanup on exit
if [[ ${exit_code} -eq 0 ]] && [[ "${VERBOSE}" != "true" ]]; then
    echo ""
    print_info "For detailed logs, run with --verbose flag"
fi

exit ${exit_code}
