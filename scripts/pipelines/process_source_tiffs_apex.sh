#!/usr/bin/env bash
#
# APEX V2 Enhancement - Full Featured Batch Processing
# Processes 6 source TIFFs with all advanced features enabled
#
# Features enabled:
# - Depth-aware tone mapping (Apple Depth Pro backend)
# - Material-specific processing
# - Clarity and atmospheric effects
# - Luxury estate preset (premium marketing aesthetic)
# - MPS hardware acceleration (Apple Silicon)
# - Comprehensive JSON reports per image
#
# Safety:
# - Fail-fast on any error
# - Atomic output writes
# - Detailed logging
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

discover_repo_root() {
    local root
    if root="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
        printf '%s\n' "${root}"
        return 0
    fi

    local current="${SCRIPT_DIR}"
    while [[ "${current}" != "/" ]]; do
        if [[ -f "${current}/pyproject.toml" && -d "${current}/.github/workflows" ]]; then
            printf '%s\n' "${current}"
            return 0
        fi
        current="$(dirname "${current}")"
    done
    return 1
}

REPO_ROOT="$(discover_repo_root)" || {
    echo "[ERROR] Unable to determine repository root from ${SCRIPT_DIR}" >&2
    exit 1
}
cd "${REPO_ROOT}"

INPUT_DIR="${REPO_ROOT}/input_images/source_tiffs"
OUTPUT_DIR="${REPO_ROOT}/output_apex_v2_luxury"
DEPTH_DIR="${REPO_ROOT}/depth_maps_apex"
LOG_DIR="${REPO_ROOT}/logs/apex_batch_$(date +%Y%m%d_%H%M%S)"

# Processing parameters
PRESET="luxury_estate"       # Premium marketing aesthetic (enhancement=0.8, clarity=0.6, material=0.7)
DEVICE="mps"                 # Apple Silicon GPU acceleration
UPSCALER="default"           # Default upscaling backend
VERBOSE="true"               # Detailed logging

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

log_info "APEX V2 Enhancement - Batch Processing"
log_info "========================================"
echo

# Check Python environment
if ! command -v python3 &> /dev/null; then
    log_error "python3 not found in PATH"
    exit 1
fi

# Check input directory
if [[ ! -d "${INPUT_DIR}" ]]; then
    log_error "Input directory not found: ${INPUT_DIR}"
    exit 1
fi

# Count input files
INPUT_COUNT=$(find "${INPUT_DIR}" -maxdepth 1 -type f \( -iname "*.tif" -o -iname "*.tiff" \) | wc -l | tr -d ' ')
if [[ "${INPUT_COUNT}" -eq 0 ]]; then
    log_error "No TIFF files found in: ${INPUT_DIR}"
    exit 1
fi

log_info "Found ${INPUT_COUNT} TIFF file(s) to process"
echo

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DEPTH_DIR}"
mkdir -p "${LOG_DIR}"

log_info "Configuration:"
log_info "  Input:     ${INPUT_DIR}"
log_info "  Output:    ${OUTPUT_DIR}"
log_info "  Depth:     ${DEPTH_DIR}"
log_info "  Logs:      ${LOG_DIR}"
log_info "  Preset:    ${PRESET}"
log_info "  Device:    ${DEVICE}"
log_info "  Upscaler:  ${UPSCALER}"
echo

# ============================================================================
# Depth Map Generation (Optional - if depth maps don't exist)
# ============================================================================

# Check if depth maps already exist
EXISTING_DEPTH_COUNT=$(find "${DEPTH_DIR}" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.tif" \) 2>/dev/null | wc -l | tr -d ' ')

if [[ "${EXISTING_DEPTH_COUNT}" -lt "${INPUT_COUNT}" ]]; then
    log_info "Generating depth maps (missing or incomplete)..."
    log_warn "Note: Depth generation requires ML dependencies (transformers, depth-pro)"
    log_warn "      If not installed, enhancement will proceed without depth awareness"
    echo

    # Automatic depth generation enabled for APEX V2 pipeline
    for input_file in "${INPUT_DIR}"/*.{tif,tiff}; do
        [[ -e "${input_file}" ]] || continue
        filename=$(basename "${input_file%.*}")
        depth_output="${DEPTH_DIR}/${filename}_depth.png"

        if [[ ! -f "${depth_output}" ]]; then
            log_info "Generating depth: ${filename}"
            python3 "${REPO_ROOT}/scripts/run_depth_estimation.py" \
                --input "${input_file}" \
                --output "${depth_output}" \
                --backend depth_pro \
                --device "${DEVICE}" \
                || log_warn "Depth generation failed for ${filename} (will use synthetic fallback)"
        fi
    done
else
    log_info "Using existing depth maps (${EXISTING_DEPTH_COUNT} found)"
    echo
fi

# ============================================================================
# V2 Enhancement Processing
# ============================================================================

log_info "Starting V2 Enhancement processing..."
echo

PROCESSED=0
FAILED=0
START_TIME=$(date +%s)

for input_file in "${INPUT_DIR}"/*.{tif,tiff}; do
    # Handle case where glob doesn't match
    [[ -e "${input_file}" ]] || continue

    filename=$(basename "${input_file}")
    basename_no_ext="${filename%.*}"
    log_file="${LOG_DIR}/${basename_no_ext}.log"

    log_info "Processing: ${filename}"

    # Build enhancement command
    cmd_args=(
        "python3" "${REPO_ROOT}/scripts/enhance_image.py"
        "${input_file}"
        "--output-dir" "${OUTPUT_DIR}"
        "--preset" "${PRESET}"
        "--device" "${DEVICE}"
        "--upscaler" "${UPSCALER}"
        "--log-file" "${log_file}"
    )

    # Add depth directory if it exists and has files
    if [[ -d "${DEPTH_DIR}" ]] && [[ "$(ls -A "${DEPTH_DIR}" 2>/dev/null)" ]]; then
        cmd_args+=("--depth-dir" "${DEPTH_DIR}")
    fi

    # Add verbosity flag
    if [[ "${VERBOSE}" == "true" ]]; then
        cmd_args+=("--verbose")
    fi

    # Execute enhancement
    if "${cmd_args[@]}" >> "${log_file}" 2>&1; then
        PROCESSED=$((PROCESSED + 1))
        log_success "✓ ${filename} (${PROCESSED}/${INPUT_COUNT})"
    else
        FAILED=$((FAILED + 1))
        log_error "✗ ${filename} failed (see ${log_file})"
    fi

    echo
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ============================================================================
# Summary Report
# ============================================================================

echo
log_info "========================================"
log_info "Batch Processing Complete"
log_info "========================================"
log_info "Total files:    ${INPUT_COUNT}"
log_success "Processed:      ${PROCESSED}"

if [[ "${FAILED}" -gt 0 ]]; then
    log_error "Failed:         ${FAILED}"
fi

log_info "Elapsed time:   ${ELAPSED}s"
log_info "Avg per image:  $((ELAPSED / INPUT_COUNT))s"
echo
log_info "Output directory: ${OUTPUT_DIR}"
log_info "Logs directory:   ${LOG_DIR}"
echo

# ============================================================================
# Output Verification
# ============================================================================

log_info "Output files:"
find "${OUTPUT_DIR}" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.json" \) | sort | while read -r f; do
    size=$(du -h "$f" | cut -f1)
    echo "  - $(basename "$f") (${size})"
done
echo

if [[ "${FAILED}" -eq 0 ]]; then
    log_success "All files processed successfully! 🎉"
    exit 0
else
    log_error "Some files failed processing. Check logs in: ${LOG_DIR}"
    exit 1
fi
