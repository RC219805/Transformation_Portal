#!/usr/bin/env bash
#
# APEX V2 Enhancement - Individual Commands for Manual Execution
# Generated on: $(date)
#
# This file contains ready-to-run commands for processing each of the 6 source TIFFs
# with all advanced V2 features enabled.
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

# Use environment variables with repo-relative defaults
INPUT_DIR="${INPUT_DIR:-${REPO_ROOT}/input_images/source_tiffs}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output_apex_v2_luxury}"
DEPTH_DIR="${DEPTH_DIR:-${REPO_ROOT}/depth_maps_apex}"  # Optional: for depth-aware processing

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DEPTH_DIR}"

# ============================================================================
# Individual Commands (Copy-paste ready)
# ============================================================================

# Advanced Features Enabled:
# - Preset: luxury_estate (enhancement=0.8, clarity=0.6, material=0.7)
# - Device: mps (Apple Silicon GPU acceleration)
# - Depth-aware tone mapping (if depth maps available in DEPTH_DIR)
# - Material-specific processing (wood, metal, glass, textiles, leather)
# - Atmospheric effects (ambient occlusion, depth haze)
# - Comprehensive JSON report per image

echo "Command 1 of 6: V2_750Picacho_Aerial.tiff"
python3 "${REPO_ROOT}/scripts/enhance_image.py" \
    "${INPUT_DIR}/V2_750Picacho_Aerial.tiff" \
    --output-dir "${OUTPUT_DIR}" \
    --preset luxury_estate \
    --device mps \
    --depth-dir "${DEPTH_DIR}" \
    --verbose

echo ""
echo "Command 2 of 6: V2_750Picacho_GreatRoom.tiff"
python3 "${REPO_ROOT}/scripts/enhance_image.py" \
    "${INPUT_DIR}/V2_750Picacho_GreatRoom.tiff" \
    --output-dir "${OUTPUT_DIR}" \
    --preset luxury_estate \
    --device mps \
    --depth-dir "${DEPTH_DIR}" \
    --verbose

echo ""
echo "Command 3 of 6: V2_750Picacho_Kitchen.tiff"
python3 "${REPO_ROOT}/scripts/enhance_image.py" \
    "${INPUT_DIR}/V2_750Picacho_Kitchen.tiff" \
    --output-dir "${OUTPUT_DIR}" \
    --preset luxury_estate \
    --device mps \
    --depth-dir "${DEPTH_DIR}" \
    --verbose

echo ""
echo "Command 4 of 6: V2_750Picacho_Pool.tiff"
python3 "${REPO_ROOT}/scripts/enhance_image.py" \
    "${INPUT_DIR}/V2_750Picacho_Pool.tiff" \
    --output-dir "${OUTPUT_DIR}" \
    --preset luxury_estate \
    --device mps \
    --depth-dir "${DEPTH_DIR}" \
    --verbose

echo ""
echo "Command 5 of 6: V2_750Picacho_PrimaryBathroom.tiff"
python3 "${REPO_ROOT}/scripts/enhance_image.py" \
    "${INPUT_DIR}/V2_750Picacho_PrimaryBathroom.tiff" \
    --output-dir "${OUTPUT_DIR}" \
    --preset luxury_estate \
    --device mps \
    --depth-dir "${DEPTH_DIR}" \
    --verbose

echo ""
echo "Command 6 of 6: V2_750Picacho_PrimaryBedroom.tiff"
python3 "${REPO_ROOT}/scripts/enhance_image.py" \
    "${INPUT_DIR}/V2_750Picacho_PrimaryBedroom.tiff" \
    --output-dir "${OUTPUT_DIR}" \
    --preset luxury_estate \
    --device mps \
    --depth-dir "${DEPTH_DIR}" \
    --verbose

echo ""
echo "============================================"
echo "All 6 files processed!"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================"
