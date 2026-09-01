#!/bin/bash
#
# Montecito Shores APEX Processing - Essential Features (Faster, Less Disk)
# Generated: 2026-02-10
# Input: 10 TIFF files @ 300 DPI
# Est. Runtime: ~1.5 minutes
#

set -euo pipefail

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
    echo "❌ Error: Unable to determine repository root from ${SCRIPT_DIR}" >&2
    exit 1
}
cd "${REPO_ROOT}"

INPUT_DIR="${REPO_ROOT}/input_images/Montecito-Shores_press_300dpi_TIFFs"
OUTPUT_DIR="${REPO_ROOT}/output_montecito_shores_apex_lean_$(date +%Y%m%d_%H%M%S)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Transformation Portal APEX - Montecito Shores (Lean)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Features: APEX Quality + Materials V3 + Standard Outputs"
echo "Est. Time: ~7s per image"
echo ""

python -m transformation_portal.lux_depth_v3 \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --quality-tier apex \
  --depth-device mps \
  --materials-v3 on \
  --cache-depth on \
  --verbose

echo ""
echo "✅ Complete: ${OUTPUT_DIR}"
