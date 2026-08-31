#!/bin/bash
#
# Montecito Shores APEX Processing - All Advanced Features Enabled
# Generated: 2026-02-10
# Input: 10 TIFF files @ 300 DPI (press-ready luxury real estate)
# Est. Runtime: ~2 minutes (with MPS acceleration)
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

# Configuration
INPUT_DIR="${REPO_ROOT}/input_images/Montecito-Shores_press_300dpi_TIFFs"
OUTPUT_DIR="${REPO_ROOT}/output_montecito_shores_apex_full_$(date +%Y%m%d_%H%M%S)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Transformation Portal APEX - Montecito Shores Batch"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Features Enabled:"
echo "  ✓ Quality Tier: APEX (maximum quality)"
echo "  ✓ Depth Backend: Depth Anything V3 (commercial, MPS-accelerated)"
echo "  ✓ Materials V3: Surface-aware finishing (glass, stone, water, foliage)"
echo "  ✓ PBR Maps: Normal, Roughness, Ambient Occlusion"
echo "  ✓ V2 Enhancement: AI-powered color grading"
echo "  ✓ Content-Addressable Cache: SHA256-based deduplication"
echo "  ✓ All Deliverables: Master16, Upscaled16, Marketing, Reports"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Pre-flight checks
if [ ! -d "${INPUT_DIR}" ]; then
    echo "❌ Error: Input directory not found: ${INPUT_DIR}"
    exit 1
fi

IMAGE_COUNT=$(find "${INPUT_DIR}" -name "*.tif" -o -name "*.tiff" | wc -l | tr -d ' ')
echo "📁 Found ${IMAGE_COUNT} TIFF images"
echo ""

# Confirm before processing
read -p "Proceed with APEX processing? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "🚀 Starting APEX pipeline..."
echo ""

# Run the pipeline
python -m transformation_portal.lux_depth_v3 \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  \
  --quality-tier apex \
  --depth-backend da3 \
  --depth-device mps \
  \
  --materials-v3 on \
  --pbr on \
  --cache-depth on \
  \
  --emit-master16 on \
  --emit-upscaled16 on \
  --emit-run-card on \
  \
  --enable-v2 on \
  --v2-preset default \
  \
  --max-workers 8 \
  --max-gpu-workers 2 \
  \
  --verbose

EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ APEX Processing Complete"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Output directory: ${OUTPUT_DIR}"
    echo ""
    echo "Deliverables:"
    echo "  📁 depth/         - 16-bit depth maps + visualizations"
    echo "  📁 pbr/           - Normal, Roughness, AO maps"
    echo "  📁 v2/            - AI-enhanced images"
    echo "  📁 master16/      - Audit-grade 16-bit linear TIFFs"
    echo "  📁 upscaled16/    - 2x resolution 16-bit TIFFs"
    echo "  📁 manifests/     - Processing metadata + run card"
    echo "  📁 logs/          - Pipeline execution logs"
    echo ""
else
    echo "  ❌ APEX Processing Failed (exit code: ${EXIT_CODE})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Check logs in: ${OUTPUT_DIR}/logs/"
fi

exit $EXIT_CODE
