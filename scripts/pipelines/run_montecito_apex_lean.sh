#!/bin/bash
#
# Montecito Shores APEX Processing - Essential Features (Faster, Less Disk)
# Generated: 2026-02-10
# Input: 10 TIFF files @ 300 DPI
# Est. Runtime: ~1.5 minutes
#

set -euo pipefail

INPUT_DIR="input_images/Montecito-Shores_press_300dpi_TIFFs"
OUTPUT_DIR="output_montecito_shores_apex_lean_$(date +%Y%m%d_%H%M%S)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Transformation Portal APEX - Montecito Shores (Lean)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Features: APEX Quality + Materials V3 + Marketing Deliverables Only"
echo "Est. Time: ~7s per image"
echo ""

python -m transformation_portal.lux_depth_v3 \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --quality-tier apex \
  --depth-device mps \
  --materials-v3 on \
  --emit-marketing on \
  --cache-depth on \
  --verbose

echo ""
echo "✅ Complete: ${OUTPUT_DIR}"
