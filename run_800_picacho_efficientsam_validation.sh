#!/usr/bin/env bash
#
# Production validation: 800 Picacho with EfficientSAM segmentation
#
# This script validates the EfficientSAM backend integration by running
# the full APEX pipeline on 20 luxury real estate images with:
# - EfficientSAM-based material segmentation (new)
# - Materials V3 pixel operations
# - Depth Anything V3
# - V2 tone mapping
# - PBR enhancements
#
# Expected outcomes:
# - Material masks generated for each image
# - Pixel operations applied based on detected materials
# - Enhanced images with surface-aware finishing
# - Manifest contains segmentation telemetry
#

set -euo pipefail

# Configuration
INPUT_DIR="input_images/800 Picacho"
OUTPUT_DIR="output_800_picacho_efficientsam_$(date +%Y%m%d_%H%M%S)"
PRESET="apex/base_v2"

echo "========================================="
echo "EfficientSAM Production Validation"
echo "========================================="
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Preset: ${PRESET}"
echo "Images: $(ls "${INPUT_DIR}"/*.jpg | wc -l)"
echo ""
echo "Features enabled:"
echo "  ✓ Depth Anything V3 (commercial-safe)"
echo "  ✓ EfficientSAM segmentation (NEW)"
echo "  ✓ Materials V3 pixel operations"
echo "  ✓ V2 tone mapping integration"
echo "  ✓ PBR enhancements"
echo "  ✓ Provenance capture"
echo "  ✓ Depth caching"
echo ""
echo "Starting pipeline..."
echo ""

# Run APEX pipeline with EfficientSAM enabled
python -m transformation_portal.lux_depth_v3.cli \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --preset "${PRESET}" \
  --depth-backend "da3" \
  --depth-device "auto" \
  --enable-materials-v3 \
  --enable-material-segmentation \
  --material-segmentation-backend "efficientsam" \
  --strict-backend \
  --apply-pixel-ops \
  --enable-v2 \
  --enable-pbr \
  --enable-provenance \
  --use-depth-cache \
  --parallel 1

echo ""
echo "========================================="
echo "Validation Complete"
echo "========================================="
echo ""

# Analyze outputs
echo "Output summary:"
echo "  Total files: $(find "${OUTPUT_DIR}" -type f | wc -l)"
echo "  Total size:  $(du -sh "${OUTPUT_DIR}" | cut -f1)"
echo ""

# Check for segmentation artifacts
if [ -d "${OUTPUT_DIR}/materials_v3" ]; then
  echo "Materials V3 outputs:"
  echo "  Masks: $(find "${OUTPUT_DIR}/materials_v3" -name "*_mask_*.png" 2>/dev/null | wc -l)"
  echo "  Enhanced: $(find "${OUTPUT_DIR}/materials_v3" -name "*_enhanced.png" 2>/dev/null | wc -l)"
else
  echo "⚠️  No materials_v3 subdirectory found"
fi

# Check manifests for segmentation metadata
echo ""
echo "Checking manifests for segmentation telemetry..."
MANIFEST_COUNT=$(find "${OUTPUT_DIR}" -name "*_manifest.json" | wc -l)
echo "  Manifests found: ${MANIFEST_COUNT}"

if [ ${MANIFEST_COUNT} -gt 0 ]; then
  # Sample first manifest for structure
  FIRST_MANIFEST=$(find "${OUTPUT_DIR}" -name "*_manifest.json" | head -1)
  echo ""
  echo "Sample manifest (first 50 lines):"
  head -50 "${FIRST_MANIFEST}"
fi

echo ""
echo "Validation outputs saved to: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Review material masks in ${OUTPUT_DIR}/materials_v3/"
echo "  2. Check manifest.json files for segmentation_backend field"
echo "  3. Compare pixel ops effectiveness vs stub backend"
echo "  4. Validate performance metrics (expected ~400ms/image on MPS)"
echo ""
