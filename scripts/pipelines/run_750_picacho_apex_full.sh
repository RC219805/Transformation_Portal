#!/usr/bin/env bash
# Production APEX Pipeline for 750 Picacho Drive
# ALL advanced features enabled - Materials V3, DA3, V2, PBR, 16-bit output
#
# Usage:
#   ./scripts/pipelines/run_750_picacho_apex_full.sh
#
# Features:
#   - Materials V3 with real SAM2 segmentation (superior quality)
#   - Depth Anything V3 (commercial-safe)
#   - V2 enhancement with material-aware tone mapping
#   - PBR texture generation (normal, roughness, AO)
#   - 16-bit TIFF archival output
#   - Comprehensive manifests and reports

set -euo pipefail

# Configuration
INPUT_DIR="input_images/750_picacho/source_jpegs"
OUTPUT_DIR="output_750_picacho_apex_full_$(date +%Y%m%d_%H%M%S)"
QUALITY_TIER="apex"
DEPTH_BACKEND="da3"
DEPTH_DEVICE="mps"  # Change to "cuda" for NVIDIA GPUs, "cpu" for CPU-only

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "750 Picacho Drive - APEX Pipeline"
echo "========================================="
echo ""

# Pre-flight checks
echo "Running pre-flight checks..."

# Check input directory
if [ ! -d "${INPUT_DIR}" ]; then
    echo -e "${RED}ERROR: Input directory not found: ${INPUT_DIR}${NC}"
    exit 1
fi

# Count input images
IMAGE_COUNT=$(find "${INPUT_DIR}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l | tr -d ' ')
if [ "${IMAGE_COUNT}" -eq 0 ]; then
    echo -e "${RED}ERROR: No images found in ${INPUT_DIR}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Found ${IMAGE_COUNT} images in ${INPUT_DIR}${NC}"

# Check Python environment
if ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: Python not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python available${NC}"

# Check if MPS is available (optional)
if [ "${DEPTH_DEVICE}" = "mps" ]; then
    MPS_CHECK=$(python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null || echo "false")
    if [ "${MPS_CHECK}" = "True" ]; then
        echo -e "${GREEN}✓ MPS (Apple Neural Engine) available${NC}"
    else
        echo -e "${YELLOW}⚠ MPS not available, falling back to CPU${NC}"
        DEPTH_DEVICE="cpu"
    fi
fi

# Check SAM2 backend (Materials V3)
BACKEND_CHECK=$(python -c "from transformation_portal.lux_depth_v3.segmentation_backend import _get_backend_instance; print(_get_backend_instance('sam2').__class__.__name__)" 2>/dev/null || echo "failed")
if [ "${BACKEND_CHECK}" = "SAM2MaterialsAdapter" ]; then
    echo -e "${GREEN}✓ SAM2 backend available (Materials V3)${NC}"
else
    echo -e "${YELLOW}⚠ SAM2 backend not available, will fall back to stub${NC}"
    echo "  Install with: pip install -e \".[ml]\""
fi

echo ""
echo "========================================="
echo "Pipeline Configuration"
echo "========================================="
echo "Input:           ${INPUT_DIR}"
echo "Output:          ${OUTPUT_DIR}"
echo "Image Count:     ${IMAGE_COUNT}"
echo "Quality Tier:    ${QUALITY_TIER}"
echo "Depth Backend:   ${DEPTH_BACKEND}"
echo "Depth Device:    ${DEPTH_DEVICE}"
echo "Materials V3:    ENABLED (with SAM2 segmentation)"
echo "PBR Maps:        ENABLED"
echo "V2 Enhancement:  ENABLED"
echo "16-bit Output:   ENABLED"
echo "========================================="
echo ""

# Confirm execution
read -p "Proceed with APEX processing? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting APEX pipeline..."
echo ""

# Start timer
START_TIME=$(date +%s)

# Run pipeline
python -m transformation_portal.lux_depth_v3 \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --quality-tier "${QUALITY_TIER}" \
  --depth-backend "${DEPTH_BACKEND}" \
  --depth-device "${DEPTH_DEVICE}" \
  --materials-v3 "on" \
  --enable-segmentation "on" \
  --segmentation-backend "sam2" \
  --pbr "on" \
  --enable-v2 "on" \
  --v2-preset "default" \
  --emit-master16 "on" \
  --emit-upscaled16 "on" \
  --emit-marketing "on" \
  --emit-report "on" \
  --emit-run-card "on" \
  --cache-depth "on" \
  --overwrite \
  --verbose

# Capture exit code
EXIT_CODE=$?

# End timer
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo -e "${GREEN}✓ Pipeline completed successfully${NC}"
else
    echo -e "${RED}✗ Pipeline failed with exit code ${EXIT_CODE}${NC}"
fi
echo "Duration: ${DURATION} seconds"
echo "Output:   ${OUTPUT_DIR}"
echo "========================================="
echo ""

# Post-processing verification
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Running post-processing verification..."
    echo ""

    # Count outputs
    DEPTH_COUNT=$(find "${OUTPUT_DIR}/depth" -name "*_depth.png" 2>/dev/null | wc -l | tr -d ' ')
    PBR_COUNT=$(find "${OUTPUT_DIR}/pbr" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
    ENHANCED_COUNT=$(find "${OUTPUT_DIR}/enhanced" -name "*_enhanced.jpg" 2>/dev/null | wc -l | tr -d ' ')
    MASTER16_COUNT=$(find "${OUTPUT_DIR}/master16" -name "*.tiff" 2>/dev/null | wc -l | tr -d ' ')

    echo "Output Summary:"
    echo "  Depth maps:      ${DEPTH_COUNT} / ${IMAGE_COUNT}"
    echo "  PBR maps:        ${PBR_COUNT} / $((IMAGE_COUNT * 3))"
    echo "  Enhanced:        ${ENHANCED_COUNT} / ${IMAGE_COUNT}"
    echo "  Master 16-bit:   ${MASTER16_COUNT} / ${IMAGE_COUNT}"
    echo ""

    # Check if all outputs match expected counts
    ALL_GOOD=true
    if [ "${DEPTH_COUNT}" -ne "${IMAGE_COUNT}" ]; then
        echo -e "${YELLOW}⚠ Missing depth maps${NC}"
        ALL_GOOD=false
    fi
    if [ "${ENHANCED_COUNT}" -ne "${IMAGE_COUNT}" ]; then
        echo -e "${YELLOW}⚠ Missing enhanced outputs${NC}"
        ALL_GOOD=false
    fi
    if [ "${PBR_COUNT}" -ne "$((IMAGE_COUNT * 3))" ]; then
        echo -e "${YELLOW}⚠ Missing PBR maps (expected 3 per image)${NC}"
        ALL_GOOD=false
    fi

    if $ALL_GOOD; then
        echo -e "${GREEN}✓ All outputs generated successfully${NC}"
    fi
    echo ""

    # Check Materials V3 segmentation backend
    MANIFEST_FILE=$(find "${OUTPUT_DIR}/manifests" -name "*.json" -type f 2>/dev/null | head -1)
    if [ -n "${MANIFEST_FILE}" ]; then
        BACKEND=$(cat "${MANIFEST_FILE}" | python -c "import sys, json; print(json.load(sys.stdin).get('stages', {}).get('materials_v3', {}).get('segmentation_backend', 'unknown'))" 2>/dev/null || echo "unknown")
        if [ "${BACKEND}" = "sam2" ]; then
            echo -e "${GREEN}✓ Materials V3 used real segmentation (SAM2)${NC}"
        else
            echo -e "${YELLOW}⚠ Materials V3 used fallback backend: ${BACKEND}${NC}"
        fi

        # Check materials detected
        MATERIALS=$(cat "${MANIFEST_FILE}" | python -c "import sys, json; print(len(json.load(sys.stdin).get('stages', {}).get('materials_v3', {}).get('materials_detected', [])))" 2>/dev/null || echo "0")
        if [ "${MATERIALS}" -gt 0 ]; then
            echo -e "${GREEN}✓ Materials detected: ${MATERIALS} types${NC}"
        else
            echo -e "${YELLOW}⚠ No materials detected${NC}"
        fi
    fi
    echo ""

    # Estimate throughput
    SECONDS_PER_IMAGE=$(echo "scale=2; ${DURATION} / ${IMAGE_COUNT}" | bc)
    IMAGES_PER_HOUR=$(echo "scale=0; 3600 / ${SECONDS_PER_IMAGE}" | bc)
    echo "Performance:"
    echo "  Time per image:  ${SECONDS_PER_IMAGE}s"
    echo "  Throughput:      ~${IMAGES_PER_HOUR} images/hour"
    echo ""
fi

# Open output directory
if command -v open &> /dev/null; then
    echo "Opening output directory..."
    open "${OUTPUT_DIR}"
fi

exit ${EXIT_CODE}
