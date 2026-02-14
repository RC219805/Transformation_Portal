#!/usr/bin/env bash
# Research-Grade APEX Pipeline with Depth Pro
# Ultra High Quality (UHQ) depth processing for source TIFFs
#
# ⚠️  LICENSE WARNING: Research and Non-Commercial Use ONLY
#
# This script uses Depth Pro from Apple ML Research, which is licensed
# under the Apple Machine Learning Research License (AMLR). This license
# PROHIBITS commercial use.
#
# Permitted Uses:
#   ✅ Academic research (university, institute)
#   ✅ Non-profit projects (no revenue generation)
#   ✅ Personal experimentation (non-commercial)
#   ✅ Benchmarking and comparative studies
#
# Prohibited Uses:
#   ❌ Commercial products or services
#   ❌ Revenue-generating applications
#   ❌ Enterprise/business deployments
#   ❌ Proprietary software distribution
#
# By running this script, you acknowledge and agree to these restrictions.
#
# Usage:
#   ./scripts/pipelines/run_source_tiffs_depth_pro_research.sh
#
# Features:
#   - Depth Pro metric depth (16-bit, meters)
#   - Materials V3 with real SAM2 segmentation (superior quality)
#   - V2 enhancement with material-aware tone mapping
#   - PBR texture generation (normal, roughness, AO)
#   - 16-bit TIFF archival output
#   - Research-grade quality validation
#
# See: docs/research/DEPTH_PRO_RESEARCH_GUIDE.md

set -euo pipefail

# Configuration
INPUT_DIR="input_images/source_tiffs"
OUTPUT_DIR="output_source_tiffs_depth_pro_$(date +%Y%m%d_%H%M%S)"
QUALITY_TIER="apex"
PRESET="depth-pro-research-uhq"
DEPTH_BACKEND="depth_pro"
DEPTH_DEVICE="mps"  # Change to "cuda" for NVIDIA GPUs, "cpu" for CPU-only

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# License acknowledgment flag
LICENSE_ACKNOWLEDGED=false

echo ""
echo "========================================="
echo "🔬 RESEARCH-GRADE APEX with Depth Pro"
echo "========================================="
echo ""

# ===== LICENSE ACKNOWLEDGMENT =====
echo -e "${RED}⚠️  LICENSE RESTRICTION WARNING ⚠️${NC}"
echo ""
echo "This script uses Depth Pro from Apple ML Research."
echo "License: Apple Machine Learning Research License (AMLR)"
echo ""
echo -e "${YELLOW}Permitted Uses:${NC}"
echo "  ✅ Academic research (university, institute)"
echo "  ✅ Non-profit projects (no revenue generation)"
echo "  ✅ Personal experimentation (non-commercial)"
echo "  ✅ Benchmarking and comparative studies"
echo ""
echo -e "${RED}Prohibited Uses:${NC}"
echo "  ❌ Commercial products or services"
echo "  ❌ Revenue-generating applications"
echo "  ❌ Enterprise/business deployments"
echo "  ❌ Proprietary software distribution"
echo ""
echo "License URL: https://github.com/apple/ml-depth-pro/blob/main/LICENSE"
echo ""
echo -e "${MAGENTA}By proceeding, you acknowledge this is for research/non-commercial use ONLY.${NC}"
echo ""
read -p "Do you accept these license restrictions? (yes/no) " -r
echo ""
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo -e "${RED}License not accepted. Exiting.${NC}"
    echo ""
    echo "For commercial use, please use the standard APEX preset with DA3:"
    echo "  ./scripts/pipelines/run_750_picacho_apex_full.sh"
    exit 1
fi

LICENSE_ACKNOWLEDGED=true
echo -e "${GREEN}✓ License restrictions acknowledged${NC}"
echo ""

# ===== PRE-FLIGHT CHECKS =====
echo "========================================="
echo "Pre-Flight Checks"
echo "========================================="
echo ""

# Check input directory
if [ ! -d "${INPUT_DIR}" ]; then
    echo -e "${RED}ERROR: Input directory not found: ${INPUT_DIR}${NC}"
    exit 1
fi

# Count input images (TIFF files)
IMAGE_COUNT=$(find "${INPUT_DIR}" -type f \( -iname "*.tif" -o -iname "*.tiff" \) | wc -l | tr -d ' ')
if [ "${IMAGE_COUNT}" -eq 0 ]; then
    echo -e "${RED}ERROR: No TIFF files found in ${INPUT_DIR}${NC}"
    echo "Expected: High-resolution TIFF files (architectural/real estate)"
    exit 1
fi
echo -e "${GREEN}✓ Found ${IMAGE_COUNT} TIFF images in ${INPUT_DIR}${NC}"

# Check Python environment
if ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: Python not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python available${NC}"

# Check lux-depth-v3 CLI installed
if ! python -m transformation_portal.lux_depth_v3 --help &> /dev/null; then
    echo -e "${RED}ERROR: lux-depth-v3 CLI not installed${NC}"
    echo "Install with: pip install -e ."
    exit 1
fi
echo -e "${GREEN}✓ lux-depth-v3 CLI installed${NC}"

# Check Depth Pro checkpoint
CHECKPOINT_PATH="checkpoints/depth_pro.pt"
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo -e "${YELLOW}⚠ Depth Pro checkpoint not found: ${CHECKPOINT_PATH}${NC}"
    echo ""
    echo "Downloading Depth Pro checkpoint (1.9 GB)..."
    mkdir -p checkpoints
    curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt \
      -o "${CHECKPOINT_PATH}" \
      --progress-bar

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Depth Pro checkpoint downloaded${NC}"
    else
        echo -e "${RED}ERROR: Failed to download Depth Pro checkpoint${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Depth Pro checkpoint exists: ${CHECKPOINT_PATH}${NC}"
fi

# Verify checkpoint hash (optional but recommended)
EXPECTED_SHA256="3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce"
if command -v shasum &> /dev/null; then
    echo "Verifying checkpoint integrity..."
    ACTUAL_SHA256=$(shasum -a 256 "${CHECKPOINT_PATH}" | awk '{print $1}')
    if [ "${ACTUAL_SHA256}" = "${EXPECTED_SHA256}" ]; then
        echo -e "${GREEN}✓ Checkpoint hash verified${NC}"
    else
        echo -e "${YELLOW}⚠ Checkpoint hash mismatch (may be outdated or corrupted)${NC}"
        echo "  Expected: ${EXPECTED_SHA256}"
        echo "  Actual:   ${ACTUAL_SHA256}"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    fi
fi

# Check if MPS is available (Apple Silicon)
if [ "${DEPTH_DEVICE}" = "mps" ]; then
    MPS_CHECK=$(python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null || echo "false")
    if [ "${MPS_CHECK}" = "True" ]; then
        echo -e "${GREEN}✓ MPS (Apple Neural Engine) available${NC}"
    else
        echo -e "${YELLOW}⚠ MPS not available, falling back to CPU${NC}"
        echo "  Note: CPU processing will be significantly slower (~10x)"
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

# Check Depth Pro package
DEPTH_PRO_CHECK=$(python -c "import depth_pro; print('ok')" 2>/dev/null || echo "failed")
if [ "${DEPTH_PRO_CHECK}" = "ok" ]; then
    echo -e "${GREEN}✓ Depth Pro package installed${NC}"
else
    echo -e "${YELLOW}⚠ Depth Pro package not installed${NC}"
    echo "  Install with: pip install depth-pro"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo ""
echo "========================================="
echo "Pipeline Configuration"
echo "========================================="
echo "Input:              ${INPUT_DIR}"
echo "Output:             ${OUTPUT_DIR}"
echo "Image Count:        ${IMAGE_COUNT} TIFF files"
echo "Quality Tier:       ${QUALITY_TIER}"
echo "Preset:             ${PRESET}"
echo ""
echo -e "${BLUE}Depth Configuration:${NC}"
echo "  Backend:          ${DEPTH_BACKEND} (Apple ML Research)"
echo "  Device:           ${DEPTH_DEVICE}"
echo "  Output:           16-bit metric depth (meters)"
echo "  Focal Length:     Estimated by Depth Pro"
echo ""
echo -e "${BLUE}Advanced Features:${NC}"
echo "  Materials V3:     ENABLED (with SAM2 segmentation)"
echo "  PBR Maps:         ENABLED (normal, roughness, AO)"
echo "  V2 Enhancement:   ENABLED (material-aware)"
echo "  16-bit Output:    ENABLED (depth + enhanced)"
echo ""
echo -e "${BLUE}License Mode:${NC}"
echo "  Mode:             RESEARCH-ONLY"
echo "  Acknowledged:     ${LICENSE_ACKNOWLEDGED}"
echo "========================================="
echo ""

# Confirm execution
read -p "Proceed with research-grade processing? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting Depth Pro APEX pipeline..."
echo ""

# Start timer
START_TIME=$(date +%s)

# Run pipeline with explicit license flags
python -m transformation_portal.lux_depth_v3 \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --quality-tier "${QUALITY_TIER}" \
  --preset "${PRESET}" \
  --depth-backend "${DEPTH_BACKEND}" \
  --depth-device "${DEPTH_DEVICE}" \
  --non-commercial-ok "true" \
  --accept-apple-depth-pro-research-license "true" \
  --materials-v3 "on" \
  --enable-segmentation "on" \
  --segmentation-backend "sam2" \
  --pbr "on" \
  --enable-v2 "on" \
  --v2-preset "luxury_estate" \
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
    echo "========================================="
    echo "Research-Grade Quality Verification"
    echo "========================================="
    echo ""

    # Count outputs
    DEPTH_COUNT=$(find "${OUTPUT_DIR}/depth" -name "*_depth.png" 2>/dev/null | wc -l | tr -d ' ')
    PBR_COUNT=$(find "${OUTPUT_DIR}/pbr" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
    ENHANCED_COUNT=$(find "${OUTPUT_DIR}/enhanced" -name "*_enhanced.jpg" 2>/dev/null | wc -l | tr -d ' ')
    MASTER16_COUNT=$(find "${OUTPUT_DIR}/master16" -name "*.tiff" 2>/dev/null | wc -l | tr -d ' ')

    echo "Output Summary:"
    echo "  Depth maps (16-bit):  ${DEPTH_COUNT} / ${IMAGE_COUNT}"
    echo "  PBR maps:             ${PBR_COUNT} / $((IMAGE_COUNT * 3))"
    echo "  Enhanced:             ${ENHANCED_COUNT} / ${IMAGE_COUNT}"
    echo "  Master 16-bit:        ${MASTER16_COUNT} / ${IMAGE_COUNT}"
    echo ""

    # Verify 16-bit depth maps
    if [ "${DEPTH_COUNT}" -gt 0 ]; then
        FIRST_DEPTH=$(find "${OUTPUT_DIR}/depth" -name "*_depth.png" -type f 2>/dev/null | head -1)
        if [ -n "${FIRST_DEPTH}" ]; then
            # Check bit depth using Python
            BIT_DEPTH=$(python -c "from PIL import Image; img = Image.open('${FIRST_DEPTH}'); print(img.mode)" 2>/dev/null || echo "unknown")
            if [ "${BIT_DEPTH}" = "I;16" ] || [ "${BIT_DEPTH}" = "I" ]; then
                echo -e "${GREEN}✓ Depth maps are 16-bit (verified)${NC}"
            else
                echo -e "${YELLOW}⚠ Depth maps may not be 16-bit (mode: ${BIT_DEPTH})${NC}"
            fi
        fi
    fi

    # Check Materials V3 segmentation backend
    MANIFEST_FILE=$(find "${OUTPUT_DIR}/manifests" -name "*.json" -type f 2>/dev/null | head -1)
    if [ -n "${MANIFEST_FILE}" ]; then
        echo ""
        echo "Manifest Verification:"

        # Check depth backend
        DEPTH_BACKEND_USED=$(cat "${MANIFEST_FILE}" | python -c "import sys, json; d = json.load(sys.stdin); print(d.get('stages', {}).get('depth', {}).get('backend', 'unknown'))" 2>/dev/null || echo "unknown")
        if [ "${DEPTH_BACKEND_USED}" = "depth_pro" ]; then
            echo -e "${GREEN}  ✓ Depth backend: depth_pro${NC}"
        else
            echo -e "${YELLOW}  ⚠ Depth backend: ${DEPTH_BACKEND_USED} (expected: depth_pro)${NC}"
        fi

        # Check segmentation backend
        SEG_BACKEND=$(cat "${MANIFEST_FILE}" | python -c "import sys, json; d = json.load(sys.stdin); print(d.get('stages', {}).get('materials_v3', {}).get('segmentation_backend', 'unknown'))" 2>/dev/null || echo "unknown")
        if [ "${SEG_BACKEND}" = "sam2" ]; then
            echo -e "${GREEN}  ✓ Materials V3 used real segmentation (SAM2)${NC}"
        else
            echo -e "${YELLOW}  ⚠ Materials V3 used fallback backend: ${SEG_BACKEND}${NC}"
        fi

        # Check materials detected
        MATERIALS=$(cat "${MANIFEST_FILE}" | python -c "import sys, json; d = json.load(sys.stdin); print(len(d.get('stages', {}).get('materials_v3', {}).get('materials_detected', [])))" 2>/dev/null || echo "0")
        if [ "${MATERIALS}" -gt 0 ]; then
            echo -e "${GREEN}  ✓ Materials detected: ${MATERIALS} types${NC}"
        else
            echo -e "${YELLOW}  ⚠ No materials detected${NC}"
        fi

        # Check license mode
        LICENSE_MODE=$(cat "${MANIFEST_FILE}" | python -c "import sys, json; d = json.load(sys.stdin); print(d.get('compliance', {}).get('license_mode', 'unknown'))" 2>/dev/null || echo "unknown")
        if [ "${LICENSE_MODE}" = "research_only" ] || [ "${LICENSE_MODE}" = "non_commercial" ]; then
            echo -e "${GREEN}  ✓ License mode: ${LICENSE_MODE}${NC}"
        else
            echo -e "${YELLOW}  ⚠ License mode: ${LICENSE_MODE}${NC}"
        fi

        # Check focal length (Depth Pro specific)
        FOCAL_LENGTH=$(cat "${MANIFEST_FILE}" | python -c "import sys, json; d = json.load(sys.stdin); print(d.get('stages', {}).get('depth', {}).get('focal_length_px', 'none'))" 2>/dev/null || echo "none")
        if [ "${FOCAL_LENGTH}" != "none" ]; then
            echo -e "${GREEN}  ✓ Focal length estimated: ${FOCAL_LENGTH}px (Depth Pro feature)${NC}"
        fi
    fi
    echo ""

    # Estimate throughput
    SECONDS_PER_IMAGE=$(echo "scale=2; ${DURATION} / ${IMAGE_COUNT}" | bc 2>/dev/null || echo "N/A")
    if [ "${SECONDS_PER_IMAGE}" != "N/A" ]; then
        IMAGES_PER_HOUR=$(echo "scale=0; 3600 / ${SECONDS_PER_IMAGE}" | bc 2>/dev/null || echo "N/A")
        echo "Performance:"
        echo "  Time per image:  ${SECONDS_PER_IMAGE}s"
        echo "  Throughput:      ~${IMAGES_PER_HOUR} images/hour"
        echo ""
        echo -e "${BLUE}Note: Depth Pro is slower than DA3 but produces higher quality depth.${NC}"
    fi
    echo ""

    # Research notes
    echo "========================================="
    echo "Research Notes"
    echo "========================================="
    echo ""
    echo "Depth Pro Advantages:"
    echo "  • Metric depth in meters (not normalized)"
    echo "  • Focal length estimation for 3D reconstruction"
    echo "  • Superior edge preservation"
    echo "  • Better performance on reflective surfaces"
    echo ""
    echo "Output Structure:"
    echo "  ${OUTPUT_DIR}/"
    echo "    ├── depth/              # 16-bit depth maps (PNG)"
    echo "    ├── enhanced/           # V2 enhanced images"
    echo "    ├── pbr/                # PBR maps (normal, roughness, AO)"
    echo "    ├── master16/           # 16-bit archival TIFFs"
    echo "    ├── manifests/          # Research metadata (JSON)"
    echo "    └── reports/            # Quality reports"
    echo ""
    echo "Next Steps:"
    echo "  1. Review manifests for focal length and depth metrics"
    echo "  2. Compare depth quality to DA3 (if available)"
    echo "  3. Validate material segmentation accuracy"
    echo "  4. Check 16-bit depth precision (use GIMP/Photoshop)"
    echo ""
    echo "Documentation: docs/research/DEPTH_PRO_RESEARCH_GUIDE.md"
    echo "========================================="
    echo ""
fi

# Open output directory (macOS only)
if command -v open &> /dev/null; then
    echo "Opening output directory..."
    open "${OUTPUT_DIR}"
fi

exit ${EXIT_CODE}
