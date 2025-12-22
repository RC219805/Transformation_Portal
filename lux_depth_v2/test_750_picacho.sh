#!/bin/bash
# Test Script for 750 Picacho TIFF Processing with Lux Depth V2
# ============================================================

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}750 Picacho TIFF Processing Test${NC}"
echo -e "${GREEN}Lux Depth V2 Pipeline${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Configuration
REPO_ROOT="/home/runner/work/Transformation_Portal/Transformation_Portal"
INPUT_DIR="${REPO_ROOT}/projects/750_picacho_lane/Final_Production_UltraQuality"
OUTPUT_DIR="${REPO_ROOT}/lux_depth_v2/test_outputs/750_picacho"
PRESET="interior_luxury"
DEVICE="cpu"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            DEVICE="cuda"
            shift
            ;;
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu              Use GPU instead of CPU"
            echo "  --preset NAME      Use specific preset (default: interior_luxury)"
            echo "  --output-dir DIR   Output directory (default: lux_depth_v2/test_outputs/750_picacho)"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Test with CPU, interior_luxury preset"
            echo "  $0 --gpu                     # Test with GPU"
            echo "  $0 --preset balanced         # Test with balanced preset"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Step 1: Pre-flight Checks${NC}"
echo "----------------------------------------"

# Check if we're in the right directory
cd "${REPO_ROOT}" || {
    echo -e "${RED}Error: Could not cd to ${REPO_ROOT}${NC}"
    exit 1
}
echo -e "${GREEN}✓${NC} Repository root: ${REPO_ROOT}"

# Check if input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo -e "${RED}✗ Input directory not found: ${INPUT_DIR}${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Input directory exists: ${INPUT_DIR}"

# Count TIFF files
TIFF_COUNT=$(find "${INPUT_DIR}" -name "*.tif" -type f | wc -l)
echo -e "${GREEN}✓${NC} Found ${TIFF_COUNT} TIFF files"

if [ "${TIFF_COUNT}" -ne 6 ]; then
    echo -e "${YELLOW}⚠${NC} Warning: Expected 6 TIFF files, found ${TIFF_COUNT}"
fi

# Check Python dependencies
echo ""
echo -e "${YELLOW}Step 2: Dependency Check${NC}"
echo "----------------------------------------"

DEPS_OK=true

# Check each required dependency
for dep in numpy cv2 tifffile torch tqdm; do
    if python -c "import ${dep}" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} ${dep} installed"
    else
        echo -e "${RED}✗${NC} ${dep} NOT installed"
        DEPS_OK=false
    fi
done

if [ "${DEPS_OK}" = false ]; then
    echo ""
    echo -e "${RED}Missing dependencies detected!${NC}"
    echo "Install with:"
    echo "  pip install numpy opencv-python tifffile torch tqdm"
    echo "Or:"
    echo "  pip install -r lux_depth_v2/requirements-repo.txt"
    exit 1
fi

# Check if lux_depth_v2 module is importable
echo ""
if python -c "from lux_depth_v2.pipeline import LuxPipelineV2; from lux_depth_v2.config import PipelineConfig" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} lux_depth_v2 module importable"
else
    echo -e "${RED}✗ lux_depth_v2 module import failed${NC}"
    echo "This may indicate a configuration issue"
    exit 1
fi

# Create output directory
echo ""
echo -e "${YELLOW}Step 3: Prepare Output Directory${NC}"
echo "----------------------------------------"

mkdir -p "${OUTPUT_DIR}"
echo -e "${GREEN}✓${NC} Output directory: ${OUTPUT_DIR}"

# Run processing
echo ""
echo -e "${YELLOW}Step 4: Process TIFF Files${NC}"
echo "----------------------------------------"
echo "Configuration:"
echo "  Preset: ${PRESET}"
echo "  Device: ${DEVICE}"
echo "  Input:  ${INPUT_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo ""
echo "Starting processing..."
echo ""

START_TIME=$(date +%s)

# Run lux_depth_v2 CLI
python -m lux_depth_v2.cli \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --preset "${PRESET}" \
    --device "${DEVICE}" \
    --upscaler-backend torch \
    --file-pattern "*.tif" || {
    echo -e "${RED}✗ Processing failed${NC}"
    exit 1
}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}✓ Processing complete${NC}"
echo "Total time: ${DURATION} seconds ($(echo "scale=2; ${DURATION}/60" | bc) minutes)"

# Validation
echo ""
echo -e "${YELLOW}Step 5: Validate Outputs${NC}"
echo "----------------------------------------"

# Count output files
OUTPUT_COUNT=$(find "${OUTPUT_DIR}" -name "*_master16.tif" -type f | wc -l)
echo -e "${GREEN}✓${NC} Generated ${OUTPUT_COUNT} master TIFF files"

if [ "${OUTPUT_COUNT}" -ne "${TIFF_COUNT}" ]; then
    echo -e "${YELLOW}⚠${NC} Warning: Expected ${TIFF_COUNT} outputs, got ${OUTPUT_COUNT}"
fi

# List outputs
echo ""
echo "Output files:"
find "${OUTPUT_DIR}" -type f -name "*.tif" -o -name "*.png" -o -name "*.jpg" -o -name "*.json" | sort | while read -r file; do
    SIZE=$(du -h "$file" | cut -f1)
    echo "  ${SIZE}  $(basename "$file")"
done

# Verify 16-bit TIFF format
echo ""
echo "Verifying 16-bit TIFF format..."
python -c "
import tifffile
import glob
import sys

files = glob.glob('${OUTPUT_DIR}/*_master16.tif')
all_ok = True

for f in files:
    try:
        img = tifffile.imread(f)
        is_16bit = str(img.dtype) == 'uint16'
        status = '✓' if is_16bit else '✗'
        print(f'  {status} {f.split(\"/\")[-1]}: {img.dtype} {img.shape}')
        if not is_16bit:
            all_ok = False
    except Exception as e:
        print(f'  ✗ {f.split(\"/\")[-1]}: Error - {e}')
        all_ok = False

sys.exit(0 if all_ok else 1)
" || {
    echo -e "${RED}✗ 16-bit verification failed${NC}"
    exit 1
}

# Generate summary report
echo ""
echo -e "${YELLOW}Step 6: Generate Summary Report${NC}"
echo "----------------------------------------"

SUMMARY_FILE="${OUTPUT_DIR}/TEST_SUMMARY.txt"

cat > "${SUMMARY_FILE}" << EOF
750 Picacho TIFF Processing Test - Summary
==========================================

Date: $(date)
Pipeline: Lux Depth V2
Preset: ${PRESET}
Device: ${DEVICE}

Results
-------
Input Files: ${TIFF_COUNT}
Output Files: ${OUTPUT_COUNT}
Processing Time: ${DURATION} seconds ($(echo "scale=2; ${DURATION}/60" | bc) minutes)
Average Time: $(echo "scale=2; ${DURATION}/${TIFF_COUNT}" | bc) seconds/file

Output Directory: ${OUTPUT_DIR}

Files Processed:
EOF

find "${INPUT_DIR}" -name "*.tif" -type f | sort | while read -r file; do
    echo "  - $(basename "$file")" >> "${SUMMARY_FILE}"
done

echo "" >> "${SUMMARY_FILE}"
echo "Output Files Generated:" >> "${SUMMARY_FILE}"
find "${OUTPUT_DIR}" -type f | sort | while read -r file; do
    SIZE=$(du -h "$file" | cut -f1)
    echo "  - ${SIZE}  $(basename "$file")" >> "${SUMMARY_FILE}"
done

echo -e "${GREEN}✓${NC} Summary report saved to: ${SUMMARY_FILE}"

# Final summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}TEST COMPLETE${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Summary:"
echo "  Input files:    ${TIFF_COUNT}"
echo "  Output files:   ${OUTPUT_COUNT}"
echo "  Processing time: ${DURATION} seconds"
echo "  Output location: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Review output files in: ${OUTPUT_DIR}"
echo "  2. Check summary report: ${SUMMARY_FILE}"
echo "  3. Visually inspect output quality"
echo "  4. Compare input vs output side-by-side"
echo ""
echo -e "${GREEN}✓ All tests passed successfully!${NC}"
