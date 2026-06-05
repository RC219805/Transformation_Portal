#!/usr/bin/env bash
# ============================================================================
# 750 Picacho Elite Processing - Quick Start Script
# ============================================================================
# Batch process all 6 images from 750 Picacho luxury estate with optimized
# room-specific settings and complete quality pipeline.
#
# Usage:
#   ./process_750_picacho_elite_batch.sh [output_dir]
#
# Author: Transformation Portal
# Date: 2025-11-10
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INPUT_DIR="input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs"
OUTPUT_DIR="${1:-output_750_picacho_elite_$(date +%Y%m%d_%H%M%S)}"
PIPELINE_SCRIPT="${SCRIPT_DIR}/luxury_estate_master_pipeline.py"
LOG_FILE="750_picacho_processing_$(date +%Y%m%d_%H%M%S).log"

cd "$REPO_ROOT"

# Print header
echo ""
echo "============================================================================"
echo "  750 PICACHO ELITE PROCESSING"
echo "============================================================================"
echo "  Source: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Log: $LOG_FILE"
echo "============================================================================"
echo ""

# Check prerequisites
echo -e "${BLUE}[1/5]${NC} Checking prerequisites..."

if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}✗ Error: Input directory not found: $INPUT_DIR${NC}"
    exit 1
fi

if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo -e "${RED}✗ Error: Pipeline script not found: $PIPELINE_SCRIPT${NC}"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Error: Python 3 not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites OK${NC}"

# Count input images
IMAGE_COUNT=$(find "$INPUT_DIR" -name "*.tif" -o -name "*.tiff" | wc -l | tr -d ' ')
echo -e "${BLUE}[2/5]${NC} Found ${GREEN}$IMAGE_COUNT${NC} images to process"

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo -e "${RED}✗ Error: No TIFF files found in $INPUT_DIR${NC}"
    exit 1
fi

# Create output directory
echo -e "${BLUE}[3/5]${NC} Creating output directory..."
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓ Created: $OUTPUT_DIR${NC}"

# Define room type mappings for optimized processing
declare -A ROOM_TYPES=(
    ["750Picacho_Aerial_HDR_32-bit.tif"]="aerial"
    ["750Picacho_Bathroom_HDR_32-bit.tif"]="bathroom"
    ["750Picacho_Bedroom_HDR_32-bit.tif"]="bedroom"
    ["750Picacho_Great_Room_HDR_32-bit.tif"]="great_room"
    ["750Picacho_Kitchen_HDR_32-bit.tif"]="kitchen"
    ["750Picacho_Pool_HDR_32-bit.tif"]="pool"
)

# Process images
echo ""
echo -e "${BLUE}[4/5]${NC} Processing images..."
echo "============================================================================"

START_TIME=$(date +%s)
PROCESSED=0
FAILED=0

for image_path in "$INPUT_DIR"/*.tif; do
    if [ ! -f "$image_path" ]; then
        continue
    fi

    image_name=$(basename "$image_path")
    room_type=${ROOM_TYPES[$image_name]:-"interior"}

    echo ""
    echo -e "${YELLOW}Processing:${NC} $image_name"
    echo -e "${YELLOW}Room Type:${NC} $room_type"
    echo "----------------------------------------------------------------------------"

    # Select preset based on room type
    PRESET="750_picacho"
    if [ "$room_type" == "aerial" ] || [ "$room_type" == "pool" ]; then
        PRESET="aerial"
    fi

    # Process image
    if python3 "$PIPELINE_SCRIPT" \
        "$image_path" \
        --room-type "$room_type" \
        --preset "$PRESET" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$LOG_FILE"; then

        PROCESSED=$((PROCESSED + 1))
        echo -e "${GREEN}✓ Completed: $image_name${NC}"
    else
        FAILED=$((FAILED + 1))
        echo -e "${RED}✗ Failed: $image_name${NC}"
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

# Summary
echo ""
echo "============================================================================"
echo -e "${BLUE}[5/5]${NC} Processing Complete"
echo "============================================================================"
echo -e "  Processed: ${GREEN}$PROCESSED${NC} / $IMAGE_COUNT"
echo -e "  Failed: ${RED}$FAILED${NC}"
echo -e "  Time: ${YELLOW}${MINUTES}m ${SECONDS}s${NC}"
echo "  Output: $OUTPUT_DIR"
echo "  Log: $LOG_FILE"
echo "============================================================================"
echo ""

# List output files
if [ "$PROCESSED" -gt 0 ]; then
    echo -e "${GREEN}Output Files:${NC}"
    echo "----------------------------------------------------------------------------"

    # Count output files
    MASTER_COUNT=$(find "$OUTPUT_DIR" -name "*_master.tif" | wc -l | tr -d ' ')
    JPEG_COUNT=$(find "$OUTPUT_DIR" -name "*_delivery.jpg" | wc -l | tr -d ' ')

    echo "  Master TIFFs: $MASTER_COUNT"
    echo "  Delivery JPEGs: $JPEG_COUNT"
    echo ""

    # Show first few outputs
    echo "  Sample outputs:"
    find "$OUTPUT_DIR" -type f \( -name "*_master.tif" -o -name "*_delivery.jpg" \) | head -6 | while read -r file; do
        size=$(du -h "$file" | cut -f1)
        echo "    • $(basename "$file") ($size)"
    done

    echo ""
    echo "  All files: $OUTPUT_DIR"

    # Check for processing report
    if [ -f "$OUTPUT_DIR/processing_report.json" ]; then
        echo ""
        echo "  📊 Processing Report: $OUTPUT_DIR/processing_report.json"
    fi
fi

# Final status
echo ""
if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✅ All images processed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Processing completed with $FAILED failed images${NC}"
    echo "   Check log file for details: $LOG_FILE"
    exit 1
fi
