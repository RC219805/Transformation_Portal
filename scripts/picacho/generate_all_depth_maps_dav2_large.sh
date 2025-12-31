#!/usr/bin/env bash
set -euo pipefail

# ========================================
# Batch Depth Map Generation
# Depth Anything V2 Large | 16-bit TIFF
# ========================================

SOURCE_DIR="750Picacho_Source_TIFFs"
OUTPUT_DIR="750Picacho_Depth_Maps_APEX"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="depth_generation_apex_${TIMESTAMP}.log"

echo "========================================" | tee "$LOG_FILE"
echo "APEX Depth Map Generation - All Scenes" | tee -a "$LOG_FILE"
echo "Depth Anything V2 Large | 16-bit TIFF" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Source files array
SOURCE_FILES=(
    "750Picacho_Aerial.tif"
    "750Picacho_GreatRoom.tif"
    "750Picacho_Kitchen.tif"
    "750Picacho_Pool_16bit.tiff"
    "750Picacho_PrimaryBathroom.tif"
    "750Picacho_PrimaryBedroom.tif"
)

# Counters
TOTAL=${#SOURCE_FILES[@]}
SUCCESS=0
FAILED=0

echo "Total files to process: $TOTAL" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Process each file
for i in "${!SOURCE_FILES[@]}"; do
    FILE="${SOURCE_FILES[$i]}"
    NUM=$((i + 1))

    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "[$NUM/$TOTAL] Processing: $FILE" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"

    INPUT_PATH="${SOURCE_DIR}/${FILE}"

    if [ ! -f "$INPUT_PATH" ]; then
        echo "⚠️  ERROR: File not found: $INPUT_PATH" | tee -a "$LOG_FILE"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Determine scene-specific preset
    PRESET="interior_luxury_max_quality"
    SCENE_TYPE="interior"

    if [[ "$FILE" == *"Aerial"* ]]; then
        PRESET="exterior_showcase_max_quality"
        SCENE_TYPE="exterior"
    elif [[ "$FILE" == *"Pool"* ]]; then
        PRESET="exterior_showcase_max_quality"
        SCENE_TYPE="exterior"
    fi

    echo "Scene type: $SCENE_TYPE" | tee -a "$LOG_FILE"
    echo "Preset: $PRESET" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Create scene-specific output directory
    SCENE_OUTPUT="${OUTPUT_DIR}/${SCENE_TYPE}"
    mkdir -p "$SCENE_OUTPUT"

    # Execute lux-depth-v2 with APEX settings
    echo "Starting depth generation..." | tee -a "$LOG_FILE"

    if lux-depth-v2 \
        --input "$INPUT_PATH" \
        --output-dir "$SCENE_OUTPUT" \
        --preset "$PRESET" \
        --quality-tier apex \
        --device auto \
        --precision fp32 \
        --tile 1024 \
        --tile-pad 32 \
        --seg-backend segformer \
        --seg-long-side 2048 \
        --materials-v2 \
        --materials-v2-backend segformer \
        --materials-v2-long-side 2048 \
        --materials-v2-confidence 0.15 \
        --edge-refinement \
        --refinement-preset aggressive \
        --cache-masks \
        --model-cache \
        --depth-cache \
        --tiff-compression lzw \
        2>&1 | tee -a "$LOG_FILE"; then

        echo "✅ SUCCESS: $FILE" | tee -a "$LOG_FILE"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "❌ FAILED: $FILE" | tee -a "$LOG_FILE"
        FAILED=$((FAILED + 1))
    fi

    echo "" | tee -a "$LOG_FILE"
done

# Summary
echo "========================================" | tee -a "$LOG_FILE"
echo "Batch Processing Complete" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Total:   $TOTAL" | tee -a "$LOG_FILE"
echo "Success: $SUCCESS" | tee -a "$LOG_FILE"
echo "Failed:  $FAILED" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# List generated depth maps
echo "" | tee -a "$LOG_FILE"
echo "Generated depth maps:" | tee -a "$LOG_FILE"
find "$OUTPUT_DIR" -name "*depth_raw_16bit.tiff" -type f -exec ls -lh {} \; | tee -a "$LOG_FILE"

exit 0
