#!/usr/bin/env bash
#
# APEX-100 Depth Map Generation for 750 Picacho Lane
# Generates production-grade 16-bit depth maps using lux-depth-v2
# with APEX-tier tiled inference, MaterialsV2 SegFormer, and edge refinement
#
# Date: 2025-12-31
# Status: Production-ready (feat/apex-100-production merged)
# Fixes: Pool path, invalid flags, tile padding

set -euo pipefail

REPO_ROOT="/Users/rc/Transformation_Portal"
OUTPUT_DIR="${REPO_ROOT}/750Picacho_Depth_Maps_APEX"
LOG_DIR="${OUTPUT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output structure
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# APEX-100 Configuration (validated against actual CLI)
DEPTH_CONFIG=(
    --quality-tier apex
    --device auto
    --precision fp32
    --tile 1024
    --tile-pad 128
    --materials-v2
    --materials-v2-backend segformer
    --materials-v2-long-side 2048
    --materials-v2-confidence 0.40
    --refinement-preset balanced
    --cache-masks
    --tiff-compression lzw
    --intent production
)

# Source files with correct paths
declare -a SCENES=(
    "interior|interior_luxury_max_quality|${REPO_ROOT}/750Picacho_Source_TIFFs/750Picacho_GreatRoom.tif"
    "interior|interior_luxury_max_quality|${REPO_ROOT}/750Picacho_Source_TIFFs/750Picacho_Kitchen.tif"
    "interior|interior_luxury_max_quality|${REPO_ROOT}/750Picacho_Source_TIFFs/750Picacho_PrimaryBathroom.tif"
    "interior|interior_luxury_max_quality|${REPO_ROOT}/750Picacho_Source_TIFFs/750Picacho_PrimaryBedroom.tif"
    "exterior|exterior_showcase_max_quality|${REPO_ROOT}/750Picacho_Source_TIFFs/750Picacho_Aerial.tif"
    "exterior|exterior_showcase_max_quality|${REPO_ROOT}/projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_Pool_UltraQuality.tif"
)

MAIN_LOG="${LOG_DIR}/apex_batch_${TIMESTAMP}.log"

echo "======================================================================" | tee "$MAIN_LOG"
echo "APEX-100 Depth Map Generation - True 100% APEX" | tee -a "$MAIN_LOG"
echo "======================================================================" | tee -a "$MAIN_LOG"
echo "Output:       ${OUTPUT_DIR}" | tee -a "$MAIN_LOG"
echo "Total Scenes: ${#SCENES[@]}" | tee -a "$MAIN_LOG"
echo "Timestamp:    ${TIMESTAMP}" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "APEX Configuration:" | tee -a "$MAIN_LOG"
echo "  • Depth Anything V2 Large (tiled inference)" | tee -a "$MAIN_LOG"
echo "  • MaterialsV2: SegFormer backend @ 2048px" | tee -a "$MAIN_LOG"
echo "  • Edge Refinement: Balanced (pool-safe)" | tee -a "$MAIN_LOG"
echo "  • Precision: FP32 (full precision)" | tee -a "$MAIN_LOG"
echo "  • Tile Size: 1024x1024 (pad: 128)" | tee -a "$MAIN_LOG"
echo "  • Confidence: 0.40 (production-stable)" | tee -a "$MAIN_LOG"
echo "  • Caching: Enabled (masks)" | tee -a "$MAIN_LOG"
echo "======================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

SUCCESS=0
FAILED=0

# Process each scene
for idx in "${!SCENES[@]}"; do
    IFS="|" read -r SCENE_TYPE PRESET SOURCE_PATH <<< "${SCENES[$idx]}"
    BASENAME=$(basename "${SOURCE_PATH}" .tif | sed 's/_UltraQuality$//')
    NUM=$((idx + 1))

    echo "----------------------------------------------------------------------" | tee -a "$MAIN_LOG"
    echo "[$NUM/${#SCENES[@]}] Processing: ${BASENAME}" | tee -a "$MAIN_LOG"
    echo "    Type:   ${SCENE_TYPE}" | tee -a "$MAIN_LOG"
    echo "    Preset: ${PRESET}" | tee -a "$MAIN_LOG"
    echo "    Source: ${SOURCE_PATH}" | tee -a "$MAIN_LOG"
    echo "----------------------------------------------------------------------" | tee -a "$MAIN_LOG"

    if [ ! -f "${SOURCE_PATH}" ]; then
        echo "    ❌ SKIP: Source file not found" | tee -a "$MAIN_LOG"
        echo "" | tee -a "$MAIN_LOG"
        FAILED=$((FAILED + 1))
        continue
    fi

    SCENE_OUTPUT="${OUTPUT_DIR}/${SCENE_TYPE}"
    mkdir -p "${SCENE_OUTPUT}"

    SCENE_LOG="${LOG_DIR}/${BASENAME}_depth_${TIMESTAMP}.log"

    # Use balanced refinement for pool (avoid aggressive edge artifacts on water)
    REFINEMENT="balanced"
    if [[ "$BASENAME" == *Pool* ]]; then
        echo "    ℹ Pool scene detected - using conservative refinement" | tee -a "$MAIN_LOG"
        REFINEMENT="balanced"
    fi

    echo "    Starting APEX-100 depth generation..." | tee -a "$MAIN_LOG"

    if lux-depth-v2 \
        --input "${SOURCE_PATH}" \
        --output-dir "${SCENE_OUTPUT}" \
        --preset "${PRESET}" \
        "${DEPTH_CONFIG[@]}" \
        --refinement-preset "${REFINEMENT}" \
        2>&1 | tee "${SCENE_LOG}"; then

        echo "    ✅ SUCCESS: ${BASENAME}" | tee -a "$MAIN_LOG"
        SUCCESS=$((SUCCESS + 1))

        # Log key quality indicators from output
        if grep -q "MaterialsV2Engine initialized | backend=segformer" "${SCENE_LOG}"; then
            echo "       ✓ MaterialsV2 SegFormer backend confirmed" | tee -a "$MAIN_LOG"
        fi
        if grep -q "Guided filter applied" "${SCENE_LOG}"; then
            echo "       ✓ Guided filter refinement applied" | tee -a "$MAIN_LOG"
        fi
    else
        echo "    ❌ FAILED: ${BASENAME}" | tee -a "$MAIN_LOG"
        FAILED=$((FAILED + 1))
    fi

    echo "" | tee -a "$MAIN_LOG"
done

echo "======================================================================" | tee -a "$MAIN_LOG"
echo "APEX-100 Batch Complete" | tee -a "$MAIN_LOG"
echo "======================================================================" | tee -a "$MAIN_LOG"
echo "Total:   ${#SCENES[@]}" | tee -a "$MAIN_LOG"
echo "Success: ${SUCCESS}" | tee -a "$MAIN_LOG"
echo "Failed:  ${FAILED}" | tee -a "$MAIN_LOG"
echo "Output:  ${OUTPUT_DIR}" | tee -a "$MAIN_LOG"
echo "Logs:    ${LOG_DIR}" | tee -a "$MAIN_LOG"
echo "======================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

echo "Generated depth maps:" | tee -a "$MAIN_LOG"
find "${OUTPUT_DIR}" -type f \( \
    -iname "*depth*.tif" -o -iname "*depth*.tiff" -o \
    -iname "*master16*.tif" -o -iname "*upscaled16*.tif" \
\) -exec ls -lh {} \; | tee -a "$MAIN_LOG"

echo "" | tee -a "$MAIN_LOG"
echo "Master log: ${MAIN_LOG}" | tee -a "$MAIN_LOG"
