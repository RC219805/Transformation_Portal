#!/usr/bin/env bash
set -euo pipefail

# ========================================
# APEX Depth Map Generation - DA3 Large v1.1
# Exports: mini_npz
# ========================================

OUTPUT_DIR="750Picacho_Depth_Maps_DAV3_1.1_APEX"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="depth_generation_dav3_1.1_apex_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_DIR"

echo "========================================" | tee "$LOG_FILE"
echo "APEX Depth Map Generation - DA3 Large v1.1" | tee -a "$LOG_FILE"
echo "Exports: mini_npz" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Explicit file map with scene types
declare -a ITEMS=(
  "interior|750Picacho_Source_TIFFs/750Picacho_GreatRoom.tif"
  "interior|750Picacho_Source_TIFFs/750Picacho_Kitchen.tif"
  "interior|750Picacho_Source_TIFFs/750Picacho_PrimaryBathroom.tif"
  "interior|750Picacho_Source_TIFFs/750Picacho_PrimaryBedroom.tif"
  "exterior|750Picacho_Source_TIFFs/750Picacho_Aerial.tif"
  "exterior|projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_Pool_UltraQuality.tif"
)

TOTAL="${#ITEMS[@]}"
SUCCESS=0
FAILED=0

echo "Total files to process: $TOTAL" | tee -a "$LOG_FILE"
echo "Model: large-v1.1" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for idx in "${!ITEMS[@]}"; do
  IFS="|" read -r SCENE_TYPE INPUT_PATH <<< "${ITEMS[$idx]}"
  NUM=$((idx + 1))

  echo "----------------------------------------" | tee -a "$LOG_FILE"
  echo "[$NUM/$TOTAL] Processing: $(basename "$INPUT_PATH")" | tee -a "$LOG_FILE"
  echo "Scene type: $SCENE_TYPE" | tee -a "$LOG_FILE"
  echo "Input:  $INPUT_PATH" | tee -a "$LOG_FILE"
  echo "----------------------------------------" | tee -a "$LOG_FILE"

  if [[ ! -f "$INPUT_PATH" ]]; then
    echo "❌ ERROR: File not found: $INPUT_PATH" | tee -a "$LOG_FILE"
    FAILED=$((FAILED + 1))
    echo "" | tee -a "$LOG_FILE"
    continue
  fi

  SCENE_OUTPUT="${OUTPUT_DIR}/${SCENE_TYPE}"
  mkdir -p "$SCENE_OUTPUT"

  echo "Starting DA3 depth generation..." | tee -a "$LOG_FILE"

  # Use lux-depth-v3 DA3 API processing (see: lux-depth-v3 api-process --help)
  if python -m lux_depth_v3.cli api-process \
      "$INPUT_PATH" \
      --output-dir "$SCENE_OUTPUT" \
      --model "large-v1.1" \
      --export-format "mini_npz" \
      --device "auto" \
      --process-res 2048 \
      2>&1 | tee -a "$LOG_FILE"; then

    echo "✅ SUCCESS: $(basename "$INPUT_PATH")" | tee -a "$LOG_FILE"
    SUCCESS=$((SUCCESS + 1))
  else
    echo "❌ FAILED: $(basename "$INPUT_PATH")" | tee -a "$LOG_FILE"
    FAILED=$((FAILED + 1))
  fi

  echo "" | tee -a "$LOG_FILE"
done

echo "========================================" | tee -a "$LOG_FILE"
echo "Batch Processing Complete" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Total:   $TOTAL" | tee -a "$LOG_FILE"
echo "Success: $SUCCESS" | tee -a "$LOG_FILE"
echo "Failed:  $FAILED" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Generated outputs:" | tee -a "$LOG_FILE"
find "$OUTPUT_DIR" -type f \( -name "*.npz" -o -name "*.png" -o -name "*.npy" \) -exec ls -lh {} \; | tee -a "$LOG_FILE"
