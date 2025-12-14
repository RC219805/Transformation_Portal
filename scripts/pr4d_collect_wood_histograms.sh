#!/bin/bash
# PR-4D Wood Material Data Collection Script
# Batch processes 4 scenes to collect histogram data for wood pixel ops implementation
# Usage: bash scripts/pr4d_collect_wood_histograms.sh

set -e

# Configuration
SCENES=(
  "750Picacho_Kitchen_UltraQuality.tif"
  "750Picacho_PrimaryBedroom_UltraQuality.tif"
  "750Picacho_GreatRoom_UltraQuality.tif"
  "750Picacho_Pool_UltraQuality.tif"
)

INPUT_DIR="projects/750_picacho_lane/Final_Production_UltraQuality"
OUTPUT_BASE="outputs/pr4d_wood_data"
PRESET="interior_luxury_apex_quality_materials_v3_glass"
DEVICE="auto"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Metadata
START_TIME=$(date +%s)
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
HOSTNAME=$(hostname)

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}PR-4D Wood Material Data Collection${NC}"
echo -e "${BLUE}==================================================${NC}"
echo "Started: $TIMESTAMP"
echo "Git commit: $GIT_COMMIT"
echo "Device: $DEVICE"
echo "Hostname: $HOSTNAME"
echo "Preset: $PRESET"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Summary tracking
PROCESSED_SCENES=()
SKIPPED_SCENES=()
FAILED_SCENES=()
SCENE_TIMES=()

# Process each scene
for SCENE in "${SCENES[@]}"; do
  SCENE_NAME="${SCENE%.*}"  # Remove extension
  INPUT_FILE="$INPUT_DIR/$SCENE"
  OUTPUT_DIR="$OUTPUT_BASE/$SCENE_NAME"
  
  echo -e "${BLUE}--------------------------------------------------${NC}"
  echo -e "${BLUE}Processing: $SCENE_NAME${NC}"
  echo -e "${BLUE}--------------------------------------------------${NC}"
  
  # Check if input file exists
  if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}❌ ERROR: Input file not found: $INPUT_FILE${NC}"
    FAILED_SCENES+=("$SCENE_NAME (file not found)")
    echo ""
    continue
  fi
  
  # Check if already processed
  REPORT_FILE="$OUTPUT_DIR/${SCENE_NAME}_report.json"
  if [ -f "$REPORT_FILE" ]; then
    echo -e "${YELLOW}⚠️  Scene already processed (report.json exists)${NC}"
    echo "   Using existing results at: $OUTPUT_DIR"
    SKIPPED_SCENES+=("$SCENE_NAME (cached)")
    echo ""
    continue
  fi
  
  # Process the scene
  SCENE_START=$(date +%s)
  echo "Input: $INPUT_FILE"
  echo "Output: $OUTPUT_DIR"
  echo ""
  
  if lux-depth-v2 \
    --input "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --preset "$PRESET" \
    --device "$DEVICE" \
    --allow-canary; then
    
    SCENE_END=$(date +%s)
    SCENE_DURATION=$((SCENE_END - SCENE_START))
    SCENE_TIMES+=("$SCENE_NAME: ${SCENE_DURATION}s")
    
    echo -e "${GREEN}✓ Successfully processed $SCENE_NAME in ${SCENE_DURATION}s${NC}"
    PROCESSED_SCENES+=("$SCENE_NAME")
  else
    echo -e "${RED}❌ ERROR: Processing failed for $SCENE_NAME${NC}"
    FAILED_SCENES+=("$SCENE_NAME (processing error)")
  fi
  
  echo ""
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

# Generate summary report
SUMMARY_FILE="$OUTPUT_BASE/pr4d_collection_summary.txt"

cat > "$SUMMARY_FILE" << EOF
PR-4D Wood Material Data Collection Summary
============================================

Collection Metadata:
-------------------
Timestamp: $TIMESTAMP
Git Commit: $GIT_COMMIT
Hostname: $HOSTNAME
Device: $DEVICE
Preset: $PRESET
Total Duration: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s

Scenes Processed (${#PROCESSED_SCENES[@]}):
-----------------
EOF

for scene in "${PROCESSED_SCENES[@]}"; do
  echo "  ✓ $scene" >> "$SUMMARY_FILE"
done

if [ ${#SKIPPED_SCENES[@]} -gt 0 ]; then
  cat >> "$SUMMARY_FILE" << EOF

Scenes Skipped (${#SKIPPED_SCENES[@]}):
---------------
EOF
  for scene in "${SKIPPED_SCENES[@]}"; do
    echo "  ⚠ $scene" >> "$SUMMARY_FILE"
  done
fi

if [ ${#FAILED_SCENES[@]} -gt 0 ]; then
  cat >> "$SUMMARY_FILE" << EOF

Scenes Failed (${#FAILED_SCENES[@]}):
--------------
EOF
  for scene in "${FAILED_SCENES[@]}"; do
    echo "  ✗ $scene" >> "$SUMMARY_FILE"
  done
fi

cat >> "$SUMMARY_FILE" << EOF

Processing Times:
-----------------
EOF

for time_entry in "${SCENE_TIMES[@]}"; do
  echo "  $time_entry" >> "$SUMMARY_FILE"
done

cat >> "$SUMMARY_FILE" << EOF

Output Locations:
-----------------
  Base Directory: $OUTPUT_BASE/
  Summary Report: $SUMMARY_FILE
  Scene Outputs:
EOF

for scene in "${PROCESSED_SCENES[@]}" "${SKIPPED_SCENES[@]%%(*}"; do
  scene_clean=$(echo "$scene" | sed 's/ (cached)//')
  if [ -d "$OUTPUT_BASE/$scene_clean" ]; then
    echo "    - $OUTPUT_BASE/$scene_clean/" >> "$SUMMARY_FILE"
  fi
done

cat >> "$SUMMARY_FILE" << EOF

Next Steps:
-----------
1. Review histogram data: python scripts/pr4d_aggregate_histograms.py
2. Analyze material recommendations in generated markdown
3. Implement top-ranked material pixel ops in lux_depth_v2/materials_v3_pixel_ops.py

EOF

# Print summary to console
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}Collection Summary${NC}"
echo -e "${BLUE}==================================================${NC}"
cat "$SUMMARY_FILE"

# Exit status
if [ ${#FAILED_SCENES[@]} -gt 0 ]; then
  echo -e "${YELLOW}⚠️  Collection completed with ${#FAILED_SCENES[@]} failure(s)${NC}"
  exit 1
else
  echo -e "${GREEN}✓ Collection completed successfully!${NC}"
  exit 0
fi
