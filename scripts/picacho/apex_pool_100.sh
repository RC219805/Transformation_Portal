#!/bin/bash
# 100% APEX Pool Processing - Production Ready
# Commit: 021b3db (feat(lux-depth-v2): 100% APEX Quality - MaterialsV2 SegFormer Backend #628)

set -euo pipefail

INPUT="projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_Pool_UltraQuality.tif"
OUTPUT="750Picacho_Processed/apex_100_pool"

echo "=== 100% APEX Pool Processing ==="
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo ""

mkdir -p "$OUTPUT"

lux-depth-v2 \
  --allow-downloads \
  --input "$INPUT" \
  --output-dir "$OUTPUT" \
  --preset exterior_pool_apex_quality \
  --quality-tier apex \
  --device auto \
  --precision fp32 \
  --tile 1024 --tile-pad 32 \
  --seg-backend segformer \
  --seg-long-side 2048 \
  --materials-v2 \
  --materials-v2-backend segformer \
  --max-segmentation-side 2048 \
  --edge-refinement --refinement-preset aggressive \
  --cache-masks --model-cache --depth-cache \
  --marketing-png-compression 0 \
  --tiff-compression lzw \
  2>&1 | tee "$OUTPUT/processing.log"

echo ""
echo "✅ Complete. Check log for:"
echo "   - MaterialsV2Engine initialized | backend=segformer"
echo "   - Loading segmentation model: segformer"
echo ""
