#!/bin/bash
# ============================================================================
# 750 Picacho - APEX Quality Batch Processing
# ============================================================================
# Quality Level: 85% APEX (Production-Ready)
# - Depth: 100% APEX (FP32, 1024px tiles, guided filter)
# - Scene Seg: 100% APEX (SegFormer-B5 @ 2048px)
# - Materials: 60% APEX (heuristic backend - still excellent)
# - Export: 100% APEX (lossless PNG, LZW TIFF)
# ============================================================================

set -e  # Exit on error
cd /Users/rc/Transformation_Portal

# Output directory
OUTPUT_BASE="750Picacho_Processed/apex_production_batch"
mkdir -p "$OUTPUT_BASE"

# Log file
LOG_FILE="750picacho_apex_batch_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================" | tee -a "$LOG_FILE"
echo "750 Picacho APEX Production Batch" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Counter
TOTAL=0
SUCCESS=0
FAILED=0

# Process each TIFF
for file in 750Picacho_Source_TIFFs/*.tif*; do
    if [ ! -f "$file" ]; then
        continue
    fi

    TOTAL=$((TOTAL + 1))
    basename=$(basename "$file" .tif)
    basename=$(basename "$basename" .tiff)

    echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "[$TOTAL] Processing: $basename" | tee -a "$LOG_FILE"
    echo "------------------------------------------------------------" | tee -a "$LOG_FILE"

    OUTPUT_DIR="$OUTPUT_BASE/$basename"

    if lux-depth-v2 \
      --input "$file" \
      --output-dir "$OUTPUT_DIR" \
      --preset interior_luxury_max_quality \
      --quality-tier apex \
      --intent hero \
      --device auto \
      --precision fp32 \
      --tile 1024 \
      --tile-pad 32 \
      --seg-backend segformer \
      --seg-long-side 2048 \
      --seg-min-conf 0.15 \
      --materials-v2 \
      --confidence-threshold 0.3 \
      --max-segmentation-side 2048 \
      --edge-refinement \
      --refinement-preset aggressive \
      --cache-masks \
      --model-cache \
      --depth-cache \
      --tiff-compression lzw \
      --marketing-png-compression 0 2>&1 | tee -a "$LOG_FILE"
    then
        SUCCESS=$((SUCCESS + 1))
        echo "✅ SUCCESS: $basename" | tee -a "$LOG_FILE"
    else
        FAILED=$((FAILED + 1))
        echo "❌ FAILED: $basename" | tee -a "$LOG_FILE"
    fi

    echo "" | tee -a "$LOG_FILE"
done

echo "============================================================" | tee -a "$LOG_FILE"
echo "Batch Complete!" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Summary:" | tee -a "$LOG_FILE"
echo "  Total:   $TOTAL" | tee -a "$LOG_FILE"
echo "  Success: $SUCCESS" | tee -a "$LOG_FILE"
echo "  Failed:  $FAILED" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Calculate total output size
if [ -d "$OUTPUT_BASE" ]; then
    TOTAL_SIZE=$(du -sh "$OUTPUT_BASE" | cut -f1)
    echo "" | tee -a "$LOG_FILE"
    echo "Total Output Size: $TOTAL_SIZE" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
