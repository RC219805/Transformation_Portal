#!/bin/bash
# Run 18-Image Validation with HF-Energy Texture Gate Fix
# Tests the not-flat + HF-energy texture scene validation

set -e

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMMIT_SHA=$(git rev-parse --short HEAD)
OUTPUT_DIR="outputs/validation_hf_fixed_${TIMESTAMP}_${COMMIT_SHA}"
INPUT_DIR="data/validation_expanded"

echo "=================================================="
echo "18-Image Validation - HF Energy Texture Fix"
echo "=================================================="
echo ""
echo "Commit:  $COMMIT_SHA"
echo "Input:   $INPUT_DIR"
echo "Output:  $OUTPUT_DIR"
echo ""

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Count images
IMAGE_COUNT=$(find "$INPUT_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.JPG" -o -name "*.JPEG" -o -name "*.PNG" \) | wc -l | tr -d ' ')
echo "Found $IMAGE_COUNT images to process"
echo ""

if [ "$IMAGE_COUNT" -ne 18 ]; then
    echo "⚠️  Warning: Expected 18 images, found $IMAGE_COUNT"
    echo "Continue? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Aborted"
        exit 1
    fi
fi

# Run validation
echo "Starting validation with HF-energy texture gates..."
echo ""

python scripts/automation/production_depth_validation_fixed.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --tile-size 1024 \
  --overlap 128 \
  --device auto

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Validation Complete"
    echo "=================================================="
    echo ""
    echo "Results: $OUTPUT_DIR"
    echo ""
    
    # Show summary if it exists
    if [ -f "$OUTPUT_DIR/validation_report.json" ]; then
        echo "Pass rates:"
        python3 -c "
import json
with open('$OUTPUT_DIR/validation_report.json') as f:
    data = json.load(f)
    lenient = data.get('quality', {}).get('lenient', {})
    strict = data.get('quality', {}).get('strict', {})
    print(f\"  Lenient: {lenient.get('passed', 0)}/{data.get('total_images', 0)} ({lenient.get('pass_rate', 0)*100:.1f}%)\")
    print(f\"  Strict:  {strict.get('passed', 0)}/{data.get('total_images', 0)} ({strict.get('pass_rate', 0)*100:.1f}%)\")
" 2>/dev/null || echo "  (Summary parsing failed)"
    fi
    
    echo ""
    echo "Next steps:"
    echo "  1. Review: cat $OUTPUT_DIR/validation_report.json | jq '.quality'"
    echo "  2. Analyze: python scripts/analyze_validation_v2.py $OUTPUT_DIR"
    echo "  3. Compare: diff old vs new pass rates"
    echo ""
else
    echo ""
    echo "❌ Validation failed"
    exit 1
fi
