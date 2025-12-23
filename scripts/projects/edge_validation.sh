#!/bin/bash
# Run 7-Image Validation with Structure-Aware Edge Gating
# This script runs the production validation suite on the quick validation dataset

set -e

# Configuration
OUTPUT_DIR="outputs/validation_structure_edges_$(date +%Y%m%d_%H%M%S)"
INPUT_DIR="data/validation_quick"

echo "=================================================="
echo "7-Image Validation with Structure-Aware Edges"
echo "=================================================="
echo ""
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Count images
IMAGE_COUNT=$(find "$INPUT_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.tif" -o -name "*.tiff" \) | wc -l | tr -d ' ')
echo "Found $IMAGE_COUNT images to process"
echo ""

# Run validation
echo "Starting validation..."
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
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Summary files:"
    find "$OUTPUT_DIR" -name "summary*.json" -o -name "validation_summary.json" 2>/dev/null
    echo ""
    echo "To view results:"
    echo "  cat $OUTPUT_DIR/validation_summary.json | jq ."
    echo ""
    echo "To compare with baseline:"
    echo "  python generate_validation_report.py \\"
    echo "    --baseline outputs/validation_resolution_policy_* \\"
    echo "    --current $OUTPUT_DIR \\"
    echo "    --output structure_edges_comparison.md"
else
    echo ""
    echo "❌ Validation failed"
    exit 1
fi
