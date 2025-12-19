#!/bin/bash
# Run validation on expanded 18-image dataset

set -euo pipefail

OUTPUT_DIR="outputs/validation_expanded_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Starting validation on 18-image expanded dataset..."
echo "Output directory: $OUTPUT_DIR"
echo ""

python scripts/automation/production_depth_validation_fixed.py \
  --input-dir data/validation_expanded \
  --output-dir "$OUTPUT_DIR" \
  --tile-size 1024 \
  --overlap 192

echo ""
echo "Validation complete. Results in: $OUTPUT_DIR"
