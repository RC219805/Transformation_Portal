#!/usr/bin/env bash
# Full 18-image validation with fail-fast post-check

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SHA=$(git rev-parse --short HEAD)
OUTPUT_DIR="outputs/validation_v2_${TIMESTAMP}_${SHA}"

echo "▶ Running 18-image validation"
echo "  Output: $OUTPUT_DIR"
echo "  Commit: $SHA"

# Run validation
python scripts/automation/production_depth_validation_fixed.py \
  --input-dir data/validation_expanded \
  --output-dir "$OUTPUT_DIR" \
  --tile-size 1024 \
  --overlap 192

# Post-check: verify no null metrics
echo ""
echo "▶ Post-check: verifying metrics complete"
NULL_COUNT=0
for json in "$OUTPUT_DIR"/*_metrics.json; do
  if [ -f "$json" ]; then
    if grep -q '"scene_type": null' "$json" || \
       grep -q '"edge_f1": null' "$json" || \
       grep -q '"lenient_pass": null' "$json"; then
      echo "❌ FAIL: Null metrics in $(basename $json)"
      NULL_COUNT=$((NULL_COUNT + 1))
    fi
  fi
done

if [ $NULL_COUNT -gt 0 ]; then
  echo ""
  echo "❌ VALIDATION FAILED: $NULL_COUNT files contain null metrics"
  echo "This is a regression - validation script not calling V2 classifier"
  exit 1
fi

echo "✅ All metrics complete (no nulls)"
echo ""
echo "Next steps:"
echo "  1. Generate confusion matrix:"
echo "     python scripts/validation/generate_confusion_matrix.py --output-dir $OUTPUT_DIR"
echo "  2. Review: $OUTPUT_DIR/validation_summary.json"
