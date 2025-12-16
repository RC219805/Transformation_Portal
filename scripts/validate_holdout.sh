#!/bin/bash
# Holdout Set Validation Runner
# Usage: ./scripts/validate_holdout.sh [output_file]

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "  Water Detection Holdout Validation"
echo "========================================="
echo ""

# Check for holdout directory
if [ -z "$WATER_HOLDOUT_DIR" ]; then
    echo -e "${RED}Error: WATER_HOLDOUT_DIR environment variable not set${NC}"
    echo ""
    echo "Usage:"
    echo "  export WATER_HOLDOUT_DIR=/path/to/holdout/images"
    echo "  ./scripts/validate_holdout.sh [output_file]"
    echo ""
    exit 1
fi

# Check directory exists
if [ ! -d "$WATER_HOLDOUT_DIR" ]; then
    echo -e "${RED}Error: Holdout directory not found: $WATER_HOLDOUT_DIR${NC}"
    echo ""
    exit 1
fi

# Output file (default: holdout_validation.json)
OUTPUT="${1:-holdout_validation_v1.json}"

# Ground truth manifest path
MANIFEST="data/water_v0/holdout_manifest.json"

# Check manifest exists
if [ ! -f "$MANIFEST" ]; then
    echo -e "${RED}Error: Holdout manifest not found: $MANIFEST${NC}"
    echo ""
    exit 1
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Holdout dir: $WATER_HOLDOUT_DIR"
echo "  Manifest:    $MANIFEST"
echo "  Output:      $OUTPUT"
echo ""

# Run validation
echo -e "${YELLOW}Running validation...${NC}"
python scripts/prw_water_validation.py \
  --ground-truth "$MANIFEST" \
  --output "$OUTPUT" \
  --seed 42 \
  --verbose

# Check if validation succeeded
if [ $? -ne 0 ]; then
    echo -e "${RED}Validation failed${NC}"
    exit 1
fi

echo ""
echo "========================================="
echo "  Holdout Validation Results"
echo "========================================="

# Extract summary metrics (check if jq is available)
if command -v jq &> /dev/null; then
    echo ""
    jq '.summary | {
      total_images,
      detected_count,
      false_trigger_count,
      false_trigger_rate,
      average_confidence
    }' "$OUTPUT"
    
    echo ""
    echo -e "${YELLOW}Acceptance Gates Check:${NC}"
    
    # Extract false trigger rate
    FT_RATE=$(jq -r '.summary.false_trigger_rate' "$OUTPUT")
    
    # Check acceptance gates (≤5% = 0.05)
    if (( $(echo "$FT_RATE > 0.05" | bc -l 2>/dev/null || echo "1") )); then
        echo -e "${RED}⚠️  FAILED: Holdout FT rate ($FT_RATE) exceeds 5% threshold${NC}"
        echo ""
        echo "False triggers detected:"
        jq -r '.results[] | select(.detected == true) | "  - \(.image_path): confidence=\(.confidence_final), tags=\(.tags | join(", "))"' "$OUTPUT"
        echo ""
        echo "Review suppressor telemetry for explanations:"
        echo "  jq '.results[] | select(.detected == true) | .suppressor_telemetry' $OUTPUT"
        echo ""
        exit 1
    else
        echo -e "${GREEN}✅ PASSED: Holdout FT rate ($FT_RATE) within acceptable range (≤5%)${NC}"
    fi
    
    # Check if any false triggers exist (require telemetry review)
    FT_COUNT=$(jq -r '.summary.false_trigger_count' "$OUTPUT")
    if [ "$FT_COUNT" -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}Note: $FT_COUNT false trigger(s) detected (within tolerance)${NC}"
        echo "Review telemetry for explainability:"
        echo "  jq '.results[] | select(.detected == true)' $OUTPUT"
    fi
else
    echo ""
    echo -e "${YELLOW}Warning: jq not found. Install jq for detailed metrics parsing.${NC}"
    echo "Results saved to: $OUTPUT"
fi

echo ""
echo "========================================="
echo -e "${GREEN}Holdout validation complete${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Review results: jq '.summary' $OUTPUT"
echo "  2. Check false triggers: jq '.results[] | select(.detected == true)' $OUTPUT"
echo "  3. Review telemetry: jq '.results[].suppressor_telemetry' $OUTPUT"
echo "  4. If passed: proceed with baseline v2 promotion"
echo ""
