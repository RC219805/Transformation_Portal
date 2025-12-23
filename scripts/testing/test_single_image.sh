#!/bin/bash
# Single Image Test - Phase 1 Pipeline Verification
# Tests pipeline on one image before full sweep execution

set -euo pipefail

echo "╔════════════════════════════════════════════════════════════╗"
echo "║      Phase 1 Single-Image Pipeline Verification           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
TEST_IMAGE="/Users/rc/Transformation_Portal/750Picacho_Source_TIFFs/750Picacho_Pool_16bit.tiff"
TEST_OUTPUT="sweep_runs/phase1_test_single_image"
PRESET="interior_luxury"

# Check if image exists
if [ ! -f "$TEST_IMAGE" ]; then
    echo "❌ Test image not found: $TEST_IMAGE"
    exit 1
fi

echo "✓ Test image: $(basename $TEST_IMAGE)"
echo "✓ Output directory: $TEST_OUTPUT"
echo "✓ Preset: $PRESET"
echo ""

# Create output directory
mkdir -p "$TEST_OUTPUT"

# Display image info
echo "📊 Image Information:"
file "$TEST_IMAGE"
identify "$TEST_IMAGE" 2>/dev/null || echo "  (identify command not available)"
echo ""

# Run pipeline test
echo "🚀 Running pipeline test..."
echo ""

START_TIME=$(date +%s)

python lux_depth_v2/test_750_picacho.py \
    --preset "$PRESET" \
    --device cpu \
    --output-dir "$TEST_OUTPUT" <<EOF || {
    echo ""
    echo "❌ Pipeline test failed!"
    exit 1
}
EOF

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Pipeline Test Complete"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Duration: ${DURATION} seconds"
echo ""
echo "Output files:"
ls -lh "$TEST_OUTPUT" | tail -20
echo ""
echo "Total output size: $(du -sh $TEST_OUTPUT | cut -f1)"
echo ""

# Verify outputs
echo "Verifying outputs..."
MASTER_TIFFS=$(find "$TEST_OUTPUT" -name "*_master16.tif" | wc -l | tr -d ' ')
REPORTS=$(find "$TEST_OUTPUT" -name "*_report.json" | wc -l | tr -d ' ')

echo "  Master TIFFs: $MASTER_TIFFS"
echo "  JSON reports: $REPORTS"
echo ""

if [ "$MASTER_TIFFS" -gt 0 ] && [ "$REPORTS" -gt 0 ]; then
    echo "✅ Test successful - pipeline is working correctly!"
    echo ""
    echo "Sample report:"
    cat $(find "$TEST_OUTPUT" -name "*_report.json" | head -1) | python -m json.tool | head -40
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo "Ready to run full Phase 1:"
    echo "  bash exploration/phase1_live_monitor.sh --all"
    echo "══════════════════════════════════════════════════════════"
else
    echo "⚠️  Test completed but outputs may be incomplete"
fi
