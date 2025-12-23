#!/bin/bash
# Phase A Isolation Test - Forensics Mode Validation
# Purpose: Verify clean baseline rendering with no upscaling or post-tiling

set -e  # Exit on error

INPUT_IMAGE="projects/750_picacho_lane/Kitchen_Only_Test/750Picacho_Kitchen_UltraQuality.tif"
OUTPUT_DIR="phase_a_forensics_baseline"
PRESET="interior_luxury"

echo "═══════════════════════════════════════════════════════════════"
echo "Phase A Isolation Test - Forensics Mode"
echo "═══════════════════════════════════════════════════════════════"
echo "Input:  $INPUT_IMAGE"
echo "Output: $OUTPUT_DIR"
echo "Preset: $PRESET"
echo "Mode:   --master16-only (forensics baseline)"
echo ""

# Clean previous run
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Run forensics mode
echo "Running forensics mode..."
python -m lux_depth_v2.cli \
  --input "$INPUT_IMAGE" \
  --output-dir "$OUTPUT_DIR" \
  --preset "$PRESET" \
  --master16-only 2>&1 | tee "$OUTPUT_DIR/phase_a_run.log"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Validation Checks"
echo "═══════════════════════════════════════════════════════════════"

# Extract validation metrics
REPORT="$OUTPUT_DIR/750Picacho_Kitchen_UltraQuality_report.json"

if [ ! -f "$REPORT" ]; then
    echo "❌ FAILED: Report file not found"
    exit 1
fi

# Parse report
POST_TILE=$(cat "$REPORT" | python -m json.tool | grep '"post_tile"' | head -1 | awk '{print $2}' | tr -d ',')
UPSCALE=$(cat "$REPORT" | python -m json.tool | grep '"upscale"' | head -1 | awk '{print $2}' | tr -d ',')
UPSCALER=$(cat "$REPORT" | python -m json.tool | grep '"upscaler"' | head -1 | awk '{print $2}' | tr -d ',"')
TIMING=$(cat "$REPORT" | python -m json.tool | grep '"timing_s"' | head -1 | awk '{print $2}' | tr -d ',')

# Count output files (should be 2: master16.tif + report.json)
OUTPUT_COUNT=$(ls -1 "$OUTPUT_DIR" | grep -v "phase_a_run.log" | wc -l | xargs)

echo "post_tile:    $POST_TILE (expected: 0)"
echo "upscale:      $UPSCALE (expected: 1)"
echo "upscaler:     $UPSCALER (expected: none)"
echo "timing_s:     $TIMING (expected: <2)"
echo "output_files: $OUTPUT_COUNT (expected: 2 - master16.tif + report.json)"
echo ""

# Validation
FAILED=0

if [ "$POST_TILE" != "0" ]; then
    echo "❌ FAILED: post_tile=$POST_TILE (expected 0)"
    FAILED=1
else
    echo "✅ PASS: post_tile=0"
fi

if [ "$UPSCALE" != "1" ]; then
    echo "❌ FAILED: upscale=$UPSCALE (expected 1)"
    FAILED=1
else
    echo "✅ PASS: upscale=1"
fi

if [ "$UPSCALER" != "none" ]; then
    echo "❌ FAILED: upscaler=$UPSCALER (expected none)"
    FAILED=1
else
    echo "✅ PASS: upscaler=none"
fi

# Check timing (<2 seconds for minimal processing)
TIMING_NUMERIC=$(echo "$TIMING" | bc 2>/dev/null || echo "$TIMING")
if [ $(echo "$TIMING_NUMERIC > 2.0" | bc 2>/dev/null || echo "0") -eq 1 ]; then
    echo "⚠️  WARNING: timing=$TIMING (expected <2s) - may indicate overhead"
else
    echo "✅ PASS: timing=$TIMING (<2s)"
fi

if [ "$OUTPUT_COUNT" -ne 2 ]; then
    echo "❌ FAILED: output_files=$OUTPUT_COUNT (expected 2)"
    FAILED=1
else
    echo "✅ PASS: output_files=2 (master16.tif + report.json only)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
if [ $FAILED -eq 0 ]; then
    echo "✅ PHASE A VALIDATION PASSED"
    echo "═══════════════════════════════════════════════════════════════"
    echo "Forensics mode is working correctly."
    echo "Clean baseline established for Phase 1 sweep."
    exit 0
else
    echo "❌ PHASE A VALIDATION FAILED"
    echo "═══════════════════════════════════════════════════════════════"
    echo "Forensics mode is NOT working correctly."
    echo "Review logs and investigate control-plane issues."
    exit 1
fi
