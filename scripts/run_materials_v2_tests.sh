#!/bin/bash
# Materials v2 Production Testing Suite
# Runs comprehensive tests on 750 Picacho images

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "========================================"
echo "Materials v2 Production Testing Suite"
echo "========================================"
echo ""

# Configuration
INPUT_DIR="input_images/750_Picacho/Optimized_TIFFs"
PRESET="photo_realistic"
UPSCALE=2
DEVICE="auto"

# Output directories
OUTPUT_BASE="output_Materials_V2_Tests_$(date +%Y%m%d_%H%M%S)"
BASELINE_DIR="${OUTPUT_BASE}/Baseline"
ENHANCED_DIR="${OUTPUT_BASE}/Enhanced_0.6"
CONSERVATIVE_DIR="${OUTPUT_BASE}/Conservative_0.8"
POOL_TEST_DIR="${OUTPUT_BASE}/Pool_Edge_Case"
BATHROOM_TEST_DIR="${OUTPUT_BASE}/Bathroom_Edge_Case"
KITCHEN_TEST_DIR="${OUTPUT_BASE}/Kitchen_Edge_Case"

# Cache directory
CACHE_DIR=".materials_v2_cache"

echo "Input directory: $INPUT_DIR"
echo "Output base: $OUTPUT_BASE"
echo ""

# Check input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Count test images
IMAGE_COUNT=$(find "$INPUT_DIR" -name "*.tif" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "Found $IMAGE_COUNT test images"
echo ""

# ========================================
# Test 1: Baseline (Materials v2 disabled)
# ========================================
echo "========================================"
echo "Test 1: Baseline (Materials v2 OFF)"
echo "========================================"
echo ""

python3 -m lux_depth_v2.cli \
    --input-dir "$INPUT_DIR" \
    --output-dir "$BASELINE_DIR" \
    --preset "$PRESET" \
    --device "$DEVICE" \
    --upscale $UPSCALE \
    2>&1 | tee "${OUTPUT_BASE}/baseline_log.txt"

echo ""
echo "✓ Baseline test complete"
echo ""

# ========================================
# Test 2: Materials v2 Enabled (0.6)
# ========================================
echo "========================================"
echo "Test 2: Materials v2 (confidence=0.6)"
echo "========================================"
echo ""

python3 -m lux_depth_v2.cli \
    --input-dir "$INPUT_DIR" \
    --output-dir "$ENHANCED_DIR" \
    --preset "$PRESET" \
    --device "$DEVICE" \
    --upscale $UPSCALE \
    --materials-v2 \
    --confidence-threshold 0.6 \
    --cache-masks \
    --cache-dir "$CACHE_DIR" \
    2>&1 | tee "${OUTPUT_BASE}/enhanced_log.txt"

echo ""
echo "✓ Materials v2 (0.6) test complete"
echo ""

# ========================================
# Test 3: Materials v2 Conservative (0.8)
# ========================================
echo "========================================"
echo "Test 3: Materials v2 (confidence=0.8)"
echo "========================================"
echo ""

python3 -m lux_depth_v2.cli \
    --input-dir "$INPUT_DIR" \
    --output-dir "$CONSERVATIVE_DIR" \
    --preset "$PRESET" \
    --device "$DEVICE" \
    --upscale $UPSCALE \
    --materials-v2 \
    --confidence-threshold 0.8 \
    --cache-masks \
    --cache-dir "$CACHE_DIR" \
    2>&1 | tee "${OUTPUT_BASE}/conservative_log.txt"

echo ""
echo "✓ Materials v2 (0.8) test complete"
echo ""

# ========================================
# Test 4: Pool Edge Case (Water)
# ========================================
echo "========================================"
echo "Test 4: Pool Edge Case (Water)"
echo "========================================"
echo ""

POOL_IMAGE=$(find "$INPUT_DIR" -name "*Pool*.tif" -type f | head -1)

if [ -n "$POOL_IMAGE" ]; then
    echo "Testing: $POOL_IMAGE"
    
    python3 -m lux_depth_v2.cli \
        --input "$POOL_IMAGE" \
        --output-dir "$POOL_TEST_DIR" \
        --preset "$PRESET" \
        --device "$DEVICE" \
        --upscale $UPSCALE \
        --materials-v2 \
        --confidence-threshold 0.6 \
        --cache-masks \
        --cache-dir "$CACHE_DIR" \
        2>&1 | tee "${OUTPUT_BASE}/pool_test_log.txt"
    
    echo ""
    echo "✓ Pool edge case test complete"
else
    echo "Warning: Pool image not found"
fi
echo ""

# ========================================
# Test 5: Bathroom Edge Case (Glass/Stone)
# ========================================
echo "========================================"
echo "Test 5: Bathroom Edge Case (Glass/Stone)"
echo "========================================"
echo ""

BATHROOM_IMAGE=$(find "$INPUT_DIR" -name "*Bathroom*.tif" -type f | head -1)

if [ -n "$BATHROOM_IMAGE" ]; then
    echo "Testing: $BATHROOM_IMAGE"
    
    python3 -m lux_depth_v2.cli \
        --input "$BATHROOM_IMAGE" \
        --output-dir "$BATHROOM_TEST_DIR" \
        --preset "$PRESET" \
        --device "$DEVICE" \
        --upscale $UPSCALE \
        --materials-v2 \
        --confidence-threshold 0.7 \
        --cache-masks \
        --cache-dir "$CACHE_DIR" \
        2>&1 | tee "${OUTPUT_BASE}/bathroom_test_log.txt"
    
    echo ""
    echo "✓ Bathroom edge case test complete"
else
    echo "Warning: Bathroom image not found"
fi
echo ""

# ========================================
# Test 6: Kitchen Edge Case (Mixed)
# ========================================
echo "========================================"
echo "Test 6: Kitchen Edge Case (Mixed)"
echo "========================================"
echo ""

KITCHEN_IMAGE=$(find "$INPUT_DIR" -name "*Kitchen*.tif" -type f | head -1)

if [ -n "$KITCHEN_IMAGE" ]; then
    echo "Testing: $KITCHEN_IMAGE"
    
    python3 -m lux_depth_v2.cli \
        --input "$KITCHEN_IMAGE" \
        --output-dir "$KITCHEN_TEST_DIR" \
        --preset "$PRESET" \
        --device "$DEVICE" \
        --upscale $UPSCALE \
        --materials-v2 \
        --confidence-threshold 0.65 \
        --cache-masks \
        --cache-dir "$CACHE_DIR" \
        2>&1 | tee "${OUTPUT_BASE}/kitchen_test_log.txt"
    
    echo ""
    echo "✓ Kitchen edge case test complete"
else
    echo "Warning: Kitchen image not found"
fi
echo ""

# ========================================
# Quality Comparison
# ========================================
echo "========================================"
echo "Analyzing Quality Differences"
echo "========================================"
echo ""

if [ -d "$BASELINE_DIR" ] && [ -d "$ENHANCED_DIR" ]; then
    python3 scripts/compare_materials_quality.py \
        --baseline-dir "$BASELINE_DIR" \
        --enhanced-dir "$ENHANCED_DIR" \
        --cache-dir "$CACHE_DIR" \
        --output "${OUTPUT_BASE}/quality_report.json" \
        2>&1 | tee "${OUTPUT_BASE}/quality_analysis_log.txt"
    
    echo ""
    echo "✓ Quality comparison complete"
fi
echo ""

# ========================================
# Test Summary
# ========================================
echo "========================================"
echo "TEST SUMMARY"
echo "========================================"
echo ""
echo "Output directory: $OUTPUT_BASE"
echo ""
echo "Test results:"
echo "  1. Baseline (no Materials v2): $BASELINE_DIR"
echo "  2. Enhanced (confidence=0.6): $ENHANCED_DIR"
echo "  3. Conservative (confidence=0.8): $CONSERVATIVE_DIR"
echo "  4. Pool edge case: $POOL_TEST_DIR"
echo "  5. Bathroom edge case: $BATHROOM_TEST_DIR"
echo "  6. Kitchen edge case: $KITCHEN_TEST_DIR"
echo ""
echo "Reports:"
echo "  - Quality comparison: ${OUTPUT_BASE}/quality_report.json"
echo ""
echo "Logs saved in: $OUTPUT_BASE/"
echo ""
echo "✓ All tests complete!"
