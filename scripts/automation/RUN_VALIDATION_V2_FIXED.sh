#!/bin/bash
# Re-run 18-Image Validation with Texture-Scene Fixes
# This script validates the fixes for texture-dominated scene handling

set -e

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/validation_v2_fixed_${TIMESTAMP}"

# Use the same input as the previous run for comparison
INPUT_DIR="input_images"

# Find all source images from the previous validation run
# We'll process the same 18 images to enable direct comparison
IMAGES=(
    "750_Picacho/Source_JPEGS/750Picacho_Aerial.jpg"
    "750_Picacho/Source_JPEGS/750Picacho_GreatRoom.jpg"
    "750_Picacho/Source_JPEGS/750Picacho_Kitchen.jpg"
    "750_Picacho/Source_JPEGS/750Picacho_Pool.jpg"
    "750_Picacho/Source_JPEGS/750Picacho_PrimaryBathroom.jpg"
    "800_Picacho/Source_JPEGS/800-picacho-1.jpg"
    "800_Picacho/Source_JPEGS/800-picacho-6.jpg"
    "800_Picacho/Source_JPEGS/800-picacho-11.jpg"
    "800_Picacho/Source_JPEGS/800-picacho-28.jpg"
    "800_Picacho/Source_JPEGS/800-picacho-38.jpg"
    "16_Seaview/Source_JPEGS/Montecito-Shores-3.jpg"
    "16_Seaview/Source_JPEGS/Montecito-Shores-7.jpg"
    "16_Seaview/Source_JPEGS/Montecito-Shores-10.jpg"
    "16_Seaview/Source_JPEGS/Montecito-Shores-12.jpg"
    "16_Seaview/Source_JPEGS/Montecito-Shores-16.jpg"
    "16_Seaview/Source_JPEGS/Montecito-Shores-18.jpg"
    "16_Seaview/Source_JPEGS/Montecito-shores-aerial-2.jpg"
    "16_Seaview/Source_JPEGS/Montecito-shores-aerial-4.jpg"
)

echo "=================================================="
echo "18-Image Validation with Texture-Scene Fixes"
echo "=================================================="
echo ""
echo "Fixes applied:"
echo "  1. Save ALL EdgeMetrics fields (edge_overlap, halo_score, etc.)"
echo "  2. Use high-frequency energy instead of global variance for texture scenes"
echo "  3. Pass filename to classifier for weak supervision"
echo "  4. Include depth_gradient_var in classification factors"
echo ""
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create a temporary directory with just the 18 images
TEMP_INPUT="$OUTPUT_DIR/input_images"
mkdir -p "$TEMP_INPUT"

echo "Copying 18 images to temporary directory..."
copied=0
for img_path in "${IMAGES[@]}"; do
    src="$INPUT_DIR/$img_path"
    if [ -f "$src" ]; then
        dst="$TEMP_INPUT/$(basename "$src")"
        cp "$src" "$dst"
        ((copied++))
    else
        echo "⚠️  Warning: Not found: $src"
    fi
done

echo "✅ Copied $copied images"
echo ""

if [ $copied -lt 15 ]; then
    echo "❌ Error: Too few images copied ($copied < 15)"
    echo "Check image paths in script"
    exit 1
fi

# Run validation
echo "Starting validation..."
echo ""

python scripts/automation/production_depth_validation_fixed.py \
  --input-dir "$TEMP_INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --tile-size 1024 \
  --overlap 192 \
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
    
    # Run analysis
    echo "Running analysis..."
    python scripts/analyze_validation_v2.py --results-dir "$OUTPUT_DIR"
    
    echo ""
    echo "Comparison with baseline:"
    echo "  Baseline: outputs/validation_v2_20251218_170022_8197588"
    echo "  Current:  $OUTPUT_DIR"
    echo ""
    echo "To compare:"
    echo "  diff outputs/validation_v2_20251218_170022_8197588/classification_report.txt $OUTPUT_DIR/classification_report.txt"
    
else
    echo ""
    echo "❌ Validation failed"
    exit 1
fi
