#!/bin/bash
set -e

# PNG Compression Benchmarks (M1.1)
# Test compression levels 0, 1, 3, 6, 9 on representative images

cd /Users/rc/Transformation_Portal

# Test images
IMAGES=(
  "input_images/750_Picacho/Pool.tif"
  "input_images/750_Picacho/Aerial.tif"
  "input_images/750_Picacho/GreatRoom.tif"
)

# Compression levels to test
LEVELS=(0 1 3 6 9)

# Create benchmark directory
BENCHMARK_DIR="benchmarks_png_compression_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BENCHMARK_DIR"

echo "========================================"
echo "PNG Compression Benchmark Matrix"
echo "========================================"
echo "Images: ${#IMAGES[@]}"
echo "Levels: ${LEVELS[@]}"
echo "Output: $BENCHMARK_DIR"
echo "========================================"
echo ""

# Run matrix
for img in "${IMAGES[@]}"; do
  img_name=$(basename "$img" .tif)
  
  for level in "${LEVELS[@]}"; do
    output_dir="$BENCHMARK_DIR/${img_name}_png${level}"
    
    echo "----------------------------------------"
    echo "Testing: $img_name | Compression: $level"
    echo "Output: $output_dir"
    echo "----------------------------------------"
    
    lux-depth-v2 \
      --input "$img" \
      --output-dir "$output_dir" \
      --marketing-png-compression ${level} \
      --preset exterior_showcase \
      2>&1 | grep -E "(Done|ERROR|WARNING)" || true
    
    echo ""
  done
done

echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo "Results in: $BENCHMARK_DIR"
echo ""
echo "Analyze with:"
echo "  python scripts/analyze_marketing_export.py $BENCHMARK_DIR/*/"
