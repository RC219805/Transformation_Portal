#!/bin/bash
set -e

# Heavy Quality Benchmark
# Test "max quality" configuration with all heavy features enabled

cd /Users/rc/Transformation_Portal

# Test images
IMAGES=(
  "input_images/750_Picacho/Aerial.tif"
  "input_images/750_Picacho/Pool.tif"
  "input_images/750_Picacho/GreatRoom.tif"
)

# Benchmark configurations
CONFIGS=(
  "baseline"    # Current production defaults (PNG level 1, materials v2 off)
  "heavy"       # Max quality (materials v2 on, higher seg resolution, etc.)
  "heavy_depth" # Max quality + depth-aware processing
)

# Create benchmark directory
BENCHMARK_DIR="benchmarks_heavy_quality_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BENCHMARK_DIR"

echo "========================================"
echo "Heavy Quality Benchmark"
echo "========================================"
echo "Images: ${#IMAGES[@]}"
echo "Configs: ${CONFIGS[@]}"
echo "Output: $BENCHMARK_DIR"
echo "========================================"
echo ""

# Function to run a single benchmark
run_benchmark() {
  local img=$1
  local config=$2
  local img_name=$(basename "$img" .tif)
  local output_dir="$BENCHMARK_DIR/${config}_${img_name}"
  
  echo "----------------------------------------"
  echo "Config: $config | Image: $img_name"
  echo "Output: $output_dir"
  echo "----------------------------------------"
  
  if [ "$config" = "baseline" ]; then
    # Baseline: Current production defaults
    lux-depth-v2 \
      --input "$img" \
      --output-dir "$output_dir" \
      --preset exterior_showcase \
      --marketing-png-compression 1 \
      2>&1 | tee "$output_dir.log" | grep -E "(Done|ERROR|WARNING|stage)" || true
      
  elif [ "$config" = "heavy" ]; then
    # Heavy: Max quality with all features enabled
    lux-depth-v2 \
      --input "$img" \
      --output-dir "$output_dir" \
      --preset exterior_showcase \
      --marketing-png-compression 1 \
      --materials-v2 \
      --max-segmentation-side 1536 \
      --cache-masks \
      2>&1 | tee "$output_dir.log" | grep -E "(Done|ERROR|WARNING|stage)" || true
      
  elif [ "$config" = "heavy_depth" ]; then
    # Heavy + Depth: Max quality with depth-aware processing
    lux-depth-v2 \
      --input "$img" \
      --output-dir "$output_dir" \
      --preset exterior_showcase \
      --marketing-png-compression 1 \
      --materials-v2 \
      --max-segmentation-side 1536 \
      --cache-masks \
      --depth-dir depth_maps/750_Picacho \
      2>&1 | tee "$output_dir.log" | grep -E "(Done|ERROR|WARNING|stage)" || true
  fi
  
  echo ""
}

# Run benchmark matrix
for config in "${CONFIGS[@]}"; do
  for img in "${IMAGES[@]}"; do
    run_benchmark "$img" "$config"
  done
done

echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo "Results in: $BENCHMARK_DIR"
echo ""
echo "Analyze with:"
echo "  python scripts/analyze_heavy_benchmark.py $BENCHMARK_DIR/"
