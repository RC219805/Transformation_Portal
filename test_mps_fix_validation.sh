#!/bin/bash
# Test script for MPS bicubic fix validation

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║             MPS BICUBIC FIX - VALIDATION TEST                               ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Testing V3+V2 pipeline with MPS-compatible bilinear interpolation..."
echo ""

# Test with single image first
python -m lux_depth_v3.cli enhance \
  --input-dir data/validation_expanded \
  --output-dir output/mps_fix_validation \
  --preset interior_luxury \
  --model nested-giant-large-v1.1 \
  --non-commercial-ok \
  --include "750Picacho_Aerial.jpg"

echo ""
echo "Check output/mps_fix_validation/v2/ for successful upscaled output"
echo "If successful, run full batch with:"
echo ""
echo "python -m lux_depth_v3.cli enhance \\"
echo "  --input-dir data/validation_expanded \\"
echo "  --output-dir output/final_optimized \\"
echo "  --preset interior_luxury \\"
echo "  --model nested-giant-large-v1.1 \\"
echo "  --non-commercial-ok"
