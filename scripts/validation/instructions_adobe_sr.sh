#!/usr/bin/env bash
set -euo pipefail

INPUT="data/benchmark_datasets/validation_v1/input"
OUTPUT="data/benchmark_datasets/validation_v1/baselines/adobe_sr"

mkdir -p "$OUTPUT"

echo "Adobe Super Resolution baseline generation (Photoshop / Camera Raw)"
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo

cat <<EOF

Manual workflow:

  1) In Photoshop, open TIFFs from:

       $INPUT

  2) Apply Camera Raw Super Resolution:
       - Right-click on image in Camera Raw
       - Select "Enhance..." 
       - Enable "Super Resolution" (2x upscale)
       - Apply minimal adjustments (Lux_Validation_SR preset):
         * Sharpening ~30
         * Clarity 0, Texture 0
         * Noise Reduction ~15 (as needed for AI artifacts)
         * No creative grading
         * Lens corrections OFF (baseline)

  3) Save as 16-bit TIFF, ProPhoto RGB, into:

       $OUTPUT

After export, run:

  python scripts/validation/generate_manifest.py

Note: Adobe Super Resolution uses AI to upscale 2x while preserving detail.
      This tests against Topaz Gigapixel's 4x upscaling capability.

EOF
