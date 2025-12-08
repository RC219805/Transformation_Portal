#!/usr/bin/env bash
set -euo pipefail

INPUT="data/benchmark_datasets/validation_v1/input"
OUTPUT="data/benchmark_datasets/validation_v1/baselines/adobe_neutral"

mkdir -p "$OUTPUT"

echo "Adobe Neutral baseline generation (Photoshop / Camera Raw)"
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo

cat <<EOF

Manual workflow:

  1) In Photoshop, open TIFFs from:

       $INPUT

  2) Apply Camera Raw preset "Lux_Validation_Neutral":
       - Sharpening ~25
       - Clarity 0, Texture 0
       - Noise Reduction 0
       - No creative grading
       - Lens corrections OFF (baseline)

  3) Save as 16-bit TIFF, ProPhoto RGB, into:

       $OUTPUT

After export, run:

  python scripts/validation/generate_manifest.py

EOF
