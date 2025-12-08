#!/usr/bin/env bash
set -euo pipefail

INPUT="data/benchmark_datasets/validation_v1/input"
OUTPUT="data/benchmark_datasets/validation_v1/baselines/topaz_video"

mkdir -p "$OUTPUT"

echo "Topaz Video AI baseline generation"
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo

if ! [ -d "$INPUT" ]; then
  echo "ERROR: Input directory not found: $INPUT" >&2
  exit 1
fi

APP="Topaz Video AI"
if ! open -a "$APP" >/dev/null 2>&1; then
  echo "NOTE: Could not open app name '$APP'. If your app name differs, edit this script." >&2
fi

echo "Opening all TIFFs in Topaz Video AI..."
shopt -s nullglob
for f in "$INPUT"/*.tif "$INPUT"/*.tiff; do
  echo "  -> $f"
  open -a "$APP" "$f" || true
  sleep 0.5
done

cat <<EOF

Manual step required:

  1) In Topaz Video AI, select all imported images.
  2) Apply preset: "Lux_Validation_Video" (upscaling + enhancement):
       - Enhancement mode: Standard enhancement
       - Temporal processing: ON (frame-aware smoothing)
       - Output resolution: Match input (no upscaling)
       - Noise reduction: Minimal/Auto
  3) Export as 16-bit TIFF (ProPhoto/original), no lossy compression.
  4) Export directory:

     $OUTPUT

After export, run:

  python scripts/validation/generate_manifest.py

Note: Topaz Video AI can process still images using its video enhancement
      algorithms. This baseline tests its temporal-aware processing on stills.

EOF
