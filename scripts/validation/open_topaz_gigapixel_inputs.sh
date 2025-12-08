#!/usr/bin/env bash
set -euo pipefail

INPUT="data/benchmark_datasets/validation_v1/input"
OUTPUT="data/benchmark_datasets/validation_v1/baselines/topaz_gigapixel"

mkdir -p "$OUTPUT"

echo "Topaz Gigapixel baseline generation"
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo

if ! [ -d "$INPUT" ]; then
  echo "ERROR: Input directory not found: $INPUT" >&2
  exit 1
fi

APP="Topaz Gigapixel"
if ! open -a "$APP" >/dev/null 2>&1; then
  # Some installations use "Topaz Gigapixel AI"
  APP="Topaz Gigapixel AI"
  if ! open -a "$APP" >/dev/null 2>&1; then
    echo "NOTE: Could not open Topaz Gigapixel app (tried 'Topaz Gigapixel' and 'Topaz Gigapixel AI'). Edit APP in this script." >&2
  fi
fi

echo "Opening all TIFFs in Topaz Gigapixel..."
shopt -s nullglob
for f in "$INPUT"/*.tif "$INPUT"/*.tiff; do
  echo "  -> $f"
  open -a "$APP" "$f" || true
  sleep 0.5
done

cat <<EOF

Manual step required:

  1) In Topaz Gigapixel, select all imported images.
  2) Apply preset: "Lux_Validation_Giga4x"
       - 4x upscale
       - Standard mode
       - Artifact suppression ON
  3) Export as 16-bit TIFF into:

     $OUTPUT

After export, run:

  python scripts/validation/generate_manifest.py

EOF
