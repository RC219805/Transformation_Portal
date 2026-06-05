#!/bin/bash
# Quick Start Script for 750 Picacho Elite Processing
# Processes all 6 images with optimized presets

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

echo "=============================================================================="
echo "750 Picacho Elite Pipeline - Quick Start"
echo "=============================================================================="
echo ""

# Configuration
INPUT_DIR="input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs"
OUTPUT_DIR="output_750_picacho_elite_$(date +%Y%m%d_%H%M%S)"
PRESET="auto"  # Auto-detect from filenames
VERBOSE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-ai)
      NO_AI="--no-ai"
      shift
      ;;
    --no-upscale)
      NO_UPSCALE="--no-upscale"
      shift
      ;;
    --fast)
      NO_AI="--no-ai"
      NO_UPSCALE="--no-upscale"
      shift
      ;;
    --verbose)
      VERBOSE="--verbose"
      shift
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --help)
      echo "Usage: ./process_750_picacho_elite.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --no-ai          Disable AI enhancement (faster)"
      echo "  --no-upscale     Disable 4x upscaling (faster)"
      echo "  --fast           Disable both AI and upscaling (fastest)"
      echo "  --verbose        Verbose output"
      echo "  --output DIR     Custom output directory"
      echo "  --help           Show this help"
      echo ""
      echo "Examples:"
      echo "  ./process_750_picacho_elite.sh                    # Full processing"
      echo "  ./process_750_picacho_elite.sh --fast             # Fast mode"
      echo "  ./process_750_picacho_elite.sh --no-upscale       # No upscaling"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
  echo "❌ ERROR: Input directory not found: $INPUT_DIR"
  echo ""
  echo "Expected location: input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/"
  echo ""
  echo "Please ensure the 750 Picacho TIFF files are in the correct location."
  exit 1
fi

# Count images
IMAGE_COUNT=$(find "$INPUT_DIR" -name "*.tif" -type f | wc -l | tr -d ' ')

if [ "$IMAGE_COUNT" -eq 0 ]; then
  echo "❌ ERROR: No TIFF images found in $INPUT_DIR"
  exit 1
fi

echo "Configuration:"
echo "  Input directory: $INPUT_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Images found: $IMAGE_COUNT"
echo "  Preset: $PRESET"
echo "  AI Enhancement: $([ -z "$NO_AI" ] && echo "✓ Enabled" || echo "✗ Disabled")"
echo "  4x Upscaling: $([ -z "$NO_UPSCALE" ] && echo "✓ Enabled" || echo "✗ Disabled")"
echo ""

# Confirm processing
read -p "Start processing? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Cancelled."
  exit 0
fi

echo ""
echo "=============================================================================="
echo "Starting batch processing..."
echo "=============================================================================="
echo ""

# Run pipeline
python "${SCRIPT_DIR}/elite_architectural_pipeline.py" \
  --directory "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --preset "$PRESET" \
  --pattern "*.tif" \
  $NO_AI \
  $NO_UPSCALE \
  $VERBOSE

# Check exit code
if [ $? -eq 0 ]; then
  echo ""
  echo "=============================================================================="
  echo "✅ Processing complete!"
  echo "=============================================================================="
  echo ""
  echo "Output location: $OUTPUT_DIR"
  echo ""
  echo "Generated files per image:"
  echo "  • *_DELIVERY.jpg      - Final delivery JPEG (98% quality)"
  echo "  • *_MASTER.tiff       - 16-bit TIFF master file"
  echo "  • *_depth.png         - Depth map visualization"
  echo "  • *_material.tiff     - Material Response stage"
  echo "  • *_graded.tiff       - Color grading stage"
  echo "  • *_ai_enhanced.png   - AI enhancement stage (if enabled)"
  echo "  • *_4x_upscaled.png   - 4x upscaled (if enabled)"
  echo "  • *_processing_report.json - Processing metadata"
  echo ""
  echo "Quick review:"
  echo "  ls -lh $OUTPUT_DIR/*_DELIVERY.jpg"
  echo "  open $OUTPUT_DIR/*_DELIVERY.jpg"
  echo ""

  # Show summary statistics
  if [ -f "$OUTPUT_DIR"/*.json ]; then
    echo "Processing time summary:"
    echo "  (See individual *_processing_report.json files for details)"
    echo ""
  fi

  echo "Next steps:"
  echo "  1. Review delivery JPEGs for quality"
  echo "  2. Check processing reports for any warnings"
  echo "  3. Re-process individual images with custom settings if needed"
  echo ""
else
  echo ""
  echo "=============================================================================="
  echo "❌ Processing failed"
  echo "=============================================================================="
  echo ""
  echo "Check the error messages above for details."
  echo "Common issues:"
  echo "  • Missing dependencies (run: pip install -r requirements.txt)"
  echo "  • Out of memory (try --fast mode or --no-upscale)"
  echo "  • Invalid TIFF format (ensure 32-bit float TIFFs with tifffile)"
  echo ""
  exit 1
fi
