#!/bin/bash
# Pre-cache all DA3 model variants for production/benchmarking

set -euo pipefail

# Configuration
CACHE_DIR="${DA3_CACHE_DIR:-$HOME/.cache/huggingface/hub}"
MODEL_SET="${DA3_MODEL_SET:-production}"
VERIFY="${DA3_VERIFY:-true}"

echo "=================================================="
echo "DA3 Model Pre-caching Script"
echo "=================================================="
echo "Cache Directory: $CACHE_DIR"
echo "Model Set: $MODEL_SET"
echo "Verify Downloads: $VERIFY"
echo "=================================================="

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.10+"
    exit 1
fi

# Check lux-depth-v3 installation
if ! python -c "import lux_depth_v3" 2>/dev/null; then
    echo "❌ lux_depth_v3 not installed. Run: pip install -e ."
    exit 1
fi

# Download models
echo ""
echo "📥 Starting download..."
lux-depth-v3 cache-download \
    --set "$MODEL_SET" \
    --cache-dir "$CACHE_DIR" \
    $([ "$VERIFY" = "true" ] && echo "--verify" || echo "--no-verify")

# Show statistics
echo ""
lux-depth-v3 cache-stats

echo ""
echo "✅ Pre-caching complete!"
echo "=================================================="
