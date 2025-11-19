#!/bin/bash
# Training with 750 Picacho BIM Data
# 
# This script trains Hyper-Reality Enhancement models using real project data
# from the 750 Picacho Lane BIM model, including:
# - 6 UltraQuality TIFF renders (23-42 MB each)
# - 2,488 BIM-extracted architectural images
# - Architectural context from BIM model
#
# Usage:
#   ./scripts/train_with_750picacho.sh [--epochs 50] [--batch-size 4]

set -e  # Exit on error

# Parse arguments
EPOCHS=50
BATCH_SIZE=4
MAX_BIM=500

while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max-bim)
      MAX_BIM="$2"
      shift 2
      ;;
    --help)
      echo "Training with 750 Picacho BIM Data"
      echo ""
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --epochs N        Train for N epochs (default: 50)"
      echo "  --batch-size N    Training batch size (default: 4)"
      echo "  --max-bim N       Max BIM images to use (default: 500)"
      echo "  --help            Show this help message"
      echo ""
      echo "This uses real architectural data from 750 Picacho Lane project:"
      echo "  - 6 UltraQuality renders"
      echo "  - Up to 500 BIM images"
      echo "  - Architectural context from BIM"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Hyper-Reality Training with 750 Picacho BIM Data            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ Error: Must run from repository root${NC}"
    exit 1
fi

# Step 1: Prepare 750 Picacho data
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 1: Preparing 750 Picacho BIM Training Data${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}Using real project data from 750 Picacho Lane:${NC}"
echo "  • 6 UltraQuality architectural renders"
echo "  • Up to $MAX_BIM BIM-extracted images"
echo "  • Architectural context from BIM model"
echo ""

python src/enhancements/prepare_750picacho_training_data.py \
    --output-dir data/training_750picacho \
    --max-bim-images "$MAX_BIM"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Training data prepared${NC}"
else
    echo ""
    echo -e "${RED}❌ Failed to prepare data${NC}"
    exit 1
fi

# Count pairs created
PAIR_COUNT=$(ls -1 data/training_750picacho/high_quality/*.png 2>/dev/null | wc -l)
echo ""
echo -e "${BLUE}Dataset ready: $PAIR_COUNT training pairs${NC}"

# Step 2: Train models
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 2: Training Models on Real Project Data${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Training configuration:"
echo "  Dataset: 750 Picacho BIM ($PAIR_COUNT pairs)"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Estimated time: ~$((EPOCHS * 3 / 60)) hours on M4 Max"
echo ""
echo -e "${YELLOW}⚠️  Training on real project data will produce better results${NC}"
echo "   than synthetic data. Quality improvements should be visible."
echo ""
read -p "Press Enter to start training, or Ctrl+C to cancel..."

python src/enhancements/train_hyper_reality.py \
    --data-dir data/training_750picacho \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr 1e-4 \
    --checkpoint-dir weights/hyper_reality_750picacho

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Training completed${NC}"
else
    echo ""
    echo -e "${RED}❌ Training failed or was interrupted${NC}"
    exit 1
fi

# Step 3: Test trained models
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 3: Testing Trained Models${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Test on one of the actual project renders
TEST_IMAGE="projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_Kitchen_UltraQuality.tif"

if [ -f "$TEST_IMAGE" ]; then
    echo "Testing on actual project render: 750Picacho_Kitchen"
    
    # Copy weights to default location so they're auto-loaded
    cp weights/hyper_reality_750picacho/best_model.pth weights/hyper_reality/best_model.pth 2>/dev/null || true
    
    python src/enhancements/hyper_reality_enhancement.py \
        "$TEST_IMAGE" \
        -o "output_kitchen_750picacho_trained.jpg" \
        -q 105
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Test completed${NC}"
        echo "  Output: output_kitchen_750picacho_trained.jpg"
        echo "  Compare with original UltraQuality render to see improvements"
    fi
else
    echo -e "${YELLOW}⚠️  Project render not found, skipping test${NC}"
fi

# Summary
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Training Complete with Real Project Data!                    ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}✓ Models trained on 750 Picacho BIM data${NC}"
echo -e "${GREEN}✓ Weights saved to: weights/hyper_reality_750picacho/${NC}"
echo ""
echo "Dataset statistics:"
echo "  • Total training pairs: $PAIR_COUNT"
echo "  • Source: Real architectural project"
echo "  • Quality: UltraQuality renders + BIM images"
echo ""
echo "Next steps:"
echo "  1. Test on your architectural renders:"
echo "     python src/enhancements/hyper_reality_enhancement.py your_render.jpg"
echo ""
echo "  2. Compare against baseline to validate improvements"
echo ""
echo "  3. Fine-tune on additional data if needed:"
echo "     python src/enhancements/train_hyper_reality.py \\"
echo "         --data-dir your_custom_data/"
echo ""
echo "  4. Share results and feedback on quality improvements!"
echo ""
