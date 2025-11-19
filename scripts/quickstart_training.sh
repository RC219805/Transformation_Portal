#!/bin/bash
# Quick Start Training Script for Hyper-Reality Enhancement
# 
# This script automates the complete training workflow:
# 1. Generate synthetic training data
# 2. Train models for 50 epochs
# 3. Test trained models on sample image
#
# Usage:
#   ./scripts/quickstart_training.sh [--num-pairs 1000] [--epochs 50]

set -e  # Exit on error

# Parse arguments
NUM_PAIRS=1000
EPOCHS=50
BATCH_SIZE=4

while [[ $# -gt 0 ]]; do
  case $1 in
    --num-pairs)
      NUM_PAIRS="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --help)
      echo "Quick Start Training for Hyper-Reality Enhancement"
      echo ""
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --num-pairs N     Generate N training pairs (default: 1000)"
      echo "  --epochs N        Train for N epochs (default: 50)"
      echo "  --batch-size N    Training batch size (default: 4)"
      echo "  --help            Show this help message"
      echo ""
      echo "Example:"
      echo "  $0 --num-pairs 2000 --epochs 100"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Hyper-Reality Enhancement - Quick Start Training            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ Error: Must run from repository root${NC}"
    echo "   cd to the Transformation_Portal directory first"
    exit 1
fi

# Step 1: Generate training data
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 1: Generating Synthetic Training Data${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Generating $NUM_PAIRS training pairs..."
echo "This will take approximately $((NUM_PAIRS / 10)) seconds..."
echo ""

python src/enhancements/train_hyper_reality.py \
    --generate-data \
    --num-pairs "$NUM_PAIRS"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Training data generated successfully${NC}"
else
    echo ""
    echo -e "${RED}❌ Failed to generate training data${NC}"
    exit 1
fi

# Step 2: Train models
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 2: Training Models${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Training configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Estimated time: ~$((EPOCHS * 3 / 60)) hours on M4 Max"
echo ""
echo -e "${YELLOW}⚠️  Training can take several hours. You can:${NC}"
echo "   - Let it run in background: Ctrl+Z, then 'bg'"
echo "   - Resume later: Training saves checkpoints every 5 epochs"
echo "   - Stop anytime: Ctrl+C (best model is saved)"
echo ""
read -p "Press Enter to start training, or Ctrl+C to cancel..."

python src/enhancements/train_hyper_reality.py \
    --data-dir data/training \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr 1e-4 \
    --checkpoint-dir weights/hyper_reality

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Training completed successfully${NC}"
else
    echo ""
    echo -e "${RED}❌ Training failed or was interrupted${NC}"
    echo "   Checkpoints may have been saved in weights/hyper_reality/"
    exit 1
fi

# Step 3: Test trained models
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 3: Testing Trained Models${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if test image exists
TEST_IMAGE="data/training/high_quality/image_0000.png"
if [ ! -f "$TEST_IMAGE" ]; then
    echo -e "${YELLOW}⚠️  No test image found, skipping test${NC}"
else
    echo "Testing on: $TEST_IMAGE"
    
    python src/enhancements/hyper_reality_enhancement.py \
        "$TEST_IMAGE" \
        -o "output_test_trained.jpg" \
        -q 105
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Test completed${NC}"
        echo "  Output: output_test_trained.jpg"
    else
        echo ""
        echo -e "${RED}❌ Test failed${NC}"
    fi
fi

# Summary
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Training Complete!                                           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}✓ Trained models available in: weights/hyper_reality/${NC}"
echo ""
echo "Next steps:"
echo "  1. Test on your images:"
echo "     python src/enhancements/hyper_reality_enhancement.py your_image.jpg"
echo ""
echo "  2. Integrate with pipelines:"
echo "     python examples/hyper_reality_example.py --example 1"
echo ""
echo "  3. Fine-tune on custom data:"
echo "     python src/enhancements/train_hyper_reality.py --data-dir your_data/"
echo ""
echo "  4. Check training guide:"
echo "     cat docs/TRAINING_GUIDE.md"
echo ""
