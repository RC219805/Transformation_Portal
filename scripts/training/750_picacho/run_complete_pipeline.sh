#!/bin/bash
# ============================================================================
# 750 Picacho Lane Complete Training Pipeline
# ============================================================================
# Master script for running the complete property-specific training protocol.
#
# This script executes all 6 stages of the training pipeline:
#   1. Property Analysis
#   2. Depth Synthesis
#   3. Dataset Generation
#   4. Model Training
#   5. Model Validation
#   6. Final Output Processing
#
# Usage:
#   ./scripts/training/750_picacho/run_complete_pipeline.sh [options]
#
# Options:
#   --skip-training     Skip training (use existing model)
#   --quick-train       Use reduced epochs for quick testing
#   --device            Compute device (auto/cuda/mps/cpu)
#   --output-dir        Custom output directory
#   --help              Show this help message
#
# Author: Transformation_Portal Enhancement Team
# Version: 1.0.0
# ============================================================================

set -e  # Exit on error

# Default values
SKIP_TRAINING=false
QUICK_TRAIN=false
DEVICE="auto"
OUTPUT_DIR="output/750_picacho"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --quick-train)
            QUICK_TRAIN=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "750 Picacho Lane Complete Training Pipeline"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-training     Skip training (use existing model)"
            echo "  --quick-train       Use reduced epochs for quick testing"
            echo "  --device DEVICE     Compute device (auto/cuda/mps/cpu)"
            echo "  --output-dir DIR    Custom output directory"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Header
echo ""
echo "============================================================================"
echo -e "${BLUE}750 PICACHO LANE - COMPLETE TRAINING PIPELINE${NC}"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Project root: $PROJECT_ROOT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo "  Skip training: $SKIP_TRAINING"
echo "  Quick train: $QUICK_TRAIN"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Training epochs based on mode
if [ "$QUICK_TRAIN" = true ]; then
    STAGE1_EPOCHS=2
    STAGE2_EPOCHS=2
    STAGE3_EPOCHS=1
    NUM_SAMPLES=50
else
    STAGE1_EPOCHS=20
    STAGE2_EPOCHS=20
    STAGE3_EPOCHS=10
    NUM_SAMPLES=600
fi

# Track timing
START_TIME=$(date +%s)
STAGE_TIMES=()

run_stage() {
    local stage_num=$1
    local stage_name=$2
    local stage_script=$3
    shift 3
    local stage_args=("$@")
    
    echo ""
    echo "------------------------------------------------------------------------"
    echo -e "${YELLOW}STAGE $stage_num: $stage_name${NC}"
    echo "------------------------------------------------------------------------"
    
    STAGE_START=$(date +%s)
    
    if python "$stage_script" "${stage_args[@]}"; then
        STAGE_END=$(date +%s)
        STAGE_DURATION=$((STAGE_END - STAGE_START))
        STAGE_TIMES+=("$stage_name: ${STAGE_DURATION}s")
        echo -e "${GREEN}✓ Stage $stage_num completed in ${STAGE_DURATION}s${NC}"
    else
        echo -e "${RED}✗ Stage $stage_num failed${NC}"
        exit 1
    fi
}

# ============================================================================
# Stage 1: Property Analysis
# ============================================================================
run_stage 1 "Property Analysis" \
    "scripts/training/750_picacho/01_analyze_property.py" \
    --output "$OUTPUT_DIR/property_analysis.json"

# ============================================================================
# Stage 2: Depth Synthesis
# ============================================================================
run_stage 2 "Depth Synthesis" \
    "scripts/training/750_picacho/02_synthesize_depth.py" \
    --output-dir "data/training_750picacho/depth" \
    --model "large"

# ============================================================================
# Stage 3: Dataset Generation
# ============================================================================
run_stage 3 "Dataset Generation" \
    "scripts/training/750_picacho/03_generate_dataset.py" \
    --output-dir "data/training_750picacho" \
    --num-samples $NUM_SAMPLES \
    --seed 42

# ============================================================================
# Stage 4: Model Training (optional)
# ============================================================================
if [ "$SKIP_TRAINING" = false ]; then
    run_stage 4 "Model Training" \
        "scripts/training/750_picacho/04_train_model.py" \
        --data-dir "data/training_750picacho" \
        --checkpoint-dir "weights/750_picacho" \
        --stage1-epochs $STAGE1_EPOCHS \
        --stage2-epochs $STAGE2_EPOCHS \
        --stage3-epochs $STAGE3_EPOCHS \
        --device "$DEVICE"
else
    echo ""
    echo "------------------------------------------------------------------------"
    echo -e "${YELLOW}STAGE 4: Model Training (SKIPPED)${NC}"
    echo "------------------------------------------------------------------------"
    echo "Using existing model weights."
    STAGE_TIMES+=("Model Training: SKIPPED")
fi

# ============================================================================
# Stage 5: Model Validation
# ============================================================================
run_stage 5 "Model Validation" \
    "scripts/training/750_picacho/05_validate_model.py" \
    --model "weights/750_picacho/best_model.pth" \
    --test-dir "data/training_750picacho/test" \
    --output-dir "$OUTPUT_DIR/validation" \
    --device "$DEVICE"

# ============================================================================
# Stage 6: Final Output Processing
# ============================================================================
run_stage 6 "Final Output Processing" \
    "scripts/training/750_picacho/06_process_final_output.py" \
    --model "weights/750_picacho/best_model.pth" \
    --output-dir "$OUTPUT_DIR/final_deliverables" \
    --format "16bit_tiff" \
    --enhancement-level "balanced" \
    --device "$DEVICE"

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "============================================================================"
echo -e "${GREEN}750 PICACHO LANE TRAINING PIPELINE COMPLETE${NC}"
echo "============================================================================"
echo ""
echo "Stage Timing:"
for stage_time in "${STAGE_TIMES[@]}"; do
    echo "  • $stage_time"
done
echo ""
echo "Total Duration: ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60))m $(($TOTAL_DURATION % 60))s)"
echo ""
echo "Output Locations:"
echo "  • Property Analysis: $OUTPUT_DIR/property_analysis.json"
echo "  • Depth Maps: data/training_750picacho/depth/"
echo "  • Training Dataset: data/training_750picacho/"
echo "  • Model Weights: weights/750_picacho/"
echo "  • Validation Results: $OUTPUT_DIR/validation/"
echo "  • Final Deliverables: $OUTPUT_DIR/final_deliverables/"
echo ""
echo "============================================================================"
echo -e "${GREEN}SUCCESS: 6 enhanced 4K 16-bit TIFF files ready for delivery${NC}"
echo "============================================================================"
echo ""
