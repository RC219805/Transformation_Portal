#!/bin/bash
#
# 50-Image Validation Baseline Execution
#
# This script runs the complete validation pipeline with:
# - Balanced classifier evaluation
# - Input-size sweep (518→768→896→1022)
# - Stratified threshold calibration
#
# Prerequisites:
# - All DA2 models cached (~5.4 GB) ✅
# - 50-image dataset labeled ✅
# - Validation scripts ready ✅
#
# Expected runtime: 2-4 hours (full pipeline) or 30-60 min (baseline only)
#

set -euo pipefail

# Configuration
REPO_ROOT="/Users/rc/Transformation_Portal"
VALIDATION_DIR="${REPO_ROOT}/data/validation_full"
LABELS_FILE="${VALIDATION_DIR}/labels.csv"
STRUCTURE_SUBSET="${REPO_ROOT}/data/structure_subset"
OUTPUT_ROOT="${REPO_ROOT}/outputs/full_validation_baseline"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMMIT_SHA=$(git rev-parse --short HEAD)

# Verify prerequisites
echo "=== Prerequisites Check ==="
echo "✓ Checking dataset..."
if [ ! -f "${LABELS_FILE}" ]; then
    echo "❌ Labels file not found: ${LABELS_FILE}"
    exit 1
fi

IMAGE_COUNT=$(find "${VALIDATION_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
echo "✓ Found ${IMAGE_COUNT} images in validation dataset"

echo "✓ Checking models cache..."
MODEL_CACHE="${HOME}/.cache/huggingface"
if [ ! -d "${MODEL_CACHE}" ]; then
    echo "⚠️  Model cache not found at ${MODEL_CACHE}"
    echo "Run: python3 scripts/download_depth_models.py"
    exit 1
fi

echo "✓ All prerequisites satisfied"
echo ""

# Create output directory
OUTPUT_DIR="${OUTPUT_ROOT}/run_${TIMESTAMP}_${COMMIT_SHA}"
mkdir -p "${OUTPUT_DIR}"

echo "=== Starting Validation Pipeline ==="
echo "Timestamp: ${TIMESTAMP}"
echo "Commit: ${COMMIT_SHA}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Option 1: Full pipeline with input-size sweep (2-4 hours)
echo "Choose validation mode:"
echo "  1) Full pipeline (classifier + input-size sweep + calibration) [2-4 hours]"
echo "  2) Quick baseline (50-image validation only) [30-60 min]"
echo "  3) Smoke test (2 images) [5 min]"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo "=== Running Full Pipeline ==="
        python3 "${REPO_ROOT}/scripts/run_full_validation_pipeline.py" \
            --validation-dir "${VALIDATION_DIR}" \
            --labels "${LABELS_FILE}" \
            --structure-input-dir "${STRUCTURE_SUBSET}" \
            --sweep-sizes 518 768 896 1022 \
            --output-root "${OUTPUT_DIR}" \
            2>&1 | tee "${OUTPUT_DIR}/full_pipeline.log"
        ;;
    2)
        echo "=== Running Quick 50-Image Baseline ==="
        python3 "${REPO_ROOT}/scripts/production_depth_validation_fixed.py" \
            --input-dir "${VALIDATION_DIR}" \
            --output-dir "${OUTPUT_DIR}" \
            2>&1 | tee "${OUTPUT_DIR}/baseline_validation.log"

        # Generate reports
        echo "=== Generating Analysis Reports ==="
        python3 "${REPO_ROOT}/scripts/evaluate_classifier_balanced.py" \
            --metrics-dir "${OUTPUT_DIR}" \
            --labels "${LABELS_FILE}" \
            > "${OUTPUT_DIR}/classifier_report.txt"

        python3 "${REPO_ROOT}/scripts/analyze_validation_v2.py" \
            --metrics-dir "${OUTPUT_DIR}" \
            --labels "${LABELS_FILE}" \
            > "${OUTPUT_DIR}/validation_analysis.txt"
        ;;
    3)
        echo "=== Running Smoke Test (2 images) ==="
        SMOKE_DIR="${REPO_ROOT}/data/validation_smoke"
        mkdir -p "${SMOKE_DIR}"

        # Copy 2 test images (1 texture, 1 structure)
        cp "${VALIDATION_DIR}/750Picacho_Pool.jpg" "${SMOKE_DIR}/" 2>/dev/null || true
        cp "${VALIDATION_DIR}/750Picacho_Kitchen.jpg" "${SMOKE_DIR}/" 2>/dev/null || true

        python3 "${REPO_ROOT}/scripts/production_depth_validation_fixed.py" \
            --input-dir "${SMOKE_DIR}" \
            --output-dir "${OUTPUT_DIR}/smoke" \
            2>&1 | tee "${OUTPUT_DIR}/smoke_test.log"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=== Validation Complete ==="
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Review results:"
echo "     cat ${OUTPUT_DIR}/*.txt"
echo "  2. Check confusion matrix"
echo "  3. If healthy (≥85% classifier, ≥70% lenient):"
echo "     git tag -a v2-baseline-50img -m '50-image validation baseline'"
echo "  4. If structure needs improvement:"
echo "     Review input-size sweep results in ${OUTPUT_DIR}/sweep_*"
echo ""
echo "Documentation: docs/guides/SESSION_END_SUMMARY_2025-12-19_VALIDATION_READY.md"
