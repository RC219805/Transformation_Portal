#!/usr/bin/env bash
#
# run_model_comparison_suite.sh
#
# Complete workflow: multi-model validation + statistical analysis
#
# Usage:
#   ./scripts/run_model_comparison_suite.sh [quick|full]
#
# Modes:
#   quick - 7-10 images, 2 models, 2 input sizes (fast iteration)
#   full  - 50+ images, all models, 4 input sizes (production validation)
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

MODE="${1:-quick}"

echo "=================================================="
echo "Multi-Model Depth Validation Suite"
echo "=================================================="
echo "Mode: $MODE"
echo "Repo: $REPO_ROOT"
echo ""

# Set parameters based on mode
if [ "$MODE" = "quick" ]; then
    INPUT_DIR="data/validation_quick"
    LABELS="data/validation_quick/labels.csv"
    MODELS="DA2_Large DA2_Metric_Indoor"
    SIZES="518 768"
    echo "⚡ Quick mode: 2 models, 2 input sizes"
elif [ "$MODE" = "full" ]; then
    INPUT_DIR="data/validation_full"
    LABELS="data/validation_full/labels.csv"
    MODELS="DA2_Large DA2_Metric_Indoor DA2_Metric_Outdoor"
    SIZES="518 768 896 1022"
    echo "🔥 Full mode: 3 models, 4 input sizes"
else
    echo "❌ Unknown mode: $MODE"
    echo "Usage: $0 [quick|full]"
    exit 1
fi

# Check prerequisites
echo ""
echo "▶ Checking prerequisites..."

if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Input directory not found: $INPUT_DIR"
    exit 1
fi

if [ ! -f "$LABELS" ]; then
    echo "❌ Labels file not found: $LABELS"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found"
    exit 1
fi

# Check Python dependencies
python3 -c "import scipy, sklearn, pandas" 2>/dev/null || {
    echo "❌ Missing Python dependencies (scipy, sklearn, pandas)"
    echo "Install with: pip install scipy scikit-learn pandas"
    exit 1
}

echo "✅ Prerequisites OK"

# Step 1: Run multi-model comparison
echo ""
echo "=================================================="
echo "Step 1: Multi-Model Validation"
echo "=================================================="
echo ""

python3 scripts/run_multi_model_comparison.py \
    --input-dir "$INPUT_DIR" \
    --labels "$LABELS" \
    --models $MODELS \
    --sweep-sizes $SIZES \
    --output-root outputs/model_comparison \
    || {
        echo "❌ Multi-model validation failed"
        exit 1
    }

# Find the latest run directory
LATEST_RUN=$(ls -td outputs/model_comparison/run_* 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "❌ No run directory found"
    exit 1
fi

echo ""
echo "✅ Validation complete: $LATEST_RUN"

# Step 2: Statistical analysis
echo ""
echo "=================================================="
echo "Step 2: Statistical Analysis"
echo "=================================================="
echo ""

python3 scripts/analyze_model_comparison.py \
    --comparison-dir "$LATEST_RUN" \
    --baseline-model DA2_Large \
    --confidence-level 0.95 \
    || {
        echo "❌ Statistical analysis failed"
        exit 1
    }

echo ""
echo "✅ Analysis complete"

# Step 3: Generate summary report
echo ""
echo "=================================================="
echo "Step 3: Summary Report"
echo "=================================================="
echo ""

# Display best configurations
if [ -f "$LATEST_RUN/best_per_model.csv" ]; then
    echo "📊 Best Configurations per Model:"
    echo ""
    cat "$LATEST_RUN/best_per_model.csv" | column -t -s,
    echo ""
fi

# Display statistical summary
if [ -f "$LATEST_RUN/analysis/statistical_summary.csv" ]; then
    echo "📈 Statistical Summary (vs Baseline):"
    echo ""
    cat "$LATEST_RUN/analysis/statistical_summary.csv" | column -t -s,
    echo ""
fi

# Display stratified results
if [ -f "$LATEST_RUN/analysis/statistical_comparison.json" ]; then
    echo "🔍 Stratified Analysis Available:"
    echo "   $LATEST_RUN/analysis/statistical_comparison.json"
    echo ""
fi

# Final summary
echo "=================================================="
echo "✅ Multi-Model Validation Suite Complete"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  $LATEST_RUN"
echo ""
echo "Key files:"
echo "  - comparison_overall.csv (all results)"
echo "  - best_per_model.csv (optimal configs)"
echo "  - analysis/statistical_summary.csv (significance tests)"
echo "  - model_comparison_summary.json (full data)"
echo ""

# Offer to open results
if command -v open &> /dev/null; then
    read -p "Open results directory? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open "$LATEST_RUN"
    fi
fi

echo "Done! 🎉"
