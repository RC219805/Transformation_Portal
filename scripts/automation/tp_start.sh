#!/usr/bin/env bash
#
# Transformation Portal - Quick Start
# One command to get back into production environment
#

set -e

cd /Users/rc/Transformation_Portal || exit 1
source .venv/bin/activate

echo ""
echo "✅ Transformation Portal - Ready"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Python: $(python --version)"
echo "  Location: $(pwd)"
echo "  Virtual Env: Active"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Quick Commands:"
echo ""
echo "  # List available recipes"
echo "  python -c \"from transformation_portal.cli import app; app()\" pipeline list-recipes"
echo ""
echo "  # Baseline assessment"
echo "  python -c \"from transformation_portal.cli import app; app()\" pipeline process \\"
echo "    -i \"input_images/PROJECT/Source_JPEGS/*.jpg\" \\"
echo "    -o \"output_PROJECT_Baseline\" \\"
echo "    -r config/recipes/baseline_quality.yaml"
echo ""
echo "  # Get RAG recommendation"
echo "  python scripts/rag/suggest_recipe.py \\"
echo "    --scene-type interior_bedroom \\"
echo "    --baseline-score 58.3"
echo ""
echo "📖 Full docs: docs/PRODUCTION_OPERATIONS.md"
echo ""
