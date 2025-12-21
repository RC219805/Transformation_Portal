#!/bin/bash
# Validate MaterialsV3 CI configuration locally

set -e

echo "=== MaterialsV3 CI Validation ==="

# Check workflow file exists
if [ ! -f .github/workflows/materialsv3_tests.yml ]; then
    echo "❌ Workflow file not found"
    exit 1
fi
echo "✅ Workflow file exists"

# Check test files exist
if [ ! -f tests/test_materials_v3_edge_cases.py ]; then
    echo "❌ Edge case tests not found"
    exit 1
fi
echo "✅ Edge case tests exist"

if [ ! -f tests/test_materials_v3_stress.py ]; then
    echo "❌ Stress tests not found"
    exit 1
fi
echo "✅ Stress tests exist"

# Check verification script
if [ ! -f scripts/utilities/verify_phase1_safety.py ]; then
    echo "❌ Verification script not found"
    exit 1
fi
echo "✅ Verification script exists"

# Run quick validation
echo "Running edge case tests..."
python -m pytest tests/test_materials_v3_edge_cases.py -v --tb=short -x

echo "Running verification (quick mode)..."
python scripts/utilities/verify_phase1_safety.py --mode quick

echo ""
echo "=== CI Validation Complete ==="
echo "✅ Ready for commit and push"
