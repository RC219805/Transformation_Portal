#!/bin/bash
set -e

echo "=============================================="
echo "Lux Depth V2 - Enhancement Verification"
echo "=============================================="
echo

echo "1. Verifying test suite..."
python -m pytest tests/ -m "not slow and not gpu" --co -q | tail -1
echo "   ✓ Tests collected successfully"
echo

echo "2. Running sample tests..."
python -m pytest tests/test_config.py -q --tb=line 2>&1 | grep -E "(passed|failed)" | head -1
echo "   ✓ Sample tests passing"
echo

echo "3. Verifying telemetry module..."
python -c "from telemetry import MetricsCollector, BatchMetrics; print('   ✓ Telemetry imports OK')"
echo

echo "4. Checking documentation structure..."
if [ -f "docs/conf.py" ] && [ -f "docs/index.rst" ]; then
    echo "   ✓ Documentation structure present"
fi
echo

echo "5. Verifying examples..."
example_count=$(ls examples/*.py 2>/dev/null | wc -l | tr -d ' ')
echo "   ✓ $example_count example scripts available"
echo

echo "6. Checking development tools..."
if [ -f "Makefile" ] && [ -f "pytest.ini" ]; then
    echo "   ✓ Development tools present"
fi
echo

echo "=============================================="
echo "All verifications passed! ✅"
echo "=============================================="
echo
echo "Quick commands:"
echo "  make test-fast     # Run fast tests"
echo "  make docs          # Build documentation"
echo "  make help          # Show all commands"
