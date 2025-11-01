#!/bin/bash
# Verification script for PR #98
# This script verifies that all tests pass and the CLI tool works

set -e  # Exit on error

echo "=========================================="
echo "PR #98 Verification Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "format_utils.py" ]; then
    echo -e "${RED}Error: format_utils.py not found. Please run from repository root.${NC}"
    exit 1
fi

# Check Python version
echo "1. Checking Python version..."
python --version
echo ""

# Check if pytest is available
echo "2. Checking pytest availability..."
if ! command -v pytest &> /dev/null; then
    echo -e "${YELLOW}pytest not found. Installing dependencies...${NC}"
    pip install -r requirements-dev.txt -q
fi
echo -e "${GREEN}✓ pytest available${NC}"
echo ""

# Run tests
echo "3. Running format_utils tests..."
echo "----------------------------------------"
if pytest tests/test_format_utils.py -v --tb=short; then
    echo ""
    echo -e "${GREEN}✓ All tests PASSED${NC}"
else
    echo ""
    echo -e "${RED}✗ Some tests FAILED${NC}"
    exit 1
fi
echo ""

# Test CLI tool
echo "4. Testing CLI tool..."
echo "----------------------------------------"
echo "Testing --help:"
python examples/validate_file_formats.py --help
echo ""
echo -e "${GREEN}✓ CLI help works${NC}"
echo ""

echo "Testing directory scan on tests/:"
python examples/validate_file_formats.py tests/ | head -15
echo "... (output truncated)"
echo ""
echo -e "${GREEN}✓ CLI directory scan works${NC}"
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}✓ PR #98 Verification COMPLETE${NC}"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Tests: PASS"
echo "  - CLI Tool: WORKS"
echo "  - Ready for: Review and merge"
echo ""
echo "Next steps:"
echo "  1. Mark PR as ready for review (GitHub UI)"
echo "  2. Request reviews from maintainers"
echo "  3. Ensure CI checks pass"
echo "  4. Merge when approved"
