#!/usr/bin/env bash
# Check for ML test isolation violations
#
# This pre-commit hook enforces ADR-031 (Test Dependency Isolation Contract):
# - Tests using @patch("transformers.*") or @patch("torch.*") must have import guards
# - Prevents import-before-mock failures in offline CI environments
#
# Exit codes:
#   0 - No violations detected
#   1 - Violations found (blocks commit)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Track violations
VIOLATIONS_FOUND=0

echo "🔍 Checking ML test isolation (ADR-031)..."

# Find Python test files
TEST_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '^tests/.*\.py$' || true)

if [ -z "$TEST_FILES" ]; then
    echo -e "${GREEN}✓${NC} No test files modified"
    exit 0
fi

# Check each test file
for FILE in $TEST_FILES; do
    # Skip if file doesn't exist (deleted)
    [ ! -f "$FILE" ] && continue

    # Check for @patch patterns (multiple variants):
    # - @patch("transformers.CLIPModel") or @patch('transformers.CLIPModel')
    # - @patch("torch") or @patch('torch')  # whole module
    # - @patch(target="transformers.AutoModel")
    # - @unittest.mock.patch("torch.cuda.is_available")
    # Extended regex to handle:
    # - Both single and double quotes
    # - With or without target= kwarg
    # - Full unittest.mock prefix
    # - Whole module or module.submodule patterns
    PATCH_LINES=$(grep -nE '@(unittest\.mock\.)?patch\((target=)?["\x27'"'"'](transformers|torch)(["\x27'"'"']|\.)' "$FILE" || true)

    if [ -n "$PATCH_LINES" ]; then
        # File has ML mocking - check for import guard
        HAS_IMPORT_GUARD=$(grep -q 'HAS_ML_DEPS\|HAS_.*_DEPS\|pytest\.importorskip' "$FILE" && echo "yes" || echo "no")

        # Also check if the test is marked as skipped (allowed to have violations)
        IS_SKIPPED=$(grep -q '@pytest\.mark\.skip\|@pytest\.mark\.skipif.*HAS_ML_DEPS' "$FILE" && echo "yes" || echo "no")

        if [ "$HAS_IMPORT_GUARD" = "no" ] && [ "$IS_SKIPPED" = "no" ]; then
            echo -e "${RED}✗ ML mock without import guard:${NC} $FILE"
            echo "$PATCH_LINES" | while IFS=: read -r LINE_NUM LINE_CONTENT; do
                echo -e "  ${YELLOW}Line $LINE_NUM:${NC} $LINE_CONTENT"
            done
            echo ""
            VIOLATIONS_FOUND=$((VIOLATIONS_FOUND + 1))
        fi
    fi
done

# Report results
if [ $VIOLATIONS_FOUND -gt 0 ]; then
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}ERROR: $VIOLATIONS_FOUND ML test isolation violation(s) detected${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "ML dependencies (transformers/torch) may not be installed in CI (offline mode)."
    echo "Tests using @patch() on these modules must have import guards."
    echo ""
    echo -e "${YELLOW}Fix options:${NC}"
    echo ""
    echo "1. Module-level import guard (RECOMMENDED):"
    echo ""
    echo "   try:"
    echo "       import transformers"
    echo "       import torch"
    echo "       HAS_ML_DEPS = True"
    echo "   except ImportError:"
    echo "       HAS_ML_DEPS = False"
    echo ""
    echo "   @pytest.mark.skipif(not HAS_ML_DEPS, reason=\"ML dependencies required\")"
    echo "   def test_something(self):"
    echo "       ..."
    echo ""
    echo "2. Use pytest.importorskip() in test:"
    echo ""
    echo "   def test_something(self):"
    echo "       transformers = pytest.importorskip(\"transformers\")"
    echo "       ..."
    echo ""
    echo -e "${YELLOW}Reference:${NC} docs/architecture/ADR-031-test-dependency-isolation.md"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓${NC} ML test isolation verified ($TEST_FILES)"
exit 0
