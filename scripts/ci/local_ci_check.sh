#!/usr/bin/env bash
# Local CI simulation for Transformation Portal
# Replicates exact CI environment checks before pushing
#
# Usage:
#   ./scripts/local_ci_check.sh [--quick] [--python VERSION]
#
# Options:
#   --quick         Run only fast checks (skip full test suite)
#   --python 3.10   Test with specific Python version (default: current)

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
QUICK_MODE=0
PYTHON_VERSION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=1
            shift
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick] [--python VERSION]"
            exit 1
            ;;
    esac
done

# Determine Python command
if [ -n "$PYTHON_VERSION" ]; then
    PYTHON="python$PYTHON_VERSION"
    if ! command -v "$PYTHON" >/dev/null 2>&1; then
        echo -e "${RED}✗ Python $PYTHON_VERSION not found${NC}"
        exit 1
    fi
else
    # Prefer venv python, then python3, then python
    if [ -x .venv/bin/python ]; then
        PYTHON=.venv/bin/python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    else
        PYTHON=python
    fi
fi

PYTHON_VER_FULL=$($PYTHON --version 2>&1 | cut -d' ' -f2)
echo -e "${BLUE}🐍 Using Python $PYTHON_VER_FULL${NC}"

# Get repository root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$REPO_ROOT"

echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Transformation Portal - Local CI Check   ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"

FAILED=0

# ============================================================================
# 1. ENVIRONMENT SETUP
# ============================================================================
echo -e "\n${BLUE}[1/6] Environment Setup${NC}"
echo -e "${YELLOW}→ Checking required tools...${NC}"

# Check for required commands
for cmd in git; do
    if command -v $cmd >/dev/null 2>&1; then
        echo -e "${GREEN}  ✓ $cmd${NC}"
    else
        echo -e "${RED}  ✗ $cmd not found${NC}"
        echo -e "${YELLOW}  💡 Install with: pip install $cmd${NC}"
        FAILED=1
    fi
done

for module in flake8 pylint pytest; do
    if "$PYTHON" -m "$module" --version >/dev/null 2>&1; then
        echo -e "${GREEN}  ✓ $module ($PYTHON)${NC}"
    else
        echo -e "${RED}  ✗ $module not available for $PYTHON${NC}"
        echo -e "${YELLOW}  💡 Install with: $PYTHON -m pip install $module${NC}"
        FAILED=1
    fi
done

if [ $FAILED -eq 1 ]; then
    echo -e "${RED}✗ Missing required tools${NC}"
    exit 1
fi

# ============================================================================
# 2. LINTING
# ============================================================================
echo -e "\n${BLUE}[2/6] Shared Lint Checks${NC}"
echo -e "${YELLOW}→ Running shared lint policy...${NC}"

if PYTHON_BIN="$PYTHON" ./scripts/ci/lint_runner.sh local; then
    echo -e "${GREEN}✓ Shared lint checks passed${NC}"
else
    echo -e "${RED}✗ Shared lint checks failed${NC}"
    FAILED=1
fi

# ============================================================================
# 3. JSON SERIALIZATION GUARDRAILS
# ============================================================================
echo -e "\n${BLUE}[3/6] JSON Serialization Guardrails${NC}"
echo -e "${YELLOW}→ Checking for raw json.dump/json.dumps usage outside approved modules...${NC}"
if "$PYTHON" scripts/validation/check_raw_json_usage.py; then
    echo -e "${GREEN}✓ JSON serialization guardrails passed${NC}"
else
    echo -e "${RED}✗ JSON serialization guardrails failed${NC}"
    FAILED=1
fi

echo -e "${YELLOW}→ Checking for tracked pip-tools cache artifacts...${NC}"
if "$PYTHON" scripts/validation/check_piptools_cache_tracked.py; then
    echo -e "${GREEN}✓ pip-tools cache guardrails passed${NC}"
else
    echo -e "${RED}✗ pip-tools cache guardrails failed${NC}"
    FAILED=1
fi

# ============================================================================
# 4. DOCUMENTATION STRUCTURE
# ============================================================================
echo -e "\n${BLUE}[4/6] Documentation Structure${NC}"
echo -e "${YELLOW}→ Checking root file placement policy...${NC}"

if ./scripts/setup/pre-commit-check.sh --all; then
    echo -e "${GREEN}✓ Root file placement policy passed${NC}"
else
    echo -e "${RED}✗ Root file placement policy failed${NC}"
    FAILED=1
fi

# ============================================================================
# 5. PYTEST (TESTS)
# ============================================================================
echo -e "\n${BLUE}[5/6] Test Suite${NC}"

if [ $QUICK_MODE -eq 1 ]; then
    echo -e "${YELLOW}→ Running fast tests only (--quick mode)...${NC}"
    if $PYTHON -m pytest -q \
        tests/test_material_response.py \
        tests/test_depth_tools.py \
        tests/test_float_roundtrip.py \
        --maxfail=3 2>&1; then
        echo -e "${GREEN}✓ Fast tests passed${NC}"
    else
        echo -e "${RED}✗ Fast tests failed${NC}"
        FAILED=1
    fi
else
    echo -e "${YELLOW}→ Running full test suite...${NC}"

    # Check if pytest-xdist is available for parallel testing
    if $PYTHON -c "import xdist" 2>/dev/null; then
        echo -e "${YELLOW}   Using parallel execution (pytest-xdist)${NC}"
        PYTEST_ARGS="-n auto"
    else
        echo -e "${YELLOW}   Using serial execution (install pytest-xdist for parallel)${NC}"
        PYTEST_ARGS=""
    fi

    if $PYTHON -m pytest $PYTEST_ARGS -v tests/ --maxfail=5; then
        echo -e "${GREEN}✓ All tests passed${NC}"
    else
        echo -e "${RED}✗ Tests failed${NC}"
        FAILED=1
    fi
fi

# ============================================================================
# 6. ADDITIONAL CHECKS
# ============================================================================
echo -e "\n${BLUE}[6/6] Additional Quality Checks${NC}"

# Check for debugging statements
echo -e "${YELLOW}→ Checking for debugging statements...${NC}"
DEBUG_FILES=$(git ls-files '*.py' | xargs grep -l 'import pdb\|pdb.set_trace()\|breakpoint()' 2>/dev/null || true)
if [ -n "$DEBUG_FILES" ]; then
    echo -e "${YELLOW}⚠ Found debugging statements in:${NC}"
    echo "$DEBUG_FILES"
else
    echo -e "${GREEN}✓ No debugging statements${NC}"
fi

# Check for TODO/FIXME
echo -e "${YELLOW}→ Checking for TODO/FIXME comments...${NC}"
TODO_COUNT=$(git ls-files '*.py' | xargs grep -c 'TODO\|FIXME' 2>/dev/null | awk -F: '{sum+=$2} END {print sum+0}')
if [ "$TODO_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Found $TODO_COUNT TODO/FIXME comments${NC}"
else
    echo -e "${GREEN}✓ No TODO/FIXME comments${NC}"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [ $FAILED -eq 1 ]; then
    echo -e "${RED}✗ Local CI checks FAILED${NC}"
    echo -e "${YELLOW}💡 Fix the issues above before pushing${NC}"
    exit 1
else
    echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✓ ALL CHECKS PASSED - READY TO PUSH!     ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
    exit 0
fi
