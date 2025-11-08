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
for cmd in git flake8 pylint pytest; do
    if command -v $cmd >/dev/null 2>&1; then
        echo -e "${GREEN}  ✓ $cmd${NC}"
    else
        echo -e "${RED}  ✗ $cmd not found${NC}"
        echo -e "${YELLOW}  💡 Install with: pip install $cmd${NC}"
        FAILED=1
    fi
done

if [ $FAILED -eq 1 ]; then
    echo -e "${RED}✗ Missing required tools${NC}"
    exit 1
fi

# ============================================================================
# 2. FLAKE8 LINTING
# ============================================================================
echo -e "\n${BLUE}[2/6] Flake8 (Critical Errors)${NC}"
echo -e "${YELLOW}→ Running flake8 with CI configuration...${NC}"
echo -e "${YELLOW}   Checks: E9 (syntax), F63 (invalid), F7 (syntax), F82 (undefined)${NC}"

if flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics; then
    echo -e "${GREEN}✓ Flake8 passed (no critical errors)${NC}"
else
    echo -e "${RED}✗ Flake8 found critical errors${NC}"
    FAILED=1
fi

# ============================================================================
# 3. PYLINT
# ============================================================================
echo -e "\n${BLUE}[3/6] Pylint (Changed Files)${NC}"
echo -e "${YELLOW}→ Running pylint on changed files...${NC}"

# Get changed Python files (exclude deprecated, src/transformation_portal, scripts)
CHANGED_FILES=$(git diff --name-only origin/main...HEAD 2>/dev/null | grep '\.py$' | grep -v -e '/deprecated/' -e 'src/transformation_portal/' -e 'scripts/' || echo "")

if [ -z "$CHANGED_FILES" ]; then
    echo -e "${YELLOW}⚠ No changed Python files (checking all files)${NC}"
    CHANGED_FILES=$(git ls-files '*.py' | grep -v -e '/deprecated/' -e 'src/transformation_portal/' -e 'scripts/' | head -20)
fi

if [ -n "$CHANGED_FILES" ]; then
    set +e
    pylint $CHANGED_FILES
    PYLINT_EXIT=$?
    set -e

    # Check pylint exit code (bitwise flags)
    # 1=fatal, 2=error, 4=warning, 8=refactor, 16=convention, 32=usage
    if [ $((PYLINT_EXIT & 1)) -ne 0 ] || [ $((PYLINT_EXIT & 2)) -ne 0 ] || [ $((PYLINT_EXIT & 32)) -ne 0 ]; then
        echo -e "${RED}✗ Pylint found critical issues (exit code: $PYLINT_EXIT)${NC}"
        FAILED=1
    elif [ $PYLINT_EXIT -ne 0 ]; then
        echo -e "${YELLOW}⚠ Pylint found warnings/suggestions (exit code: $PYLINT_EXIT)${NC}"
        echo -e "${GREEN}✓ No critical errors (warnings acceptable)${NC}"
    else
        echo -e "${GREEN}✓ Pylint passed (no issues)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ No Python files to check${NC}"
fi

# ============================================================================
# 4. DOCUMENTATION STRUCTURE
# ============================================================================
echo -e "\n${BLUE}[4/6] Documentation Structure${NC}"
echo -e "${YELLOW}→ Checking markdown file count in root...${NC}"

MD_COUNT=$(find . -maxdepth 1 -name "*.md" -type f | wc -l | tr -d ' ')
MAX_MD=10

if [ "$MD_COUNT" -gt "$MAX_MD" ]; then
    echo -e "${RED}✗ Too many markdown files in root: $MD_COUNT (max: $MAX_MD)${NC}"
    echo -e "${YELLOW}💡 Run: ./scripts/organize_docs.sh${NC}"
    FAILED=1
else
    echo -e "${GREEN}✓ Markdown file count OK ($MD_COUNT/$MAX_MD)${NC}"
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
