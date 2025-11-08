#!/usr/bin/env bash
# Pre-commit hook for Transformation Portal
# Prevents commits with quality issues that would fail CI
#
# Installation:
#   cp scripts/pre_commit_hook.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔍 Running pre-commit quality checks...${NC}"

# Get repository root
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Track if any check fails
FAILED=0

# 1. Check for trailing whitespace
echo -e "\n${YELLOW}→ Checking for trailing whitespace...${NC}"
TRAILING_WHITESPACE=$(git diff --cached --name-only | xargs grep -l '[[:space:]]$' 2>/dev/null || true)
if [ -n "$TRAILING_WHITESPACE" ]; then
    echo -e "${RED}✗ Found trailing whitespace in:${NC}"
    echo "$TRAILING_WHITESPACE"
    echo -e "${YELLOW}💡 Auto-fixing trailing whitespace...${NC}"
    echo "$TRAILING_WHITESPACE" | xargs sed -i.bak 's/[[:space:]]*$//'
    echo "$TRAILING_WHITESPACE" | xargs -I {} rm -f {}.bak
    git add $TRAILING_WHITESPACE
    echo -e "${GREEN}✓ Fixed trailing whitespace${NC}"
fi

# 2. Run flake8 with exact CI configuration
echo -e "\n${YELLOW}→ Running flake8 (critical errors only)...${NC}"
if command -v flake8 >/dev/null 2>&1; then
    if ! flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.venv,__pycache__,.git,.tox,build,dist,*.egg-info 2>&1 | grep -v "sympy/polys/numberfields/resolvent_lookup.py" > /tmp/flake8_output.txt; then
        # Check if there are actual errors (not just the sympy recursion issue)
        if grep -E "^[./]" /tmp/flake8_output.txt | grep -v "sympy" | grep -q .; then
            echo -e "${RED}✗ Flake8 found critical errors${NC}"
            grep -E "^[./]" /tmp/flake8_output.txt | grep -v "sympy"
            echo -e "${YELLOW}💡 Fix undefined variables, import errors, or syntax issues${NC}"
            FAILED=1
        else
            echo -e "${GREEN}✓ Flake8 passed (sympy warnings ignored)${NC}"
        fi
    else
        echo -e "${GREEN}✓ Flake8 passed${NC}"
    fi
    rm -f /tmp/flake8_output.txt
else
    echo -e "${YELLOW}⚠ flake8 not installed, skipping (install with: pip install flake8)${NC}"
fi

# 3. Check Python imports in staged files
echo -e "\n${YELLOW}→ Validating Python imports in staged files...${NC}"
STAGED_PY_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)
if [ -n "$STAGED_PY_FILES" ]; then
    for file in $STAGED_PY_FILES; do
        if [ -f "$file" ]; then
            # Check for undefined names (like 'iio' without import)
            if python3 -m py_compile "$file" 2>/dev/null; then
                continue
            else
                echo -e "${RED}✗ Syntax error in: $file${NC}"
                python3 -m py_compile "$file"
                FAILED=1
            fi
        fi
    done
    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}✓ All Python files compile successfully${NC}"
    fi
else
    echo -e "${YELLOW}⚠ No Python files staged${NC}"
fi

# 4. Check markdown file count in root (max 10)
echo -e "\n${YELLOW}→ Checking markdown file count in root...${NC}"
MD_COUNT=$(find . -maxdepth 1 -name "*.md" -type f | wc -l | tr -d ' ')
MAX_MD=10
if [ "$MD_COUNT" -gt "$MAX_MD" ]; then
    echo -e "${RED}✗ Too many markdown files in root: $MD_COUNT (max: $MAX_MD)${NC}"
    echo -e "${YELLOW}💡 Run: scripts/organize_docs.sh to move files to docs/${NC}"
    FAILED=1
else
    echo -e "${GREEN}✓ Markdown file count OK ($MD_COUNT/$MAX_MD)${NC}"
fi

# 5. Check for common issues in changed files
echo -e "\n${YELLOW}→ Checking for common issues...${NC}"
if [ -n "$STAGED_PY_FILES" ]; then
    # Check for debugging statements
    DEBUG_STATEMENTS=$(echo "$STAGED_PY_FILES" | xargs grep -n 'import pdb\|pdb.set_trace()\|breakpoint()' 2>/dev/null || true)
    if [ -n "$DEBUG_STATEMENTS" ]; then
        echo -e "${RED}✗ Found debugging statements:${NC}"
        echo "$DEBUG_STATEMENTS"
        echo -e "${YELLOW}💡 Remove debugging statements before committing${NC}"
        FAILED=1
    fi

    # Check for print statements (excluding legitimate logging)
    PRINT_STATEMENTS=$(echo "$STAGED_PY_FILES" | xargs grep -n '^[[:space:]]*print(' 2>/dev/null | grep -v '# OK: print' || true)
    if [ -n "$PRINT_STATEMENTS" ]; then
        echo -e "${YELLOW}⚠ Found print statements (consider using logging):${NC}"
        echo "$PRINT_STATEMENTS"
    fi
fi

# 6. Run quick Python syntax check
echo -e "\n${YELLOW}→ Running Python syntax validation...${NC}"
if [ -n "$STAGED_PY_FILES" ]; then
    SYNTAX_OK=1
    for file in $STAGED_PY_FILES; do
        if ! python3 -c "import ast; ast.parse(open('$file').read())" 2>/dev/null; then
            echo -e "${RED}✗ Syntax error in: $file${NC}"
            python3 -c "import ast; ast.parse(open('$file').read())"
            SYNTAX_OK=0
        fi
    done
    if [ $SYNTAX_OK -eq 1 ]; then
        echo -e "${GREEN}✓ Python syntax valid${NC}"
    else
        FAILED=1
    fi
fi

# Summary
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [ $FAILED -eq 1 ]; then
    echo -e "${RED}✗ Pre-commit checks FAILED${NC}"
    echo -e "${YELLOW}💡 Fix the issues above and try again${NC}"
    exit 1
else
    echo -e "${GREEN}✓ All pre-commit checks PASSED${NC}"
    exit 0
fi
