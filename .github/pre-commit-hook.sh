#!/bin/bash
# Pre-commit hook for quality control
# Install: cp .github/pre-commit-hook.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

set -eo pipefail

echo "🔍 Running pre-commit quality checks..."

if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi

staged_files=()
while IFS= read -r file; do
    [ -n "$file" ] && staged_files+=("$file")
done < <(git diff --cached --name-only --diff-filter=ACM)

staged_py=()
for file in "${staged_files[@]}"; do
    case "$file" in
        *.py) staged_py+=("$file") ;;
    esac
done

# Block known core files from being accidentally omitted from version control.
if git ls-files --others --exclude-standard | grep -E '^(app\.py|portal\.html|tests/test_app_orchestrator_runtime\.py)$' >/dev/null; then
    echo "❌ Core files are untracked (app.py, portal.html, or tests/test_app_orchestrator_runtime.py). Add them explicitly before committing."
    exit 1
fi

# 1. Check for undefined names on staged Python files only.
if [ "${#staged_py[@]}" -gt 0 ]; then
    echo "Checking for undefined names..."
    flake8_out="$(mktemp)"
    trap 'rm -f "$flake8_out"' EXIT
    if ! "$PYTHON_BIN" -m flake8 --select=F821 "${staged_py[@]}" >"$flake8_out" 2>&1; then
        echo "❌ ERROR: Undefined names detected. Fix before committing."
        cat "$flake8_out"
        exit 1
    fi
fi

# 2. Auto-fix trailing whitespace in staged Python files.
if [ "${#staged_py[@]}" -gt 0 ]; then
    echo "Auto-fixing trailing whitespace..."
    for file in "${staged_py[@]}"; do
        if [ -f "$file" ]; then
            sed -i '' 's/[[:space:]]*$//' "$file" 2>/dev/null || sed -i 's/[[:space:]]*$//' "$file"
            git add "$file"
        fi
    done
fi

# 3. Check markdown file count in root (warning only).
markdown_count="$(find . -maxdepth 1 -name '*.md' -type f | wc -l | tr -d ' ')"
if [ "$markdown_count" -gt 10 ]; then
    echo "⚠️  WARNING: $markdown_count markdown files in root (max: 10)"
    echo "Consider moving some to docs/"
fi

# 4. Quick staged-file import heuristics.
if [ "${#staged_py[@]}" -gt 0 ]; then
    echo "Checking for missing imports..."
    if ! "$PYTHON_BIN" - "${staged_py[@]}" <<'PYEOF'
import ast
import sys

files = [f for f in sys.argv[1:] if f.endswith(".py")]
errors = []
for filepath in files:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filepath)
    except Exception:
        continue

    names_used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    undefined = names_used - {"self", "cls", "True", "False", "None"}
    if "iio" in undefined:
        errors.append(f"{filepath}: 'iio' used but imageio not imported")
    if "cv2" in undefined:
        errors.append(f"{filepath}: 'cv2' used but cv2 not imported")

if errors:
    print("❌ Import errors detected:")
    for error in errors:
        print(f"  {error}")
    sys.exit(1)
PYEOF
    then
        echo "Fix import errors before committing"
        exit 1
    fi
fi

echo "✅ Pre-commit checks passed"
