#!/bin/bash
# Legacy pre-commit hook for quality control
#
# DEPRECATED: This file is retained for compatibility with workflows that may
# reference it directly. The canonical approach is to use the pre-commit
# framework via `.pre-commit-config.yaml`.
#
# Preferred installation method:
#   make install-hooks
#   # or
#   pre-commit install -f
#
# For manual execution of the unified quality gate:
#   ./scripts/pre_commit_hook.sh
#
# Direct manual installation (not recommended):
#   cp .github/pre-commit-hook.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine repository root from script location
if [[ "$SCRIPT_DIR" == */.git/hooks ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
elif [[ "$SCRIPT_DIR" == */.github ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
fi

# Delegate to the unified quality gate if it exists
UNIFIED_GATE="$REPO_ROOT/scripts/pre_commit_hook.sh"
if [[ -x "$UNIFIED_GATE" ]]; then
    echo "🔍 Delegating to unified pre-commit quality gate..."
    exec "$UNIFIED_GATE" "$@"
fi

# Fallback: run inline checks if the unified gate is missing
echo "🔍 Running pre-commit quality checks (legacy fallback)..."

cd "$REPO_ROOT"

if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
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

# 1. Ensure local lint tool versions match CI pins.
if [ "${#staged_py[@]}" -gt 0 ]; then
    echo "Checking lint tool version parity..."
    if ! "$PYTHON_BIN" - <<'PYEOF'
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

req_path = Path("requirements-lint.txt")
if not req_path.exists():
    print("❌ requirements-lint.txt not found")
    sys.exit(1)

pinned = {}
for raw in req_path.read_text(encoding="utf-8").splitlines():
    line = raw.split("#", 1)[0].strip()
    if not line or "==" not in line:
        continue
    name, ver = line.split("==", 1)
    pinned[name.strip().lower()] = ver.strip()

errors = []
for tool in ("black", "isort", "flake8", "pylint"):
    expected = pinned.get(tool)
    if not expected:
        errors.append(f"{tool}: missing exact pin in requirements-lint.txt")
        continue
    try:
        actual = version(tool)
    except PackageNotFoundError:
        errors.append(f"{tool}: not installed (expected {expected})")
        continue
    if actual != expected:
        errors.append(f"{tool}: expected {expected}, found {actual}")

if errors:
    print("❌ Lint tool versions do not match CI pins:")
    for item in errors:
        print(f"  - {item}")
    sys.exit(1)
PYEOF
    then
        echo "Install exact lint tooling with: $PYTHON_BIN -m pip install -r requirements-lint.txt"
        exit 1
    fi
fi

# 2. Check for undefined names on staged Python files only.
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

# 3. Auto-fix trailing whitespace in staged Python files.
if [ "${#staged_py[@]}" -gt 0 ]; then
    echo "Auto-fixing trailing whitespace..."
    for file in "${staged_py[@]}"; do
        if [ -f "$file" ]; then
            # Portable sed: try GNU-style first, fall back to BSD-style
            if sed --version >/dev/null 2>&1; then
                sed -i 's/[[:space:]]*$//' "$file"
            else
                sed -i '' 's/[[:space:]]*$//' "$file"
            fi
            git add "$file"
        fi
    done
fi

# 4. Enforce black/isort checks using CI-pinned versions.
if [ "${#staged_py[@]}" -gt 0 ]; then
    echo "Checking black formatting parity with CI..."
    if ! "$PYTHON_BIN" -m black --check --diff --line-length=127 "${staged_py[@]}"; then
        echo "❌ Black check failed (CI parity)."
        exit 1
    fi
    echo "Checking isort ordering parity with CI..."
    if ! "$PYTHON_BIN" -m isort --check-only --diff --profile=black --line-length=127 "${staged_py[@]}"; then
        echo "❌ isort check failed (CI parity)."
        exit 1
    fi
fi

# 5. Check markdown file count in root (warning only).
markdown_count="$(find . -maxdepth 1 -name '*.md' -type f | wc -l | tr -d ' ')"
if [ "$markdown_count" -gt 10 ]; then
    echo "⚠️  WARNING: $markdown_count markdown files in root (max: 10)"
    echo "Consider moving some to docs/"
fi

# 6. Quick staged-file import heuristics.
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
