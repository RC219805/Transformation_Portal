#!/bin/bash
# Pre-commit hook for quality control
# Install: cp .github/pre-commit-hook.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

echo "🔍 Running pre-commit quality checks..."

# 1. Check for undefined variables (like 'iio' issue)
echo "Checking for undefined names..."
python3 -m flake8 --select=F821 $(git diff --cached --name-only --diff-filter=ACM | grep '\.py$') 2>&1 | tee /tmp/flake8-undefined.txt
if [ -s /tmp/flake8-undefined.txt ]; then
    echo "❌ ERROR: Undefined names detected. Fix before committing."
    cat /tmp/flake8-undefined.txt
    exit 1
fi

# 2. Auto-fix trailing whitespace
echo "Auto-fixing trailing whitespace..."
for file in $(git diff --cached --name-only --diff-filter=ACM | grep '\.py$'); do
    if [ -f "$file" ]; then
        sed -i '' 's/[[:space:]]*$//' "$file" 2>/dev/null || sed -i 's/[[:space:]]*$//' "$file"
        git add "$file"
    fi
done

# 3. Check markdown file count in root
markdown_count=$(find . -maxdepth 1 -name "*.md" -type f | wc -l | tr -d ' ')
if [ "$markdown_count" -gt 10 ]; then
    echo "⚠️  WARNING: $markdown_count markdown files in root (max: 10)"
    echo "Consider moving some to docs/"
    # Non-blocking warning
fi

# 4. Check for missing imports
echo "Checking for missing imports..."
python3 << 'PYEOF'
import sys
import ast
import subprocess

files = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'], text=True).strip().split('\n')
python_files = [f for f in files if f.endswith('.py')]

errors = []
for filepath in python_files:
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read(), filepath)
            # Basic check for common missing imports
            names_used = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    names_used.add(node.id)

            # Check for common undefined names
            undefined = names_used - {'self', 'cls', 'True', 'False', 'None'}
            if 'iio' in undefined:
                errors.append(f"{filepath}: 'iio' used but imageio not imported")
            if 'cv2' in undefined:
                errors.append(f"{filepath}: 'cv2' used but cv2 not imported")
    except Exception as e:
        continue

if errors:
    print("❌ Import errors detected:")
    for error in errors:
        print(f"  {error}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "Fix import errors before committing"
    exit 1
fi

echo "✅ Pre-commit checks passed"
exit 0
