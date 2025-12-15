#!/bin/bash
# Pre-commit hook to prevent accidentally committing water_v0 images
#
# Install:
#   cp scripts/pre-commit-water-safety.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit

echo "🔍 Checking for accidentally staged images..."

# Check if any images are staged
STAGED_IMAGES=$(git diff --cached --name-only | grep -E '^data/water_v0/images/.*\.(jpg|jpeg|png)$' || true)

if [ -n "$STAGED_IMAGES" ]; then
    echo ""
    echo "❌ ERROR: Water dataset images are staged for commit!"
    echo ""
    echo "The following image files are staged:"
    echo "$STAGED_IMAGES" | sed 's/^/  • /'
    echo ""
    echo "Images should NOT be committed to git (they are generated synthetically)."
    echo ""
    echo "To unstage these files, run:"
    echo "  git reset HEAD data/water_v0/images/"
    echo ""
    echo "To proceed anyway (NOT RECOMMENDED), use:"
    echo "  git commit --no-verify"
    echo ""
    exit 1
fi

# Also check for large files (> 1MB) anywhere
LARGE_FILES=$(git diff --cached --name-only | while read file; do
    if [ -f "$file" ]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        if [ "$size" -gt 1048576 ]; then
            echo "$file ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "${size}B"))"
        fi
    fi
done)

if [ -n "$LARGE_FILES" ]; then
    echo ""
    echo "⚠️  WARNING: Large files detected in commit:"
    echo "$LARGE_FILES" | sed 's/^/  • /'
    echo ""
    echo "Consider whether these files should be committed to git."
    echo ""
    # Warning only, not blocking
fi

echo "✅ OK: No images staged"
exit 0
