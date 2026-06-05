#!/bin/bash
# =============================================================================
# Fix Large TIFF Files in Git Push
# =============================================================================
# This script removes large TIFF files from git tracking while preserving
# them locally. Run this to fix the feat/rag-integration-complete branch.
# =============================================================================

set -e  # Exit on error

echo "🔍 Analyzing current git state..."
echo ""

# Check if we're on the right branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "feat/rag-integration-complete" ]; then
    echo "⚠️  WARNING: You're on branch '$CURRENT_BRANCH'"
    echo "   Expected: feat/rag-integration-complete"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📊 Current repository size:"
du -sh .git
echo ""

echo "📁 TIFF files in input_images/:"
find input_images -name "*.tif*" -type f | wc -l
echo ""

echo "🗑️  Step 1: Removing TIFF files from git tracking (keeps local files)..."
git rm --cached input_images/*.tif 2>/dev/null || echo "   No .tif files tracked"
git rm --cached input_images/*.tiff 2>/dev/null || echo "   No .tiff files tracked"
echo "   ✓ Files removed from tracking"
echo ""

echo "📝 Step 2: Adding updated .gitignore and .gitkeep..."
git add .gitignore input_images/.gitkeep
echo "   ✓ .gitignore and .gitkeep staged"
echo ""

echo "💾 Step 3: Committing the fix..."
git commit -m "fix: Add input_images/ to .gitignore to prevent large binary files

- Mirrors existing data/sample_images/ pattern
- Prevents 2.7GB of TIFF files from bloating repository
- Maintains directory structure with .gitkeep
- Follows repository best practices for binary asset handling

Refs: GIT_TIFF_ANALYSIS.md for full analysis"
echo "   ✓ Changes committed"
echo ""

echo "✅ Local fix complete!"
echo ""
echo "Next steps:"
echo "1. Review the commit: git show HEAD"
echo "2. Force push to clean remote: git push origin $CURRENT_BRANCH --force"
echo ""
echo "⚠️  WARNING: Force push will rewrite remote history!"
echo "   Only proceed if others are not working on this branch."
echo ""

# Verification
echo "🔍 Verification:"
echo "Files now ignored:"
git check-ignore input_images/*.tiff 2>/dev/null | head -5 || echo "   (Check manually with: git check-ignore input_images/*.tiff)"
echo ""

echo "📊 Repository size after fix:"
du -sh .git
echo ""

echo "✨ Done! Review the changes above and force-push when ready."
