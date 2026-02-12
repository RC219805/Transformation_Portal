#!/bin/bash
set -e

echo "📦 Adding remaining useful documentation and scripts..."

# Add process documentation (useful for future reference)
git add PUSH_INSTRUCTIONS.md
git add GIT_TIFF_ANALYSIS.md

# Add cleanup/fix scripts (useful tools)
git add FIX_CRITICAL_ISSUES.sh
git add FIX_TIFF_PUSH.sh
git add EXECUTE_CLEANUP.sh

# Add .gitignore additions (reference for what was added)
git add .gitignore.additions

echo "✅ Files staged"
echo ""
echo "📋 Summary of what was added:"
git status --short | grep "^A"
echo ""
echo "💾 Ready to commit. Run:"
echo "   git commit -m 'chore: Add process documentation and utility scripts'"
