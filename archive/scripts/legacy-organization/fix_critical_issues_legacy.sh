#!/bin/bash
# Critical Issue Fixes for feat/rag-integration-complete
# Run this script to address P0 issues before push

set -e

echo "🔧 Fixing Critical Issues - feat/rag-integration-complete"
echo "=========================================================="
echo ""

# Verify we're on the right branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "feat/rag-integration-complete" ]; then
    echo "❌ ERROR: Not on feat/rag-integration-complete branch"
    echo "   Current branch: $CURRENT_BRANCH"
    exit 1
fi

echo "✅ On correct branch: $CURRENT_BRANCH"
echo ""

# ==============================================================================
# PHASE 1: Remove PNG Previews (Critical - 361 MB)
# ==============================================================================
echo "📸 Phase 1: Removing PNG preview files from git tracking..."
echo ""

PNG_COUNT=$(git ls-files | grep "input_images/.*\.png$" | wc -l | tr -d ' ')
if [ "$PNG_COUNT" -gt 0 ]; then
    echo "   Found $PNG_COUNT PNG files to remove:"
    git ls-files | grep "input_images/.*\.png$" | sed 's/^/     - /'

    git rm --cached input_images/*.png
    echo "   ✅ PNG files removed from git tracking"
else
    echo "   ℹ️  No PNG files to remove (already clean)"
fi
echo ""

# ==============================================================================
# PHASE 2: Move Client-Specific Files to Local Backup
# ==============================================================================
echo "🔒 Phase 2: Moving client-specific files to local backup..."
echo ""

# Create local backup directory (not tracked)
mkdir -p .local_backup/client_750picacho/scripts
mkdir -p .local_backup/client_750picacho/documentation

# Move Python scripts
echo "   Moving Python scripts..."
git mv conservative_enhance_greatroom*.py .local_backup/client_750picacho/scripts/ 2>/dev/null || true
git mv conservative_enhance_pool*.py .local_backup/client_750picacho/scripts/ 2>/dev/null || true
git mv conservative_enhance_kitchen.py .local_backup/client_750picacho/scripts/ 2>/dev/null || true
git mv conservative_enhance.py .local_backup/client_750picacho/scripts/ 2>/dev/null || true
git mv ai_enhance_750picacho*.py .local_backup/client_750picacho/scripts/ 2>/dev/null || true
git mv compare_pool_outputs.py .local_backup/client_750picacho/scripts/ 2>/dev/null || true
git mv process_renderings_750.py .local_backup/client_750picacho/scripts/ 2>/dev/null || true

# Move documentation
echo "   Moving documentation..."
git mv GREATROOM_*.md .local_backup/client_750picacho/documentation/ 2>/dev/null || true
git mv POOL_*.md .local_backup/client_750picacho/documentation/ 2>/dev/null || true
git mv KITCHEN_*.md .local_backup/client_750picacho/documentation/ 2>/dev/null || true
git mv ANALYSIS_750Picacho_*.md .local_backup/client_750picacho/documentation/ 2>/dev/null || true
git mv PHOTOREALISTIC_4K_WORKFLOW.md .local_backup/client_750picacho/documentation/ 2>/dev/null || true
git mv BUG_REPORT_2025-11-05.md .local_backup/client_750picacho/documentation/ 2>/dev/null || true

echo "   ✅ Client files moved to .local_backup/"
echo ""

# ==============================================================================
# PHASE 3: Remove Temporary Planning Files
# ==============================================================================
echo "🗑️  Phase 3: Removing temporary planning files..."
echo ""

git rm STOP_PUSH_NOW.md 2>/dev/null || true
git rm EXECUTE_CLEANUP.sh 2>/dev/null || true
git rm FIX_TIFF_PUSH.sh 2>/dev/null || true
git rm .gitignore.additions 2>/dev/null || true
git rm EXEC_SUMMARY.txt 2>/dev/null || true
git rm BINARY_CLEANUP_ACTION_PLAN.md 2>/dev/null || true
git rm BINARY_MANAGEMENT_SUMMARY.md 2>/dev/null || true
git rm BINARY_QUICK_REFERENCE.md 2>/dev/null || true
git rm GIT_TIFF_ANALYSIS.md 2>/dev/null || true

echo "   ✅ Temporary files removed"
echo ""

# ==============================================================================
# PHASE 4: Update .gitignore
# ==============================================================================
echo "📝 Phase 4: Updating .gitignore..."
echo ""

# Add local backup directory to gitignore
if ! grep -q ".local_backup/" .gitignore; then
    cat >> .gitignore << 'EOF'

# Local backup directory (client-specific work)
.local_backup/

# Additional PNG exclusions
input_images/**/*.png
!input_images/.gitkeep
EOF
    echo "   ✅ .gitignore updated"
else
    echo "   ℹ️  .gitignore already contains .local_backup/"
fi
echo ""

# ==============================================================================
# PHASE 5: Organize RAG Demo Files
# ==============================================================================
echo "📁 Phase 5: Organizing RAG demonstration files..."
echo ""

mkdir -p docs/examples/rag_demonstration/
git mv step*.md docs/examples/rag_demonstration/ 2>/dev/null || true
git mv step*.json docs/examples/rag_demonstration/ 2>/dev/null || true
git mv stats.json docs/examples/rag_demonstration/ 2>/dev/null || true
git mv artifacts.json docs/examples/rag_demonstration/ 2>/dev/null || true
git mv artifacts_catalog.json docs/examples/rag_demonstration/ 2>/dev/null || true

echo "   ✅ RAG demo files organized"
echo ""

# ==============================================================================
# PHASE 6: Remove .DS_Store files
# ==============================================================================
echo "🍎 Phase 6: Removing macOS .DS_Store files..."
echo ""

find . -name ".DS_Store" -type f -not -path "./.git/*" -delete
git rm --cached .DS_Store 2>/dev/null || true

echo "   ✅ .DS_Store files removed"
echo ""

# ==============================================================================
# PHASE 7: Stage Changes
# ==============================================================================
echo "💾 Phase 7: Staging all changes..."
echo ""

git add .gitignore
git add input_images/.gitkeep 2>/dev/null || (touch input_images/.gitkeep && git add input_images/.gitkeep)
git add docs/examples/rag_demonstration/ 2>/dev/null || true
git add -u

echo "   ✅ Changes staged"
echo ""

# ==============================================================================
# SUMMARY & NEXT STEPS
# ==============================================================================
echo "=========================================================="
echo "✅ Critical Issues Fixed!"
echo "=========================================================="
echo ""
echo "📊 Summary of Changes:"
git status --short
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Review the changes:"
echo "   git diff --cached --stat"
echo ""
echo "2. Commit the fixes:"
echo "   git commit -m \"chore: Fix critical issues before push"
echo ""
echo "   - Remove PNG previews from git tracking (361 MB)"
echo "   - Move client-specific files to local backup"
echo "   - Remove temporary planning files"
echo "   - Organize RAG demo files into docs/examples/"
echo "   - Update .gitignore for comprehensive binary exclusion"
echo "   - Remove macOS .DS_Store files"
echo ""
echo "   Fixes privacy, repository bloat, and organization issues"
echo "   identified in PRE_PUSH_AUDIT_REPORT.md\""
echo ""
echo "3. Verify no large files:"
echo "   git ls-files | xargs du -ch 2>/dev/null | tail -1"
echo ""
echo "4. Push to GitHub:"
echo "   git push origin feat/rag-integration-complete"
echo ""
echo "⚠️  NOTE: If you already pushed commits with large files,"
echo "   you may need to force push after amending:"
echo "   git commit --amend --no-edit"
echo "   git push origin feat/rag-integration-complete --force-with-lease"
echo ""
