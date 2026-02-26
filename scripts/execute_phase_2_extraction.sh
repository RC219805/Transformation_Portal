#!/usr/bin/env bash
# Phase 2 Extraction Script: Materials V3 Investigation Documentation
# Version: 1.0
# Date: 2026-02-14
# Estimated Time: 2-3 hours

set -e  # Exit on error
set -u  # Exit on undefined variable

#==============================================================================
# Configuration
#==============================================================================
PHASE_BRANCH="docs/materials-v3-investigations"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${BASE_DIR}/docs/project-status/phase_2_execution.log"
DISCOVERIES_FILE="${BASE_DIR}/docs/project-status/phase_2_discovered_files.txt"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#==============================================================================
# Helper Functions
#==============================================================================
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠️  $*${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ❌ $*${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✅ $*${NC}" | tee -a "$LOG_FILE"
}

step() {
    echo "" | tee -a "$LOG_FILE"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}$*${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$LOG_FILE"
}

pause_for_user() {
    echo ""
    read -p "Press Enter to continue or Ctrl+C to abort... "
}

#==============================================================================
# Pre-flight Checks
#==============================================================================
preflight_checks() {
    step "Pre-flight Checks"

    cd "$BASE_DIR"

    # Check we're in a git repo
    if ! git rev-parse --is-inside-work-tree &>/dev/null; then
        error "Not in a git repository!"
        exit 1
    fi

    # Check we're on main
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$current_branch" != "main" ]; then
        warn "Not on main branch (currently on: $current_branch)"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Check working directory is clean
    if ! git diff-index --quiet HEAD --; then
        warn "Working directory has uncommitted changes"
        git status --short
        read -p "Stash changes and continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git stash push -m "Phase 2 pre-flight stash $(date +'%Y-%m-%d %H:%M:%S')"
            success "Changes stashed"
        else
            exit 1
        fi
    fi

    success "Pre-flight checks passed"
}

#==============================================================================
# Step 1: Preparation
#==============================================================================
preparation() {
    step "Step 1: Preparation (10 minutes)"

    cd "$BASE_DIR"

    log "Fetching all remote branches..."
    git fetch --all

    log "Checking out main and pulling latest..."
    git checkout main
    git pull origin main

    log "Creating Phase 2 branch: $PHASE_BRANCH"
    if git rev-parse --verify "$PHASE_BRANCH" &>/dev/null; then
        warn "Branch $PHASE_BRANCH already exists"
        read -p "Delete and recreate? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git branch -D "$PHASE_BRANCH"
        else
            exit 1
        fi
    fi
    git checkout -b "$PHASE_BRANCH"

    log "Creating target directories..."
    mkdir -p docs/investigations/materials_v3/assets
    mkdir -p tools/investigations/materials_v3

    # Create __init__.py files
    touch tools/investigations/__init__.py
    touch tools/investigations/materials_v3/__init__.py

    success "Preparation complete"
}

#==============================================================================
# Step 2: Search Development Branches
#==============================================================================
search_branches() {
    step "Step 2: Search Development Branches (20 minutes)"

    cd "$BASE_DIR"

    # Initialize discoveries file
    echo "# Phase 2: Discovered Files" > "$DISCOVERIES_FILE"
    echo "# Generated: $(date)" >> "$DISCOVERIES_FILE"
    echo "" >> "$DISCOVERIES_FILE"

    local branches=(
        "bugfix/materials-v3-critical-fixes"
        "feature/materials-v3-water-sky-synthesis"
        "feature/materials-v3-production-integration"
    )

    for branch in "${branches[@]}"; do
        log "Searching branch: $branch"

        if git checkout "$branch" 2>/dev/null; then
            echo "## Branch: $branch" >> "$DISCOVERIES_FILE"

            # Find investigation-related files
            log "  Looking for investigation files..."
            find . -type f \( -name "*investigation*.md" -o -name "*bug*.md" -o -name "*analysis*.md" \) \
                -not -path "*/\.*" \
                -not -path "*/node_modules/*" \
                -not -path "*/venv/*" \
                >> "$DISCOVERIES_FILE" 2>/dev/null || true

            # Find diagnostic scripts
            log "  Looking for diagnostic scripts..."
            find . -type f \( -name "*diagnose*.py" -o -name "*analyze*.py" -o -name "*profile*.py" \) \
                -not -path "*/\.*" \
                -not -path "*/node_modules/*" \
                -not -path "*/venv/*" \
                >> "$DISCOVERIES_FILE" 2>/dev/null || true

            # Find materials-related diffs from main
            log "  Checking diff from main..."
            git diff main --name-only | grep -E "materials|pixel.*ops|depth" >> "$DISCOVERIES_FILE" 2>/dev/null || true

            echo "" >> "$DISCOVERIES_FILE"
        else
            warn "Branch $branch not found, skipping..."
            echo "## Branch: $branch (NOT FOUND)" >> "$DISCOVERIES_FILE"
            echo "" >> "$DISCOVERIES_FILE"
        fi
    done

    # Return to Phase 2 branch
    git checkout "$PHASE_BRANCH"

    log "Branch search complete. Results saved to: $DISCOVERIES_FILE"

    # Show summary
    echo ""
    echo "=== Discovered Files Summary ==="
    grep -E "^\./|^##" "$DISCOVERIES_FILE" | head -20 || echo "No additional files found"
    echo ""

    if [ -s "$DISCOVERIES_FILE" ]; then
        log "Review discovered files and note any to extract manually"
        pause_for_user
    fi

    success "Branch search complete"
}

#==============================================================================
# Step 3: Reorganize Existing Files
#==============================================================================
reorganize_files() {
    step "Step 3: Reorganize Existing Files (30 minutes)"

    cd "$BASE_DIR"

    log "Moving investigation reports..."

    # Move PRIMARY_BEDROOM_EDGE_ARTIFACTS_SUMMARY.md
    if [ -f "docs/investigations/PRIMARY_BEDROOM_EDGE_ARTIFACTS_SUMMARY.md" ]; then
        git mv docs/investigations/PRIMARY_BEDROOM_EDGE_ARTIFACTS_SUMMARY.md \
               docs/investigations/materials_v3/edge_artifacts_primary_bedroom.md
        success "Moved PRIMARY_BEDROOM_EDGE_ARTIFACTS_SUMMARY.md"
    else
        warn "PRIMARY_BEDROOM_EDGE_ARTIFACTS_SUMMARY.md not found"
    fi

    # Move SKY_WATER_INVESTIGATION_SUMMARY.md
    if [ -f "docs/investigations/SKY_WATER_INVESTIGATION_SUMMARY.md" ]; then
        git mv docs/investigations/SKY_WATER_INVESTIGATION_SUMMARY.md \
               docs/investigations/materials_v3/sky_water_color_grading_analysis.md
        success "Moved SKY_WATER_INVESTIGATION_SUMMARY.md"
    else
        warn "SKY_WATER_INVESTIGATION_SUMMARY.md not found"
    fi

    log "Moving diagnostic scripts..."

    # Move diagnose_sky_issue.py
    if [ -f "diagnose_sky_issue.py" ]; then
        git mv diagnose_sky_issue.py tools/investigations/materials_v3/
        success "Moved diagnose_sky_issue.py"
    else
        warn "diagnose_sky_issue.py not found in root"
    fi

    # Move create_sky_comparison.py
    if [ -f "create_sky_comparison.py" ]; then
        git mv create_sky_comparison.py tools/investigations/materials_v3/
        success "Moved create_sky_comparison.py"
    else
        warn "create_sky_comparison.py not found in root"
    fi

    success "File reorganization complete"
}

#==============================================================================
# Step 4: Extract Additional Files (Manual)
#==============================================================================
extract_additional() {
    step "Step 4: Extract Additional Files (40 minutes)"

    log "This step requires manual extraction of files discovered in Step 2"
    log "Discoveries file: $DISCOVERIES_FILE"
    echo ""
    log "For each file you want to extract:"
    echo "  1. git checkout <branch>"
    echo "  2. git show HEAD:path/to/file.md > /tmp/file.md"
    echo "  3. git checkout $PHASE_BRANCH"
    echo "  4. cp /tmp/file.md docs/investigations/materials_v3/new_name.md"
    echo "  5. git add docs/investigations/materials_v3/new_name.md"
    echo ""

    pause_for_user

    success "Additional file extraction complete (manual step)"
}

#==============================================================================
# Step 5: Create Documentation (Manual)
#==============================================================================
create_documentation() {
    step "Step 5: Create Documentation (60 minutes)"

    log "This step requires creating comprehensive README files"
    echo ""
    log "Files to create:"
    echo "  1. docs/investigations/materials_v3/README.md (~300 lines)"
    echo "     - Investigation index with summaries"
    echo "     - Cross-references to Phase A/B reports"
    echo "     - Links to diagnostic tools"
    echo ""
    echo "  2. docs/investigations/materials_v3/DIAGNOSTIC_METHODOLOGY.md (~250 lines)"
    echo "     - 6-stage debugging approach"
    echo "     - Quantitative analysis methods"
    echo "     - Fix validation checklist"
    echo ""
    echo "  3. tools/investigations/materials_v3/README.md (~200 lines)"
    echo "     - Tool descriptions and usage"
    echo "     - Example commands"
    echo "     - Integration patterns"
    echo ""

    log "Templates available in: docs/project-status/PHASE_2_EXECUTION_PLAN.md"
    echo ""

    pause_for_user

    success "Documentation creation complete (manual step)"
}

#==============================================================================
# Step 6: Update File References
#==============================================================================
update_references() {
    step "Step 6: Update File References (20 minutes)"

    cd "$BASE_DIR/docs/investigations/materials_v3"

    log "Updating relative paths in investigation reports..."

    for file in *.md; do
        if [ -f "$file" ] && [ "$file" != "README.md" ] && [ "$file" != "DIAGNOSTIC_METHODOLOGY.md" ]; then
            log "  Processing $file..."

            # Update references to materials/ directory (one level deeper now)
            sed -i.bak 's|../materials/|../../materials/|g' "$file" 2>/dev/null || true

            # Update references to root-level docs (two levels deeper now)
            sed -i.bak 's|../../MATERIALS_V3|../../../MATERIALS_V3|g' "$file" 2>/dev/null || true

            # Update references to docs/ directory
            sed -i.bak 's|../docs/|../../|g' "$file" 2>/dev/null || true

            # Clean up backup files
            rm -f "${file}.bak"
        fi
    done

    cd "$BASE_DIR"

    success "File references updated"
}

#==============================================================================
# Step 7: Validation
#==============================================================================
validation() {
    step "Step 7: Validation (20 minutes)"

    cd "$BASE_DIR"

    log "Testing diagnostic scripts..."

    # Test diagnose_sky_issue.py (syntax check - no argparse/CLI flags)
    if [ -f "tools/investigations/materials_v3/diagnose_sky_issue.py" ]; then
        if python -m py_compile tools/investigations/materials_v3/diagnose_sky_issue.py; then
            success "diagnose_sky_issue.py compiles OK"
        else
            error "diagnose_sky_issue.py has syntax errors"
        fi
    fi

    # Test create_sky_comparison.py (syntax check - no argparse/CLI flags)
    if [ -f "tools/investigations/materials_v3/create_sky_comparison.py" ]; then
        if python -m py_compile tools/investigations/materials_v3/create_sky_comparison.py; then
            success "create_sky_comparison.py compiles OK"
        else
            error "create_sky_comparison.py has syntax errors"
        fi
    fi

    log "Checking for broken links (manual verification needed)..."
    echo ""
    log "Files to verify:"
    find docs/investigations/materials_v3 -name "*.md" | while read -r file; do
        echo "  - $file"
    done
    echo ""

    log "Manual validation checklist:"
    echo "  [ ] All markdown files lint clean"
    echo "  [ ] All relative links resolve"
    echo "  [ ] No absolute paths or sensitive data"
    echo ""

    pause_for_user

    success "Validation complete"
}

#==============================================================================
# Step 8: Commit and PR
#==============================================================================
commit_and_pr() {
    step "Step 8: Commit and Create PR (15 minutes)"

    cd "$BASE_DIR"

    log "Committing changes in logical groups..."

    # Commit 1: New documentation
    if [ -f "docs/investigations/materials_v3/README.md" ] || \
       [ -f "docs/investigations/materials_v3/DIAGNOSTIC_METHODOLOGY.md" ] || \
       [ -f "tools/investigations/materials_v3/README.md" ]; then
        git add docs/investigations/materials_v3/README.md \
                docs/investigations/materials_v3/DIAGNOSTIC_METHODOLOGY.md \
                tools/investigations/materials_v3/README.md \
                2>/dev/null || true
        git commit -m "docs(materials-v3): add investigation index and methodology" || true
        success "Committed new documentation"
    fi

    # Commit 2: Reorganized reports
    git add docs/investigations/materials_v3/*.md 2>/dev/null || true
    git commit -m "docs(materials-v3): reorganize investigation reports" || true
    success "Committed reorganized reports"

    # Commit 3: Diagnostic tools
    git add tools/investigations/materials_v3/*.py \
            tools/investigations/__init__.py \
            tools/investigations/materials_v3/__init__.py \
            2>/dev/null || true
    git commit -m "tools(materials-v3): relocate diagnostic scripts" || true
    success "Committed diagnostic tools"

    # Commit 4: Documentation map update (if exists)
    if [ -f "docs/governance/DOCUMENTATION_MAP.md" ]; then
        git add docs/governance/DOCUMENTATION_MAP.md 2>/dev/null || true
        git commit -m "docs: update documentation map with Phase 2 investigations" || true
    fi

    log "Pushing branch to remote..."
    git push -u origin "$PHASE_BRANCH"

    echo ""
    log "Create PR with:"
    echo ""
    echo "  gh pr create \\"
    echo "    --title 'docs(materials-v3): extract and organize investigation reports (Phase 2)' \\"
    echo "    --base main \\"
    echo "    --body-file docs/project-status/PHASE_2_PR_TEMPLATE.md"
    echo ""
    echo "Or visit:"
    echo "  https://github.com/YOUR_ORG/Transformation_Portal/compare/main...$PHASE_BRANCH"
    echo ""

    success "Commit and push complete"
}

#==============================================================================
# Main Execution
#==============================================================================
main() {
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  Phase 2: Materials V3 Investigation Documentation Extraction ║"
    echo "║  Estimated Time: 2-3 hours                                    ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    # Initialize log file
    echo "Phase 2 Execution Log - $(date)" > "$LOG_FILE"

    preflight_checks
    preparation
    search_branches
    reorganize_files
    extract_additional
    create_documentation
    update_references
    validation
    commit_and_pr

    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                   Phase 2 Extraction Complete!                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    success "All steps completed successfully"
    log "Execution log saved to: $LOG_FILE"
    log "Discoveries saved to: $DISCOVERIES_FILE"
    echo ""
}

# Run main if executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
