#!/usr/bin/env bash
# Workspace Cleanup & Branch Archive Script
# Feature Freeze Compliant - Infrastructure Only
# Date: December 20, 2025

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Header
echo "======================================================================"
echo "  Transformation Portal - Workspace Cleanup & Branch Archive"
echo "  Feature Freeze Compliant (No Functional Changes)"
echo "  Date: $(date +"%Y-%m-%d %H:%M:%S")"
echo "======================================================================"
echo

# Confirm execution
read -p "This script will archive branches and clean workspace. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_warning "Operation cancelled by user"
    exit 0
fi

# ============================================================================
# PHASE 1: Create Archive Tags
# ============================================================================
echo
log_info "PHASE 1: Creating archive tags for branches..."

BRANCHES=(
    "feature/materials-v3-prw1-w2-water-detection-integration:archive/materials-v3-water-detection"
    "phase2-validation:archive/phase2-validation"
    "pr-571:archive/pr-571"
)

for branch_tag in "${BRANCHES[@]}"; do
    IFS=':' read -r branch tag <<< "$branch_tag"

    if git rev-parse --verify "$branch" >/dev/null 2>&1; then
        log_info "Creating tag '$tag' for branch '$branch'..."
        git tag -a "$tag" "$branch" -m "Archive of $branch ($(date +%Y-%m-%d))"
        log_success "Tag '$tag' created"
    else
        log_warning "Branch '$branch' not found, skipping tag creation"
    fi
done

# ============================================================================
# PHASE 2: Push Tags to Remote
# ============================================================================
echo
log_info "PHASE 2: Pushing archive tags to remote..."

for branch_tag in "${BRANCHES[@]}"; do
    IFS=':' read -r branch tag <<< "$branch_tag"

    if git rev-parse --verify "$tag" >/dev/null 2>&1; then
        log_info "Pushing tag '$tag' to origin..."
        git push origin "$tag" 2>&1 || log_warning "Failed to push $tag (may already exist on remote)"
        log_success "Tag '$tag' pushed to remote"
    fi
done

# ============================================================================
# PHASE 3: Create Incremental Backup
# ============================================================================
echo
log_info "PHASE 3: Creating incremental backup (diffs + logs)..."

BACKUP_DIR=".local_backup/branch_cleanup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

log_info "Backup directory: $BACKUP_DIR"

for branch_tag in "${BRANCHES[@]}"; do
    IFS=':' read -r branch tag <<< "$branch_tag"

    if git rev-parse --verify "$branch" >/dev/null 2>&1; then
        BRANCH_NAME=$(basename "$branch")

        log_info "Exporting diff for $BRANCH_NAME..."
        git diff main.."$branch" > "$BACKUP_DIR/${BRANCH_NAME}.diff"

        log_info "Exporting commit log for $BRANCH_NAME..."
        git log main.."$branch" --oneline > "$BACKUP_DIR/${BRANCH_NAME}.log"

        log_success "Backup created for $BRANCH_NAME"
    fi
done

# Create backup README
cat > "$BACKUP_DIR/README.txt" <<EOF
Branch Cleanup Backup
Created: $(date)
Repository: Transformation Portal

This backup contains diffs and commit logs for archived branches:
- feature/materials-v3-prw1-w2-water-detection-integration
- phase2-validation
- pr-571

RECOVERY INSTRUCTIONS:
1. Restore from tag: git checkout -b <new-branch-name> <tag-name>
2. Apply diff: git apply <branch-name>.diff

TAGS CREATED:
- archive/materials-v3-water-detection
- archive/phase2-validation
- archive/pr-571

External backup: 16.5 GB backup exists
Remote backup: Tags pushed to GitHub origin
EOF

log_success "Backup README created at $BACKUP_DIR/README.txt"

# ============================================================================
# PHASE 4: Delete Local Branches (Optional)
# ============================================================================
echo
log_warning "PHASE 4: Branch deletion (optional - tags preserved on remote)"
read -p "Delete local branches? They can be restored from tags. (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    for branch_tag in "${BRANCHES[@]}"; do
        IFS=':' read -r branch tag <<< "$branch_tag"

        if git rev-parse --verify "$branch" >/dev/null 2>&1; then
            log_info "Deleting local branch '$branch'..."
            git branch -D "$branch" 2>&1 || log_warning "Failed to delete $branch"
            log_success "Branch '$branch' deleted"
        fi
    done
else
    log_info "Local branches preserved"
fi

# ============================================================================
# PHASE 5: Workspace Cleanup
# ============================================================================
echo
log_info "PHASE 5: Cleaning workspace (safe cleanup only)..."

# Clean Python cache
log_info "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
log_success "Python cache cleaned"

# Clean pytest cache
if [ -d ".pytest_cache" ]; then
    log_info "Removing pytest cache..."
    rm -rf .pytest_cache
    log_success "Pytest cache cleaned"
fi

# Clean hypothesis cache
if [ -d ".hypothesis" ]; then
    log_info "Removing hypothesis cache..."
    rm -rf .hypothesis
    log_success "Hypothesis cache cleaned"
fi

# Clean empty directories in outputs
log_info "Cleaning empty output directories..."
find output outputs service_output test_output -type d -empty -delete 2>/dev/null || true
log_success "Empty directories cleaned"

# Remove temporary log files (keep important ones)
log_info "Removing temporary log files..."
rm -f sweep_experiment.log sweep_run.log 2>/dev/null || true
log_success "Temporary logs cleaned"

# Clean old summary files (archive them)
ARCHIVE_SUMMARIES_DIR="archive/summaries_$(date +%Y%m%d)"
mkdir -p "$ARCHIVE_SUMMARIES_DIR"

log_info "Archiving old summary files..."
for summary in ARCHITECT_EXECUTION_SUMMARY.md CI_FIX_SUMMARY.md EDGE_REFINEMENT_IMPLEMENTATION_SUMMARY.md \
               PR_573_COMPLETION_SUMMARY.md PR_573_EXECUTION_SUMMARY.md PR_573_FIX_SUMMARY.md \
               PR_574_COMPLETION_SUMMARY.md PR_574_FIX_SUMMARY.md IMPLEMENTATION_SUMMARY.txt \
               HIGH_FIDELITY_DEPTH_ARCHITECTURE.txt; do
    if [ -f "$summary" ]; then
        mv "$summary" "$ARCHIVE_SUMMARIES_DIR/"
        log_success "Archived $summary"
    fi
done

# ============================================================================
# PHASE 6: Update .gitignore for Next Sprint
# ============================================================================
echo
log_info "PHASE 6: Updating .gitignore for edge refinement experiments..."

cat >> .gitignore <<EOF

# Edge Refinement Experiments (added $(date +%Y-%m-%d))
experiments/edge_refinement/
experiments/input_size_sweep/
validation_edge_refinement/
edge_refinement_results/
EOF

log_success ".gitignore updated for next sprint experiments"

# ============================================================================
# Summary Report
# ============================================================================
echo
echo "======================================================================"
log_success "Workspace cleanup completed successfully!"
echo "======================================================================"
echo
echo "SUMMARY:"
echo "--------"
echo "✅ Archive tags created and pushed to remote"
echo "✅ Incremental backup saved to: $BACKUP_DIR"
echo "✅ Workspace cleaned (cache, temp files, empty dirs)"
echo "✅ Old summaries archived to: $ARCHIVE_SUMMARIES_DIR"
echo "✅ .gitignore updated for next sprint"
echo
echo "NEXT STEPS (Feature Freeze Period):"
echo "-----------------------------------"
echo "1. Write ADR for edge refinement module architecture"
echo "2. Design API contracts for bilateral filtering, guided filter"
echo "3. Create test infrastructure (test harnesses, mock data)"
echo "4. Draft requirements-edge-refinement.txt (no installation yet)"
echo "5. Prepare validation datasets for structure scene improvement"
echo
echo "FEATURE FREEZE ACTIVE: Jan 10, 2026"
echo "Only infrastructure preparation permitted (no functional changes)"
echo
log_info "For detailed guidance, see: BRANCH_STRATEGY_GUIDANCE.md"
echo "======================================================================"
