#!/bin/bash
# =============================================================================
# Phase 2 RAG System - Git Hook Installation
# Action 1: Enable real-time index synchronization
# =============================================================================
#
# This script installs git hooks that trigger incremental RAG index updates
# on commit, merge, checkout, and pre-push operations.
#
# Expected Outcomes:
#   - Post-commit: Index updated with committed changes (<500ms)
#   - Post-merge: Index updated after pull/merge operations
#   - Post-checkout: Cache validated on branch switches
#   - Pre-push: Consistency verification before push
#
# Usage:
#   ./1_install_git_hooks.sh [--dry-run] [--verbose]
#
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--dry-run] [--verbose]"
            echo ""
            echo "Options:"
            echo "  --dry-run   Preview changes without installing"
            echo "  --verbose   Enable verbose output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# =============================================================================
# Prerequisite Checks
# =============================================================================

log_step "Checking prerequisites..."

# Find repository root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "")
if [[ -z "$REPO_ROOT" ]]; then
    log_error "Not inside a git repository. Please run from within the Transformation Portal repo."
    exit 1
fi
log_info "Repository root: $REPO_ROOT"

# Check for git_hooks.py
GIT_HOOKS_PY="$REPO_ROOT/.github/agents/rag_system/git_hooks.py"
if [[ ! -f "$GIT_HOOKS_PY" ]]; then
    log_error "git_hooks.py not found at expected location: $GIT_HOOKS_PY"
    log_info "Ensure Phase 2 deployment is complete before running this script."
    exit 1
fi
log_info "Found git_hooks.py: $GIT_HOOKS_PY"

# Check Python availability
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is required but not found in PATH"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
log_info "Python version: $PYTHON_VERSION"

# =============================================================================
# Hook Installation
# =============================================================================

log_step "Installing git hooks..."

if $DRY_RUN; then
    log_warning "DRY RUN MODE - No changes will be made"
    echo ""
    echo "Would execute:"
    echo "  cd $REPO_ROOT"
    echo "  python3 $GIT_HOOKS_PY install"
    echo ""
    echo "This would create the following hooks:"
    echo "  - .git/hooks/post-commit"
    echo "  - .git/hooks/post-merge"
    echo "  - .git/hooks/post-checkout"
    echo "  - .git/hooks/pre-push"
    echo ""
    log_info "Run without --dry-run to perform actual installation"
else
    cd "$REPO_ROOT"
    
    # Run the Python hook installer
    # Note: The install command doesn't support --verbose, but we can get
    # additional output from the status command if verbose is enabled
    python3 "$GIT_HOOKS_PY" install
    
    INSTALL_STATUS=$?
    
    if [[ $INSTALL_STATUS -eq 0 ]]; then
        log_success "Git hooks installed successfully"
    else
        log_error "Hook installation failed with status: $INSTALL_STATUS"
        exit 1
    fi
fi

# =============================================================================
# Verification
# =============================================================================

log_step "Verifying installation..."

HOOKS_DIR="$REPO_ROOT/.git/hooks"
EXPECTED_HOOKS=("post-commit" "post-merge" "post-checkout" "pre-push")
INSTALLED_COUNT=0

for hook in "${EXPECTED_HOOKS[@]}"; do
    HOOK_PATH="$HOOKS_DIR/$hook"
    if [[ -f "$HOOK_PATH" ]] && [[ -x "$HOOK_PATH" ]]; then
        if $VERBOSE; then
            log_info "✓ $hook hook installed and executable"
        fi
        INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
    else
        if ! $DRY_RUN; then
            log_warning "✗ $hook hook not found or not executable"
        fi
    fi
done

if ! $DRY_RUN; then
    echo ""
    log_info "Installed hooks: $INSTALLED_COUNT/${#EXPECTED_HOOKS[@]}"
fi

# =============================================================================
# Status Check
# =============================================================================

log_step "Checking hook status..."

if ! $DRY_RUN; then
    python3 "$GIT_HOOKS_PY" status
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================================="
echo "                    GIT HOOK INSTALLATION COMPLETE"
echo "============================================================================="
echo ""

if $DRY_RUN; then
    echo "  Mode: DRY RUN (no changes made)"
else
    echo "  Mode: INSTALLED"
fi

echo ""
echo "  Hooks Active:"
echo "    • post-commit  - Updates index after each commit"
echo "    • post-merge   - Updates index after merge/pull"
echo "    • post-checkout - Validates cache on branch switch"
echo "    • pre-push     - Verifies consistency before push"
echo ""
echo "  Expected Performance:"
echo "    • Change detection: <50ms"
echo "    • Incremental index: 200-500ms"
echo "    • Full validation: 100-300ms"
echo ""
echo "  Commands:"
echo "    • Validate cache:  python3 $GIT_HOOKS_PY validate"
echo "    • Manual update:   python3 $GIT_HOOKS_PY update"
echo "    • Check status:    python3 $GIT_HOOKS_PY status"
echo "    • Uninstall:       python3 $GIT_HOOKS_PY uninstall"
echo ""
echo "============================================================================="

if ! $DRY_RUN; then
    log_success "Action 1 complete: Git hooks are now operational"
fi
