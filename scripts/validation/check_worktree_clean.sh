#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# check_worktree_clean.sh
#
# Verify the git worktree is clean after build/validation operations.
# Used to ensure build artifacts don't dirty the repository.
#
# Exit codes:
#   0 - Worktree is clean
#   1 - Worktree has uncommitted changes
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Options
IGNORE_UNTRACKED=false
SHOW_DIFF=false

# Colors
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

log_error() {
    echo -e "${RED}✗ $1${NC}" >&2
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Verify the git worktree is clean after build/validation operations.

Options:
    --ignore-untracked  Ignore untracked files
    --show-diff         Show diff of changed files
    -h, --help          Show this help message

Examples:
    $(basename "$0")                    # Check for any changes
    $(basename "$0") --ignore-untracked # Check only tracked files
    $(basename "$0") --show-diff        # Show what changed
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ignore-untracked)
            IGNORE_UNTRACKED=true
            shift
            ;;
        --show-diff)
            SHOW_DIFF=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

cd "$REPO_ROOT"

# Check if we're in a git repository
if ! git rev-parse --git-dir &> /dev/null; then
    log_error "Not in a git repository"
    exit 1
fi

# Build git status command
STATUS_OPTS=("--porcelain")
if [[ "$IGNORE_UNTRACKED" == "true" ]]; then
    STATUS_OPTS+=("-uno")
fi

# Get status
STATUS=$(git status "${STATUS_OPTS[@]}")

if [[ -z "$STATUS" ]]; then
    log_success "Worktree is clean"
    exit 0
else
    log_error "Worktree has uncommitted changes"
    echo ""
    
    # Parse and display changes
    MODIFIED_COUNT=0
    ADDED_COUNT=0
    DELETED_COUNT=0
    UNTRACKED_COUNT=0
    
    while IFS= read -r line; do
        status_code="${line:0:2}"
        file_path="${line:3}"
        
        case "$status_code" in
            " M"|"M "|"MM")
                ((MODIFIED_COUNT++))
                echo "  modified: $file_path"
                ;;
            " A"|"A "|"AM")
                ((ADDED_COUNT++))
                echo "  added:    $file_path"
                ;;
            " D"|"D ")
                ((DELETED_COUNT++))
                echo "  deleted:  $file_path"
                ;;
            "??")
                ((UNTRACKED_COUNT++))
                echo "  untracked: $file_path"
                ;;
            *)
                echo "  changed:  $file_path ($status_code)"
                ;;
        esac
    done <<< "$STATUS"
    
    echo ""
    echo "Summary:"
    [[ $MODIFIED_COUNT -gt 0 ]] && echo "  Modified:  $MODIFIED_COUNT"
    [[ $ADDED_COUNT -gt 0 ]] && echo "  Added:     $ADDED_COUNT"
    [[ $DELETED_COUNT -gt 0 ]] && echo "  Deleted:   $DELETED_COUNT"
    [[ $UNTRACKED_COUNT -gt 0 ]] && echo "  Untracked: $UNTRACKED_COUNT"
    
    if [[ "$SHOW_DIFF" == "true" ]]; then
        echo ""
        echo "Diff of tracked changes:"
        echo "─────────────────────────────────────────────────────────────"
        git diff
    fi
    
    echo ""
    log_warning "Build artifacts may have dirtied the worktree"
    echo "  Check .gitignore to ensure build outputs are ignored"
    echo "  Common issues:"
    echo "    - Missing .next-build-verify/ in .gitignore"
    echo "    - Generated files committed by mistake"
    echo ""
    echo "  To reset: git checkout -- ."
    echo "  To see what changed: git diff"
    
    exit 1
fi
