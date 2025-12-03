#!/usr/bin/env bash
#
# .auto-organize.sh
# Automated Repository Organization System
#
# Automatically organizes files in the Transformation Portal repository
# to maintain a clean, structured directory hierarchy.
#
# Usage:
#   ./.auto-organize.sh [--dry-run] [--verbose]
#
# Options:
#   --dry-run   Show what would be done without making changes
#   --verbose   Show detailed output
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=false
VERBOSE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --verbose)
            VERBOSE=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--dry-run] [--verbose]"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    if [[ "$VERBOSE" == "true" || "$DRY_RUN" == "true" ]]; then
        echo "[INFO] $*"
    fi
}

log_move() {
    echo "[MOVE] $1 → $2"
}

log_skip() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo "[SKIP] $*"
    fi
}

# Move file function
move_file() {
    local src="$1"
    local dest_dir="$2"
    local filename="$(basename "$src")"
    local dest="$dest_dir/$filename"
    
    # Skip if source doesn't exist
    if [[ ! -f "$src" ]]; then
        return
    fi
    
    # Skip if already in the right place
    if [[ "$(cd "$(dirname "$src")" && pwd)" == "$(cd "$dest_dir" && pwd)" ]]; then
        log_skip "$filename (already in correct location)"
        return
    fi
    
    # Create destination directory
    if [[ "$DRY_RUN" == "false" ]]; then
        mkdir -p "$dest_dir"
    fi
    
    # Move the file
    log_move "$src" "$dest"
    if [[ "$DRY_RUN" == "false" ]]; then
        mv "$src" "$dest"
    fi
}

# Main organization logic
organize_repository() {
    log_info "Starting repository organization..."
    
    cd "$SCRIPT_DIR"
    
    # ========================================
    # Documentation Files
    # ========================================
    log_info "Organizing documentation files..."
    
    # Strategy and planning documents → docs/guides/
    for file in \
        CI_WORKFLOW_OPTIMIZATION.md \
        DIRECTORY_OPTIMIZATION_PLAN.md \
        DIRECTORY_STRUCTURE_OPTIMIZATION.md \
        FINAL_STRUCTURE.md \
        OPTIMIZATION_COMPLETE.md \
        OPTIONAL_FEATURES_INSTALLED.md \
        PHASE1_COMPLETION_SUMMARY.md \
        SYSTEM_STATUS.md \
        START_HERE.md
    do
        move_file "$file" "docs/guides"
    done
    
    # ========================================
    # Scripts
    # ========================================
    log_info "Organizing scripts..."
    
    # Utility scripts → scripts/utilities/
    for file in \
        navigate.sh \
        verify_organization.sh
    do
        move_file "$file" "scripts/utilities"
    done
    
    # Python CLI wrappers → scripts/utilities/
    for file in \
        luxury_tiff_batch_processor_cli.py
    do
        move_file "$file" "scripts/utilities"
    done
    
    # ========================================
    # Text Files and Summaries
    # ========================================
    log_info "Organizing text files..."
    
    # Project summaries → docs/guides/
    for file in \
        750_PICACHO_QUICK_SUMMARY.txt \
        FILES_CHANGED_SUMMARY.txt \
        PUSH_STATUS.txt \
        QUALITY_BOOST_SUMMARY.txt \
        step5_lut_examples.txt
    do
        move_file "$file" "docs/guides"
    done
    
    # ========================================
    # Data Files
    # ========================================
    log_info "Organizing data files..."
    
    # JSON files → data/
    for file in \
        depth_model_comparison.json \
        index_stats.json
    do
        move_file "$file" "data"
    done
    
    # Images → archive/ (unless actively used)
    for file in \
        debug_after_white_balance.jpg
    do
        move_file "$file" "archive"
    done
    
    # ========================================
    # Configuration and Build Files
    # ========================================
    log_info "Checking configuration files..."
    
    # These stay in root:
    # - README.md
    # - Makefile
    # - pyproject.toml
    # - requirements*.txt
    # - pytest.ini
    # - mypy.ini
    # - .pylintrc
    # - Dockerfile
    # - docker-compose.yml
    # - .gitignore
    # - .gitattributes (to be created)
    
    # ========================================
    # Hidden/System Files
    # ========================================
    log_info "Organizing hidden files..."
    
    # Move old organization scripts to archive
    for file in \
        .organize_docs.sh
    do
        if [[ -f "$file" ]]; then
            move_file "$file" "archive"
        fi
    done
    
    # Quality check scripts → scripts/utilities/
    for file in \
        .codebase_health_monitor.py \
        .pre-commit-quality-check.py \
        .quality_fix.py
    do
        if [[ -f "$file" ]]; then
            # Remove leading dot for organized version
            local new_name="${file#.}"
            if [[ "$DRY_RUN" == "false" ]]; then
                mkdir -p "scripts/utilities"
                mv "$file" "scripts/utilities/$new_name"
                log_move "$file" "scripts/utilities/$new_name"
            else
                log_move "$file" "scripts/utilities/$new_name"
            fi
        fi
    done
    
    # TypeScript code files → archive/ (unless actively used)
    for file in \
        code.ts
    do
        if [[ -f "$file" ]]; then
            move_file "$file" "archive"
        fi
    done
    
    log_info "Organization complete!"
}

# Main execution
main() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "=== DRY RUN MODE - No changes will be made ==="
        echo ""
    fi
    
    organize_repository
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo ""
        echo "=== DRY RUN COMPLETE ==="
        echo "Run without --dry-run to apply changes"
    else
        echo ""
        echo "✓ Repository organization complete!"
        echo "  Review the changes with: git status"
    fi
}

main "$@"
