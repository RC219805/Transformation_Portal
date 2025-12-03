#!/usr/bin/env bash
#
# .auto-organize.sh
# Automated Repository Organization & RAG Knowledge Management System
#
# Organizes loose/temporary files and artifacts in the repository root:
# 1. Moves orphaned RAG memory dumps and embeddings to data/knowledge_base/
# 2. Relocates stray feedback loop artifacts to data/feedback_loops/
# 3. Archives debug images and temporary files
# 4. Organizes loose documentation files
#
# NOTE: This script does NOT move production CLI tools (lux_render_pipeline.py, etc.)
# which are intentionally placed in root as user-facing entry points.
#
# Usage:
#   ./.auto-organize.sh [--dry-run] [--verbose]
#

set -euo pipefail

# Configuration
# shellcheck disable=SC2034  # SCRIPT_DIR may be used by sourcing scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=false
VERBOSE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --verbose) VERBOSE=true ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--dry-run] [--verbose]"
            exit 1
            ;;
    esac
done

# --- Logging & Utility Functions ---

# Enable nullglob for safer wildcard handling
shopt -s nullglob

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

move_file() {
    local src="$1"
    local dest_dir="$2"
    
    # Handle wildcards passed as strings or non-existent files
    if [[ ! -e "$src" ]]; then
        return
    fi
    
    local filename
    filename=$(basename "$src")
    local dest="$dest_dir/$filename"
    
    # Skip if already in the right place (use realpath -m for dest_dir that may not exist)
    if [[ "$(dirname "$(realpath "$src")")" == "$(realpath -m "$dest_dir")" ]]; then
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

# --- Organization Logic ---

organize_rag_and_memory() {
    log_info "Organizing RAG & Knowledge Memory artifacts from root directory..."
    
    # 1. Knowledge Base & Memory Dumps
    # Moves context files and memory logs from ROOT ONLY to the data/knowledge layer
    local kb_dir="data/knowledge_base"
    
    for file in \
        *memory_dump.json \
        *rag_context.md \
        *knowledge_graph.json \
        *semantic_index.faiss \
        *_embeddings.pkl
    do
        # Only move files from root directory
        if [[ -f "$file" ]] && [[ "$(dirname "$(realpath "$file")")" == "$SCRIPT_DIR" ]]; then
            move_file "$file" "$kb_dir/memory_snapshots"
        else
            log_skip "$file (not in root directory, skipping)"
        fi
    done

    # 2. Decision Decay & Feedback Loops
    # Organizes system learning feedback for the auditing tools (ROOT ONLY)
    local feedback_dir="data/feedback_loops"
    
    for file in \
        *decision_decay.json \
        *learning_feedback.log \
        *performance_metrics.csv \
        *auditor_report.json
    do
        # Only move files from root directory
        if [[ -f "$file" ]] && [[ "$(dirname "$(realpath "$file")")" == "$SCRIPT_DIR" ]]; then
            move_file "$file" "$feedback_dir/audits"
        else
            log_skip "$file (not in root directory, skipping)"
        fi
    done
}

organize_lut_files() {
    log_info "Organizing loose LUT files from root directory..."
    
    # Move only loose LUT files from the root directory (not subdirectories)
    for ext in cube 3dl; do
        for file in *."$ext"; do
            if [[ -f "$file" ]] && [[ "$(dirname "$(realpath "$file")")" == "$SCRIPT_DIR" ]]; then
                move_file "$file" "assets/luts/imported"
            fi
        done
    done
}

organize_standard_docs() {
    log_info "Organizing standard documentation from root..."
    
    # Specific documentation files to move (not broad wildcards)
    for file in \
        CI_WORKFLOW_OPTIMIZATION.md \
        DIRECTORY_OPTIMIZATION_PLAN.md \
        FINAL_STRUCTURE.md \
        SYSTEM_STATUS.md
    do
        move_file "$file" "docs/guides"
    done
    
    # Summary and report files - only from root directory with explicit patterns
    for file in \
        PROJECT_*_SUMMARY.txt \
        PHASE*_SUMMARY.txt \
        OPTIMIZATION_*_SUMMARY.txt \
        BUILD_report.txt \
        CI_report.txt
    do
        if [[ -f "$file" ]] && [[ "$(dirname "$(realpath "$file")")" == "$SCRIPT_DIR" ]]; then
            move_file "$file" "docs/guides"
        fi
    done
}

organize_utilities() {
    log_info "Organizing utility scripts..."
    
    # Maintenance scripts - only move if they exist in root and not already in scripts/utilities
    for file in \
        navigate.sh \
        verify_organization.sh \
        codebase_philosophy_auditor.py \
        decision_decay_dashboard.py
    do
        local dest_path="scripts/utilities/$file"
        if [[ -f "$dest_path" ]]; then
            log_skip "$file already exists in scripts/utilities"
        elif [[ -f "$file" ]]; then
            move_file "$file" "scripts/utilities"
        fi
    done
    
    # Hidden maintenance hooks (cleaning up root)
    for file in \
        .codebase_health_monitor.py \
        .pre-commit-quality-check.py
    do
        # Move hidden health monitors to scripts/utilities (without dot prefix)
        if [[ -f "$file" ]]; then
            local new_name="${file#.}"
            local dest_path="scripts/utilities/$new_name"
            if [[ -f "$dest_path" ]]; then
                log_skip "$file already exists as $new_name in scripts/utilities"
            elif [[ "$DRY_RUN" == "false" ]]; then
                mkdir -p "scripts/utilities"
                mv "$file" "scripts/utilities/$new_name"
                log_move "$file" "scripts/utilities/$new_name"
            else
                log_move "$file" "scripts/utilities/$new_name"
            fi
        fi
    done
}

# --- Main Execution ---

main() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "=== DRY RUN MODE - No changes will be made ==="
        echo ""
    fi
    
    log_info "Starting Intelligent Repository Organization..."
    
    # Ensure we are in the script directory for consistent relative paths
    cd "$SCRIPT_DIR"
    
    # Execute Modules
    organize_rag_and_memory
    organize_lut_files
    organize_standard_docs
    organize_utilities
    
    # Archive cleanup - only from root directory
    for file in ./debug_*.jpg ./temp_*.png; do
        if [[ -f "$file" ]]; then
            move_file "$file" "archive/debug_artifacts"
        fi
    done
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo ""
        echo "=== DRY RUN COMPLETE ==="
    else
        echo ""
        echo "✓ Repository organization complete."
        echo "  RAG Memory & Feedback loops updated."
    fi
}

main "$@"