#!/usr/bin/env bash
#
# .auto-organize.sh
# Automated Repository Organization & RAG Knowledge Management System
#
# Integrates with the Transformation Portal's enhanced capabilities to:
# 1. Structure AI/ML pipelines and model assets
# 2. Manage RAG system knowledge memory and embeddings
# 3. Process feedback loop data for continuous learning
# 4. Maintain architectural cleanliness
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
    
    # Skip if already in the right place
    if [[ "$(dirname "$(realpath "$src")")" == "$(realpath "$dest_dir")" ]]; then
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
    log_info "Organizing RAG & Knowledge Memory artifacts..."
    
    # 1. Knowledge Base & Memory Dumps
    # Moves context files and memory logs to the data/knowledge layer
    local kb_dir="data/knowledge_base"
    
    for file in \
        *memory_dump.json \
        *rag_context.md \
        *knowledge_graph.json \
        *semantic_index.faiss \
        *_embeddings.pkl
    do
        move_file "$file" "$kb_dir/memory_snapshots"
    done

    # 2. Decision Decay & Feedback Loops
    # Organizes system learning feedback for the auditing tools
    local feedback_dir="data/feedback_loops"
    
    for file in \
        *decision_decay.json \
        *learning_feedback.log \
        *performance_metrics.csv \
        *auditor_report.json
    do
        move_file "$file" "$feedback_dir/audits"
    done
}

organize_ai_pipelines() {
    log_info "Structuring AI Pipeline components..."
    
    # 1. Core Processing Scripts
    # Moves loose python pipeline scripts to src/transformation_portal/pipelines
    # or scripts/production based on type
    local pipeline_dir="src/transformation_portal/pipelines"
    local scripts_dir="scripts/production"
    
    # Production CLI tools
    for file in \
        luxury_tiff_batch_processor.py \
        luxury_video_master_grader.py \
        hdr_production_pipeline.sh
    do
        move_file "$file" "$scripts_dir"
    done
    
    # Core Logic Pipelines
    for file in \
        lux_render_pipeline.py \
        depth_pipeline.py \
        material_response.py \
        depth_tools.py
    do
        move_file "$file" "$pipeline_dir"
    done

    # 2. Model Weights & LUTS
    # Ensures assets are correctly placed in the assets structure
    for file in *.cube *.3dl; do
        move_file "$file" "assets/luts/imported"
    done
}

organize_standard_docs() {
    log_info "Organizing standard documentation..."
    
    # Documentation Files
    for file in \
        CI_WORKFLOW_OPTIMIZATION.md \
        DIRECTORY_OPTIMIZATION_PLAN.md \
        FINAL_STRUCTURE.md \
        SYSTEM_STATUS.md \
        *_SUMMARY.txt \
        *_report.txt
    do
        move_file "$file" "docs/guides"
    done
}

organize_utilities() {
    log_info "Organizing utility scripts..."
    
    # Maintenance scripts
    for file in \
        navigate.sh \
        verify_organization.sh \
        codebase_philosophy_auditor.py \
        decision_decay_dashboard.py
    do
        move_file "$file" "scripts/utilities"
    done
    
    # Hidden maintenance hooks (cleaning up root)
    for file in \
        .codebase_health_monitor.py \
        .pre-commit-quality-check.py
    do
        # Move hidden health monitors to scripts/utilities (without dot prefix)
        if [[ -f "$file" ]]; then
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
}

# --- Main Execution ---

main() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "=== DRY RUN MODE - No changes will be made ==="
        echo ""
    fi
    
    log_info "Starting Intelligent Repository Organization..."
    
    # Execute Modules
    organize_rag_and_memory
    organize_ai_pipelines
    organize_standard_docs
    organize_utilities
    
    # Archive cleanup
    for file in debug_*.jpg temp_*.png; do
        move_file "$file" "archive/debug_artifacts"
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