#!/usr/bin/env bash
#
# .auto-organize.sh
# Automated Repository Organization System
#
# Orchestrates file organization in the Transformation Portal repository
# by delegating to modular helper scripts and validation tools.
#
# This script is the canonical entry point for repository organization.
# It coordinates helper scripts in a fixed sequence and integrates with
# the pre-commit quality gate system.
#
# Usage:
#   ./.auto-organize.sh [OPTIONS]
#
# Options:
#   --dry-run       Show what would be done without making changes
#   --check         Validate current organization (exit 1 if violations found)
#   --verbose       Show detailed output including skipped files
#   --docs-only     Only organize documentation files
#   --skip-root     Skip root file placement validation
#   -h, --help      Show this help message
#
# Helper Scripts (executed in sequence when running full organization):
#   1. scripts/organize_docs.sh         - Classify docs into approved locations
#   2. scripts/setup/pre-commit-check.sh - Validate root file placement
#
# Documentation:
#   See docs/governance/REPO_ORGANIZATION.md for organization rules.
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

# Color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Mode flags
DRY_RUN=false
CHECK_MODE=false
VERBOSE=false
DOCS_ONLY=false
SKIP_ROOT=false

# Exit codes
EXIT_SUCCESS=0
EXIT_FAILURE=1
EXIT_USAGE=2

# ============================================================================
# Helper Functions
# ============================================================================

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Orchestrate file organization in the Transformation Portal repository.

Options:
  --dry-run       Show what would be done without making changes
  --check         Validate current organization (exit 1 if violations)
  --verbose       Show detailed output including skipped files
  --docs-only     Only organize documentation files
  --skip-root     Skip root file placement validation
  -h, --help      Show this help message

Examples:
  $0 --dry-run              # Preview organization changes
  $0                        # Apply organization changes
  $0 --check                # CI validation mode (fail if violations)
  $0 --docs-only --dry-run  # Preview documentation moves only

Documentation:
  docs/governance/REPO_ORGANIZATION.md - Organization rules and guidelines
  scripts/setup/README.md              - Setup and installation guide

EOF
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $*"
}

log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[INFO]${NC} $*"
    fi
}

log_section() {
    echo ""
    echo -e "${BLUE}═════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $*${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
}

# Check if a helper script exists and is executable
check_helper_script() {
    local script="$1"
    local full_path="$REPO_ROOT/$script"

    if [[ ! -f "$full_path" ]]; then
        log_error "Helper script not found: $script"
        return 1
    fi

    if [[ ! -x "$full_path" ]]; then
        log_warn "Helper script not executable: $script (fixing...)"
        chmod +x "$full_path"
    fi

    return 0
}

# ============================================================================
# Argument Parsing
# ============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --check)
                CHECK_MODE=true
                DRY_RUN=true  # Check mode implies dry-run
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --docs-only)
                DOCS_ONLY=true
                shift
                ;;
            --skip-root)
                SKIP_ROOT=true
                shift
                ;;
            -h|--help)
                usage
                exit $EXIT_SUCCESS
                ;;
            *)
                log_error "Unknown option: $1"
                usage >&2
                exit $EXIT_USAGE
                ;;
        esac
    done
}

# ============================================================================
# Organization Steps
# ============================================================================

# Step 1: Organize documentation files using the dedicated helper
organize_docs() {
    log_section "Organizing Documentation Files"

    local helper="scripts/organize_docs.sh"

    if ! check_helper_script "$helper"; then
        log_error "Cannot organize docs: helper script missing"
        return 1
    fi

    local args=()
    if [[ "$DRY_RUN" == "true" ]]; then
        args+=("--dry-run")
    else
        args+=("--apply")
    fi

    if [[ "$VERBOSE" == "true" ]]; then
        args+=("--verbose")
    fi

    log_info "Running: $helper ${args[*]}"
    if ! "$REPO_ROOT/$helper" "${args[@]}"; then
        log_error "Documentation organization failed"
        return 1
    fi

    log_success "Documentation organization completed"
    return 0
}

# Step 2: Validate root file placement
validate_root_files() {
    log_section "Validating Root File Placement"

    if [[ "$SKIP_ROOT" == "true" ]]; then
        log_info "Skipping root file validation (--skip-root)"
        return 0
    fi

    local helper="scripts/setup/pre-commit-check.sh"

    if ! check_helper_script "$helper"; then
        log_error "Cannot validate root files: helper script missing"
        return 1
    fi

    log_info "Running: $helper --all"
    if ! "$REPO_ROOT/$helper" --all; then
        if [[ "$CHECK_MODE" == "true" ]]; then
            log_error "Root file placement violations detected"
            return 1
        else
            log_warn "Root file placement issues found"
            log_info "Review docs/governance/REPO_ORGANIZATION.md for allowed root files"
        fi
    else
        log_success "Root file placement validated"
    fi

    return 0
}

# Step 3: Check for misplaced Python scripts in root (dynamic detection)
check_root_scripts() {
    log_section "Checking for Misplaced Root Scripts"

    local misplaced=()

    # Find Python files in root that should be elsewhere
    while IFS= read -r -d '' file; do
        local basename
        basename="$(basename "$file")"

        # Skip allowed root Python files
        case "$basename" in
            app.py|__init__.py|setup.py|conftest.py)
                continue
                ;;
        esac

        # Classify based on naming patterns
        local dest=""
        case "$basename" in
            test_*.py)
                dest="tests/"
                ;;
            process_*.py|run_*.py|*_pipeline*.py)
                dest="scripts/pipelines/"
                ;;
            convert_*.py|fix_*.py|verify_*.py|save_*.py|update_*.py)
                dest="scripts/utilities/"
                ;;
            analyze_*.py|diagnose_*.py|audit_*.py)
                dest="scripts/analysis/"
                ;;
            install_*.py|download_*.py)
                dest="scripts/setup/"
                ;;
            example_*.py|*_example*.py)
                dest="examples/"
                ;;
        esac

        if [[ -n "$dest" ]]; then
            misplaced+=("$basename → $dest")
        fi

    done < <(find "$REPO_ROOT" -maxdepth 1 -name "*.py" -type f -print0 2>/dev/null)

    if [[ ${#misplaced[@]} -gt 0 ]]; then
        log_warn "Found Python scripts that may be misplaced in root:"
        for entry in "${misplaced[@]}"; do
            echo "  - $entry"
        done
        return 1
    fi

    log_success "No misplaced Python scripts detected in root"
    return 0
}

# Step 4: Check for misplaced shell scripts in root
check_root_shell_scripts() {
    log_section "Checking for Misplaced Shell Scripts in Root"

    local misplaced=()

    while IFS= read -r -d '' file; do
        local basename
        basename="$(basename "$file")"

        # Skip allowed root shell scripts
        case "$basename" in
            .auto-organize.sh)
                continue
                ;;
        esac

        misplaced+=("$basename → scripts/")

    done < <(find "$REPO_ROOT" -maxdepth 1 -name "*.sh" -type f ! -name ".*" -print0 2>/dev/null)

    if [[ ${#misplaced[@]} -gt 0 ]]; then
        log_warn "Found shell scripts that should be in scripts/:"
        for entry in "${misplaced[@]}"; do
            echo "  - $entry"
        done
        return 1
    fi

    log_success "No misplaced shell scripts detected in root"
    return 0
}

# Step 5: Validate documentation structure
validate_docs_structure() {
    log_section "Validating Documentation Structure"

    local helper="scripts/governance/check_docs_structure.py"

    if [[ ! -f "$REPO_ROOT/$helper" ]]; then
        log_verbose "Documentation structure validator not found, skipping"
        return 0
    fi

    log_info "Running: python3 $helper --all"
    if ! python3 "$REPO_ROOT/$helper" --all 2>/dev/null; then
        if [[ "$CHECK_MODE" == "true" ]]; then
            log_error "Documentation structure violations detected"
            return 1
        else
            log_warn "Documentation structure issues found"
        fi
    else
        log_success "Documentation structure validated"
    fi

    return 0
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    parse_args "$@"

    cd "$REPO_ROOT"

    local total_steps=0
    local passed_steps=0
    local failed_steps=0

    # Header
    echo ""
    if [[ "$CHECK_MODE" == "true" ]]; then
        echo "═══════════════════════════════════════════════════════════════"
        echo "  Transformation Portal - Organization Validation"
        echo "  Mode: CHECK (CI validation - fail on violations)"
        echo "═══════════════════════════════════════════════════════════════"
    elif [[ "$DRY_RUN" == "true" ]]; then
        echo "═══════════════════════════════════════════════════════════════"
        echo "  Transformation Portal - Repository Organization"
        echo "  Mode: DRY RUN (no changes will be made)"
        echo "═══════════════════════════════════════════════════════════════"
    else
        echo "═══════════════════════════════════════════════════════════════"
        echo "  Transformation Portal - Repository Organization"
        echo "  Mode: APPLY (changes will be made)"
        echo "═══════════════════════════════════════════════════════════════"
    fi

    # Step 1: Organize documentation
    total_steps=$((total_steps + 1))
    if organize_docs; then
        passed_steps=$((passed_steps + 1))
    else
        failed_steps=$((failed_steps + 1))
    fi

    # Step 2: Validate root files (unless docs-only mode)
    if [[ "$DOCS_ONLY" != "true" ]]; then
        total_steps=$((total_steps + 1))
        if validate_root_files; then
            passed_steps=$((passed_steps + 1))
        else
            failed_steps=$((failed_steps + 1))
        fi

        # Step 3: Check for misplaced Python scripts
        total_steps=$((total_steps + 1))
        if check_root_scripts; then
            passed_steps=$((passed_steps + 1))
        else
            failed_steps=$((failed_steps + 1))
        fi

        # Step 4: Check for misplaced shell scripts
        total_steps=$((total_steps + 1))
        if check_root_shell_scripts; then
            passed_steps=$((passed_steps + 1))
        else
            failed_steps=$((failed_steps + 1))
        fi

        # Step 5: Validate documentation structure
        total_steps=$((total_steps + 1))
        if validate_docs_structure; then
            passed_steps=$((passed_steps + 1))
        else
            failed_steps=$((failed_steps + 1))
        fi
    fi

    # Summary
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Summary: $passed_steps/$total_steps steps passed"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    if [[ "$failed_steps" -gt 0 ]]; then
        if [[ "$CHECK_MODE" == "true" ]]; then
            log_error "Organization validation failed with $failed_steps issue(s)"
            echo ""
            echo "To fix organization issues, run:"
            echo "  ./.auto-organize.sh"
            echo ""
            echo "For documentation, see:"
            echo "  docs/governance/REPO_ORGANIZATION.md"
            echo ""
            return $EXIT_FAILURE
        else
            log_warn "Organization completed with $failed_steps warning(s)"
            echo ""
            echo "Review the warnings above and address any issues."
            echo "For documentation, see: docs/governance/REPO_ORGANIZATION.md"
            echo ""
        fi
    else
        if [[ "$DRY_RUN" == "true" ]]; then
            log_success "Dry run completed successfully"
            echo ""
            echo "To apply changes, run without --dry-run:"
            echo "  ./.auto-organize.sh"
            echo ""
        else
            log_success "Repository organization completed successfully"
            echo ""
            echo "Review changes with: git status"
            echo ""
        fi
    fi

    return $EXIT_SUCCESS
}

main "$@"
