#!/usr/bin/env bash
# Classify misplaced documentation into approved locations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODE="dry-run"
VERBOSE=0

usage() {
    cat <<EOF
Usage: $0 [--dry-run|--apply] [--verbose]

Options:
  --dry-run   Print proposed moves without mutating the repository (default).
  --apply     Move files using git mv when tracked.
  --verbose   Print skipped files and extra summary detail.
  -h, --help  Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            MODE="dry-run"
            shift
            ;;
        --apply)
            MODE="apply"
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

log_verbose() {
    if [[ "$VERBOSE" -eq 1 ]]; then
        echo "$1"
    fi
}

is_git_tracked() {
    git -C "$REPO_ROOT" ls-files --error-unmatch "$1" >/dev/null 2>&1
}

is_allowed_root_doc() {
    local file="$1"
    case "$file" in
        README.md|CONTRIBUTING.md|SECURITY.md|CHANGELOG.md|AGENTS.md|LICENSE)
            return 0
            ;;
        requirements*.txt)
            return 0
            ;;
    esac
    return 1
}

apply_action() {
    local src="$1"
    local dest_dir="$2"
    local dest=""

    if [[ "$dest_dir" == "__REMOVE__" ]]; then
        echo "REMOVE $src"
        if [[ "$MODE" == "apply" ]]; then
            if is_git_tracked "$src"; then
                git -C "$REPO_ROOT" rm -f "$src"
            else
                rm -f "$REPO_ROOT/$src"
            fi
        fi
        return 0
    fi

    dest="$dest_dir/$(basename "$src")"
    echo "MOVE $src -> $dest"
    if [[ "$MODE" == "apply" ]]; then
        mkdir -p "$REPO_ROOT/$dest_dir"
        if is_git_tracked "$src"; then
            git -C "$REPO_ROOT" mv "$src" "$dest"
        else
            mv "$REPO_ROOT/$src" "$REPO_ROOT/$dest"
        fi
    fi
}

classify_destination() {
    local path="$1"
    local basename upper

    basename="$(basename "$path")"
    upper="$(printf '%s' "$basename" | tr '[:lower:]' '[:upper:]')"

    case "$upper" in
        README.MD)
            return 1
            ;;
        .DS_STORE)
            echo "__REMOVE__"
            return 0
            ;;
        PR_*|PUSH_*|MERGE_*|REVIEW_* )
            echo "docs/pr_archive"
            return 0
            ;;
        ADR-*|*ARCHITECT*|*DESIGN*|*ROADMAP* )
            echo "docs/architecture"
            return 0
            ;;
        *POLICY*|*GOVERNANCE*|*ORGANIZATION* )
            echo "docs/governance"
            return 0
            ;;
        *CI*|*WORKFLOW*|*BRANCH_PROTECTION* )
            echo "docs/ci"
            return 0
            ;;
        *DEPLOY*|*PRODUCTION* )
            echo "docs/deployment"
            return 0
            ;;
        *CLI* )
            echo "docs/cli"
            return 0
            ;;
        *CONTRACT* )
            echo "docs/contracts"
            return 0
            ;;
        *SCHEMA* )
            echo "docs/schemas"
            return 0
            ;;
        *REFERENCE*|*QUICK_REF*|*QUICKREF*|*CHEATSHEET* )
            echo "docs/reference"
            return 0
            ;;
        *SETUP*|*INSTALL*|*TROUBLESHOOT*|*GUIDE*|*BEST_PRACTICES*|*SUPPORTED_FILE_FORMATS* )
            echo "docs/guides"
            return 0
            ;;
        *PERFORMANCE*|*OPTIMIZATION* )
            echo "docs/performance"
            return 0
            ;;
        *DEPTH_MODEL* )
            echo "docs/depth_model"
            return 0
            ;;
        *DEPTH*|*LUX_DEPTH* )
            echo "docs/depth_pipeline"
            return 0
            ;;
        *STATUS*|*NEXT_STEPS* )
            echo "docs/status"
            return 0
            ;;
        *REPORT*|*SUMMARY*|*RESULTS*|*CHECKLIST*|*COMPLETE*|*COMPLETION*|*VERIFICATION*|*FIXES*|*RAW_TEST* )
            echo "docs/historical"
            return 0
            ;;
        *.CSV )
            echo "docs/compliance"
            return 0
            ;;
        *.TXT )
            echo "docs/reports"
            return 0
            ;;
        *.MD )
            echo "docs/guides"
            return 0
            ;;
    esac

    return 1
}

collect_candidates() {
    local file
    CANDIDATES=()

    while IFS= read -r -d '' file; do
        file="${file#$REPO_ROOT/}"
        if [[ "$file" != "docs/README.md" ]]; then
            CANDIDATES+=("$file")
        fi
    done < <(find "$REPO_ROOT/docs" -maxdepth 1 -type f -print0)

    while IFS= read -r -d '' file; do
        if [[ "$file" == */* ]]; then
            continue
        fi
        if is_allowed_root_doc "$file"; then
            continue
        fi
        case "$file" in
            *.md|*.MD|*.txt|*.TXT|*.csv|*.CSV)
                CANDIDATES+=("$file")
                ;;
        esac
    done < <(git -C "$REPO_ROOT" ls-files -z)
}

main() {
    local moves=0
    local skipped=0
    local seen=()
    local file dest current_dir already_listed

    cd "$REPO_ROOT"
    collect_candidates

    if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
        echo "No candidate documentation files found."
        return 0
    fi

    echo "Documentation organization plan ($MODE):"

    for file in "${CANDIDATES[@]}"; do
        already_listed=0
        if [[ ${#seen[@]} -gt 0 ]]; then
            for current_dir in "${seen[@]}"; do
                if [[ "$current_dir" == "$file" ]]; then
                    already_listed=1
                    break
                fi
            done
        fi
        if [[ "$already_listed" -eq 1 ]]; then
            continue
        fi
        seen+=("$file")

        if ! dest="$(classify_destination "$file")"; then
            log_verbose "SKIP $file (no deterministic classification rule)"
            skipped=$((skipped + 1))
            continue
        fi

        current_dir="$(dirname "$file")"
        if [[ "$dest" != "__REMOVE__" && "$current_dir" == "$dest" ]]; then
            log_verbose "SKIP $file (already in $dest)"
            continue
        fi

        apply_action "$file" "$dest"
        moves=$((moves + 1))
    done

    if [[ "$MODE" == "dry-run" ]]; then
        echo "Dry run only. Re-run with --apply to perform the proposed moves."
    fi
    echo "Summary: $moves proposed move(s), $skipped skipped file(s)."
}

CANDIDATES=()
main "$@"
