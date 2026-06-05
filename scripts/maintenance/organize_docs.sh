#!/usr/bin/env bash
# Classify misplaced documentation into approved locations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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
        README.md|CONTRIBUTING.md|SECURITY.md|CHANGELOG.md|AGENTS.md|CLAUDE.md|LICENSE)
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

is_supported_destination() {
    local dest="$1"

    case "$dest" in
        docs/architecture|docs/ci|docs/cli|docs/compliance|docs/contracts|docs/deployment|docs/depth_model|docs/depth_pipeline|docs/governance|docs/guides|docs/historical|docs/performance|docs/pr_archive|docs/reference|docs/reports|docs/schemas|docs/status)
            return 0
            ;;
    esac
    return 1
}

normalize_tokens() {
    local input="$1"
    local normalized

    normalized="$(printf '%s' "$input" | sed -E 's/[^A-Z0-9]+/ /g; s/^ +//; s/ +$//; s/ +/ /g')"
    printf ' %s ' "$normalized"
}

has_token() {
    local haystack="$1"
    local needle="$2"

    [[ "$haystack" == *" $needle "* ]]
}

has_phrase() {
    local haystack="$1"
    local phrase="$2"

    [[ "$haystack" == *" $phrase "* ]]
}

starts_with_token() {
    local haystack="$1"
    local needle="$2"

    [[ "$haystack" == " $needle "* ]]
}

classify_destination() {
    local path="$1"
    local basename upper upper_stem tokens dest=""

    basename="$(basename "$path")"
    upper="$(printf '%s' "$basename" | tr '[:lower:]' '[:upper:]')"

    if [[ "$upper" == "README.MD" ]]; then
        return 1
    fi

    if [[ "$upper" == ".DS_STORE" ]]; then
        echo "__REMOVE__"
        return 0
    fi

    upper_stem="$(printf '%s' "${basename%.*}" | tr '[:lower:]' '[:upper:]')"
    tokens="$(normalize_tokens "$upper_stem")"

    if starts_with_token "$tokens" "PR" || starts_with_token "$tokens" "PUSH" || starts_with_token "$tokens" "MERGE" || starts_with_token "$tokens" "REVIEW"; then
        dest="docs/pr_archive"
    elif starts_with_token "$tokens" "ADR" || has_token "$tokens" "ARCHITECT" || has_token "$tokens" "ARCHITECTURE" || has_token "$tokens" "ARCHITECTURAL" || has_token "$tokens" "DESIGN" || has_token "$tokens" "ROADMAP"; then
        dest="docs/architecture"
    elif has_token "$tokens" "POLICY" || has_token "$tokens" "GOVERNANCE" || has_token "$tokens" "ORGANIZATION"; then
        dest="docs/governance"
    elif has_token "$tokens" "CI" || has_token "$tokens" "WORKFLOW" || has_phrase "$tokens" "BRANCH PROTECTION"; then
        dest="docs/ci"
    elif has_token "$tokens" "DEPLOY" || has_token "$tokens" "DEPLOYMENT" || has_token "$tokens" "PRODUCTION"; then
        dest="docs/deployment"
    elif has_token "$tokens" "CLI"; then
        dest="docs/cli"
    elif has_token "$tokens" "CONTRACT"; then
        dest="docs/contracts"
    elif has_token "$tokens" "SCHEMA"; then
        dest="docs/schemas"
    elif has_token "$tokens" "REFERENCE" || has_token "$tokens" "QUICKREF" || has_token "$tokens" "CHEATSHEET" || has_phrase "$tokens" "QUICK REF"; then
        dest="docs/reference"
    elif has_token "$tokens" "SETUP" || has_token "$tokens" "INSTALL" || has_token "$tokens" "TROUBLESHOOTING" || has_token "$tokens" "GUIDE" || has_phrase "$tokens" "BEST PRACTICES" || has_phrase "$tokens" "SUPPORTED FILE FORMATS"; then
        dest="docs/guides"
    elif has_token "$tokens" "PERFORMANCE" || has_token "$tokens" "OPTIMIZATION"; then
        dest="docs/performance"
    elif has_phrase "$tokens" "DEPTH MODEL"; then
        dest="docs/depth_model"
    elif has_token "$tokens" "DEPTH" || has_phrase "$tokens" "LUX DEPTH"; then
        dest="docs/depth_pipeline"
    elif has_token "$tokens" "STATUS" || has_phrase "$tokens" "NEXT STEPS"; then
        dest="docs/status"
    elif has_token "$tokens" "REPORT" || has_token "$tokens" "SUMMARY" || has_token "$tokens" "RESULTS" || has_token "$tokens" "CHECKLIST" || has_token "$tokens" "COMPLETE" || has_token "$tokens" "COMPLETION" || has_token "$tokens" "VERIFICATION" || has_token "$tokens" "FIXES" || has_phrase "$tokens" "RAW TEST"; then
        dest="docs/historical"
    else
        case "$upper" in
            *.CSV)
                dest="docs/compliance"
                ;;
            *.TXT)
                dest="docs/reports"
                ;;
            *.MD)
                dest="docs/guides"
                ;;
        esac
    fi

    if [[ -n "$dest" ]] && is_supported_destination "$dest"; then
        echo "$dest"
        return 0
    fi

    return 1
}

collect_candidates() {
    local file
    local sorted=()
    CANDIDATES=()

    while IFS= read -r -d '' file; do
        case "$file" in
            docs/README.md)
                continue
                ;;
            docs/*)
                if [[ "$file" == docs/* && "$file" != docs/*/* ]]; then
                    CANDIDATES+=("$file")
                fi
                ;;
            */*)
                continue
                ;;
            *)
                if is_allowed_root_doc "$file"; then
                    continue
                fi
                case "$file" in
                    *.md|*.MD|*.txt|*.TXT|*.csv|*.CSV)
                        CANDIDATES+=("$file")
                        ;;
                esac
                ;;
        esac
    done < <(git -C "$REPO_ROOT" ls-files -z)

    if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
        return 0
    fi

    while IFS= read -r file; do
        if [[ -n "$file" ]]; then
            sorted+=("$file")
        fi
    done < <(printf '%s\n' "${CANDIDATES[@]}" | LC_ALL=C sort -u)

    CANDIDATES=("${sorted[@]}")
}

main() {
    local moves=0
    local skipped=0
    local file dest current_dir

    cd "$REPO_ROOT"
    collect_candidates

    if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
        echo "No candidate documentation files found."
        return 0
    fi

    echo "Documentation organization plan ($MODE):"

    for file in "${CANDIDATES[@]}"; do
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
