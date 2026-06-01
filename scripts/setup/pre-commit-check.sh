#!/usr/bin/env bash
#
# pre-commit-check.sh
# Canonical root-file placement validator for staged changes and full-repo scans.
#

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODE="staged"
LEGACY_ALLOWLIST_PATH="$REPO_ROOT/scripts/governance/root_structure_legacy_allowlist.txt"

ALLOWED_ROOT_FILES=(
    "README.md"
    "LICENSE"
    "CONTRIBUTING.md"
    "SECURITY.md"
    "CHANGELOG.md"
    "AGENTS.md"
    "CLAUDE.md"
    "Makefile"
    "package.json"
    "package-lock.json"
    "pyproject.toml"
    "setup.py"
    "setup.cfg"
    "requirements.txt"
    "requirements-dev.txt"
    "requirements-ci.txt"
    "requirements-test.txt"
    "requirements-lint.txt"
    "Pipfile"
    "Pipfile.lock"
    "poetry.lock"
    "pytest.ini"
    "tox.ini"
    ".coveragerc"
    ".pylintrc"
    ".flake8"
    "mypy.ini"
    "Dockerfile"
    "docker-compose.yml"
    "docker-compose.yaml"
    ".gitignore"
    ".dockerignore"
    ".gitattributes"
    ".gitmodules"
    ".git-blame-ignore-revs"
    ".pre-commit-config.yaml"
    "wrangler.jsonc"
    ".auto-organize.sh"
    ".architect_directive_status.yml"
    ".env.example"
    "PKG-INFO"
    "MANIFEST.in"
    "__init__.py"
    "app.py"
    "portal.html"
)

ALLOWED_ROOT_PATTERNS=(
    '^requirements.*\.txt$'
    '^\.git.*$'
    '^\..*rc$'
)

BLOCKED_PATH_PREFIXES=(
    "productivity/"
)

usage() {
    cat <<EOF
Usage: $0 [--staged|--all] [--legacy-allowlist PATH]

Options:
  --staged               Validate only staged root files (default).
  --all                  Validate all tracked root files in the repository.
  --legacy-allowlist     Newline-delimited baseline of grandfathered root files for --all scans.
  -h, --help             Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --staged)
            MODE="staged"
            shift
            ;;
        --all)
            MODE="all"
            shift
            ;;
        --legacy-allowlist)
            if [[ $# -lt 2 ]]; then
                echo "Missing path for --legacy-allowlist" >&2
                exit 2
            fi
            LEGACY_ALLOWLIST_PATH="$2"
            shift 2
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

is_allowed_in_root() {
    local file="$1"
    local basename
    basename="$(basename "$file")"
    local allowed
    local pattern

    for allowed in "${ALLOWED_ROOT_FILES[@]}"; do
        if [[ "$basename" == "$allowed" ]]; then
            return 0
        fi
    done

    for pattern in "${ALLOWED_ROOT_PATTERNS[@]}"; do
        if [[ "$basename" =~ $pattern ]]; then
            return 0
        fi
    done

    return 1
}

is_blocked_repo_path() {
    local file="$1"
    local prefix

    for prefix in "${BLOCKED_PATH_PREFIXES[@]}"; do
        if [[ "$file" == "$prefix"* ]]; then
            return 0
        fi
    done

    return 1
}

is_legacy_root_file() {
    local file="$1"
    local legacy_file
    if [[ ${#LEGACY_ROOT_FILES[@]} -eq 0 ]]; then
        return 1
    fi
    for legacy_file in "${LEGACY_ROOT_FILES[@]}"; do
        if [[ "$file" == "$legacy_file" ]]; then
            return 0
        fi
    done
    return 1
}

suggest_destination() {
    local file="$1"
    local basename
    local ext

    basename="$(basename "$file")"
    ext="${basename##*.}"

    if [[ "$basename" =~ \.md$ ]]; then
        if [[ "$file" == productivity/* ]]; then
            echo "docs/historical/ or another approved docs archive"
            return
        fi
        if [[ "$basename" =~ ^(PR_|PUSH_|MERGE_|REVIEW_|BRANCH_) ]]; then
            echo "docs/pr_archive/"
        elif [[ "$basename" =~ (POLICY|GOVERNANCE|ORGANIZATION|STANDARD|README|GUIDE|REFERENCE|CHECKLIST|QUICK_REF|QUICKSTART|BEST_PRACTICES) ]]; then
            echo "docs/governance/ or docs/guides/"
        else
            echo "docs/historical/"
        fi
        return
    fi

    if [[ "$ext" == "csv" || "$ext" == "json" || "$ext" == "txt" ]]; then
        echo "docs/reports/ or data/"
        return
    fi

    if [[ "$basename" =~ \.(sh|py)$ ]]; then
        if [[ "$file" == productivity/* ]]; then
            echo "archive/scripts/ or scripts/"
            return
        fi
        echo "scripts/"
        return
    fi

    echo "an approved project subdirectory"
}

load_legacy_allowlist() {
    LEGACY_ROOT_FILES=()
    if [[ ! -f "$LEGACY_ALLOWLIST_PATH" ]]; then
        return
    fi

    while IFS= read -r raw_line; do
        local line
        line="${raw_line#"${raw_line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        if [[ -z "$line" || "$line" == \#* ]]; then
            continue
        fi
        LEGACY_ROOT_FILES+=("$line")
    done < "$LEGACY_ALLOWLIST_PATH"
}

collect_candidates() {
    CANDIDATES=()

    if [[ "$MODE" == "all" ]]; then
        while IFS= read -r -d '' file; do
            if is_blocked_repo_path "$file"; then
                CANDIDATES+=("$file")
                continue
            fi
            if [[ "$file" == */* ]]; then
                continue
            fi
            CANDIDATES+=("$file")
        done < <(git -C "$REPO_ROOT" ls-files -z)
        return
    fi

    while IFS= read -r -d '' file; do
        if is_blocked_repo_path "$file"; then
            CANDIDATES+=("$file")
            continue
        fi
        if [[ "$file" == */* ]]; then
            continue
        fi
        CANDIDATES+=("$file")
    done < <(git -C "$REPO_ROOT" diff --cached --name-only --diff-filter=ACMR -z)
}

main() {
    local misplaced_files=()
    local known_legacy_files=()
    local stale_legacy_files=()
    local file
    local legacy_file

    cd "$REPO_ROOT"
    load_legacy_allowlist
    collect_candidates

    if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
        echo "No root files to validate."
        return 0
    fi

    for file in "${CANDIDATES[@]}"; do
        if is_blocked_repo_path "$file"; then
            misplaced_files+=("$file")
            continue
        fi
        if is_allowed_in_root "$file"; then
            continue
        fi
        if [[ "$MODE" == "all" ]] && is_legacy_root_file "$file"; then
            known_legacy_files+=("$file")
            continue
        fi
        misplaced_files+=("$file")
    done

    if [[ "$MODE" == "all" ]] && [[ ${#LEGACY_ROOT_FILES[@]} -gt 0 ]]; then
        for legacy_file in "${LEGACY_ROOT_FILES[@]}"; do
            if [[ ! -f "$REPO_ROOT/$legacy_file" ]]; then
                stale_legacy_files+=("$legacy_file")
            fi
        done
    fi

    if [[ ${#stale_legacy_files[@]} -gt 0 ]]; then
        echo "Legacy root allowlist contains paths that are no longer present:"
        for file in "${stale_legacy_files[@]}"; do
            echo "  - $file"
        done
        echo "Update $LEGACY_ALLOWLIST_PATH so the baseline matches the repository."
        return 1
    fi

    if [[ ${#misplaced_files[@]} -gt 0 ]]; then
        echo -e "${RED}✗ Root file placement check failed${NC}"
        echo
        if [[ "$MODE" == "all" ]]; then
            echo "The following tracked root files are outside the current repository contract:"
        else
            echo "The following staged root files are outside the current repository contract:"
        fi
        echo
        for file in "${misplaced_files[@]}"; do
            echo -e "  ${YELLOW}$file${NC}"
            echo -e "    → Suggested: ${GREEN}$(suggest_destination "$file")${NC}"
        done
        echo
        echo "Move these files to approved locations or update the root policy intentionally."
        return 1
    fi

    if [[ ${#known_legacy_files[@]} -gt 0 ]]; then
        echo "Root file placement check passed with grandfathered legacy root files:"
        for file in "${known_legacy_files[@]}"; do
            echo "  - $file"
        done
        echo "A cleanup PR is still required to remove this baseline debt."
        return 0
    fi

    echo "Root file placement check passed."
    return 0
}

LEGACY_ROOT_FILES=()
CANDIDATES=()
main "$@"
