#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"

case "$MODE" in
    local|pr|advisory)
        ;;
    *)
        echo "lint_runner: invalid mode '$MODE' (expected: local|pr|advisory)"
        exit 64
        ;;
esac

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$REPO_ROOT"

if [ -n "${PYTHON_BIN:-}" ]; then
    :
elif [ -x .venv/bin/python ]; then
    PYTHON_BIN=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    PYTHON_BIN=python
fi

readonly PYTHON_BIN
readonly PYLINT_EXCLUDE_REGEX='^(external/|depth-anything-3/|deprecated/|src/transformation_portal/|src/luxury_tiff_batch_processor/|scripts/|examples/|\.github/|\.backup_local/)'

# Fallback lint surface used when no changed Python files match the lint filter.
# Ensures deterministic pylint execution during local/advisory runs.
readonly PYLINT_FALLBACK=(
    "src/tp/phase4/verify_phase4_chain.py"
    "tests/test_material_response.py"
    "tests/test_depth_tools.py"
)

log() {
    echo "lint_runner: $*"
}

require_module() {
    local module="$1"
    if ! "$PYTHON_BIN" -m "$module" --version >/dev/null 2>&1; then
        log "'$module' not available in interpreter '$PYTHON_BIN'"
        log "prepare the environment with: make install-core"
        exit 1
    fi
}

collect_filtered_pyfiles_from_diff_range() {
    local diff_range="$1"
    local diff_output=""

    if ! diff_output=$(git diff --diff-filter=d --name-only "$diff_range" 2>/dev/null); then
        return 1
    fi

    if [ -z "$diff_output" ]; then
        return 0
    fi

    printf '%s\n' "$diff_output" \
        | grep -E '\.py$' \
        | grep -vE "$PYLINT_EXCLUDE_REGEX" \
        || true
}

collect_pylint_files_from_diff_range() {
    local diff_range="$1"
    local diff_candidates=""
    local candidate=""

    if ! diff_candidates=$(collect_filtered_pyfiles_from_diff_range "$diff_range"); then
        return 1
    fi

    if [ -z "$diff_candidates" ]; then
        return 0
    fi

    while IFS= read -r candidate; do
        [ -n "$candidate" ] && PYLINT_FILES+=("$candidate")
    done <<< "$diff_candidates"

    return 0
}

resolve_local_diff_range() {
    if git rev-parse --verify origin/main >/dev/null 2>&1; then
        echo "origin/main...HEAD"
    elif git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
        echo "HEAD~1..HEAD"
    else
        echo ""
    fi
}

resolve_pr_diff_range() {
    local event_name="${LINT_RUNNER_GITHUB_EVENT_NAME:-${GITHUB_EVENT_NAME:-}}"
    local before_sha="${LINT_RUNNER_GITHUB_BEFORE:-}"
    local current_sha="${LINT_RUNNER_GITHUB_SHA:-${GITHUB_SHA:-HEAD}}"
    local diff_range=""

    case "$event_name" in
        pull_request)
            git fetch --quiet --no-tags origin main:refs/remotes/origin/main >/dev/null 2>&1 || true
            if git rev-parse --verify origin/main >/dev/null 2>&1 && git merge-base origin/main HEAD >/dev/null 2>&1; then
                diff_range="origin/main...HEAD"
            fi
            ;;
        push)
            if [ -n "$before_sha" ] && [ "$before_sha" != "0000000000000000000000000000000000000000" ]; then
                git fetch --quiet --no-tags --depth=1 origin "$before_sha" >/dev/null 2>&1 || true
                if git cat-file -e "${before_sha}^{commit}" 2>/dev/null && git cat-file -e "${current_sha}^{commit}" 2>/dev/null; then
                    diff_range="${before_sha}..${current_sha}"
                fi
            fi
            ;;
        *)
            diff_range=""
            ;;
    esac

    echo "$diff_range"
}

run_flake8() {
    local flake8_exit=0

    log "mode=$MODE"
    log "flake8 scope=src tests"

    set +e
    "$PYTHON_BIN" -m flake8 src/ tests/ \
        --count \
        --select=E9,F63,F7,F82 \
        --show-source \
        --statistics
    flake8_exit=$?
    set -e

    if [ "$flake8_exit" -eq 0 ]; then
        log "flake8 passed"
    else
        log "flake8 failed (exit=$flake8_exit)"
    fi

    return "$flake8_exit"
}

select_pylint_files() {
    local diff_range=""
    local candidate=""
    local diff_reliable=0

    PYLINT_ACTION=""
    PYLINT_FILES=()

    case "$MODE" in
        local|advisory)
            diff_range=$(resolve_local_diff_range)
            if [ -n "$diff_range" ] && collect_pylint_files_from_diff_range "$diff_range"; then
                diff_reliable=1
            elif [ -n "$diff_range" ]; then
                log "unable to diff range '$diff_range'; using fallback lint surface"
            fi

            if [ "$diff_reliable" -eq 0 ] || [ "${#PYLINT_FILES[@]}" -eq 0 ]; then
                PYLINT_FILES=("${PYLINT_FALLBACK[@]}")
                PYLINT_ACTION="fallback pylint"
            else
                PYLINT_ACTION="running pylint"
            fi
            ;;
        pr)
            diff_range=$(resolve_pr_diff_range)
            if [ -n "$diff_range" ] && collect_pylint_files_from_diff_range "$diff_range"; then
                diff_reliable=1
            elif [ -n "$diff_range" ]; then
                log "unable to diff range '$diff_range'; using fallback lint surface"
            else
                log "no reliable PR diff range available; using fallback lint surface"
            fi

            if [ "$diff_reliable" -eq 0 ]; then
                PYLINT_FILES=("${PYLINT_FALLBACK[@]}")
                PYLINT_ACTION="fallback pylint"
            elif [ "${#PYLINT_FILES[@]}" -eq 0 ]; then
                PYLINT_ACTION="no eligible Python files changed; skipping pylint"
            else
                PYLINT_ACTION="running pylint"
            fi
            ;;
    esac
}

run_pylint() {
    local pylint_exit=0
    local blocking_exit=0
    local pylint_pythonpath=""

    select_pylint_files

    log "pylint candidates=${#PYLINT_FILES[@]}"
    log "$PYLINT_ACTION"

    if [ "${#PYLINT_FILES[@]}" -eq 0 ]; then
        return 0
    fi

    pylint_pythonpath="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

    set +e
    PYTHONPATH="$pylint_pythonpath" "$PYTHON_BIN" -m pylint --jobs=1 "${PYLINT_FILES[@]}"
    pylint_exit=$?
    set -e

    # Pylint uses bitwise exit codes:
    # 1=fatal, 2=error, 4=warning, 8=refactor, 16=convention, 32=usage.
    # Only fatal, error, and usage are blocking in local/pr modes.
    if [ $((pylint_exit & 1)) -ne 0 ] || [ $((pylint_exit & 2)) -ne 0 ] || [ $((pylint_exit & 32)) -ne 0 ]; then
        blocking_exit=1
        log "pylint found blocking issues (exit=$pylint_exit)"
    elif [ "$pylint_exit" -ne 0 ]; then
        log "pylint reported advisory findings (exit=$pylint_exit)"
    else
        log "pylint passed"
    fi

    return "$blocking_exit"
}

require_module flake8
require_module pylint

FLAKE8_EXIT=0
PYLINT_BLOCKING_EXIT=0

set +e
run_flake8
FLAKE8_EXIT=$?
run_pylint
PYLINT_BLOCKING_EXIT=$?
set -e

if [ "$MODE" = "advisory" ]; then
    exit 0
fi

if [ "$FLAKE8_EXIT" -ne 0 ] || [ "$PYLINT_BLOCKING_EXIT" -ne 0 ]; then
    exit 1
fi
