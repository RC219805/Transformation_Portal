#!/usr/bin/env bash
#
# resolve_python_311.sh
# Print a usable Python 3.11+ interpreter for this repository.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_VENV_CANDIDATES=(
    "${REPO_ROOT}/.venv/bin/python"
    "${REPO_ROOT}/.venv/Scripts/python.exe"
)

MIN_MAJOR=3
MIN_MINOR=11

is_supported_python() {
    local candidate="$1"

    if [[ "$candidate" == */* ]]; then
        [[ -x "$candidate" ]] || return 1
    else
        candidate="$(command -v "$candidate" 2>/dev/null || true)"
        [[ -n "$candidate" ]] || return 1
    fi

    "$candidate" -I -S -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' >/dev/null 2>&1
}

emit_guidance() {
    cat >&2 <<EOF
[ERROR] Transformation Portal requires Python ${MIN_MAJOR}.${MIN_MINOR}+.
[ERROR] Checked candidates (in order):
[ERROR]   1. ${REPO_VENV_CANDIDATES[0]}
[ERROR]   2. ${REPO_VENV_CANDIDATES[1]}
[ERROR]   3. versioned python3.N commands on PATH (newest supported first)
[ERROR]   4. python3
[ERROR]   5. python
[ERROR]
[ERROR] Install Python ${MIN_MAJOR}.${MIN_MINOR}+ and retry, then run:
[ERROR]   make venv
[ERROR] If you need to bootstrap manually, use any available Python ${MIN_MAJOR}.${MIN_MINOR}+ interpreter, for example:
[ERROR]   python3.13 -m venv .venv
[ERROR]   python3.12 -m venv .venv
[ERROR]   python3.11 -m venv .venv
EOF
}

discover_versioned_python_candidates() {
    local path_dir candidate candidate_name minor
    local discovered=()

    IFS=: read -r -a path_dirs <<< "${PATH:-}"
    for path_dir in "${path_dirs[@]}"; do
        [[ -d "$path_dir" ]] || continue
        for candidate in "$path_dir"/python3.*; do
            [[ -e "$candidate" ]] || continue
            candidate_name="${candidate##*/}"
            if [[ "$candidate_name" =~ ^python3\.([0-9]+)(\.exe)?$ ]]; then
                minor="${BASH_REMATCH[1]}"
                (( minor < MIN_MINOR )) && continue
                discovered+=("${minor}:${candidate_name}")
            fi
        done
    done

    if ((${#discovered[@]})); then
        printf '%s\n' "${discovered[@]}" | sort -t: -k1,1nr -k2,2 | awk -F: '!seen[$2]++ { print $2 }'
    fi
}

main() {
    local candidate

    for candidate in "${REPO_VENV_CANDIDATES[@]}"; do
        if is_supported_python "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    while IFS= read -r candidate; do
        [[ -n "$candidate" ]] || continue
        if is_supported_python "$candidate"; then
            command -v "$candidate"
            return 0
        fi
    done < <(discover_versioned_python_candidates)

    for candidate in python3 python; do
        if is_supported_python "$candidate"; then
            command -v "$candidate"
            return 0
        fi
    done

    emit_guidance
    return 1
}

main "$@"
