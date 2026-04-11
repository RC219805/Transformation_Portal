#!/usr/bin/env bash
#
# resolve_python_311.sh
# Print a usable Python 3.11+ interpreter for this repository.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

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

    "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' >/dev/null 2>&1
}

emit_guidance() {
    cat >&2 <<EOF
[ERROR] Transformation Portal requires Python ${MIN_MAJOR}.${MIN_MINOR}+.
[ERROR] Checked candidates:
[ERROR]   1. ${REPO_VENV_PYTHON}
[ERROR]   2. python3.13
[ERROR]   3. python3.12
[ERROR]   4. python3.11
[ERROR]   5. python3
[ERROR]   6. python
[ERROR]
[ERROR] Install Python ${MIN_MAJOR}.${MIN_MINOR}+ and retry, then run:
[ERROR]   make venv
[ERROR] If you need to bootstrap manually:
[ERROR]   python3.11 -m venv .venv
EOF
}

main() {
    local candidate
    for candidate in \
        "${REPO_VENV_PYTHON}" \
        python3.13 \
        python3.12 \
        python3.11 \
        python3 \
        python; do
        if is_supported_python "$candidate"; then
            if [[ "$candidate" == */* ]]; then
                printf '%s\n' "$candidate"
            else
                command -v "$candidate"
            fi
            return 0
        fi
    done

    emit_guidance
    return 1
}

main "$@"
