#!/usr/bin/env bash
#
# resolve_python_311.sh
# Print a usable Python 3.11+ interpreter for this repository.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Platform-aware repo venv interpreter path
# On Windows (Git Bash/MSYS/Cygwin), check both layouts
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    REPO_VENV_PYTHON="${REPO_ROOT}/.venv/Scripts/python.exe"
    REPO_VENV_PYTHON_ALT="${REPO_ROOT}/.venv/bin/python"
else
    REPO_VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
    REPO_VENV_PYTHON_ALT=""
fi

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
[ERROR] Checked candidates (in order):
[ERROR]   1. ${REPO_VENV_PYTHON}
[ERROR]   2. python3.15, python3.14, python3.13, python3.12, python3.11
[ERROR]   3. python3
[ERROR]   4. python
[ERROR]
[ERROR] Install Python ${MIN_MAJOR}.${MIN_MINOR}+ and retry, then run:
[ERROR]   make venv
[ERROR] If you need to bootstrap manually, use any available Python ${MIN_MAJOR}.${MIN_MINOR}+ interpreter:
[ERROR]   python3.13 -m venv .venv
[ERROR]   python3.12 -m venv .venv
[ERROR]   python3.11 -m venv .venv
EOF
}

main() {
    local candidate

    # First check repo venv (platform-specific path)
    if is_supported_python "${REPO_VENV_PYTHON}"; then
        printf '%s\n' "${REPO_VENV_PYTHON}"
        return 0
    fi

    # On Windows, also check the alternative layout
    if [[ -n "${REPO_VENV_PYTHON_ALT:-}" ]] && is_supported_python "${REPO_VENV_PYTHON_ALT}"; then
        printf '%s\n' "${REPO_VENV_PYTHON_ALT}"
        return 0
    fi

    # Probe versioned interpreters from newest to oldest (3.15 down to 3.11)
    # This future-proofs against newer Python releases
    for candidate in python3.15 python3.14 python3.13 python3.12 python3.11; do
        if is_supported_python "$candidate"; then
            command -v "$candidate"
            return 0
        fi
    done

    # Fallback to generic python3/python if they meet the version requirement
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
