#!/usr/bin/env bash
# Compatibility wrapper for the unified pre-commit quality gate.
#
# Manual run:
#   ./scripts/pre_commit_hook.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
QUALITY_GATE="$REPO_ROOT/scripts/utilities/pre-commit-quality-check.py"

if [[ ! -f "$QUALITY_GATE" ]]; then
    echo "Pre-commit quality gate is missing: $QUALITY_GATE" >&2
    exit 1
fi

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "Python interpreter not found; cannot run pre-commit quality gate." >&2
    exit 1
fi

cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$QUALITY_GATE" "$@"
