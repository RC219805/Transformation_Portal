#!/usr/bin/env bash
# Unified pre-commit quality gate wrapper.
#
# This is the canonical script for running the repository's pre-commit quality
# checks outside of the pre-commit framework. It delegates to the comprehensive
# Python quality gate at scripts/utilities/pre-commit-quality-check.py.
#
# Usage:
#   ./scripts/pre_commit_hook.sh              # Check staged files only
#   ./scripts/pre_commit_hook.sh --all-files  # Check all tracked files
#   ./scripts/pre_commit_hook.sh --quick-tests # Also run legacy quick pytest smoke
#
# Preferred installation method for git hooks:
#   make install-hooks
#   # or
#   pre-commit install -f
#
# The pre-commit framework (.pre-commit-config.yaml) is the canonical way to
# install git hooks in this repository. This script is provided for:
# - Manual execution of the quality gate
# - Compatibility with workflows that call this script directly
# - CI/CD pipelines that need explicit hook invocation

set -euo pipefail

# Resolve repository root - prefer git if available, fall back to script location
if git rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git rev-parse --show-toplevel)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

QUALITY_GATE="$REPO_ROOT/scripts/utilities/pre-commit-quality-check.py"

if [[ ! -f "$QUALITY_GATE" ]]; then
    echo "Pre-commit quality gate is missing: $QUALITY_GATE" >&2
    exit 1
fi

# Resolve Python interpreter with preference for lint venv, then main venv
if [[ -x "$REPO_ROOT/.venv-lint/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv-lint/bin/python"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
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
