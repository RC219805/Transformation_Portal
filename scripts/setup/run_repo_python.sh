#!/usr/bin/env bash
#
# Run a repository Python script through the same interpreter resolver used by
# Makefile targets. This keeps pre-commit system hooks independent of a bare
# `python` binary on PATH while still failing closed on unsupported runtimes.

set -euo pipefail

if (($# < 1)); then
    echo "Usage: $0 <script.py> [args...]" >&2
    exit 64
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="$("${SCRIPT_DIR}/resolve_python_311.sh")"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" "$@"
