#!/usr/bin/env bash
# Compatibility wrapper for the unified pre-commit quality gate.
#
# Manual run:
#   ./scripts/pre_commit_hook.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
QUALITY_GATE="$REPO_ROOT/scripts/utilities/pre-commit-quality-check.py"

if [[ ! -x "$QUALITY_GATE" ]]; then
    echo "Pre-commit quality gate is missing or not executable: $QUALITY_GATE" >&2
    exit 1
fi

cd "$REPO_ROOT"
exec "$QUALITY_GATE" "$@"
