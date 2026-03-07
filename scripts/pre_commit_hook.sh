#!/usr/bin/env bash
# Backward-compatible wrapper for the canonical root placement hook.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/setup/pre-commit-check.sh" "$@"
