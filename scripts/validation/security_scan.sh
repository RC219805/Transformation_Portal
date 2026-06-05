#!/usr/bin/env bash
# Security scan script matching CI policy
# Run this locally to verify security gate before pushing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="$("$REPO_ROOT/scripts/setup/resolve_python_311.sh")"

echo "Running Bandit security scan with CI flags..."
echo "Policy: severity >= LOW, confidence >= MEDIUM"
echo ""

cd "$REPO_ROOT"
"$PYTHON_BIN" -m bandit -r src/ -ll -ii

echo ""
echo "✅ Security scan passed"
