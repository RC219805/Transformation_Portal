#!/usr/bin/env bash
#
# Compatibility verifier for repository organization.
#
# Keep policy in .auto-organize.sh and the governed validators instead of
# duplicating path rules here.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

echo "============================================"
echo "Repository Organization Verification"
echo "============================================"
echo ""

./.auto-organize.sh --check --verbose
