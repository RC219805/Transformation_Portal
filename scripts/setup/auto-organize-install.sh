#!/usr/bin/env bash
#
# auto-organize-install.sh
# Installation script for the automated repository organization system
#
# This script installs the pre-commit hook and configures the organization system.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Transformation Portal Organization System Installer ==="
echo ""

# Check if we're in a git repository
if [[ ! -d "$REPO_ROOT/.git" ]]; then
    echo "ERROR: Not in a git repository"
    echo "Please run this script from within the Transformation_Portal repository"
    exit 1
fi

echo "Repository root: $REPO_ROOT"
echo ""

# Install pre-commit hook
echo "Installing pre-commit hook..."
if ! command -v pre-commit >/dev/null 2>&1; then
    echo "ERROR: 'pre-commit' is required but not installed."
    echo "Install it with: python3 -m pip install pre-commit"
    exit 1
fi

(cd "$REPO_ROOT" && pre-commit install -f)
echo "  ✓ Pre-commit hook installed"

# Make organization script executable
ORGANIZE_SCRIPT="$REPO_ROOT/.auto-organize.sh"
if [[ -f "$ORGANIZE_SCRIPT" ]]; then
    chmod +x "$ORGANIZE_SCRIPT"
    echo "  ✓ Organization script is executable"
else
    echo "  WARNING: Organization script not found at: $ORGANIZE_SCRIPT"
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "The organization system is now active. Here's what you can do:"
echo ""
echo "1. Test the organization (dry-run):"
echo "   ./.auto-organize.sh --dry-run"
echo ""
echo "2. Organize the repository:"
echo "   ./.auto-organize.sh"
echo ""
echo "3. View organization documentation:"
echo "   cat docs/governance/REPO_ORGANIZATION.md"
echo ""
echo "The pre-commit hook will now run the repository hook set before each commit."
echo "That includes misplaced-file detection as one of its checks. To bypass (not recommended):"
echo "   git commit --no-verify"
echo ""
