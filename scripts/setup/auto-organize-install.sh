#!/usr/bin/env bash
#
# auto-organize-install.sh
# Installation script for the automated repository organization system
#
# This script installs the repo-managed pre-commit framework hook set and
# configures the organization system.
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

# Install the configured pre-commit framework hook set. The Make target uses
# the repo-managed .venv pre-commit binary and .pre-commit-config.yaml's
# default_install_hook_types, so pre-commit and pre-push stay in sync.
echo "Installing pre-commit and pre-push hooks..."
(cd "$REPO_ROOT" && make install-hooks)
echo "  ✓ Pre-commit and pre-push hooks installed"

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
echo "The repository hook set now runs before each commit and push."
echo "That includes misplaced-file detection and push-time secrets checks. To bypass (not recommended):"
echo "   git commit --no-verify"
echo "   git push --no-verify"
echo ""
