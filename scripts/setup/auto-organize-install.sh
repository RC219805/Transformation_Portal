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
PRE_COMMIT_HOOK="$REPO_ROOT/.git/hooks/pre-commit"
PRE_COMMIT_SCRIPT="$SCRIPT_DIR/pre-commit-check.sh"

# Check if pre-commit script exists
if [[ ! -f "$PRE_COMMIT_SCRIPT" ]]; then
    echo "ERROR: Pre-commit script not found at: $PRE_COMMIT_SCRIPT"
    exit 1
fi

# Make pre-commit script executable
chmod +x "$PRE_COMMIT_SCRIPT"

# Create or update pre-commit hook
if [[ -L "$PRE_COMMIT_HOOK" ]]; then
    echo "  Removing existing symbolic link..."
    rm "$PRE_COMMIT_HOOK"
elif [[ -f "$PRE_COMMIT_HOOK" ]]; then
    echo "  Backing up existing pre-commit hook..."
    mv "$PRE_COMMIT_HOOK" "$PRE_COMMIT_HOOK.backup"
    echo "  Backup saved to: $PRE_COMMIT_HOOK.backup"
fi

# Create symbolic link
ln -sf "../../scripts/setup/pre-commit-check.sh" "$PRE_COMMIT_HOOK"
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
echo "   cat REPO_ORGANIZATION.md"
echo ""
echo "The pre-commit hook will now automatically check for misplaced files"
echo "before each commit. To bypass (not recommended):"
echo "   git commit --no-verify"
echo ""
