#!/usr/bin/env python3
"""
Test the automated repository organization system.

This test verifies that:
1. The organization script exists and is executable
2. The pre-commit hook exists and is executable
3. The installation script exists and is executable
4. All documented files are in their correct locations
"""

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_organization_scripts_exist():
    """Test that all organization scripts exist and are executable."""
    repo_root = Path(__file__).parent.parent

    scripts = [
        ".auto-organize.sh",
        "scripts/governance/check_script_topology.py",
        "scripts/setup/auto-organize-install.sh",
        "scripts/setup/pre-commit-check.sh",
    ]

    for script in scripts:
        script_path = repo_root / script
        assert script_path.exists(), f"Script not found: {script}"
        assert os.access(script_path, os.X_OK), f"Script not executable: {script}"


def test_documentation_exists():
    """Test that organization documentation exists."""
    repo_root = Path(__file__).parent.parent

    docs = [
        "docs/governance/REPO_ORGANIZATION.md",
        "scripts/setup/README.md",
    ]

    for doc in docs:
        doc_path = repo_root / doc
        assert doc_path.exists(), f"Documentation not found: {doc}"


def test_gitattributes_exists():
    """Test that .gitattributes file exists."""
    repo_root = Path(__file__).parent.parent
    gitattributes = repo_root / ".gitattributes"
    assert gitattributes.exists(), ".gitattributes not found"


def test_directory_structure():
    """Test that the organized directory structure exists."""
    repo_root = Path(__file__).parent.parent

    directories = [
        "docs/guides",
        "docs/architecture",
        "docs/api",
        "docs/deployment",
        "scripts/setup",
        "scripts/automation",
        "scripts/utilities",
        "archive",
        "data",
        "assets",
    ]

    for directory in directories:
        dir_path = repo_root / directory
        assert dir_path.exists(), f"Directory not found: {directory}"
        assert dir_path.is_dir(), f"Not a directory: {directory}"


def test_organization_script_dry_run():
    """Test that the organization script runs in dry-run mode."""
    repo_root = Path(__file__).parent.parent
    script = repo_root / ".auto-organize.sh"

    result = subprocess.run([str(script), "--dry-run"], cwd=str(repo_root), capture_output=True, text=True, check=False)

    assert result.returncode == 0, f"Organization script failed: {result.stderr}"
    assert "DRY RUN" in result.stdout, "Dry run mode not detected"


def test_files_organized():
    """Test that specific files were moved to correct locations."""
    repo_root = Path(__file__).parent.parent

    # Files that should have been moved to docs/guides/
    moved_to_guides = [
        "docs/guides/START_HERE.md",
        "docs/guides/SYSTEM_STATUS.md",
        "docs/guides/CI_WORKFLOW_OPTIMIZATION.md",
    ]

    for file_path in moved_to_guides:
        full_path = repo_root / file_path
        assert full_path.exists(), f"Expected file not found: {file_path}"

    # Files that should have been moved to scripts/utilities/
    moved_to_utilities = [
        "scripts/utilities/navigate.sh",
        "scripts/utilities/verify_organization.sh",
    ]

    for file_path in moved_to_utilities:
        full_path = repo_root / file_path
        assert full_path.exists(), f"Expected file not found: {file_path}"

    # Files that should NOT be in root
    should_not_be_in_root = [
        "START_HERE.md",
        "SYSTEM_STATUS.md",
        "navigate.sh",
        "verify_organization.sh",
    ]

    for filename in should_not_be_in_root:
        full_path = repo_root / filename
        assert not full_path.exists(), f"File should not be in root: {filename}"


if __name__ == "__main__":
    # Run all tests
    test_organization_scripts_exist()
    print("✓ Organization scripts exist and are executable")

    test_documentation_exists()
    print("✓ Documentation exists")

    test_gitattributes_exists()
    print("✓ .gitattributes exists")

    test_directory_structure()
    print("✓ Directory structure is correct")

    test_organization_script_dry_run()
    print("✓ Organization script runs in dry-run mode")

    test_files_organized()
    print("✓ Files are correctly organized")

    print("\n✅ All organization system tests passed!")
