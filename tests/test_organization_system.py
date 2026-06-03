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
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).parent.parent


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)


def _write(path: Path, content: str = "test\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _copy_executable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _shell_array_values(array_name: str) -> list[str]:
    script = (REPO_ROOT / "scripts" / "setup" / "pre-commit-check.sh").read_text(encoding="utf-8")
    pattern = rf"{array_name}=\(\n(?P<body>.*?)\n\)"
    match = re.search(pattern, script, flags=re.DOTALL)
    assert match, f"{array_name} block not found"
    values = []
    for line in match.group("body").splitlines():
        candidate = line.strip().strip('"')
        if candidate:
            values.append(candidate)
    return values


def _allowed_root_files() -> list[str]:
    return _shell_array_values("ALLOWED_ROOT_FILES")


def _allowed_root_directories() -> list[str]:
    return _shell_array_values("ALLOWED_ROOT_DIRECTORIES")


def test_organization_scripts_exist():
    """Test that all organization scripts exist and are executable."""
    repo_root = REPO_ROOT

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


def test_auto_organize_header_lists_governed_validators():
    """The root organization entrypoint should document the validators it runs."""
    script = (REPO_ROOT / ".auto-organize.sh").read_text(encoding="utf-8")

    required_helpers = [
        "scripts/organize_docs.sh",
        "scripts/setup/pre-commit-check.sh",
        "scripts/governance/check_script_topology.py",
        "scripts/governance/check_docs_structure.py",
    ]
    for helper in required_helpers:
        assert helper in script


def test_documentation_exists():
    """Test that organization documentation exists."""
    repo_root = REPO_ROOT

    docs = [
        "docs/governance/REPO_ORGANIZATION.md",
        "scripts/setup/README.md",
    ]

    for doc in docs:
        doc_path = repo_root / doc
        assert doc_path.exists(), f"Documentation not found: {doc}"


def test_repo_organization_doc_uses_check_mode_for_ci():
    """CI guidance should use the fail-closed organization validation mode."""
    doc = (REPO_ROOT / "docs" / "governance" / "REPO_ORGANIZATION.md").read_text(encoding="utf-8")

    assert ".auto-organize.sh --check" in doc
    assert "CI check that runs `.auto-organize.sh --dry-run`" not in doc


def test_docs_only_guidance_mentions_topology_validation():
    """Docs-only mode should be documented as docs validation, not just move planning."""
    script = (REPO_ROOT / ".auto-organize.sh").read_text(encoding="utf-8")
    doc = (REPO_ROOT / "docs" / "governance" / "REPO_ORGANIZATION.md").read_text(encoding="utf-8")
    combined = script + "\n" + doc

    assert "Only run documentation organization and docs topology validation" in combined
    assert "Preview docs moves and validate docs topology" in combined
    assert "Only organize documentation files" not in combined
    assert "Preview documentation moves only" not in combined


def test_organization_hook_installation_guidance_uses_repo_managed_hooks():
    """Current organization guidance should not teach direct hook symlinks."""
    doc = (REPO_ROOT / "docs" / "governance" / "REPO_ORGANIZATION.md").read_text(encoding="utf-8")
    setup_readme = (REPO_ROOT / "scripts" / "setup" / "README.md").read_text(encoding="utf-8")
    quality_readme = (REPO_ROOT / "scripts" / "README_QUALITY_CONTROL.md").read_text(encoding="utf-8")
    installer = (REPO_ROOT / "scripts" / "setup" / "auto-organize-install.sh").read_text(encoding="utf-8")
    combined = "\n".join([doc, setup_readme, quality_readme, installer])

    assert "make install-hooks" in combined
    assert "pre-commit and pre-push" in combined
    assert ".git/hooks/pre-push" in combined
    assert "ln -sf ../../scripts/setup/pre-commit-check.sh .git/hooks/pre-commit" not in combined
    assert "chmod +x .git/hooks/pre-commit" not in combined
    assert "pre-commit install -f" not in combined
    assert "command -v pre-commit" not in installer


def test_repo_organization_doc_lists_allowed_root_files():
    """The governance doc should mirror the root-placement file allowlist."""
    doc = (REPO_ROOT / "docs" / "governance" / "REPO_ORGANIZATION.md").read_text(encoding="utf-8")

    assert "Current allowed root files" in doc
    for filename in _allowed_root_files():
        assert f"`{filename}`" in doc


def test_requirements_readme_documents_allowed_root_requirement_files():
    """The dependency README should describe every allowed root requirements file."""
    doc = (REPO_ROOT / "requirements" / "README.md").read_text(encoding="utf-8")
    root_requirement_files = sorted(
        filename for filename in _allowed_root_files() if filename.startswith("requirements") and filename.endswith(".txt")
    )

    assert root_requirement_files
    for filename in root_requirement_files:
        assert f"`{filename}`" in doc


def test_repo_organization_doc_documents_cloudflare_root_shim_boundary():
    """Root Node files should be documented as Cloudflare Worker deploy shims only."""
    doc = (REPO_ROOT / "docs" / "governance" / "REPO_ORGANIZATION.md").read_text(encoding="utf-8")

    assert "minimal Workers Builds deploy shim" in doc
    assert "`cloudflare/transformationportal-worker`" in doc
    assert "`tests/validation/test_cloudflare_worker_root_shim_contract.py`" in doc


def test_repo_organization_doc_lists_allowed_top_level_directories():
    """The governance doc should mirror the root-placement directory allowlist."""
    doc = (REPO_ROOT / "docs" / "governance" / "REPO_ORGANIZATION.md").read_text(encoding="utf-8")

    assert "Current allowed top-level directories" in doc
    for directory in _allowed_root_directories():
        assert f"`{directory}/`" in doc


def test_gitattributes_exists():
    """Test that .gitattributes file exists."""
    repo_root = REPO_ROOT
    gitattributes = repo_root / ".gitattributes"
    assert gitattributes.exists(), ".gitattributes not found"


def test_directory_structure():
    """Test that the organized directory structure exists."""
    repo_root = REPO_ROOT

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
    repo_root = REPO_ROOT
    script = repo_root / ".auto-organize.sh"

    result = subprocess.run([str(script), "--dry-run"], cwd=str(repo_root), capture_output=True, text=True, check=False)

    assert result.returncode == 0, f"Organization script failed: {result.stderr}"
    assert "DRY RUN" in result.stdout, "Dry run mode not detected"


def test_organization_script_check_mode_uses_validation_messaging():
    """Check mode should not tell CI users to apply organization changes."""
    result = _run(["./.auto-organize.sh", "--check"], REPO_ROOT)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Mode: CHECK" in result.stdout
    assert "Organization validation completed successfully" in result.stdout
    assert "No organization changes are required." in result.stdout
    assert "To apply changes, run without --dry-run" not in result.stdout


def test_organization_script_docs_only_check_validates_docs_structure():
    """Docs-only check mode should still run documentation topology validation."""
    result = _run(["./.auto-organize.sh", "--docs-only", "--check"], REPO_ROOT)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Organizing Documentation Files" in result.stdout
    assert "Validating Documentation Structure" in result.stdout
    assert "Summary: 2/2 steps passed" in result.stdout
    assert "Validating Root File Placement" not in result.stdout
    assert "Checking for Misplaced Root Scripts" not in result.stdout
    assert "Validating Script Topology" not in result.stdout


def test_organization_script_skip_root_check_only_skips_root_placement():
    """Skip-root mode should retain non-placement organization validators."""
    result = _run(["./.auto-organize.sh", "--skip-root", "--check"], REPO_ROOT)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Skipping root file validation (--skip-root)" in result.stdout
    assert "Checking for Misplaced Root Scripts" in result.stdout
    assert "Checking for Misplaced Shell Scripts in Root" in result.stdout
    assert "Validating Script Topology" in result.stdout
    assert "Validating Documentation Structure" in result.stdout
    assert "Summary: 6/6 steps passed" in result.stdout


def test_auto_organize_flags_retired_root_python_files(tmp_path: Path):
    """Root setup/conftest/package markers should not bypass root script checks."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    assert _run(["git", "init", "-q"], repo_root).returncode == 0

    _copy_executable(REPO_ROOT / ".auto-organize.sh", repo_root / ".auto-organize.sh")
    _copy_executable(REPO_ROOT / "scripts" / "setup" / "pre-commit-check.sh", repo_root / "scripts/setup/pre-commit-check.sh")

    _write(
        repo_root / "scripts" / "organize_docs.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\necho 'No candidate documentation files found.'\n",
    )
    _write(
        repo_root / "scripts" / "governance" / "check_script_topology.py",
        "#!/usr/bin/env python3\nprint('Script topology check passed.')\n",
    )
    _write(
        repo_root / "scripts" / "governance" / "check_docs_structure.py",
        "#!/usr/bin/env python3\nprint('Documentation structure check passed.')\n",
    )
    for helper in (
        repo_root / "scripts" / "organize_docs.sh",
        repo_root / "scripts" / "governance" / "check_script_topology.py",
        repo_root / "scripts" / "governance" / "check_docs_structure.py",
    ):
        helper.chmod(helper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    for root_python in ("app.py", "setup.py", "conftest.py", "__init__.py", "helper.py"):
        _write(repo_root / root_python, "print('placeholder')\n")

    add_result = _run(
        [
            "git",
            "add",
            ".auto-organize.sh",
            "scripts/setup/pre-commit-check.sh",
            "scripts/organize_docs.sh",
            "scripts/governance/check_script_topology.py",
            "scripts/governance/check_docs_structure.py",
            "app.py",
            "setup.py",
            "conftest.py",
            "__init__.py",
            "helper.py",
        ],
        repo_root,
    )
    assert add_result.returncode == 0, add_result.stdout + add_result.stderr

    result = _run(["./.auto-organize.sh", "--check"], repo_root)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "app.py →" not in result.stdout
    assert "setup.py → governed package, tool, or test location" in result.stdout
    assert "conftest.py → governed package, tool, or test location" in result.stdout
    assert "__init__.py → governed package, tool, or test location" in result.stdout
    assert "helper.py → scripts/ or src/transformation_portal/" in result.stdout


def test_files_organized():
    """Test that specific files were moved to correct locations."""
    repo_root = REPO_ROOT

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
