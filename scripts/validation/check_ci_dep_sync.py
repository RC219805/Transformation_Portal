#!/usr/bin/env python3
"""
Validates that CI dependency files are in sync and do not drift.

This script ensures:
1. Test runner deps (pytest-*, httpx) are NOT duplicated between:
   - requirements-ci.txt (root) and requirements/ci.in
2. Test deps in requirements-ci.txt align with requirements/dev.in
3. requirements/ci.in contains ONLY CI pipeline tools (bandit, safety, build, twine, etc.)
4. Dev-only tools in requirements/dev.in are exposed through root requirements-dev.txt

Run via:
    python scripts/validation/check_ci_dep_sync.py
    make check-ci-sync

Exit codes:
    0 - No drift detected
    1 - Drift detected between files
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION: Canonical lists maintained here as the single source of truth
# ============================================================================

# Packages that are test runners/frameworks (should be in dev.in/root CI, NOT ci.in).
# Pattern matches exact names: pytest, httpx, hypothesis, or any pytest-* plugin.
TEST_RUNNER_PATTERN = re.compile(r"^(pytest|pytest-.+|httpx|hypothesis)$")

# CI pipeline tools (should be in ci.in, NOT root requirements-ci.txt)
# These are security/packaging/release tools, not test runners
CI_TOOLS = frozenset({"bandit", "safety", "build", "twine", "tox", "pypdf"})

# Core test support packages that MUST be in both root requirements-ci.txt
# AND requirements/dev.in to ensure sync. This is a contract.
# Update this set when adding new core test dependencies.
CORE_TEST_DEPS = frozenset({
    "pytest",
    "pytest-cov",
    "pytest-asyncio",
    "pytest-json-report",
    "pytest-xdist",
    "hypothesis",
    "httpx",
    "moto",
})

# Developer-only test tooling that should be available from the documented
# root development entry point, but should not be pulled into lean CI installs.
DEV_ONLY_DEPS = frozenset({
    "pytest-rerunfailures",
})


def extract_packages(filepath: Path) -> set[str]:
    """Extract package names from a requirements file, ignoring comments and -r lines.

    Missing files are treated as errors to avoid silently passing CI drift checks
    with incomplete inputs.
    """
    if not filepath.exists():
        # Fail fast: this script is a CI gate and expects all configured files
        raise FileNotFoundError(f"Required requirements file not found: {filepath}")

    packages: set[str] = set()
    for line in filepath.read_text().splitlines():
        line = line.strip()
        # Skip empty lines, comments, and -r includes
        if not line or line.startswith("#") or line.startswith("-r"):
            continue
        # Extract package name (before any version specifier)
        # Pattern handles: standard names, dots (zope.interface), underscores, hyphens
        match = re.match(r"^([a-zA-Z0-9._-]+)", line)
        if match:
            # Normalize per PEP 503: lowercase, convert underscores/dots/hyphens to hyphens
            normalized = match.group(1).lower().replace("_", "-").replace(".", "-")
            packages.add(normalized)
    return packages


def validate_dependency_sync(repo_root: Path) -> list[str]:
    """Return dependency sync errors for the configured requirements files."""
    root_ci = repo_root / "requirements-ci.txt"
    root_dev = repo_root / "requirements-dev.txt"
    nested_ci_in = repo_root / "requirements" / "ci.in"
    nested_dev_in = repo_root / "requirements" / "dev.in"

    errors: list[str] = []

    # Extract packages
    root_ci_packages = extract_packages(root_ci)
    root_dev_packages = extract_packages(root_dev)
    nested_ci_packages = extract_packages(nested_ci_in)
    nested_dev_packages = extract_packages(nested_dev_in)

    # Check 1: Test runner deps should NOT be in nested ci.in (they belong in dev.in or root)
    test_deps_in_nested_ci = {p for p in nested_ci_packages if TEST_RUNNER_PATTERN.match(p)}
    test_deps_in_nested_ci |= nested_ci_packages & CORE_TEST_DEPS
    if test_deps_in_nested_ci:
        errors.append(
            f"ERROR: Test deps found in requirements/ci.in (should be in dev.in/root CI):\n"
            f"       {sorted(test_deps_in_nested_ci)}"
        )

    # Check 2: CI pipeline tools in nested ci.in should NOT be in root requirements-ci.txt
    # (CI tools like bandit, safety, build, twine are specialized and not needed for test runs)
    ci_tools_in_root = root_ci_packages & CI_TOOLS
    if ci_tools_in_root:
        errors.append(
            f"ERROR: CI pipeline tools found in requirements-ci.txt (should be in requirements/ci.in):\n"
            f"       {sorted(ci_tools_in_root)}"
        )

    # Check 3: Core test deps must be in both root CI and dev.in (sync requirement)
    # Uses CORE_TEST_DEPS as the canonical contract
    missing_in_root = CORE_TEST_DEPS - root_ci_packages
    if missing_in_root:
        errors.append(
            f"ERROR: Core test deps missing from requirements-ci.txt:\n"
            f"       {sorted(missing_in_root)}\n"
            f"       Add these to requirements-ci.txt so lean CI runs exercise the full test contract."
        )

    missing_in_dev = CORE_TEST_DEPS - nested_dev_packages
    if missing_in_dev:
        errors.append(
            f"ERROR: Test deps in requirements-ci.txt missing from requirements/dev.in:\n"
            f"       {sorted(missing_in_dev)}\n"
            f"       Add these to requirements/dev.in to maintain sync."
        )

    # Check 4: Developer-only tools must be present in both the governed
    # layered dev input and the documented root development entry point.
    missing_dev_only_in_root_dev = DEV_ONLY_DEPS - root_dev_packages
    if missing_dev_only_in_root_dev:
        errors.append(
            f"ERROR: Dev-only deps missing from requirements-dev.txt:\n"
            f"       {sorted(missing_dev_only_in_root_dev)}\n"
            f"       Add these to requirements-dev.txt so documented dev installs match requirements/dev.in."
        )

    missing_dev_only_in_nested_dev = DEV_ONLY_DEPS - nested_dev_packages
    if missing_dev_only_in_nested_dev:
        errors.append(
            f"ERROR: Dev-only deps missing from requirements/dev.in:\n"
            f"       {sorted(missing_dev_only_in_nested_dev)}\n"
            f"       Add these to requirements/dev.in to maintain the governed layered dev contract."
        )

    dev_only_in_root_ci = root_ci_packages & DEV_ONLY_DEPS
    if dev_only_in_root_ci:
        errors.append(
            f"ERROR: Dev-only deps found in requirements-ci.txt:\n"
            f"       {sorted(dev_only_in_root_ci)}\n"
            f"       Move these to requirements-dev.txt so lean CI installs stay minimal."
        )

    # Note: Check 5 (pytest-* detection) was removed because TEST_RUNNER_PATTERN in Check 1
    # already matches pytest-* plugins, so violations would be reported twice.
    return errors


def main() -> int:
    """Main entry point."""
    repo_root = Path(__file__).resolve().parents[2]
    errors = validate_dependency_sync(repo_root)

    # Report results
    if errors:
        print("=" * 70)
        print("CI DEPENDENCY SYNC CHECK FAILED")
        print("=" * 70)
        for error in errors:
            print(f"\n{error}")
        print("\n" + "=" * 70)
        print("To fix:")
        print("  1. Move test deps to requirements/dev.in (NOT requirements/ci.in)")
        print("  2. Keep CI pipeline tools (bandit, safety, etc.) in requirements/ci.in")
        print("  3. Root requirements-ci.txt is the source for lean CI test runs")
        print("  4. Keep dev-only tools in requirements/dev.in and requirements-dev.txt")
        print("=" * 70)
        return 1

    print("✓ CI dependency files are in sync (no drift detected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
