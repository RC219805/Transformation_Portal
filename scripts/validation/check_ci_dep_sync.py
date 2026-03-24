#!/usr/bin/env python3
"""
Validates that CI dependency files are in sync and do not drift.

This script ensures:
1. Test runner deps (pytest-*, httpx) are NOT duplicated between:
   - requirements-ci.txt (root) and requirements/ci.in
2. Test deps in requirements-ci.txt align with requirements/dev.in
3. requirements/ci.in contains ONLY CI pipeline tools (bandit, safety, build, twine, etc.)

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


def extract_packages(filepath: Path) -> set[str]:
    """Extract package names from a requirements file, ignoring comments and -r lines."""
    packages = set()
    if not filepath.exists():
        return packages

    for line in filepath.read_text().splitlines():
        line = line.strip()
        # Skip empty lines, comments, and -r includes
        if not line or line.startswith("#") or line.startswith("-r"):
            continue
        # Extract package name (before any version specifier)
        match = re.match(r"^([a-zA-Z0-9_-]+)", line)
        if match:
            packages.add(match.group(1).lower().replace("_", "-"))
    return packages


def main() -> int:
    """Main entry point."""
    repo_root = Path(__file__).resolve().parents[2]
    root_ci = repo_root / "requirements-ci.txt"
    nested_ci_in = repo_root / "requirements" / "ci.in"
    nested_dev_in = repo_root / "requirements" / "dev.in"

    errors: list[str] = []

    # Extract packages
    root_ci_packages = extract_packages(root_ci)
    nested_ci_packages = extract_packages(nested_ci_in)
    nested_dev_packages = extract_packages(nested_dev_in)

    # Check 1: Test runner deps should NOT be in nested ci.in (they belong in dev.in or root)
    test_runner_pattern = re.compile(r"^pytest-|^httpx$|^hypothesis$")
    test_deps_in_nested_ci = {
        p for p in nested_ci_packages if test_runner_pattern.match(p)
    }
    if test_deps_in_nested_ci:
        errors.append(
            f"ERROR: Test runner deps found in requirements/ci.in (should be in dev.in):\n"
            f"       {sorted(test_deps_in_nested_ci)}"
        )

    # Check 2: CI pipeline tools in nested ci.in should NOT be in root requirements-ci.txt
    # (CI tools like bandit, safety, build, twine are specialized and not needed for test runs)
    ci_tools_pattern = re.compile(r"^(bandit|safety|build|twine|tox|pypdf)$")
    ci_tools_in_root = {p for p in root_ci_packages if ci_tools_pattern.match(p)}
    if ci_tools_in_root:
        errors.append(
            f"WARNING: CI pipeline tools found in requirements-ci.txt (should be in requirements/ci.in):\n"
            f"         {sorted(ci_tools_in_root)}"
        )

    # Check 3: Core test deps in root should also be in dev.in (sync requirement)
    # These are the pytest and test framework packages
    core_test_deps = {"pytest", "pytest-cov", "pytest-asyncio", "pytest-json-report", "pytest-xdist", "hypothesis", "httpx"}
    root_test_deps = root_ci_packages & core_test_deps
    dev_test_deps = nested_dev_packages & core_test_deps

    missing_in_dev = root_test_deps - dev_test_deps
    if missing_in_dev:
        errors.append(
            f"ERROR: Test deps in requirements-ci.txt missing from requirements/dev.in:\n"
            f"       {sorted(missing_in_dev)}\n"
            f"       Add these to requirements/dev.in to maintain sync."
        )

    # Check 4: Detect any pytest-* packages in nested ci.in that should be in dev.in
    pytest_plugins_in_ci = {p for p in nested_ci_packages if p.startswith("pytest-")}
    if pytest_plugins_in_ci:
        errors.append(
            f"ERROR: pytest plugins found in requirements/ci.in (should be in dev.in):\n"
            f"       {sorted(pytest_plugins_in_ci)}"
        )

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
        print("=" * 70)
        return 1

    print("✓ CI dependency files are in sync (no drift detected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
