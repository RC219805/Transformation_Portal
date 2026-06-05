#!/usr/bin/env python3
"""
Validate CI path filters configuration.

Ensures that path filters are correctly configured and that critical
files are not accidentally excluded.
"""

import sys
from pathlib import Path
from typing import Dict, List

import yaml


def load_workflow(workflow_path: Path) -> Dict:
    """Load workflow YAML file."""
    with open(workflow_path) as f:
        return yaml.safe_load(f)


def extract_path_filters(workflow: Dict) -> List[str]:
    """Extract path filters from workflow configuration."""
    paths = []

    # Check pull_request paths (YAML parses 'on:' as True)
    on_key = "on" if "on" in workflow else True
    if on_key in workflow and isinstance(workflow[on_key], dict):
        if "pull_request" in workflow[on_key]:
            pr_config = workflow[on_key]["pull_request"]
            if isinstance(pr_config, dict) and "paths" in pr_config:
                paths = pr_config["paths"]

    return paths


def validate_critical_paths(filters: List[str]) -> List[str]:
    """Validate that critical paths are included in filters."""
    errors = []

    # Critical paths that must be included
    critical_patterns = [
        "src/**",
        "tests/**",
        "config/**",  # Runtime configuration (presets, recipes)
        "requirements*.txt",
        ".github/workflows/**",
    ]

    for pattern in critical_patterns:
        if pattern not in filters:
            errors.append(f"Missing critical path pattern: {pattern}")

    return errors


def check_filter_coverage(filters: List[str]) -> Dict[str, bool]:
    """Check coverage of different file types."""
    coverage = {
        "source_code": "src/**" in filters,
        "tests": "tests/**" in filters,
        "runtime_config": "config/**" in filters,
        "requirements": any("requirements" in p for p in filters),
        "workflows": ".github/workflows/**" in filters,
        "build_config": any(p in ["pyproject.toml", "setup.py"] for p in filters),
    }
    return coverage


def main():
    """Run validation."""
    repo_root = Path(__file__).resolve().parents[2]
    workflows_dir = repo_root / ".github" / "workflows"

    # Check build.yml (primary CI workflow)
    build_yml = workflows_dir / "build.yml"
    if not build_yml.exists():
        print(f"❌ ERROR: {build_yml} not found")
        return 1

    print(f"📋 Validating CI path filters in {build_yml.relative_to(repo_root)}")
    print()

    workflow = load_workflow(build_yml)
    filters = extract_path_filters(workflow)

    if not filters:
        print("⚠️  WARNING: No path filters found in pull_request trigger")
        print("   This means the workflow will run on ALL file changes")
        print("   This is acceptable but may not be optimal")
        return 0

    print(f"✓ Found {len(filters)} path filter patterns")
    print()

    # Validate critical paths
    errors = validate_critical_paths(filters)
    if errors:
        print("❌ ERRORS:")
        for error in errors:
            print(f"   {error}")
        return 1

    print("✓ All critical paths included")
    print()

    # Check coverage
    coverage = check_filter_coverage(filters)
    print("📊 Filter Coverage:")
    for area, covered in coverage.items():
        status = "✓" if covered else "✗"
        print(f"   {status} {area}")

    if not all(coverage.values()):
        print()
        print("⚠️  WARNING: Some areas not covered by filters")
        return 1

    print()
    print("✅ All validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
