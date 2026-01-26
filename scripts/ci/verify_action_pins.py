#!/usr/bin/env python3
"""
Verify GitHub Actions are pinned to commit SHAs for security.

Enforcement levels:
- FAIL: Security-critical workflows (release, security gates, dependency-submission)
- WARN: All other workflows (migration in progress)
"""
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Security-critical workflows that MUST use pinned SHAs
CRITICAL_WORKFLOWS = {
    "dependency-submission.yml",
    "codeql.yml",
    "submit-pypi.yml",
}

# Pattern to match action uses
ACTION_PATTERN = re.compile(r'uses:\s+([^@\s]+)@([^\s]+)')


def check_workflow(workflow_path: Path) -> List[Tuple[str, int, str, bool]]:
    """
    Check a workflow file for unpinned actions.

    Returns list of (action, line_num, version, is_critical)
    """
    issues = []
    is_critical = workflow_path.name in CRITICAL_WORKFLOWS

    with open(workflow_path) as f:
        for line_num, line in enumerate(f, 1):
            match = ACTION_PATTERN.search(line)
            if match:
                action, version = match.groups()
                # Allow official GitHub actions with @v tags
                if action.startswith("actions/") and version.startswith("v"):
                    continue
                # Check if version is a commit SHA (40 hex chars)
                if not re.match(r'^[0-9a-f]{40}$', version):
                    issues.append((action, line_num, version, is_critical))

    return issues


def main():
    workflows_dir = Path(".github/workflows")
    if not workflows_dir.exists():
        print("No .github/workflows directory found")
        return 0

    all_issues = []
    critical_issues = []

    for workflow_file in workflows_dir.glob("*.yml"):
        issues = check_workflow(workflow_file)
        if issues:
            all_issues.extend([(workflow_file.name, *issue) for issue in issues])
            critical_issues.extend([
                (workflow_file.name, *issue)
                for issue in issues
                if issue[3]  # is_critical
            ])

    if not all_issues:
        print("✅ All GitHub Actions are properly pinned")
        return 0

    # Report critical issues (hard fail)
    if critical_issues:
        print("❌ CRITICAL: Unpinned actions in security-critical workflows:\n")
        for workflow, action, line, version, _ in critical_issues:
            print(f"  {workflow}:{line} - {action}@{version}")
        print(f"\n💡 Pin to commit SHA: {action}@<40-char-sha>\n")
        return 1

    # Report non-critical issues (warning only)
    print("⚠️  WARNING: Unpinned actions in non-critical workflows:\n")
    for workflow, action, line, version, _ in all_issues:
        print(f"  {workflow}:{line} - {action}@{version}")
    print("\n💡 These will become hard failures in Phase 2")
    print("   Gradually migrate to pinned SHAs\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
