#!/usr/bin/env python3
"""
Verify banned dependencies are not installed or referenced.

Security policy enforcement for supply-chain security.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Set

import tomli


# Banned packages (security/license/quality concerns)
BANNED_PACKAGES = {
    "realesrgan": "Unmaintained, use local implementation",
    "gfpgan": "Unstable dependencies, use alternative",
}


def _is_hard_block_constraint(raw_line: str) -> bool:
    """
    Return True if a constraints.txt line intentionally hard-blocks a package.

    Expected form: >=9999 or >=9999.0.0
    """
    s = raw_line.split("#", 1)[0].strip()
    return bool(re.search(r">=\s*9999(\.0\.0)?\b", s))


def load_pyproject_dependencies() -> Set[str]:
    """Extract all dependencies from pyproject.toml (project deps + optional deps)."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        return set()

    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)

    deps: Set[str] = set()

    project = data.get("project", {})
    for dep in project.get("dependencies", []) or []:
        pkg = str(dep).split("[")[0].split(">=")[0].split("==")[0].strip()
        if pkg:
            deps.add(pkg)

    opt = project.get("optional-dependencies", {}) or {}
    for group_deps in opt.values():
        for dep in group_deps or []:
            pkg = str(dep).split("[")[0].split(">=")[0].split("==")[0].strip()
            if pkg:
                deps.add(pkg)

    return deps


def check_requirements_files() -> List[str]:
    """Check requirements/*.txt files for banned packages (constraints allowed only as hard-block pins)."""
    violations: List[str] = []
    req_dir = Path("requirements")

    if not req_dir.exists():
        return violations

    banned_lower = {b.lower(): BANNED_PACKAGES[b] for b in BANNED_PACKAGES}

    for req_file in req_dir.glob("*.txt"):
        is_constraints = (req_file.name == "constraints.txt")

        with open(req_file, encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, 1):
                line = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue

                pkg = line.split("[")[0].split(">=")[0].split("==")[0].strip()
                reason = banned_lower.get(pkg.lower())
                if reason is None:
                    continue

                # Allow intentional hard-block pins in constraints.txt
                if is_constraints and _is_hard_block_constraint(raw_line):
                    continue

                violations.append(f"{req_file}:{line_num} - {pkg} ({reason})")

    return violations


def main() -> int:
    print("🔍 Checking for banned dependencies...\n")

    banned_lower = {b.lower(): BANNED_PACKAGES[b] for b in BANNED_PACKAGES}

    pyproject_deps = load_pyproject_dependencies()
    pyproject_violations: List[str] = []
    for dep in pyproject_deps:
        reason = banned_lower.get(dep.lower())
        if reason is not None:
            pyproject_violations.append(f"pyproject.toml - {dep} ({reason})")

    req_violations = check_requirements_files()
    all_violations = pyproject_violations + req_violations

    if not all_violations:
        print("✅ No banned dependencies found")
        return 0

    print("❌ BANNED DEPENDENCIES DETECTED:\n")
    for violation in all_violations:
        print(f"  {violation}")

    print("\n💡 Remove these packages and use approved alternatives")
    print("   See docs/security/dependency-policy.md for guidance\n")

    return 1 if "--strict" in sys.argv else 0


if __name__ == "__main__":
    sys.exit(main())
