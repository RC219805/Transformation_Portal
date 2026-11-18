#!/usr/bin/env python3
"""Verify banned dependencies are not installed or referenced.

Security policy enforcement for supply-chain security.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


BANNED_REGISTRY_PATH = Path("scripts/security/banned_dependencies.json")
CONSTRAINTS_PATH = Path("requirements/constraints.txt")
DOC_GUIDANCE_PATH = "docs/architecture/ADR-032-dependency-pinning-strategy.md"


_EGG_FRAGMENT_RE = re.compile(r"[#&]egg=([A-Za-z0-9][A-Za-z0-9._-]*)")


def canonicalize_name(name: str) -> str:
    """PEP 503 normalization: case-insensitive, runs of ``-``, ``_`` and ``.`` are equivalent."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _normalize_package_name(raw: str) -> str:
    """Extract the package name from a dependency specifier.

    Handles PEP 508 specifiers, ``name @ url`` direct references and pip URL
    requirements that name the project with a ``#egg=`` fragment. Comments
    follow pip's rule (``#`` at line start or after whitespace), so URL
    fragments are not mistaken for comments.
    """
    text = re.sub(r"(^|\s)#.*$", "", raw).strip()
    if not text or text.startswith(("-r", "-c", "--requirement", "--constraint")):
        return ""
    egg = _EGG_FRAGMENT_RE.search(text)
    if egg is not None:
        return egg.group(1)
    if text.startswith(("-e", "--editable")):
        return ""
    text = text.split(";", 1)[0].strip()
    text = re.split(r"[<>=!~@\s\[(]", text, maxsplit=1)[0].strip()
    if "/" in text or ":" in text:
        # Bare URL or path without an egg fragment: no project name to check.
        return ""
    return text


def _is_hard_block_constraint(raw_line: str) -> bool:
    """Return True if a constraints.txt line intentionally hard-blocks a package."""
    s = raw_line.split("#", 1)[0].strip()
    return bool(re.search(r">=\s*9999(\.0\.0)?\b", s))


def load_banned_registry() -> Dict[str, Dict[str, str]]:
    """Load banned dependency metadata from canonical JSON registry."""
    if not BANNED_REGISTRY_PATH.exists():
        raise ValueError(f"Missing banned dependency registry: {BANNED_REGISTRY_PATH}")

    data = json.loads(BANNED_REGISTRY_PATH.read_text(encoding="utf-8"))
    entries = data.get("packages", [])
    if not isinstance(entries, list):
        raise ValueError(f"Invalid registry format in {BANNED_REGISTRY_PATH}: expected 'packages' list")

    registry: Dict[str, Dict[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        reason = str(entry.get("reason", "")).strip()
        migration = str(entry.get("migration", "")).strip()
        if not name or not reason:
            raise ValueError(
                f"Invalid banned package entry in {BANNED_REGISTRY_PATH}: each entry needs non-empty name and reason"
            )
        registry[canonicalize_name(name)] = {
            "name": name,
            "reason": reason,
            "migration": migration,
        }

    if not registry:
        raise ValueError(f"Banned registry is empty: {BANNED_REGISTRY_PATH}")
    return registry


def load_pyproject_dependencies() -> Set[str]:
    """Extract all dependencies from pyproject.toml (project deps + optional deps)."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        return set()

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    deps: Set[str] = set()

    project = data.get("project", {})
    for dep in project.get("dependencies", []) or []:
        pkg = _normalize_package_name(str(dep))
        if pkg:
            deps.add(pkg)

    opt = project.get("optional-dependencies", {}) or {}
    for group_deps in opt.values():
        for dep in group_deps or []:
            pkg = _normalize_package_name(str(dep))
            if pkg:
                deps.add(pkg)

    return deps


def _requirement_files() -> List[Path]:
    """Return every requirement source/lock the repository can install from.

    Root ``requirements*.txt`` files (what CI installs; they do not apply
    ``requirements/constraints.txt``) plus ``requirements/**/*.txt`` and
    ``requirements/**/*.in`` (lock sources and nested lock directories).
    """
    files: Set[Path] = {path for path in Path(".").glob("requirements*.txt") if path.is_file()}
    req_dir = Path("requirements")
    if req_dir.is_dir():
        for pattern in ("*.txt", "*.in"):
            files.update(path for path in req_dir.rglob(pattern) if path.is_file())
    return sorted(files)


def check_requirements_files(banned_registry: Dict[str, Dict[str, str]]) -> List[str]:
    """Check requirement files for banned packages (constraints allowed only as hard-block pins)."""
    violations: List[str] = []

    for req_file in _requirement_files():
        is_constraints = req_file == CONSTRAINTS_PATH

        with open(req_file, encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, 1):
                pkg = _normalize_package_name(raw_line)
                if not pkg:
                    continue

                meta = banned_registry.get(canonicalize_name(pkg))
                if meta is None:
                    continue

                # Allow intentional hard-block pins in constraints.txt
                if is_constraints and _is_hard_block_constraint(raw_line):
                    continue

                violations.append(f"{req_file}:{line_num} - {pkg} ({meta['reason']})")

    return violations


def check_constraints_sync(banned_registry: Dict[str, Dict[str, str]]) -> List[str]:
    """Ensure the canonical banned registry and constraints hard-blocks stay in sync."""
    violations: List[str] = []
    if not CONSTRAINTS_PATH.exists():
        return [f"Missing constraints file: {CONSTRAINTS_PATH}"]

    hard_blocked: Set[str] = set()
    with open(CONSTRAINTS_PATH, encoding="utf-8") as f:
        for raw_line in f:
            pkg = _normalize_package_name(raw_line)
            if not pkg:
                continue
            if _is_hard_block_constraint(raw_line):
                hard_blocked.add(canonicalize_name(pkg))

    for banned in sorted(banned_registry):
        if banned not in hard_blocked:
            name = banned_registry[banned]["name"]
            violations.append(f"{CONSTRAINTS_PATH} - missing hard-block pin for banned package '{name}'")

    for hard_blocked_pkg in sorted(hard_blocked):
        if hard_blocked_pkg not in banned_registry:
            violations.append(
                f"{CONSTRAINTS_PATH} - hard-blocked package '{hard_blocked_pkg}' missing from {BANNED_REGISTRY_PATH}"
            )

    return violations


def main() -> int:
    print("🔍 Checking for banned dependencies...\n")

    try:
        banned_registry = load_banned_registry()
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1

    pyproject_deps = load_pyproject_dependencies()
    pyproject_violations: List[str] = []
    for dep in sorted(pyproject_deps):
        meta = banned_registry.get(canonicalize_name(dep))
        if meta is not None:
            pyproject_violations.append(f"pyproject.toml - {dep} ({meta['reason']})")

    req_violations = check_requirements_files(banned_registry)
    sync_violations = check_constraints_sync(banned_registry)
    all_violations = pyproject_violations + req_violations + sync_violations

    if not all_violations:
        print(f"✅ No banned dependencies found ({len(banned_registry)} policy entries validated)")
        return 0

    print("❌ BANNED DEPENDENCY POLICY VIOLATIONS DETECTED:\n")
    for violation in all_violations:
        print(f"  {violation}")

    print("\n💡 Remove these packages and use approved alternatives")
    print(f"   See {DOC_GUIDANCE_PATH} for guidance\n")

    return 1 if "--strict" in sys.argv else 0


if __name__ == "__main__":
    sys.exit(main())
