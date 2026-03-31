#!/usr/bin/env python3
"""Enforce the YAML governance boundary for runtime source files.

Policy:
- Preset-like, user-facing governance YAML must be loaded via
  ``load_and_validate_preset()``.
- Raw ``yaml.safe_load()`` remains allowed only in explicitly marked files that
  are non-preset/internal loaders or in the shared compliance loader authority.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (PROJECT_ROOT / "src",)
SAFE_LOAD_PATTERN = re.compile(r"\byaml\.safe_load\s*\(")
AUTHORITY_MARKER = "YAML_GOVERNANCE_AUTHORITY:"
EXEMPT_MARKER = "YAML_GOVERNANCE_EXEMPT:"


def iter_python_files(roots: Iterable[Path]) -> list[Path]:
    """Return Python source files under the supplied roots."""
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(path for path in root.rglob("*.py") if path.is_file())
    return sorted(set(files))


def file_has_yaml_safe_load(path: Path) -> bool:
    """Return True when a file contains a raw yaml.safe_load call."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    return SAFE_LOAD_PATTERN.search(text) is not None


def file_has_boundary_marker(path: Path) -> bool:
    """Return True when a file is explicitly marked as authority or exempt."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    return AUTHORITY_MARKER in text or EXEMPT_MARKER in text


def find_violations(roots: Iterable[Path] | None = None) -> list[str]:
    """Return governance-boundary violations for Python runtime files."""
    scan_roots = tuple(roots) if roots is not None else SOURCE_ROOTS
    violations: list[str] = []
    for path in iter_python_files(scan_roots):
        if not file_has_yaml_safe_load(path):
            continue
        if file_has_boundary_marker(path):
            continue
        violations.append(
            f"{path}: raw yaml.safe_load() requires either {AUTHORITY_MARKER} or {EXEMPT_MARKER}"
        )
    return violations


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Check YAML governance boundary markers for raw yaml.safe_load() calls")
    parser.add_argument("paths", nargs="*", help="Optional source roots/files to scan instead of src/")
    args = parser.parse_args()

    roots = tuple(Path(path) for path in args.paths) if args.paths else SOURCE_ROOTS
    violations = find_violations(roots)
    if not violations:
        print("✅ YAML governance boundary markers are valid")
        return 0

    print("❌ YAML governance boundary violations detected:")
    for violation in violations:
        print(f"  - {violation}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
