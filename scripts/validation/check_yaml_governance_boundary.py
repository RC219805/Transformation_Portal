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
import ast
import re
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (PROJECT_ROOT / "src",)
SAFE_LOAD_PATTERN = re.compile(r"\byaml\.safe_load\s*\(")
AUTHORITY_MARKER = "YAML_GOVERNANCE_AUTHORITY:"
EXEMPT_MARKER = "YAML_GOVERNANCE_EXEMPT:"
AUTHORITY_FILE_PATHS = frozenset({(PROJECT_ROOT / "src" / "transformation_portal" / "compliance" / "licensing.py").resolve()})


def iter_python_files(roots: Iterable[Path]) -> list[Path]:
    """Return Python source files under the supplied roots."""
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix == ".py":
                files.append(root)
            continue
        files.extend(path for path in root.rglob("*.py") if path.is_file())
    return sorted(set(files))


class _YamlSafeLoadDetector(ast.NodeVisitor):
    """Detect raw yaml.safe_load usage, including aliased imports."""

    def __init__(self) -> None:
        self.found = False
        self.yaml_module_names = {"yaml"}
        self.safe_load_names = set[str]()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "yaml":
                self.yaml_module_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module != "yaml":
            self.generic_visit(node)
            return

        for alias in node.names:
            if alias.name == "safe_load":
                self.safe_load_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in self.safe_load_names:
            self.found = True
            return

        if (
            isinstance(func, ast.Attribute)
            and func.attr == "safe_load"
            and isinstance(func.value, ast.Name)
            and func.value.id in self.yaml_module_names
        ):
            self.found = True
            return

        self.generic_visit(node)


def file_has_yaml_safe_load(path: Path) -> bool:
    """Return True when a file contains a raw yaml.safe_load call."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return SAFE_LOAD_PATTERN.search(text) is not None or re.search(r"\bsafe_load\s*\(", text) is not None

    detector = _YamlSafeLoadDetector()
    detector.visit(tree)
    return detector.found


def _read_text(path: Path) -> str:
    """Read a source file as UTF-8, replacing undecodable bytes when needed."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def file_has_authority_marker(path: Path) -> bool:
    """Return True when a file declares YAML governance authority."""
    return AUTHORITY_MARKER in _read_text(path)


def file_has_exempt_marker(path: Path) -> bool:
    """Return True when a file is explicitly exempt from preset governance loading."""
    return EXEMPT_MARKER in _read_text(path)


def file_is_yaml_governance_authority(path: Path) -> bool:
    """Return True when a file is an approved shared authority module."""
    return path.resolve() in AUTHORITY_FILE_PATHS and file_has_authority_marker(path)


def find_violations(roots: Iterable[Path] | None = None) -> list[str]:
    """Return governance-boundary violations for Python runtime files."""
    scan_roots = tuple(roots) if roots is not None else SOURCE_ROOTS
    violations: list[str] = []
    for path in iter_python_files(scan_roots):
        if not file_has_yaml_safe_load(path):
            continue
        if file_has_exempt_marker(path) or file_is_yaml_governance_authority(path):
            continue
        if file_has_authority_marker(path):
            violations.append(
                f"{path}: {AUTHORITY_MARKER} is reserved for approved shared preset loaders; "
                f"use {EXEMPT_MARKER} for non-preset/internal YAML loaders"
            )
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
