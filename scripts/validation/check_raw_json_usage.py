#!/usr/bin/env python3
"""Enforce JSON serialization guardrails for source modules.

Policy:
- `json.dump()` / `json.dumps()` calls are disallowed by default in `src/`.
- Modules explicitly listed in the approved allowlist are exempt.

This prevents new ad-hoc serialization paths from bypassing canonical
normalization helpers.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_APPROVED_FILE = Path("policy/json_raw_approved_modules.txt")
DEFAULT_ROOTS = ("src",)
JSON_CALLS = frozenset({"dump", "dumps"})


@dataclass(frozen=True)
class JsonCallSite:
    path: Path
    lineno: int
    col_offset: int
    call_name: str


class JsonCallVisitor(ast.NodeVisitor):
    """Collect direct json.dump/json.dumps call-sites from an AST."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in JSON_CALLS:
            owner = func.value
            if isinstance(owner, ast.Name) and owner.id == "json":
                self.calls.append((node.lineno, node.col_offset, func.attr))
        self.generic_visit(node)


def _load_approved_patterns(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Approved-modules file not found: {path}")

    patterns: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def _iter_python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _is_approved(path: Path, approved_patterns: list[str]) -> bool:
    posix_path = path.as_posix()
    return any(fnmatch.fnmatch(posix_path, pattern) for pattern in approved_patterns)


def _scan_file(path: Path) -> list[tuple[int, int, str]]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise RuntimeError(f"Failed to parse {path}: {exc}") from exc

    visitor = JsonCallVisitor()
    visitor.visit(tree)
    return visitor.calls


def check_raw_json_usage(roots: list[Path], approved_file: Path) -> list[JsonCallSite]:
    approved_patterns = _load_approved_patterns(approved_file)
    violations: list[JsonCallSite] = []

    for root in roots:
        for py_file in _iter_python_files(root):
            calls = _scan_file(py_file)
            if not calls:
                continue
            if _is_approved(py_file, approved_patterns):
                continue
            for lineno, col_offset, call_name in calls:
                violations.append(
                    JsonCallSite(
                        path=py_file,
                        lineno=lineno,
                        col_offset=col_offset,
                        call_name=call_name,
                    )
                )

    return violations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail on raw json.dump/json.dumps outside approved modules.")
    parser.add_argument(
        "--approved-file",
        type=Path,
        default=DEFAULT_APPROVED_FILE,
        help=f"Path to approved module list (default: {DEFAULT_APPROVED_FILE})",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=list(DEFAULT_ROOTS),
        help="Source roots to scan (default: src)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    roots = [Path(root) for root in args.roots]

    try:
        violations = check_raw_json_usage(roots=roots, approved_file=args.approved_file)
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not violations:
        print("JSON serialization guardrail passed: no raw json.dump/json.dumps calls outside approved modules.")
        return 0

    print("ERROR: raw json.dump/json.dumps usage found outside approved modules:", file=sys.stderr)
    for violation in sorted(violations, key=lambda v: (v.path.as_posix(), v.lineno, v.col_offset)):
        print(
            f"  - {violation.path.as_posix()}:{violation.lineno}:{violation.col_offset + 1} json.{violation.call_name}",
            file=sys.stderr,
        )
    print(
        "Remediation: route writes through transformation_portal.ingest.canonical_json.dump_json/dumps_json "
        "or add module to the approved allowlist with justification.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
