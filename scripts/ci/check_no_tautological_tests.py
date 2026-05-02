#!/usr/bin/env python3
"""Fail CI when test files contain tautological assertions.

Bans `assert True`, `assert 1` (and similar always-true literals) when they
appear as direct statements inside test files. Such assertions count as passing
tests but verify nothing — they were a real-world quality smell in this repo
(see docs/testing/test_coverage_improvement_plan.md and the call-out in
test_raw_loader.py).

Implementation notes:
- Walks the AST so we ignore `assert True` inside string literals (those occur
  as fixture inputs in tests/test_retrofit_test_markers.py and similar).
- Detects the equally tautological forms: `assert <truthy literal>`,
  `assert not False`, `assert not 0`.
- Allows `assert True, "message"` only if a tag comment "tautology-ok" appears
  on the same line — escape hatch for intentional smoke checks; use sparingly.

Usage:
    python scripts/ci/check_no_tautological_tests.py [tests/]
    # Exits 0 if no offenders, 1 otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

ESCAPE_HATCH = "tautology-ok"


def _is_truthy_literal(node: ast.expr) -> bool:
    """Return True if `node` is a literal that is always truthy."""
    if isinstance(node, ast.Constant):
        return bool(node.value) and node.value is not None
    # `assert not False` / `assert not 0` are tautologies too.
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        if isinstance(node.operand, ast.Constant):
            return not node.operand.value
    return False


def find_tautological_asserts(path: Path) -> List[Tuple[int, str]]:
    """Return [(line, source-snippet), ...] for every offender in `path`."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        # Don't block CI on a syntax error here; other lints will catch that.
        return []

    source_lines = source.splitlines()
    offenders: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        if not _is_truthy_literal(node.test):
            continue
        line = node.lineno
        snippet = source_lines[line - 1] if 0 < line <= len(source_lines) else ""
        if ESCAPE_HATCH in snippet:
            continue
        offenders.append((line, snippet.strip()))
    return offenders


def iter_test_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file():
            if root.name.startswith("test_") and root.suffix == ".py":
                yield root
            continue
        for path in root.rglob("test_*.py"):
            yield path


def main(argv: List[str]) -> int:
    roots = [Path(arg) for arg in argv[1:]] or [Path("tests")]
    bad_files: List[Tuple[Path, List[Tuple[int, str]]]] = []
    for path in iter_test_files(roots):
        offenders = find_tautological_asserts(path)
        if offenders:
            bad_files.append((path, offenders))

    if not bad_files:
        return 0

    print("Tautological assertions found in test files:", file=sys.stderr)
    for path, offenders in bad_files:
        for line, snippet in offenders:
            print(f"  {path}:{line}: {snippet}", file=sys.stderr)
    print(
        f"\n{sum(len(o) for _, o in bad_files)} offender(s) in "
        f"{len(bad_files)} file(s).\n"
        "These tests pass without verifying anything. Replace them with a real "
        "assertion, or — if a placeholder is genuinely intentional — add the "
        f"comment '# {ESCAPE_HATCH}' on the same line.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
