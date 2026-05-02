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
import tokenize
from io import StringIO
from pathlib import Path
from typing import Iterable, List, Set, Tuple

ESCAPE_HATCH = "tautology-ok"


def _is_truthy_literal(node: ast.expr) -> bool:
    """Return True if `node` is a literal that is always truthy.

    Covers:
    - ``True``, non-zero numbers, non-empty strings/bytes (``ast.Constant``)
    - non-empty container literals: ``[1]``, ``(1,)``, ``{'k': 'v'}``, ``{1}``
    - ``not <falsy-constant>`` (e.g. ``assert not False``)
    """
    if isinstance(node, ast.Constant):
        return bool(node.value) and node.value is not None
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        # A non-empty list/tuple/set literal is always truthy. An empty one is
        # falsy and therefore not a tautology, so we leave it alone.
        return len(node.elts) > 0
    if isinstance(node, ast.Dict):
        return len(node.keys) > 0
    # `assert not False` / `assert not 0` are tautologies too.
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        if isinstance(node.operand, ast.Constant):
            return not node.operand.value
    return False


def _comment_lines_with_escape_hatch(source: str) -> Set[int]:
    """Return line numbers whose source contains a real ``# tautology-ok`` comment.

    Uses tokenize so the tag is only recognized inside an actual comment —
    a string like ``assert True, "tautology-ok"`` does NOT bypass the lint.
    """
    lines: Set[int] = set()
    try:
        tokens = list(tokenize.generate_tokens(StringIO(source).readline))
    except tokenize.TokenizeError:
        return lines
    for tok in tokens:
        if tok.type == tokenize.COMMENT and ESCAPE_HATCH in tok.string:
            lines.add(tok.start[0])
    return lines


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
    escape_lines = _comment_lines_with_escape_hatch(source)
    offenders: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        if not _is_truthy_literal(node.test):
            continue
        line = node.lineno
        if line in escape_lines:
            continue
        snippet = source_lines[line - 1] if 0 < line <= len(source_lines) else ""
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
