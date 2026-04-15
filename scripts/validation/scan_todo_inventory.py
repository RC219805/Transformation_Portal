#!/usr/bin/env python3
"""Scan and validate TODO inventory across the repository.

Policy:
- TODO patterns are tracked: # TODO:, # FIXME:, # HACK:, # XXX:, NotImplementedError
- Documented false positives are excluded (e.g., security regex patterns)
- New TODOs should include governance references for traceability:
  - # TODO(Phase X):  - phase-gated work
  - # TODO(ADR-XXX): - ADR-linked work
  - # TODO(#1234):   - issue-linked work
  - # TODO(@owner):  - owner-assigned work

Usage:
    # Default human-readable scan
    python scripts/validation/scan_todo_inventory.py

    # JSON output for CI consumption
    python scripts/validation/scan_todo_inventory.py --json

    # Governance enforcement mode (exit 1 on violations)
    python scripts/validation/scan_todo_inventory.py --check-governance
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directories to scan for Python files
PYTHON_SCAN_ROOTS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "tools",
)

# Directory to scan for JS/TS files
JS_TS_SCAN_ROOT = PROJECT_ROOT / "web"

# TODO patterns to detect
TODO_PATTERNS = (
    re.compile(r"#\s*(TODO|FIXME|HACK|XXX)\s*:?\s*(.*)$", re.IGNORECASE),
)

# JS/TS TODO patterns (single-line // comments and block /** */ comments)
JS_TODO_PATTERNS = (
    re.compile(r"//\s*(TODO|FIXME|HACK|XXX)\s*:?\s*(.*)$", re.IGNORECASE),
    re.compile(r"\*\s*(TODO|FIXME|HACK|XXX)\s*:?\s*(.*)$", re.IGNORECASE),
)

# Governance reference patterns - TODOs with proper tracking
GOVERNANCE_REF_PATTERNS = (
    re.compile(r"\(Phase\s*\d+[A-Za-z]?\)", re.IGNORECASE),  # (Phase 3), (Phase 4A)
    re.compile(r"\(ADR-\d+\)", re.IGNORECASE),              # (ADR-044)
    re.compile(r"\(#\d+\)"),                                 # (#1234) - issue ref
    re.compile(r"\(@[\w-]+\)"),                             # (@specialist) - owner
    re.compile(r"\([A-Za-z]+_INVENTORY\.md"),               # (TODO_INVENTORY.md §X)
)

# Excluded patterns (false positives in security code)
EXCLUDED_PATTERNS = (
    re.compile(r"TODO_REPLACE", re.IGNORECASE),  # Security scanner pattern
)

# Self-exclude: this scanner file contains TODO examples in docs
SELF_EXCLUDE_FILE = "scripts/validation/scan_todo_inventory.py"

# Excluded file paths (relative to project root)
EXCLUDED_PATH_PATTERNS = (
    "docs/",                    # Historical documentation markers
    "__pycache__/",
    ".git/",
    "node_modules/",
    ".venv/",
    ".runtime/",
    "*.egg-info/",
)


class TodoType(str, Enum):
    """Types of TODO markers."""

    TODO = "TODO"
    FIXME = "FIXME"
    HACK = "HACK"
    XXX = "XXX"
    NOT_IMPLEMENTED = "NotImplementedError"


@dataclass(frozen=True)
class TodoItem:
    """A single TODO item found in the codebase."""

    path: Path
    lineno: int
    todo_type: TodoType
    message: str
    has_governance_ref: bool
    governance_refs: tuple[str, ...]
    col_offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "path": self.path.as_posix(),
            "lineno": self.lineno,
            "col_offset": self.col_offset,
            "type": self.todo_type.value,
            "message": self.message,
            "has_governance_ref": self.has_governance_ref,
            "governance_refs": list(self.governance_refs),
        }


@dataclass
class ScanResult:
    """Result of a TODO scan."""

    items: list[TodoItem] = field(default_factory=list)
    files_scanned: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return len(self.items)

    @property
    def ungoverned_count(self) -> int:
        return sum(1 for item in self.items if not item.has_governance_ref)

    @property
    def by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.todo_type.value] = counts.get(item.todo_type.value, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "summary": {
                "total": self.total_count,
                "ungoverned": self.ungoverned_count,
                "files_scanned": self.files_scanned,
                "by_type": self.by_type,
            },
            "items": [item.to_dict() for item in self.items],
            "errors": self.errors,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Detection Logic
# ─────────────────────────────────────────────────────────────────────────────


def _is_excluded_path(path: Path) -> bool:
    """Check if a path should be excluded from scanning."""
    posix_path = path.as_posix()

    # Self-exclude: this scanner file contains TODO examples in documentation
    if SELF_EXCLUDE_FILE in posix_path:
        return True

    for pattern in EXCLUDED_PATH_PATTERNS:
        if pattern in posix_path:
            return True
    return False


def _is_excluded_content(line: str) -> bool:
    """Check if a line contains an excluded pattern (false positive)."""
    return any(pattern.search(line) for pattern in EXCLUDED_PATTERNS)


def _extract_governance_refs(text: str) -> tuple[str, ...]:
    """Extract governance references from a TODO message."""
    refs: list[str] = []
    for pattern in GOVERNANCE_REF_PATTERNS:
        matches = pattern.findall(text)
        refs.extend(matches)
    return tuple(refs)


def _has_governance_ref(text: str) -> bool:
    """Check if a TODO has proper governance tracking."""
    return bool(_extract_governance_refs(text))


def _iter_python_files(roots: tuple[Path, ...]) -> list[Path]:
    """Yield Python source files under the given roots."""
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if not _is_excluded_path(path):
                files.append(path)
    return sorted(files)


def _iter_js_ts_files(root: Path) -> list[Path]:
    """Yield JS/TS source files under the given root."""
    if not root.exists():
        return []
    files: list[Path] = []
    for ext in ("*.js", "*.ts", "*.jsx", "*.tsx", "*.mjs"):
        for path in root.rglob(ext):
            if not _is_excluded_path(path):
                files.append(path)
    return sorted(files)


def _read_file_safe(path: Path) -> str | None:
    """Read a file safely, returning None on errors."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None


class NotImplementedVisitor(ast.NodeVisitor):
    """AST visitor to find NotImplementedError raises with context."""

    def __init__(self) -> None:
        self.items: list[tuple[int, int, str]] = []

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc is None:
            self.generic_visit(node)
            return

        exc = node.exc
        is_not_implemented = False
        message = ""

        # Handle: raise NotImplementedError
        if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
            is_not_implemented = True

        # Handle: raise NotImplementedError("message")
        if isinstance(exc, ast.Call):
            if isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError":
                is_not_implemented = True
                if exc.args and isinstance(exc.args[0], ast.Constant):
                    message = str(exc.args[0].value)

        if is_not_implemented:
            self.items.append((node.lineno, node.col_offset, message))

        self.generic_visit(node)


def _scan_python_file(path: Path) -> list[TodoItem]:
    """Scan a single Python file for TODO patterns."""
    source = _read_file_safe(path)
    if source is None:
        return []

    items: list[TodoItem] = []
    lines = source.splitlines()

    # Scan for comment-based TODOs
    for lineno, line in enumerate(lines, start=1):
        if _is_excluded_content(line):
            continue

        for pattern in TODO_PATTERNS:
            match = pattern.search(line)
            if match:
                todo_type_str = match.group(1).upper()
                message = match.group(2).strip()

                # Include context for multi-line messages (next line if continuation)
                if lineno < len(lines):
                    next_line = lines[lineno].strip()
                    if next_line.startswith("#") and not any(
                        p.search(next_line) for p in TODO_PATTERNS
                    ):
                        continuation = next_line.lstrip("#").strip()
                        if continuation:
                            message = f"{message} {continuation}"

                governance_refs = _extract_governance_refs(line + " " + message)
                items.append(
                    TodoItem(
                        path=path.relative_to(PROJECT_ROOT),
                        lineno=lineno,
                        todo_type=TodoType(todo_type_str),
                        message=message,
                        has_governance_ref=bool(governance_refs),
                        governance_refs=governance_refs,
                        col_offset=match.start(),
                    )
                )
                break  # Only count each line once

    # Scan for NotImplementedError
    try:
        tree = ast.parse(source, filename=str(path))
        visitor = NotImplementedVisitor()
        visitor.visit(tree)
        for lineno, col_offset, message in visitor.items:
            governance_refs = _extract_governance_refs(message)
            items.append(
                TodoItem(
                    path=path.relative_to(PROJECT_ROOT),
                    lineno=lineno,
                    todo_type=TodoType.NOT_IMPLEMENTED,
                    message=message or "(no message)",
                    has_governance_ref=bool(governance_refs),
                    governance_refs=governance_refs,
                    col_offset=col_offset,
                )
            )
    except SyntaxError:
        pass  # Skip files that can't be parsed

    return items


def _scan_js_ts_file(path: Path) -> list[TodoItem]:
    """Scan a single JS/TS file for TODO patterns."""
    source = _read_file_safe(path)
    if source is None:
        return []

    items: list[TodoItem] = []
    lines = source.splitlines()

    for lineno, line in enumerate(lines, start=1):
        if _is_excluded_content(line):
            continue

        for pattern in JS_TODO_PATTERNS:
            match = pattern.search(line)
            if match:
                todo_type_str = match.group(1).upper()
                message = match.group(2).strip()
                governance_refs = _extract_governance_refs(line + " " + message)
                items.append(
                    TodoItem(
                        path=path.relative_to(PROJECT_ROOT),
                        lineno=lineno,
                        todo_type=TodoType(todo_type_str),
                        message=message,
                        has_governance_ref=bool(governance_refs),
                        governance_refs=governance_refs,
                        col_offset=match.start(),
                    )
                )
                break

    return items


def scan_repository() -> ScanResult:
    """Scan the entire repository for TODO patterns."""
    result = ScanResult()

    # Scan Python files
    python_files = _iter_python_files(PYTHON_SCAN_ROOTS)
    for path in python_files:
        try:
            items = _scan_python_file(path)
            result.items.extend(items)
            result.files_scanned += 1
        except Exception as e:
            result.errors.append(f"{path}: {e}")

    # Scan JS/TS files
    js_ts_files = _iter_js_ts_files(JS_TS_SCAN_ROOT)
    for path in js_ts_files:
        try:
            items = _scan_js_ts_file(path)
            result.items.extend(items)
            result.files_scanned += 1
        except Exception as e:
            result.errors.append(f"{path}: {e}")

    # Sort by path, then line number
    result.items.sort(key=lambda x: (x.path.as_posix(), x.lineno))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Output Formatting
# ─────────────────────────────────────────────────────────────────────────────


def _format_human_readable(result: ScanResult, governance_mode: bool) -> str:
    """Format scan result for human consumption."""
    lines: list[str] = []

    # Summary header
    lines.append("=" * 70)
    lines.append("TODO Inventory Scan Results")
    lines.append("=" * 70)
    lines.append(f"Files scanned: {result.files_scanned}")
    lines.append(f"Total TODOs found: {result.total_count}")
    lines.append(f"Ungoverned TODOs: {result.ungoverned_count}")
    lines.append("")

    # By-type breakdown
    if result.by_type:
        lines.append("By type:")
        for todo_type, count in sorted(result.by_type.items()):
            lines.append(f"  {todo_type}: {count}")
        lines.append("")

    # List items
    if result.items:
        if governance_mode:
            lines.append("-" * 70)
            lines.append("Ungoverned TODOs (missing tracking reference):")
            lines.append("-" * 70)
            ungoverned = [item for item in result.items if not item.has_governance_ref]
            for item in ungoverned:
                lines.append(
                    f"  {item.path}:{item.lineno} [{item.todo_type.value}] {item.message[:60]}"
                )
        else:
            lines.append("-" * 70)
            lines.append("All TODOs:")
            lines.append("-" * 70)
            for item in result.items:
                gov_marker = "✓" if item.has_governance_ref else "⚠"
                lines.append(
                    f"  {gov_marker} {item.path}:{item.lineno} [{item.todo_type.value}] {item.message[:60]}"
                )

    # Errors
    if result.errors:
        lines.append("")
        lines.append("-" * 70)
        lines.append("Scan errors:")
        lines.append("-" * 70)
        for error in result.errors:
            lines.append(f"  ⚠ {error}")

    lines.append("")

    # Governance guidance
    if governance_mode and result.ungoverned_count > 0:
        lines.append("-" * 70)
        lines.append("Governance Guidance:")
        lines.append("-" * 70)
        lines.append("New TODOs should include tracking references:")
        lines.append("  # TODO(Phase 3): Description...")
        lines.append("  # TODO(ADR-044): Description...")
        lines.append("  # TODO(#1234): Description...")
        lines.append("  # TODO(@specialist): Description...")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Scan repository for TODO patterns and validate governance compliance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default human-readable scan
    python scripts/validation/scan_todo_inventory.py

    # JSON output for CI
    python scripts/validation/scan_todo_inventory.py --json

    # Governance check (exit 1 if ungoverned TODOs found)
    python scripts/validation/scan_todo_inventory.py --check-governance

Governance Reference Patterns:
    # TODO(Phase 3): ...     - Phase-gated work
    # TODO(ADR-044): ...     - ADR-linked work
    # TODO(#1234): ...       - Issue-linked work
    # TODO(@specialist): ... - Owner-assigned work
        """,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON instead of human-readable text",
    )
    parser.add_argument(
        "--check-governance",
        action="store_true",
        help="Governance enforcement mode: exit 1 if ungoverned TODOs are found",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = _parse_args()

    result = scan_repository()

    if args.json:
        output = result.to_dict()
        output["governance_compliant"] = result.ungoverned_count == 0
        print(json.dumps(output, indent=2))
    else:
        print(_format_human_readable(result, governance_mode=args.check_governance))

    # Exit codes and final summary
    if args.check_governance and result.ungoverned_count > 0:
        if not args.json:
            print(
                f"❌ Governance check failed: {result.ungoverned_count} ungoverned TODO(s) found"
            )
        return 1

    if not args.json:
        print(f"✅ Scan complete: {result.total_count} TODO(s) found in {result.files_scanned} files")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
