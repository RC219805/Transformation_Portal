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

    # Refresh the committed scanner snapshot
    python scripts/validation/scan_todo_inventory.py --write-snapshot

    # Governance enforcement mode (exit 1 on violations)
    python scripts/validation/scan_todo_inventory.py --check-governance
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import sys
import tempfile
import tokenize
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT_PATH = PROJECT_ROOT / "docs" / "analysis" / "todo_scanner_snapshot.json"
DEFAULT_JSON_INDENT = 2

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
TODO_PATTERNS = (re.compile(r"#\s*(TODO|FIXME|HACK|XXX)\s*:?\s*(.*)$", re.IGNORECASE),)

# JS/TS TODO patterns (single-line // comments and block /** */ comments)
JS_TODO_PATTERNS = (
    re.compile(r"//\s*(TODO|FIXME|HACK|XXX)\s*:?\s*(.*)$", re.IGNORECASE),
    re.compile(r"\*\s*(TODO|FIXME|HACK|XXX)\s*:?\s*(.*)$", re.IGNORECASE),
)

# Governance reference patterns - TODOs with proper tracking
GOVERNANCE_REF_PATTERNS = (
    re.compile(r"\(Phase\s*\d+[A-Za-z]?\)", re.IGNORECASE),  # (Phase 3), (Phase 4A)
    re.compile(r"\(ADR-\d+\)", re.IGNORECASE),  # (ADR-044)
    re.compile(r"\(#\d+\)"),  # (#1234) - issue ref
    re.compile(r"\(@[\w-]+\)"),  # (@specialist) - owner
    re.compile(r"\([A-Za-z]+_INVENTORY\.md(?:\s+§[\w.-]+)?\)", re.IGNORECASE),  # (TODO_INVENTORY.md), (TODO_INVENTORY.md §3.1)
)

# Excluded patterns (false positives in security code)
EXCLUDED_PATTERNS = (re.compile(r"TODO_REPLACE", re.IGNORECASE),)  # Security scanner pattern

# Self-exclude: this scanner file contains TODO examples in docs
SELF_EXCLUDE_FILE = "scripts/validation/scan_todo_inventory.py"

# Excluded file paths (relative to project root)
EXCLUDED_PATH_PATTERNS = (
    "docs/",  # Historical documentation markers
    "__pycache__/",
    ".git/",
    ".next/",
    ".next-build-verify/",
    ".next-smoke-",
    ".next-codex-",
    "node_modules/",
    ".venv/",
    ".runtime/",
    ".egg-info/",
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


def _json_payload(result: ScanResult) -> dict[str, Any]:
    """Build the stable machine-readable scanner payload."""
    output = result.to_dict()
    output["governance_compliant"] = result.ungoverned_count == 0
    return output


def _format_json_payload(payload: dict[str, Any]) -> str:
    """Format scanner JSON with a stable trailing newline."""
    return json.dumps(payload, indent=DEFAULT_JSON_INDENT) + "\n"


def _resolve_snapshot_path(raw_path: str) -> Path:
    """Resolve a snapshot path and require it to stay under the repository."""
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    resolved = candidate.resolve()
    repo_root = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Snapshot path must stay under repository root: {raw_path}") from exc
    return resolved


def _write_json_snapshot(payload: dict[str, Any], raw_path: str) -> Path:
    """Write a scanner snapshot JSON file and return the resolved path."""
    snapshot_path = _resolve_snapshot_path(raw_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(snapshot_path, _format_json_payload(payload))
    return snapshot_path


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically write text to a path using a same-directory temp file."""
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        temp_path.replace(path)
    except Exception:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
        raise


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


# Abstract/base-class patterns that should be excluded from governance requirements.
# These are common idiomatic patterns for abstract methods and interface stubs.
ABSTRACT_METHOD_PATTERNS = (
    re.compile(r"subclass(?:es)?\s+(?:must|should)\s+(?:implement|override)", re.IGNORECASE),
    re.compile(r"must\s+be\s+(?:implemented|overridden)\s+(?:by|in)\s+subclass", re.IGNORECASE),
    re.compile(r"override\s+(?:this|in)\s+subclass", re.IGNORECASE),
    re.compile(r"abstract\s+method", re.IGNORECASE),
    re.compile(r"not\s+implemented\s+(?:in\s+)?(?:base|abstract)\s+class", re.IGNORECASE),
    re.compile(r"implement\s+in\s+(?:derived|child)\s+class", re.IGNORECASE),
)


def _is_abstract_method_pattern(message: str) -> bool:
    """Check if a NotImplementedError message indicates an abstract method pattern.

    Returns True for:
    - Empty/no-message raises (bare `raise NotImplementedError`)
    - Messages matching abstract method documentation patterns
    """
    # Bare raise NotImplementedError (no message) is an idiomatic abstract stub
    if not message:
        return True
    return any(pattern.search(message) for pattern in ABSTRACT_METHOD_PATTERNS)


class NotImplementedVisitor(ast.NodeVisitor):
    """AST visitor to find NotImplementedError raises with context."""

    def __init__(self) -> None:
        self.items: list[tuple[int, int, str, bool]] = []  # (lineno, col, message, is_abstract)

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
            is_abstract = _is_abstract_method_pattern(message)
            self.items.append((node.lineno, node.col_offset, message, is_abstract))

        self.generic_visit(node)


def _extract_comment_todos(source: str, path: Path) -> tuple[list[TodoItem], list[str]]:
    """Extract TODO items from Python comments using tokenize.

    Uses Python's tokenize module to scan only actual comment tokens,
    avoiding false positives from TODO patterns inside string literals/docstrings.

    Returns:
        Tuple of (items, errors) where errors contains tokenization failure messages.
    """
    items: list[TodoItem] = []
    errors: list[str] = []
    lines = source.splitlines()

    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type != tokenize.COMMENT:
                continue

            comment_text = tok.string
            lineno = tok.start[0]
            col_offset = tok.start[1]

            if _is_excluded_content(comment_text):
                continue

            for pattern in TODO_PATTERNS:
                match = pattern.search(comment_text)
                if match:
                    todo_type_str = match.group(1).upper()
                    message = match.group(2).strip()

                    # Include context for multi-line messages (next line if continuation)
                    if lineno < len(lines):
                        next_line = lines[lineno].strip()
                        if next_line.startswith("#") and not any(p.search(next_line) for p in TODO_PATTERNS):
                            continuation = next_line.lstrip("#").strip()
                            if continuation:
                                message = f"{message} {continuation}"

                    governance_refs = _extract_governance_refs(comment_text + " " + message)
                    items.append(
                        TodoItem(
                            path=path.relative_to(PROJECT_ROOT),
                            lineno=lineno,
                            todo_type=TodoType(todo_type_str),
                            message=message,
                            has_governance_ref=bool(governance_refs),
                            governance_refs=governance_refs,
                            col_offset=col_offset,
                        )
                    )
                    break  # Only count each comment once
    except tokenize.TokenError as e:
        errors.append(f"{path}: tokenize error: {e}")

    return items, errors


def _scan_python_file(path: Path) -> tuple[list[TodoItem], list[str]]:
    """Scan a single Python file for TODO patterns.

    Returns:
        Tuple of (items, errors) where errors contains parsing failure messages.
    """
    source = _read_file_safe(path)
    if source is None:
        return [], []

    items: list[TodoItem] = []
    errors: list[str] = []

    # Scan for comment-based TODOs using tokenize (avoids false positives in strings)
    comment_items, comment_errors = _extract_comment_todos(source, path)
    items.extend(comment_items)
    errors.extend(comment_errors)

    # Scan for NotImplementedError using AST
    try:
        tree = ast.parse(source, filename=str(path))
        visitor = NotImplementedVisitor()
        visitor.visit(tree)
        for lineno, col_offset, message, is_abstract in visitor.items:
            # Abstract method patterns are auto-governed (not actionable TODOs)
            governance_refs = _extract_governance_refs(message)
            has_governance = bool(governance_refs) or is_abstract
            items.append(
                TodoItem(
                    path=path.relative_to(PROJECT_ROOT),
                    lineno=lineno,
                    todo_type=TodoType.NOT_IMPLEMENTED,
                    message=message or "(no message)",
                    has_governance_ref=has_governance,
                    governance_refs=governance_refs,
                    col_offset=col_offset,
                )
            )
    except SyntaxError as e:
        errors.append(f"{path}: syntax error: {e}")

    return items, errors


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
            items, errors = _scan_python_file(path)
            result.items.extend(items)
            result.errors.extend(errors)
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
                lines.append(f"  {item.path}:{item.lineno} [{item.todo_type.value}] {item.message[:60]}")
        else:
            lines.append("-" * 70)
            lines.append("All TODOs:")
            lines.append("-" * 70)
            for item in result.items:
                gov_marker = "✓" if item.has_governance_ref else "⚠"
                lines.append(f"  {gov_marker} {item.path}:{item.lineno} [{item.todo_type.value}] {item.message[:60]}")

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

    # Refresh committed scanner snapshot
    python scripts/validation/scan_todo_inventory.py --write-snapshot

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
    parser.add_argument(
        "--write-snapshot",
        nargs="?",
        const=DEFAULT_SNAPSHOT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        metavar="PATH",
        help=(
            "Write the JSON scan payload to PATH, or to "
            f"{DEFAULT_SNAPSHOT_PATH.relative_to(PROJECT_ROOT).as_posix()} when PATH is omitted"
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = _parse_args()

    result = scan_repository()
    payload = _json_payload(result)

    snapshot_arg = getattr(args, "write_snapshot", None)
    snapshot_path: Path | None = None
    if snapshot_arg is not None:
        if result.errors:
            print(
                f"❌ Refusing to write TODO scanner snapshot: {len(result.errors)} scan error(s) encountered",
                file=sys.stderr,
            )
            return 2
        try:
            snapshot_path = _write_json_snapshot(payload, snapshot_arg)
        except (OSError, ValueError) as exc:
            print(f"❌ Failed to write TODO scanner snapshot: {exc}", file=sys.stderr)
            return 2

    if args.json:
        print(_format_json_payload(payload), end="")
    else:
        print(_format_human_readable(result, governance_mode=args.check_governance))
        if snapshot_path is not None:
            print(f"Wrote TODO scanner snapshot: {snapshot_path.relative_to(PROJECT_ROOT)}")

    # Exit codes and final summary
    # Exit 2: scan errors in governance mode (fail closed)
    if args.check_governance and result.errors:
        if not args.json:
            print(f"❌ Governance check failed: {len(result.errors)} scan error(s) encountered (fail closed)")
        return 2

    # Exit 1: ungoverned TODOs in governance mode
    if args.check_governance and result.ungoverned_count > 0:
        if not args.json:
            print(f"❌ Governance check failed: {result.ungoverned_count} ungoverned TODO(s) found")
        return 1

    if not args.json:
        print(f"✅ Scan complete: {result.total_count} TODO(s) found in {result.files_scanned} files")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
