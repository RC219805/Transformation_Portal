#!/usr/bin/env python3
"""Enforce pytest marker requirements for test functions.

Policy (ADR-044):
- All test functions must have at least one pytest marker.
- Tests in specific directories have required markers by convention.
- Pre-commit hook blocks unmarked tests from being added.

This script supports two modes:
1. Pre-commit (default): Only checks files passed as arguments (staged files)
2. Full audit: Scans all test files and reports comprehensive marker coverage

Usage:
    # Pre-commit mode (validate specific files)
    python scripts/validation/check_test_markers.py tests/test_foo.py tests/test_bar.py

    # Full audit mode (scan entire tests/ directory)
    python scripts/validation/check_test_markers.py --audit

    # Show detailed report
    python scripts/validation/check_test_markers.py --audit --verbose
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Configuration constants
MAX_VIOLATIONS_DISPLAY = 20

# Pytest built-in markers (not test category markers)
BUILTIN_MARKERS = frozenset(
    {
        "skip",
        "skipif",
        "xfail",
        "parametrize",
        "usefixtures",
        "filterwarnings",
        "timeout",
    }
)

# Semantic test category markers as registered in pyproject.toml
CATEGORY_MARKERS = frozenset(
    {
        "slow",
        "unit",
        "regression",
        "security",
        "integration",
        "ml",
        "golden",
        "stress",
        "benchmark",
    }
)

# Valid pytest markers: semantic markers plus built-in markers
VALID_MARKERS = CATEGORY_MARKERS | BUILTIN_MARKERS

# Directory-based marker requirements per ADR-044
# Maps directory name -> required markers
DIRECTORY_MARKER_REQUIREMENTS: dict[str, list[str]] = {
    "unit": ["unit"],
    "security": ["security"],  # Also implicitly requires unit, but we check presence of at least one
    "integration": ["integration"],
    "benchmarks": ["benchmark"],
    "stress": ["stress"],  # Also typically has slow
    # smoke tests map to unit per ADR-044 decision
}


@dataclass(frozen=True)
class TestFunction:
    """Represents a test function found in a file."""

    file_path: Path
    name: str
    lineno: int
    markers: frozenset[str]


@dataclass(frozen=True)
class MarkerViolation:
    """Represents a test function without required markers."""

    file_path: Path
    name: str
    lineno: int
    issue: str


@dataclass
class AuditReport:
    """Comprehensive marker audit report."""

    total_test_functions: int = 0
    marked_functions: int = 0
    unmarked_functions: int = 0
    violations: list[MarkerViolation] = field(default_factory=list)
    marker_counts: dict[str, int] = field(default_factory=dict)

    @property
    def coverage_percent(self) -> float:
        if self.total_test_functions == 0:
            return 100.0
        return (self.marked_functions / self.total_test_functions) * 100


class TestMarkerVisitor(ast.NodeVisitor):
    """AST visitor to extract test functions and their markers."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.test_functions: list[TestFunction] = []
        self._module_markers: frozenset[str] = frozenset()

    def visit_Module(self, node: ast.Module) -> None:
        """Extract module-level pytestmark assignment."""
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == "pytestmark":
                        self._module_markers = self._extract_markers_from_value(item.value)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract test functions and their markers."""
        if node.name.startswith("test_"):
            markers = self._extract_markers_from_decorators(node.decorator_list)
            # Combine with module-level markers
            all_markers = markers | self._module_markers
            self.test_functions.append(
                TestFunction(
                    file_path=self.file_path,
                    name=node.name,
                    lineno=node.lineno,
                    markers=all_markers,
                )
            )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Handle async test functions."""
        if node.name.startswith("test_"):
            markers = self._extract_markers_from_decorators(node.decorator_list)
            all_markers = markers | self._module_markers
            self.test_functions.append(
                TestFunction(
                    file_path=self.file_path,
                    name=node.name,
                    lineno=node.lineno,
                    markers=all_markers,
                )
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handle test classes with class-level markers."""
        if node.name.startswith("Test"):
            class_markers = self._extract_markers_from_decorators(node.decorator_list)
            # Visit methods within the class
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("test_"):
                        method_markers = self._extract_markers_from_decorators(item.decorator_list)
                        all_markers = method_markers | class_markers | self._module_markers
                        self.test_functions.append(
                            TestFunction(
                                file_path=self.file_path,
                                name=f"{node.name}.{item.name}",
                                lineno=item.lineno,
                                markers=all_markers,
                            )
                        )
        else:
            self.generic_visit(node)

    def _extract_markers_from_decorators(self, decorators: list[ast.expr]) -> frozenset[str]:
        """Extract pytest.mark.X markers from decorator list."""
        markers: set[str] = set()
        for dec in decorators:
            marker = self._parse_marker(dec)
            if marker:
                markers.add(marker)
        return frozenset(markers)

    def _extract_markers_from_value(self, value: ast.expr) -> frozenset[str]:
        """Extract markers from pytestmark = ... assignment."""
        markers: set[str] = set()

        if isinstance(value, ast.List):
            for elt in value.elts:
                marker = self._parse_marker(elt)
                if marker:
                    markers.add(marker)
        else:
            marker = self._parse_marker(value)
            if marker:
                markers.add(marker)

        return frozenset(markers)

    def _parse_marker(self, node: ast.expr) -> str | None:
        """Parse a single marker expression."""
        # Handle: @pytest.mark.unit
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Attribute):
                if isinstance(node.value.value, ast.Name) and node.value.value.id == "pytest" and node.value.attr == "mark":
                    return node.attr
        # Handle: @pytest.mark.unit() - marker with call
        if isinstance(node, ast.Call):
            return self._parse_marker(node.func)
        return None


def scan_file(file_path: Path) -> list[TestFunction]:
    """Scan a single file for test functions and their markers."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        lineno_info = f" at line {e.lineno}" if e.lineno else ""
        msg_info = f": {e.msg}" if e.msg else ""
        print(
            f"WARNING: Cannot validate markers in {file_path} due to syntax error{lineno_info}{msg_info}. "
            "Fix syntax errors before running marker validation.",
            file=sys.stderr,
        )
        return []

    visitor = TestMarkerVisitor(file_path)
    visitor.visit(tree)
    return visitor.test_functions


def get_directory_type(file_path: Path) -> str | None:
    """Determine which test directory type a file belongs to."""
    parts = file_path.parts
    if "tests" in parts:
        tests_idx = parts.index("tests")
        if tests_idx + 1 < len(parts):
            subdir = parts[tests_idx + 1]
            if subdir in DIRECTORY_MARKER_REQUIREMENTS:
                return subdir
    return None


def check_marker_requirements(test_func: TestFunction) -> MarkerViolation | None:
    """Check if a test function meets marker requirements."""
    # Filter to only semantic markers (not parametrize, skipif, etc.)
    semantic_markers = test_func.markers & VALID_MARKERS
    # Exclude built-in markers that don't indicate test category
    category_markers = semantic_markers - BUILTIN_MARKERS

    if not category_markers:
        return MarkerViolation(
            file_path=test_func.file_path,
            name=test_func.name,
            lineno=test_func.lineno,
            issue="missing required marker (add @pytest.mark.unit, @pytest.mark.ml, etc.)",
        )

    # Check directory-specific requirements
    dir_type = get_directory_type(test_func.file_path)
    if dir_type and dir_type in DIRECTORY_MARKER_REQUIREMENTS:
        required = set(DIRECTORY_MARKER_REQUIREMENTS[dir_type])
        if not (category_markers & required):
            return MarkerViolation(
                file_path=test_func.file_path,
                name=test_func.name,
                lineno=test_func.lineno,
                issue=f"tests in {dir_type}/ should have @pytest.mark.{list(required)[0]}",
            )

    return None


def audit_test_directory(tests_root: Path, verbose: bool = False) -> AuditReport:
    """Perform full audit of test marker coverage."""
    report = AuditReport()

    if not tests_root.exists():
        print(f"ERROR: Tests directory not found: {tests_root}", file=sys.stderr)
        return report

    # Collect and deduplicate test files efficiently
    test_files = sorted(set(tests_root.rglob("test_*.py")) | set(tests_root.rglob("*_test.py")))

    for test_file in test_files:
        # Skip __pycache__ and other non-source directories
        if "__pycache__" in test_file.parts:
            continue

        test_functions = scan_file(test_file)

        for func in test_functions:
            report.total_test_functions += 1

            # Count markers
            for marker in func.markers:
                if marker in VALID_MARKERS:
                    report.marker_counts[marker] = report.marker_counts.get(marker, 0) + 1

            violation = check_marker_requirements(func)
            if violation:
                report.unmarked_functions += 1
                report.violations.append(violation)
                if verbose:
                    print(f"  UNMARKED: {func.file_path}:{func.lineno} {func.name}")
            else:
                report.marked_functions += 1

    return report


def check_files(files: Sequence[Path]) -> list[MarkerViolation]:
    """Check specific files for marker violations (pre-commit mode)."""
    violations: list[MarkerViolation] = []

    for file_path in files:
        if not file_path.exists():
            continue
        if not file_path.name.startswith("test_") and not file_path.name.endswith("_test.py"):
            continue

        test_functions = scan_file(file_path)

        for func in test_functions:
            violation = check_marker_requirements(func)
            if violation:
                violations.append(violation)

    return violations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enforce pytest marker requirements (ADR-044).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Test files to check (pre-commit mode)",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run full audit on tests/ directory",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--tests-root",
        type=Path,
        default=Path("tests"),
        help="Root directory for tests (default: tests)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code on any unmarked tests (default for pre-commit mode)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.audit:
        # Full audit mode
        print(f"Auditing test markers in {args.tests_root}/...")
        report = audit_test_directory(args.tests_root, verbose=args.verbose)

        print("\n" + "=" * 60)
        print("TEST MARKER AUDIT REPORT (ADR-044)")
        print("=" * 60)
        print(f"Total test functions: {report.total_test_functions}")
        print(f"Marked functions:     {report.marked_functions}")
        print(f"Unmarked functions:   {report.unmarked_functions}")
        print(f"Coverage:             {report.coverage_percent:.1f}%")
        print()

        if report.marker_counts:
            print("Marker distribution:")
            for marker, count in sorted(report.marker_counts.items(), key=lambda x: -x[1]):
                print(f"  @pytest.mark.{marker}: {count}")
            print()

        if report.violations:
            print(f"Violations ({len(report.violations)}):")
            for v in report.violations[:MAX_VIOLATIONS_DISPLAY]:
                print(f"  - {v.file_path}:{v.lineno} {v.name}")
                print(f"    {v.issue}")
            if len(report.violations) > MAX_VIOLATIONS_DISPLAY:
                print(f"  ... and {len(report.violations) - MAX_VIOLATIONS_DISPLAY} more")

        # ADR-044 target: <5% unmarked
        target_coverage = 95.0
        if report.coverage_percent < target_coverage:
            print(f"\n⚠ Coverage {report.coverage_percent:.1f}% is below target {target_coverage:.0f}%")
            if args.strict:
                return 1
        else:
            print(f"\n✓ Coverage {report.coverage_percent:.1f}% meets target {target_coverage:.0f}%")

        return 0

    elif args.files:
        # Pre-commit mode: check specific files
        violations = check_files(args.files)

        if not violations:
            print("Test marker check passed: all test functions have markers.")
            return 0

        print("ERROR: Test functions missing required markers (ADR-044):", file=sys.stderr)
        for v in violations:
            print(f"  - {v.file_path}:{v.lineno} {v.name}", file=sys.stderr)
            print(f"    {v.issue}", file=sys.stderr)

        print(
            "\nRemediation: Add @pytest.mark.unit (or appropriate marker) to each test function.",
            file=sys.stderr,
        )
        valid_marker_list = ", ".join(sorted(CATEGORY_MARKERS))
        print(f"Valid markers: {valid_marker_list}", file=sys.stderr)
        return 1

    else:
        # No files and no --audit: show help
        print("Usage: check_test_markers.py [--audit] [--verbose] [files...]")
        print("Run with --help for full usage information.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
