#!/usr/bin/env python3
"""CI validation script to detect unsafe torch.load() usage.

This script scans the codebase for raw torch.load() calls that don't use
weights_only=True, which would bypass CVE-2025-32434 mitigation.

SECURITY CONTEXT:
    CVE-2025-32434 is a critical RCE vulnerability (CVSS 9.8) in torch.load().
    Instead of upgrading torch (which would break CAS determinism), we mitigate
    at runtime by enforcing weights_only=True on all torch.load() calls.

    This CI gate ensures no raw torch.load() usage slips through code review.

EXIT CODES:
    0: No unsafe torch.load() usage found
    1: Unsafe torch.load() usage detected (fails CI)
    2: Script error (e.g., invalid arguments)

USAGE:
    python scripts/validation/check_unsafe_torch_load.py
    python scripts/validation/check_unsafe_torch_load.py --verbose
    python scripts/validation/check_unsafe_torch_load.py --fix-suggestions
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence


# Directories to skip
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "build",
    "dist",
    "*.egg-info",
}

# Files that are allowed to use raw torch.load (approved security bypasses)
ALLOWED_FILES = {
    # The torch_security module itself uses torch.load internally
    "src/transformation_portal/core/security/torch_security.py",
    # Test files for security module
    "tests/test_torch_security.py",
    # This validation script discusses torch.load in documentation
    "scripts/validation/check_unsafe_torch_load.py",
}

# Pattern to detect torch.load() calls
# This matches: torch.load(...) where ... doesn't contain weights_only=True
TORCH_LOAD_PATTERN = re.compile(
    r'\btorch\.load\s*\('
)

# Pattern to detect safe usage with weights_only=True
SAFE_USAGE_PATTERN = re.compile(
    r'\btorch\.load\s*\([^)]*weights_only\s*=\s*True'
)

# Pattern to detect our safe_load() wrapper
SAFE_LOAD_PATTERN = re.compile(
    r'\bsafe_load\s*\('
)


@dataclass
class Violation:
    """Represents an unsafe torch.load() usage."""
    file: Path
    line_number: int
    line_content: str
    suggestion: str


def should_skip_path(path: Path) -> bool:
    """Check if a path should be skipped."""
    for part in path.parts:
        if part in SKIP_DIRS or part.endswith(".egg-info"):
            return True
    return False


def is_allowed_file(path: Path, repo_root: Path) -> bool:
    """Check if a file is in the allowed list."""
    try:
        rel_path = path.relative_to(repo_root)
        return str(rel_path) in ALLOWED_FILES
    except ValueError:
        return False


def is_in_string(line: str, pattern: re.Pattern) -> bool:
    """Check if the pattern match appears inside a string literal.

    This is a simple heuristic that counts quotes before the match.
    If the number of quotes is odd, the match is likely inside a string.
    """
    match = pattern.search(line)
    if not match:
        return False

    before_match = line[:match.start()]

    # Count unescaped quotes before the match
    single_quotes = before_match.count("'") - before_match.count("\\'")
    double_quotes = before_match.count('"') - before_match.count('\\"')

    # If either count is odd, we're likely inside a string
    return (single_quotes % 2 == 1) or (double_quotes % 2 == 1)


def check_file(path: Path, repo_root: Path) -> Iterator[Violation]:
    """Check a single file for unsafe torch.load() usage."""
    if is_allowed_file(path, repo_root):
        return

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, IOError):
        return

    lines = content.splitlines()

    # Track docstring state (simple heuristic)
    in_docstring = False
    docstring_char = None

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Track docstring boundaries (triple quotes)
        if '"""' in stripped or "'''" in stripped:
            quote = '"""' if '"""' in stripped else "'''"
            # Count occurrences to handle start and end on same line
            count = stripped.count(quote)
            if count == 1:
                in_docstring = not in_docstring
                docstring_char = quote if in_docstring else None
            # If count is even (e.g., 2), state doesn't change

        # Skip if inside docstring
        if in_docstring:
            continue

        # Skip comment lines
        if stripped.startswith("#"):
            continue

        # Skip lines that are just strings (documentation, messages)
        # These typically start with a quote or are inside parentheses after a string
        if stripped.startswith('"') or stripped.startswith("'"):
            continue

        # Check for torch.load() calls
        if TORCH_LOAD_PATTERN.search(line):
            # Check if it's safe usage
            if SAFE_USAGE_PATTERN.search(line):
                continue  # Safe: has weights_only=True

            # Check if line is commented out
            if "#" in line:
                code_part = line.split("#")[0]
                if not TORCH_LOAD_PATTERN.search(code_part):
                    continue  # torch.load is in comment

            # Check if it's inside a string literal (common in docstrings/messages)
            # Simple heuristic: if the match is inside quotes, skip it
            if is_in_string(line, TORCH_LOAD_PATTERN):
                continue

            yield Violation(
                file=path,
                line_number=line_num,
                line_content=line.strip(),
                suggestion=(
                    "Replace with: safe_load(path, map_location=device)\n"
                    "  Or add: weights_only=True parameter\n"
                    "  Import: from transformation_portal.core.security.torch_security import safe_load"
                ),
            )


def find_python_files(root: Path) -> Iterator[Path]:
    """Find all Python files in the repository."""
    for path in root.rglob("*.py"):
        if not should_skip_path(path):
            yield path


def main(args: Sequence[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check for unsafe torch.load() usage in codebase",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output including scanned files",
    )
    parser.add_argument(
        "--fix-suggestions",
        action="store_true",
        help="Show detailed fix suggestions for each violation",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root path (auto-detected if not provided)",
    )
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="Run in CI mode with machine-readable output",
    )

    parsed = parser.parse_args(args)

    # Find repository root
    if parsed.repo_root:
        repo_root = parsed.repo_root
    else:
        # Auto-detect: look for .git directory
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir
        while repo_root.parent != repo_root:
            if (repo_root / ".git").exists():
                break
            repo_root = repo_root.parent
        else:
            print("ERROR: Could not find repository root", file=sys.stderr)
            return 2

    if not repo_root.exists():
        print(f"ERROR: Repository root does not exist: {repo_root}", file=sys.stderr)
        return 2

    # Find violations
    violations: List[Violation] = []
    files_scanned = 0

    if parsed.verbose:
        print(f"Scanning repository: {repo_root}")
        print(f"Allowed files: {ALLOWED_FILES}")
        print()

    for py_file in find_python_files(repo_root):
        files_scanned += 1
        if parsed.verbose:
            print(f"  Scanning: {py_file.relative_to(repo_root)}")

        for violation in check_file(py_file, repo_root):
            violations.append(violation)

    # Report results
    print()
    print("=" * 72)
    print("TORCH.LOAD() SECURITY CHECK (CVE-2025-32434)")
    print("=" * 72)
    print()
    print(f"Files scanned: {files_scanned}")
    print(f"Violations found: {len(violations)}")
    print()

    if violations:
        print("UNSAFE TORCH.LOAD() USAGE DETECTED:")
        print("-" * 72)
        print()

        for i, v in enumerate(violations, start=1):
            rel_path = v.file.relative_to(repo_root)
            print(f"{i}. {rel_path}:{v.line_number}")
            print(f"   Line: {v.line_content[:70]}{'...' if len(v.line_content) > 70 else ''}")

            if parsed.fix_suggestions:
                print()
                print(f"   FIX: {v.suggestion}")

            print()

        print("-" * 72)
        print()
        print("⛔ CI GATE FAILED: Unsafe torch.load() usage violates security policy")
        print()
        print("Resolution:")
        print("  1. Import: from transformation_portal.core.security.torch_security import safe_load")
        print("  2. Replace: torch.load(path) → safe_load(path, map_location=device)")
        print("  3. Or add:  weights_only=True parameter to existing torch.load() calls")
        print()
        print("Documentation:")
        print("  - See: src/transformation_portal/core/security/torch_security.py")
        print("  - See: SECURITY.md for CVE-2025-32434 mitigation policy")
        print()

        if parsed.ci_mode:
            # Machine-readable output for CI
            for v in violations:
                rel_path = v.file.relative_to(repo_root)
                print(f"::error file={rel_path},line={v.line_number}::Unsafe torch.load() usage")

        return 1

    else:
        print("✅ No unsafe torch.load() usage found")
        print()
        print("All torch.load() calls either:")
        print("  - Use weights_only=True parameter")
        print("  - Use safe_load() wrapper")
        print("  - Are in approved security bypass files")
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
