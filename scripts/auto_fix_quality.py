#!/usr/bin/env python3
"""
Auto-fix utility for common quality issues in Transformation Portal.

Fixes:
- Trailing whitespace
- Common missing imports (heuristic)
- Format code with autopep8

Usage:
    python scripts/auto_fix_quality.py [--fix-all] [--dry-run] [paths...]

Options:
    --fix-all       Fix all issues automatically (no prompts)
    --dry-run       Show what would be fixed without making changes
    --whitespace    Fix trailing whitespace only
    --imports       Fix import issues only
    --format        Format code with autopep8
    --jobs N        Number of worker processes (default: 1)
    --quiet         Suppress output (only errors)
    paths...        Specific files or directories to fix (default: all tracked files)
"""

import argparse
import ast
import concurrent.futures
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger("auto_fix_quality")


class QualityFixer:
    """Auto-fix common quality issues."""

    def __init__(self, dry_run: bool = False, verbose: bool = True, jobs: int = 1):
        self.dry_run = dry_run
        self.verbose = verbose
        self.jobs = max(1, jobs)
        self.fixed_files: Set[Path] = set()
        self.repo_root = self._get_repo_root()

    # -------------------------------------------------------------------------
    # Infrastructure
    # -------------------------------------------------------------------------

    def _get_repo_root(self) -> Path:
        """Get repository root directory (via git, with sane fallback)."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True,
            )
            return Path(result.stdout.strip())
        except subprocess.CalledProcessError:
            return Path.cwd()

    def log(self, message: str, level: str = "info"):
        """Log message with color (for TTY) and via logging module."""
        if not self.verbose and level not in {"error", "warning"}:
            return

        colors = {
            "info": "\033[0;34m",  # Blue
            "success": "\033[0;32m",  # Green
            "warning": "\033[1;33m",  # Yellow
            "error": "\033[0;31m",  # Red
            "reset": "\033[0m",
        }
        color = colors.get(level, colors["reset"])

        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "success":
            logger.info(message)
        else:
            logger.info(message)

        # Keep existing colored stdout behavior
        print(f"{color}{message}{colors['reset']}")

    # -------------------------------------------------------------------------
    # Core operations (per-file)
    # -------------------------------------------------------------------------

    def _fix_trailing_whitespace_file(self, path: Path) -> bool:
        """Fix trailing whitespace in a single file. Returns True if changed.

        This function detects the first line ending style (CRLF, CR, or LF) present
        in the file and normalizes all line endings to that style. Mixed line endings are
        corrected to use a consistent style throughout the file. The final newline
        is preserved if present in the original file.
        """
        if not path.is_file():
            return False

        try:
            content = path.read_bytes().decode("utf-8")
        except (UnicodeDecodeError, PermissionError, OSError) as e:
            self.log(f"  ✗ Skipping (cannot read): {path} ({e})", "warning")
            return False

        lines = content.splitlines(keepends=True)
        has_trailing = any(line.rstrip() != line.rstrip("\n\r") for line in lines)

        if not has_trailing:
            return False

        if self.dry_run:
            self.log(
                f"  [DRY-RUN] Would fix trailing whitespace: {path.relative_to(self.repo_root)}",
                "warning",
            )
            return True

        # Detect the first line ending style encountered (CRLF, CR, or fallback to LF)
        # Note: This normalizes all line endings in the file to maintain consistency.
        # The implementation normalizes to the first non-LF ending found (CRLF or CR),
        # falling back to LF only if no other endings are detected.
        line_ending = "\n"  # default to Unix-style
        for line in lines:
            if line.endswith("\r\n"):
                line_ending = "\r\n"
                break
            elif line.endswith("\r"):
                line_ending = "\r"
                break

        # Check if file ends with a newline
        has_final_newline = lines and lines[-1].endswith(("\n", "\r"))

        # Strip trailing whitespace from all lines
        fixed_lines = [line.rstrip(" \t\r\n") for line in lines]

        # Rejoin with consistent line ending, preserving final newline behavior
        if has_final_newline:
            content_fixed = line_ending.join(fixed_lines) + line_ending
        else:
            content_fixed = line_ending.join(fixed_lines)
        try:
            path.write_bytes(content_fixed.encode("utf-8"))
        except (PermissionError, OSError) as e:
            self.log(f"  ✗ Failed to write file (read-only?): {path} ({e})", "error")
            return False

        self.log(f"  ✓ Fixed trailing whitespace: {path.relative_to(self.repo_root)}", "success")
        return True

    def _get_used_names(self, content: str) -> Set[str]:
        """
        Use AST to find names that are actually used in the code.
        This excludes matches in comments, strings, and docstrings.
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # If file has syntax errors, fall back to empty set
            return set()

        used_names = set()

        class NameVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                used_names.add(node.id)
                self.generic_visit(node)

            def visit_Attribute(self, node):
                # For attributes like np.array, we want to track 'np'
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
                self.generic_visit(node)

        NameVisitor().visit(tree)
        return used_names

    def _get_existing_imports(self, content: str) -> Set[str]:
        """
        Use AST to find all existing import statements.
        Returns the canonical import line for each import.
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # If file has syntax errors, skip import detection
            return set()

        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        imports.add(f"import {alias.name} as {alias.asname}")
                    else:
                        imports.add(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                # Handle relative imports: node.level gives number of leading dots
                dots = "." * node.level if hasattr(node, "level") and node.level else ""
                module = node.module or ""
                from_part = f"{dots}{module}" if module else dots
                for alias in node.names:
                    if alias.asname:
                        imports.add(f"from {from_part} import {alias.name} as {alias.asname}")
                    else:
                        imports.add(f"from {from_part} import {alias.name}")

        return imports

    def _fix_imports_file(self, path: Path) -> bool:
        """Fix common import issues in a single Python file. Returns True if changed."""
        if path.suffix != ".py" or not path.is_file():
            return False

        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError, OSError) as e:
            self.log(f"  ✗ Skipping (cannot read): {path} ({e})", "warning")
            return False

        import_fixes = {
            "iio": "import imageio.v3 as iio",
            "np": "import numpy as np",
            "pd": "import pandas as pd",
            "plt": "import matplotlib.pyplot as plt",
            "cv2": "import cv2",
            "Image": "from PIL import Image",
            "Path": "from pathlib import Path",
        }

        # Use AST-based analysis to find actually used names
        used_names = self._get_used_names(content)
        existing_imports = self._get_existing_imports(content)

        undefined = []
        for name, import_line in import_fixes.items():
            # Check if name is used in the AST but its canonical import line is missing
            if name in used_names and import_line not in existing_imports:
                undefined.append((name, import_line))

        if not undefined:
            return False

        if self.dry_run:
            self.log(
                f"  [DRY-RUN] Would add imports to: {path.relative_to(self.repo_root)}",
                "warning",
            )
            for _, import_line in undefined:
                self.log(f"    + {import_line}", "warning")
            return True

        lines = content.splitlines(keepends=True)
        import_section_end = 0

        # Find end of import section
        for i, line in enumerate(lines):
            if line.strip().startswith(("import ", "from ")):
                import_section_end = i + 1

        for _, import_line in reversed(undefined):
            lines.insert(import_section_end, import_line + "\n")

        try:
            path.write_text("".join(lines), encoding="utf-8")
        except (PermissionError, OSError) as e:
            self.log(f"  ✗ Failed to write imports in {path}: {e}", "error")
            return False

        self.log(f"  ✓ Fixed imports in: {path.relative_to(self.repo_root)}", "success")
        for _, import_line in undefined:
            self.log(f"    + {import_line}", "success")
        return True

    def _format_code_file(self, path: Path) -> bool:
        """Format a single Python file with autopep8. Returns True if changed."""
        if path.suffix != ".py" or not path.is_file():
            return False

        rel = path.relative_to(self.repo_root)
        cmd = [
            "autopep8",
            "--in-place",
            "--max-line-length=127",
            "--aggressive",
            "--aggressive",
            str(path),
        ]

        if self.dry_run:
            self.log(f"  [DRY-RUN] Would format: {rel}", "warning")
            return True

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            self.log(f"  ✗ Failed to format {rel}: {e}", "error")
            return False

        self.log(f"  ✓ Formatted: {rel}", "success")
        return True

    # -------------------------------------------------------------------------
    # Higher-level operations (over file sets)
    # -------------------------------------------------------------------------

    def _run_per_file(self, func, files: List[Path], label: str) -> int:
        """Run a per-file function over a set of files, optionally in parallel."""
        if not files:
            self.log(f"⚠ No files to process for {label}", "warning")
            return 0

        changed_count = 0
        changed_files = []

        if self.jobs <= 1:
            for path in files:
                try:
                    if func(path):
                        changed_count += 1
                        changed_files.append(path)
                except Exception as e:  # Defensive: keep going
                    self.log(f"✗ Error while processing {path}: {e}", "error")
        else:
            self.log(f"→ Running {label} with {self.jobs} worker processes...", "info")
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.jobs) as exe:
                futures = {exe.submit(func, p): p for p in files}
                for fut in concurrent.futures.as_completed(futures):
                    path = futures[fut]
                    try:
                        if fut.result():
                            changed_count += 1
                            changed_files.append(path)
                    except Exception as e:
                        self.log(f"✗ Error in worker for {path}: {e}", "error")

        # Update the set of fixed files
        self.fixed_files.update(changed_files)
        return changed_count

    def fix_trailing_whitespace(self, paths: List[Path]) -> int:
        """Remove trailing whitespace from files."""
        self.log("\n→ Fixing trailing whitespace...", "info")
        count = self._run_per_file(self._fix_trailing_whitespace_file, paths, "whitespace")
        if count > 0:
            self.log(f"✓ Fixed trailing whitespace in {count} files", "success")
        else:
            self.log("✓ No trailing whitespace found", "success")
        return count

    def fix_imports(self, paths: List[Path]) -> int:
        """Fix common import issues in Python files."""
        self.log("\n→ Checking for undefined imports...", "info")
        count = self._run_per_file(self._fix_imports_file, paths, "imports")
        if count > 0:
            self.log(f"✓ Fixed imports in {count} files", "success")
        else:
            self.log("✓ No import issues found", "success")
        return count

    def format_code(self, paths: List[Path]) -> int:
        """Format Python code with autopep8."""
        self.log("\n→ Formatting code with autopep8...", "info")

        # Check autopep8 availability once before processing any files
        try:
            subprocess.run(["autopep8", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log("✗ autopep8 not found (install with: pip install autopep8)", "error")
            return 0

        count = self._run_per_file(self._format_code_file, paths, "format")
        if count > 0:
            self.log(f"✓ Formatted {count} files", "success")
        return count

    # -------------------------------------------------------------------------
    # File discovery
    # -------------------------------------------------------------------------

    def get_files(self, paths: List[str]) -> List[Path]:
        """Get list of files to process."""
        if not paths:
            # Default to all tracked files
            try:
                result = subprocess.run(
                    ["git", "ls-files"],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return [self.repo_root / p for p in result.stdout.strip().split("\n") if p]
            except subprocess.CalledProcessError:
                # Fallback to all Python files
                return list(self.repo_root.rglob("*.py"))

        result: List[Path] = []
        for path_str in paths:
            path = Path(path_str)
            if not path.is_absolute():
                path = self.repo_root / path

            if path.is_file():
                result.append(path)
            elif path.is_dir():
                result.extend(path.rglob("*.py"))

        return result


def configure_logging(quiet: bool) -> None:
    """Configure logging with sensible defaults."""
    level = logging.ERROR if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Auto-fix quality issues in Transformation Portal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to fix (default: all tracked files)",
    )
    parser.add_argument(
        "--fix-all",
        action="store_true",
        help="Fix all issues automatically",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes",
    )
    parser.add_argument(
        "--whitespace",
        action="store_true",
        help="Fix trailing whitespace only",
    )
    parser.add_argument(
        "--imports",
        action="store_true",
        help="Fix import issues only",
    )
    parser.add_argument(
        "--format",
        action="store_true",
        help="Format code with autopep8",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes for per-file fixes (default: 1)",
    )

    args = parser.parse_args(argv)
    configure_logging(args.quiet)

    fixer = QualityFixer(dry_run=args.dry_run, verbose=not args.quiet, jobs=args.jobs)
    files = fixer.get_files(args.paths)

    fixer.log("╔════════════════════════════════════════════╗", "info")
    fixer.log("║  Transformation Portal - Quality Fixer    ║", "info")
    fixer.log("╚════════════════════════════════════════════╝", "info")

    if args.dry_run:
        fixer.log("\n[DRY-RUN MODE] No changes will be made\n", "warning")

    fixer.log(f"Processing {len(files)} files...", "info")

    total_fixed = 0

    # Fix based on flags
    if args.whitespace or args.fix_all or not any([args.imports, args.format]):
        total_fixed += fixer.fix_trailing_whitespace(files)

    if args.imports or args.fix_all:
        total_fixed += fixer.fix_imports(files)

    if args.format or args.fix_all:
        total_fixed += fixer.format_code(files)

    # Summary
    fixer.log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")
    if total_fixed > 0:
        fixer.log(f"✓ Fixed {total_fixed} issues in {len(fixer.fixed_files)} files", "success")
        if not args.dry_run:
            fixer.log("\n💡 Run 'git diff' to review changes", "info")
    else:
        fixer.log("✓ No issues found!", "success")


if __name__ == "__main__":
    main()
