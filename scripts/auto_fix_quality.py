#!/usr/bin/env python3
"""
Auto-fix utility for common quality issues in Transformation Portal.

Fixes:
- Trailing whitespace
- Import sorting (optional)
- Common flake8 issues
- Format code with autopep8
- Organize markdown files

Usage:
    python scripts/auto_fix_quality.py [--fix-all] [--dry-run] [paths...]

Options:
    --fix-all       Fix all issues automatically (no prompts)
    --dry-run       Show what would be fixed without making changes
    --whitespace    Fix trailing whitespace only
    --imports       Fix import issues only
    --format        Format code with autopep8
    paths...        Specific files or directories to fix (default: all)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Set


class QualityFixer:
    """Auto-fix common quality issues."""

    def __init__(self, dry_run: bool = False, verbose: bool = True):
        self.dry_run = dry_run
        self.verbose = verbose
        self.fixed_files: Set[Path] = set()
        self.repo_root = self._get_repo_root()

    def _get_repo_root(self) -> Path:
        """Get repository root directory."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--show-toplevel'],
                capture_output=True,
                text=True,
                check=True
            )
            return Path(result.stdout.strip())
        except subprocess.CalledProcessError:
            return Path.cwd()

    def log(self, message: str, level: str = 'info'):
        """Log message with color."""
        if not self.verbose:
            return

        colors = {
            'info': '\033[0;34m',    # Blue
            'success': '\033[0;32m', # Green
            'warning': '\033[1;33m', # Yellow
            'error': '\033[0;31m',   # Red
            'reset': '\033[0m'
        }
        color = colors.get(level, colors['reset'])
        print(f"{color}{message}{colors['reset']}")

    def fix_trailing_whitespace(self, paths: List[Path]) -> int:
        """Remove trailing whitespace from files."""
        self.log("\n→ Fixing trailing whitespace...", 'info')
        fixed_count = 0

        for path in paths:
            if not path.is_file():
                continue

            # Skip binary files
            try:
                content = path.read_text(encoding='utf-8')
            except (UnicodeDecodeError, PermissionError):
                continue

            # Check for trailing whitespace
            lines = content.splitlines(keepends=True)
            has_trailing = any(line.rstrip() != line.rstrip('\n\r') for line in lines)

            if has_trailing:
                if self.dry_run:
                    self.log(f"  [DRY-RUN] Would fix: {path.relative_to(self.repo_root)}", 'warning')
                else:
                    # Remove trailing whitespace
                    fixed_lines = [line.rstrip() + '\n' for line in lines]
                    # Preserve final newline
                    if lines and not lines[-1].endswith('\n'):
                        fixed_lines[-1] = fixed_lines[-1].rstrip('\n')

                    path.write_text(''.join(fixed_lines), encoding='utf-8')
                    self.log(f"  ✓ Fixed: {path.relative_to(self.repo_root)}", 'success')
                    self.fixed_files.add(path)

                fixed_count += 1

        if fixed_count > 0:
            self.log(f"✓ Fixed trailing whitespace in {fixed_count} files", 'success')
        else:
            self.log("✓ No trailing whitespace found", 'success')

        return fixed_count

    def fix_imports(self, paths: List[Path]) -> int:
        """Fix common import issues in Python files."""
        self.log("\n→ Checking for undefined imports...", 'info')
        fixed_count = 0

        # Common missing imports
        import_fixes = {
            'iio': 'import imageio.v3 as iio',
            'np': 'import numpy as np',
            'pd': 'import pandas as pd',
            'plt': 'import matplotlib.pyplot as plt',
            'cv2': 'import cv2',
            'Image': 'from PIL import Image',
            'Path': 'from pathlib import Path',
        }

        for path in paths:
            if path.suffix != '.py' or not path.is_file():
                continue

            try:
                content = path.read_text(encoding='utf-8')
            except (UnicodeDecodeError, PermissionError):
                continue

            # Check for undefined names
            undefined = []
            for name, import_line in import_fixes.items():
                # Check if name is used but not imported
                if re.search(rf'\b{name}\b', content) and import_line not in content:
                    undefined.append((name, import_line))

            if undefined:
                if self.dry_run:
                    self.log(f"  [DRY-RUN] Would add imports to: {path.relative_to(self.repo_root)}", 'warning')
                    for name, import_line in undefined:
                        self.log(f"    + {import_line}", 'warning')
                else:
                    # Add missing imports after existing imports
                    lines = content.splitlines(keepends=True)
                    import_section_end = 0

                    # Find end of import section
                    for i, line in enumerate(lines):
                        if line.strip().startswith(('import ', 'from ')):
                            import_section_end = i + 1

                    # Insert missing imports
                    for name, import_line in reversed(undefined):
                        lines.insert(import_section_end, import_line + '\n')

                    path.write_text(''.join(lines), encoding='utf-8')
                    self.log(f"  ✓ Fixed imports in: {path.relative_to(self.repo_root)}", 'success')
                    for name, import_line in undefined:
                        self.log(f"    + {import_line}", 'success')
                    self.fixed_files.add(path)

                fixed_count += 1

        if fixed_count > 0:
            self.log(f"✓ Fixed imports in {fixed_count} files", 'success')
        else:
            self.log("✓ No import issues found", 'success')

        return fixed_count

    def format_code(self, paths: List[Path]) -> int:
        """Format Python code with autopep8."""
        self.log("\n→ Formatting code with autopep8...", 'info')

        # Check if autopep8 is available
        try:
            subprocess.run(['autopep8', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log("✗ autopep8 not found (install with: pip install autopep8)", 'error')
            return 0

        py_files = [p for p in paths if p.suffix == '.py' and p.is_file()]

        if not py_files:
            self.log("⚠ No Python files to format", 'warning')
            return 0

        fixed_count = 0
        for path in py_files:
            cmd = [
                'autopep8',
                '--in-place',
                '--max-line-length=127',
                '--aggressive',
                '--aggressive',
                str(path)
            ]

            if self.dry_run:
                self.log(f"  [DRY-RUN] Would format: {path.relative_to(self.repo_root)}", 'warning')
            else:
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    self.log(f"  ✓ Formatted: {path.relative_to(self.repo_root)}", 'success')
                    self.fixed_files.add(path)
                    fixed_count += 1
                except subprocess.CalledProcessError as e:
                    self.log(f"  ✗ Failed to format {path.name}: {e}", 'error')

        if fixed_count > 0:
            self.log(f"✓ Formatted {fixed_count} files", 'success')

        return fixed_count

    def get_files(self, paths: List[str]) -> List[Path]:
        """Get list of files to process."""
        if not paths:
            # Default to all tracked files
            try:
                result = subprocess.run(
                    ['git', 'ls-files'],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    check=True
                )
                return [self.repo_root / p for p in result.stdout.strip().split('\n') if p]
            except subprocess.CalledProcessError:
                # Fallback to all Python files
                return list(self.repo_root.rglob('*.py'))

        # Process specified paths
        result = []
        for path_str in paths:
            path = Path(path_str)
            if not path.is_absolute():
                path = self.repo_root / path

            if path.is_file():
                result.append(path)
            elif path.is_dir():
                result.extend(path.rglob('*.py'))

        return result


def main():
    parser = argparse.ArgumentParser(
        description='Auto-fix quality issues in Transformation Portal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'paths',
        nargs='*',
        help='Files or directories to fix (default: all tracked files)'
    )
    parser.add_argument(
        '--fix-all',
        action='store_true',
        help='Fix all issues automatically'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be fixed without making changes'
    )
    parser.add_argument(
        '--whitespace',
        action='store_true',
        help='Fix trailing whitespace only'
    )
    parser.add_argument(
        '--imports',
        action='store_true',
        help='Fix import issues only'
    )
    parser.add_argument(
        '--format',
        action='store_true',
        help='Format code with autopep8'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output'
    )

    args = parser.parse_args()

    fixer = QualityFixer(dry_run=args.dry_run, verbose=not args.quiet)
    files = fixer.get_files(args.paths)

    fixer.log("╔════════════════════════════════════════════╗", 'info')
    fixer.log("║  Transformation Portal - Quality Fixer    ║", 'info')
    fixer.log("╚════════════════════════════════════════════╝", 'info')

    if args.dry_run:
        fixer.log("\n[DRY-RUN MODE] No changes will be made\n", 'warning')

    fixer.log(f"Processing {len(files)} files...", 'info')

    total_fixed = 0

    # Fix based on flags
    if args.whitespace or args.fix_all or not any([args.imports, args.format]):
        total_fixed += fixer.fix_trailing_whitespace(files)

    if args.imports or args.fix_all:
        total_fixed += fixer.fix_imports(files)

    if args.format or args.fix_all:
        total_fixed += fixer.format_code(files)

    # Summary
    fixer.log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 'info')
    if total_fixed > 0:
        fixer.log(f"✓ Fixed {total_fixed} issues in {len(fixer.fixed_files)} files", 'success')
        if not args.dry_run:
            fixer.log("\n💡 Run 'git diff' to review changes", 'info')
    else:
        fixer.log("✓ No issues found!", 'success')


if __name__ == '__main__':
    main()
