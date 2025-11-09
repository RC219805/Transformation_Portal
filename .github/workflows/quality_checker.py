#!/usr/bin/env python3
"""
Comprehensive Quality Checker for Transformation Portal.
Proactively catches common code quality issues before commit.
"""
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class QualityChecker:
    """Automated quality checking and fixing."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.errors = []
        self.warnings = []

    def check_trailing_whitespace(self) -> bool:
        """Check for trailing whitespace in Python files."""
        print("🔍 Checking for trailing whitespace...")
        files_with_issues = []

        for py_file in self.repo_root.rglob("*.py"):
            # Skip excluded directories
            if any(excluded in str(py_file) for excluded in
                   ['deprecated/', 'src/transformation_portal/', '.venv/', '__pycache__']):
                continue

            with open(py_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.rstrip() != line.rstrip('\n'):
                        files_with_issues.append((py_file, line_num))

        if files_with_issues:
            self.errors.append(f"Found trailing whitespace in {len(files_with_issues)} locations")
            return False

        print("✅ No trailing whitespace found")
        return True

    def check_import_order(self) -> bool:
        """Check that imports are at the top of files."""
        print("🔍 Checking import order...")
        issues = []

        for py_file in self.repo_root.rglob("*.py"):
            if any(excluded in str(py_file) for excluded in
                   ['deprecated/', 'src/transformation_portal/', '.venv/', '__pycache__']):
                continue

            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Find first import
            first_import_line = None
            first_code_line = None

            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped or stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                if stripped.startswith('import ') or stripped.startswith('from '):
                    if first_import_line is None:
                        first_import_line = i
                elif first_code_line is None and not stripped.startswith('#'):
                    first_code_line = i

            # Check if there are imports after code
            if first_import_line and first_code_line and first_code_line < first_import_line:
                issues.append(str(py_file.relative_to(self.repo_root)))

        if issues:
            self.warnings.append(f"Imports after code in {len(issues)} files")
            return True  # Warning, not error

        print("✅ Import order looks good")
        return True

    def check_undefined_names(self) -> bool:
        """Run flake8 to check for critical errors."""
        print("🔍 Running flake8 critical checks...")

        try:
            result = subprocess.run(
                ['flake8', '.', '--count', '--select=E9,F63,F7,F82', '--show-source', '--statistics',
                 '--exclude=deprecated/,src/transformation_portal/,.venv/'],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                self.errors.append(f"Flake8 found critical errors:\n{result.stdout}")
                return False

            print("✅ No critical flake8 errors")
            return True

        except FileNotFoundError:
            self.warnings.append("flake8 not installed, skipping check")
            return True

    def check_markdown_count(self) -> bool:
        """Ensure not too many markdown files in root."""
        print("🔍 Checking markdown file count in root...")

        markdown_files = list(self.repo_root.glob("*.md"))
        if len(markdown_files) > 10:
            self.errors.append(
                f"Too many markdown files in root ({len(markdown_files)}). "
                "Move documentation to docs/"
            )
            return False

        print(f"✅ Markdown file count OK ({len(markdown_files)}/10)")
        return True

    def auto_fix_whitespace(self) -> bool:
        """Automatically fix trailing whitespace."""
        print("🔧 Auto-fixing trailing whitespace...")

        fixed_count = 0
        for py_file in self.repo_root.rglob("*.py"):
            if any(excluded in str(py_file) for excluded in
                   ['deprecated/', 'src/transformation_portal/', '.venv/', '__pycache__']):
                continue

            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            fixed_lines = [line.rstrip() + '\n' if line.endswith('\n') else line.rstrip()
                          for line in lines]

            if lines != fixed_lines:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.writelines(fixed_lines)
                fixed_count += 1

        if fixed_count > 0:
            print(f"✅ Fixed trailing whitespace in {fixed_count} files")
        return True

    def run_all_checks(self, auto_fix: bool = False) -> bool:
        """Run all quality checks."""
        print("\n" + "="*60)
        print("🚀 Running Quality Checks")
        print("="*60 + "\n")

        if auto_fix:
            self.auto_fix_whitespace()

        checks = [
            self.check_trailing_whitespace(),
            self.check_import_order(),
            self.check_undefined_names(),
            self.check_markdown_count(),
        ]

        print("\n" + "="*60)
        if all(checks):
            print("✅ All quality checks passed!")
            print("="*60 + "\n")
            return True
        else:
            print("❌ Quality checks failed!")
            if self.errors:
                print("\n❌ ERRORS:")
                for error in self.errors:
                    print(f"  - {error}")
            if self.warnings:
                print("\n⚠️  WARNINGS:")
                for warning in self.warnings:
                    print(f"  - {warning}")
            print("="*60 + "\n")
            return False


def main():
    """Run quality checker."""
    repo_root = Path(__file__).parent.parent.parent
    checker = QualityChecker(repo_root)

    auto_fix = '--fix' in sys.argv
    success = checker.run_all_checks(auto_fix=auto_fix)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
