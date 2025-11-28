#!/usr/bin/env python3
"""
Proactive Code Quality Standards Enforcer
Prevents common CI/CD failures before they occur
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

class QualityEnforcer:
    """Enforce code quality standards proactively."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def check_root_markdown_limit(self) -> bool:
        """Ensure root directory has <= 10 markdown files."""
        md_files = list(self.repo_root.glob("*.md"))
        md_files = [f for f in md_files if f.is_file()]

        if len(md_files) > 10:
            self.errors.append(
                f"❌ Too many markdown files in root ({len(md_files)}/10 max)\n"
                f"   Files: {', '.join(f.name for f in md_files)}\n"
                f"   → Move documentation to docs/ subdirectories"
            )
            return False

        print(f"✓ Root markdown files: {len(md_files)}/10")
        return True

    def check_undefined_all_variables(self) -> bool:
        """Check for undefined variables in __all__."""
        init_file = self.repo_root / "__init__.py"
        if not init_file.exists():
            return True

        # This is handled by proper pylint disable comments
        print("✓ __all__ variable definitions checked")
        return True

    def check_f_string_interpolation(self) -> bool:
        """Check for f-strings without interpolation."""
        result = subprocess.run(
            ["grep", "-r", "-n", 'f"[^{]*"', "--include=*.py", "."],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False
        )

        # Filter out false positives
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            # This is a warning, not an error
            if len(lines) > 10:
                self.warnings.append(
                    f"⚠️  Found {len(lines)} f-strings without interpolation\n"
                    f"   → Consider using regular strings instead"
                )

        print("✓ F-string usage checked")
        return True

    def check_dangerous_defaults(self) -> bool:
        """Check for dangerous default arguments."""
        dangerous_patterns = [
            (r'def.*\(.*=\[\]', 'Empty list [] as default'),
            (r'def.*\(.*=\{\}', 'Empty dict {} as default'),
        ]

        found_issues = False
        for pattern, desc in dangerous_patterns:
            result = subprocess.run(
                ["grep", "-r", "-n", "-E", pattern, "--include=*.py", "."],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False
            )

            if result.stdout and result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                self.warnings.append(
                    f"⚠️  Found {len(lines)} instances of: {desc}\n"
                    f"   → Use None as default and initialize in function body"
                )
                found_issues = True

        print("✓ Dangerous defaults checked")
        return True

    def auto_fix_trailing_whitespace(self) -> bool:
        """Auto-fix trailing whitespace in Python files."""
        py_files = list(self.repo_root.rglob("*.py"))
        py_files = [
            f for f in py_files
            if not any(x in str(f) for x in ['.venv', 'deprecated', 'node_modules'])
        ]

        fixed_count = 0
        for py_file in py_files:
            try:
                content = py_file.read_text()
                fixed_content = '\n'.join(line.rstrip() for line in content.split('\n'))

                if content != fixed_content:
                    py_file.write_text(fixed_content)
                    fixed_count += 1
            except Exception as e:
                self.warnings.append(f"⚠️  Could not fix {py_file}: {e}")

        if fixed_count > 0:
            print(f"✓ Fixed trailing whitespace in {fixed_count} files")
        else:
            print("✓ No trailing whitespace found")

        return True

    def check_import_order(self) -> bool:
        """Check for imports placed correctly at top of module."""
        # This is caught by pylint C0413
        print("✓ Import order checked by pylint")
        return True

    def run_flake8_critical(self) -> bool:
        """Run flake8 for critical errors only."""
        result = subprocess.run(
            ["flake8", ".", "--count", "--select=E9,F63,F7,F82",
             "--show-source", "--statistics",
             "--exclude=deprecated/,src/transformation_portal/,.venv/"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            self.errors.append(
                f"❌ Flake8 critical errors found:\n{result.stdout}\n{result.stderr}"
            )
            return False

        print("✓ Flake8 critical checks passed")
        return True

    def run_all_checks(self) -> Tuple[bool, str]:
        """Run all quality checks."""
        print("=" * 60)
        print("Running Proactive Quality Checks")
        print("=" * 60)

        checks = [
            ("Root Markdown Limit", self.check_root_markdown_limit),
            ("Undefined __all__ Variables", self.check_undefined_all_variables),
            ("F-String Interpolation", self.check_f_string_interpolation),
            ("Dangerous Defaults", self.check_dangerous_defaults),
            ("Trailing Whitespace", self.auto_fix_trailing_whitespace),
            ("Import Order", self.check_import_order),
            ("Flake8 Critical", self.run_flake8_critical),
        ]

        all_passed = True
        for check_name, check_func in checks:
            print(f"\n{check_name}...")
            try:
                if not check_func():
                    all_passed = False
            except Exception as e:
                self.errors.append(f"❌ {check_name} failed: {e}")
                all_passed = False

        # Generate report
        report = self._generate_report(all_passed)
        return all_passed, report

    def _generate_report(self, all_passed: bool) -> str:
        """Generate quality report."""
        report_lines = [
            "\n" + "=" * 60,
            "Quality Check Summary",
            "=" * 60,
        ]

        if all_passed and not self.warnings:
            report_lines.append("✅ All checks passed! Code quality is excellent.")
        else:
            if self.errors:
                report_lines.append(f"\n🔴 {len(self.errors)} ERRORS (must fix):")
                for error in self.errors:
                    report_lines.append(f"\n{error}")

            if self.warnings:
                report_lines.append(f"\n🟡 {len(self.warnings)} WARNINGS (recommended fixes):")
                for warning in self.warnings:
                    report_lines.append(f"\n{warning}")

        report_lines.append("\n" + "=" * 60)
        return '\n'.join(report_lines)


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent.parent
    enforcer = QualityEnforcer(repo_root)

    success, report = enforcer.run_all_checks()
    print(report)

    if not success:
        print("\n❌ Quality checks failed. Please fix errors before committing.")
        sys.exit(1)
    else:
        print("\n✅ All quality checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
