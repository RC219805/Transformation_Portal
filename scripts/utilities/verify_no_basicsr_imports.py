#!/usr/bin/env python3
"""
Verification Script: No BasicSR Imports
========================================

This script verifies that no Python files in the repository import from
the vulnerable basicsr package. All imports should use basicsr_tp instead.

Security Advisory: CVE-2024-27763
- BasicSR ≤ 1.4.2 has command injection vulnerability
- We vendor basicsr_tp as a secure replacement
- This script ensures no code uses the vulnerable package

Usage:
  python verify_no_basicsr_imports.py              # Check imports only
  python verify_no_basicsr_imports.py --check-pkg  # Also check if package installed

Exit Codes:
  0 - No vulnerable imports found (success)
  1 - Vulnerable imports detected (failure)
  2 - Vulnerable package is installed (failure, with --check-pkg)
"""

import ast
import subprocess
import sys
from pathlib import Path


def find_basicsr_imports(root_dir: Path) -> list[tuple[Path, int, str]]:
    """
    Find all imports from basicsr (excluding basicsr_tp) using AST parsing.

    This uses Python's ast module for accurate import detection, avoiding
    false positives from string literals, comments, or documentation.

    Returns:
        List of (file_path, line_number, line_content) tuples
    """
    violations = []
    python_files = root_dir.rglob("*.py")

    for py_file in python_files:
        file_violations = _check_file_for_basicsr_imports(py_file)
        violations.extend(file_violations)

    return violations


def _should_skip_file(py_file: Path) -> bool:
    """Check if a file should be skipped during import scanning."""
    # Skip vendored basicsr_tp code
    if "basicsr_tp" in str(py_file):
        return True

    # Skip git, cache, and virtual environment directories
    skip_dirs = [".git", "__pycache__", ".venv", "venv", ".tox"]
    if any(skip in str(py_file) for skip in skip_dirs):
        return True

    return False


def _check_file_for_basicsr_imports(py_file: Path) -> list[tuple[Path, int, str]]:
    """Check a single file for basicsr imports."""
    if _should_skip_file(py_file):
        return []

    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.splitlines()
    except (UnicodeDecodeError, PermissionError):
        return []

    # Parse the file using AST for accurate import detection
    try:
        tree = ast.parse(content, filename=str(py_file))
        return _find_imports_in_ast(tree, py_file, lines)
    except SyntaxError:
        # Fall back to string matching for files with syntax errors
        return _find_imports_via_string_matching(py_file, lines)


def _find_imports_in_ast(tree: ast.AST, py_file: Path, lines: list[str]) -> list[tuple[Path, int, str]]:
    """Find basicsr imports using AST parsing."""
    violations = []

    for node in ast.walk(tree):
        node_violations = _check_ast_node_for_import(node, py_file, lines)
        violations.extend(node_violations)

    return violations


# Maximum directory levels to traverse when searching for repository root
_MAX_TRAVERSAL_DEPTH = 10


def _is_vulnerable_basicsr_import(module_name: str) -> bool:
    """Check if a module name refers to the vulnerable basicsr package.

    This uses exact matching to avoid false positives with similarly-named packages
    like 'basicsr_new', 'basicsrv2', etc.

    Args:
        module_name: The module name from an import statement (e.g., 'basicsr.archs')

    Returns:
        True if this is a vulnerable basicsr import, False otherwise.
    """
    if not module_name:
        return False

    # Split module path and check if first component is exactly 'basicsr'
    # This matches: 'basicsr', 'basicsr.archs', 'basicsr.utils.dist_util'
    # But NOT: 'basicsr_tp', 'basicsr_new', 'basicsrv2'
    parts = module_name.split('.')
    first_part = parts[0]

    # Must be exactly 'basicsr' (the vulnerable package)
    # and NOT 'basicsr_tp' (our secure vendored replacement)
    return first_part == 'basicsr'


def _check_ast_node_for_import(node: ast.AST, py_file: Path, lines: list[str]) -> list[tuple[Path, int, str]]:
    """Check a single AST node for basicsr imports.

    Returns a list of violations to handle cases like `import basicsr, basicsr.models`
    where multiple vulnerable imports exist on the same line.
    """
    violations = []

    # Check for "from basicsr import ..." statements
    if isinstance(node, ast.ImportFrom):
        if _is_vulnerable_basicsr_import(node.module):
            line_no = node.lineno
            line_content = lines[line_no - 1] if line_no <= len(lines) else ""
            violations.append((py_file, line_no, line_content))

    # Check for "import basicsr" statements - check ALL aliases
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if _is_vulnerable_basicsr_import(alias.name):
                line_no = node.lineno
                line_content = lines[line_no - 1] if line_no <= len(lines) else ""
                violations.append((py_file, line_no, line_content))

    return violations


def _find_imports_via_string_matching(py_file: Path, lines: list[str]) -> list[tuple[Path, int, str]]:
    """Fall back to string matching for files with syntax errors.

    Uses word boundary matching to avoid false positives with similarly-named packages.
    """
    import re
    violations = []

    # Pattern matches 'from basicsr' or 'import basicsr' as whole words
    # but NOT 'from basicsr_tp', 'import basicsr_new', 'basicsr-fork', etc.
    # The negative lookahead (?![_-]) ensures we don't match basicsr_ or basicsr- prefixed packages
    pattern = re.compile(r'\b(from|import)\s+basicsr\b(?![_-])')

    for line_no, line in enumerate(lines, start=1):
        line_stripped = line.strip()
        if not line_stripped.startswith("#") and pattern.search(line):
            violations.append((py_file, line_no, line))

    return violations


def _find_repo_root() -> Path:
    """Find repository root by looking for .git directory."""
    script_path = Path(__file__).resolve()
    current = script_path.parent

    # Walk up directory tree looking for .git directory
    for _ in range(_MAX_TRAVERSAL_DEPTH):
        if (current / '.git').exists():
            return current
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent

    # Fallback: use path traversal from script location
    # scripts/utilities/verify_no_basicsr_imports.py -> repository root
    return script_path.parent.parent.parent


def check_basicsr_installed() -> bool:
    """Check if the vulnerable basicsr package is installed.

    Returns:
        True if basicsr is installed (security violation), False otherwise.
    """
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', 'basicsr'],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, OSError):
        # If we can't check, assume it's not installed
        return False


def main():
    """Main verification function."""
    check_pkg = '--check-pkg' in sys.argv

    repo_root = _find_repo_root()
    print("=" * 70)
    print("VERIFYING: No vulnerable basicsr imports (CVE-2024-27763)")
    print("=" * 70)
    print(f"Scanning: {repo_root}")
    if check_pkg:
        print("Package installation check: ENABLED")
    print()

    # Check 1: Verify no basicsr imports in code
    violations = find_basicsr_imports(repo_root)

    if violations:
        print(f"❌ FAILURE: Found {len(violations)} vulnerable import(s):")
        print()

        for file_path, line_no, line_content in violations:
            rel_path = file_path.relative_to(repo_root)
            print(f"  {rel_path}:{line_no}")
            print(f"    {line_content}")
            print()

        print("To fix:")
        print("  Replace 'from basicsr' with 'from basicsr_tp'")
        print("  Replace 'import basicsr' with 'import basicsr_tp'")
        print()
        # Example code for fixing imports (using variable to avoid self-detection)
        vulnerable_package = "basicsr"
        print("Example:")
        print("  # OLD (vulnerable):")
        print(f"  from {vulnerable_package}.archs.rrdbnet_arch import RRDBNet")
        print()
        print("  # NEW (secure):")
        print("  from basicsr_tp.archs.rrdbnet_arch import RRDBNet")
        print()
        return 1

    print("✅ Import check passed: No vulnerable basicsr imports found!")
    print("   All imports use the secure basicsr_tp vendored package.")
    print()

    # Check 2: Verify basicsr package is not installed (optional)
    if check_pkg:
        if check_basicsr_installed():
            print("❌ SECURITY VIOLATION: basicsr package is installed!")
            print()
            print("   The vulnerable basicsr package should be blocked by")
            print("   requirements/constraints.txt. If you see this error:")
            print()
            print("   1. Ensure pip install uses: -c requirements/constraints.txt")
            print("   2. Run: pip uninstall basicsr")
            print("   3. Reinstall with constraints: pip install -c requirements/constraints.txt -r requirements.txt")
            print()
            return 2

        print("✅ Package check passed: basicsr is NOT installed")
        print()

    print("=" * 70)
    print("SECURITY VERIFICATION COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
