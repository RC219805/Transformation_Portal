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

Exit Codes:
  0 - No vulnerable imports found (success)
  1 - Vulnerable imports detected (failure)
"""

import ast
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
        # Skip vendored basicsr_tp code
        if "basicsr_tp" in str(py_file):
            continue

        # Skip git, cache, and virtual environment directories
        if any(skip in str(py_file) for skip in [".git", "__pycache__", ".venv", "venv", ".tox"]):
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()

            # Parse the file using AST for accurate import detection
            try:
                tree = ast.parse(content, filename=str(py_file))

                for node in ast.walk(tree):
                    # Check for "from basicsr import ..." statements
                    if isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith('basicsr') and 'basicsr_tp' not in node.module:
                            line_no = node.lineno
                            # Get the actual line content for reporting
                            line_content = lines[line_no - 1] if line_no <= len(lines) else ""
                            violations.append((py_file, line_no, line_content))

                    # Check for "import basicsr" statements
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith('basicsr') and 'basicsr_tp' not in alias.name:
                                line_no = node.lineno
                                line_content = lines[line_no - 1] if line_no <= len(lines) else ""
                                violations.append((py_file, line_no, line_content))

            except SyntaxError:
                # If file has syntax errors, fall back to string matching
                # (better to have false positives than miss actual imports)
                for line_no, line in enumerate(lines, start=1):
                    line_stripped = line.strip()
                    if ("from basicsr" in line or "import basicsr" in line) and "basicsr_tp" not in line:
                        if not line_stripped.startswith("#"):
                            violations.append((py_file, line_no, line))

        except (UnicodeDecodeError, PermissionError):
            # Skip files that can't be read
            continue
        except (UnicodeDecodeError, PermissionError):
            # Skip files that can't be read
            continue

    return violations


def main():
    """Main verification function."""
    repo_root = Path(__file__).parent
    print("=" * 70)
    print("VERIFYING: No vulnerable basicsr imports")
    print("=" * 70)
    print(f"Scanning: {repo_root}")
    print()

    violations = find_basicsr_imports(repo_root)

    if not violations:
        print("✅ SUCCESS: No vulnerable basicsr imports found!")
        print()
        print("All imports use the secure basicsr_tp vendored package.")
        return 0

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


if __name__ == "__main__":
    sys.exit(main())
