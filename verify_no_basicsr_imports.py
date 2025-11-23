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

import sys
from pathlib import Path


def find_basicsr_imports(root_dir: Path) -> list[tuple[Path, int, str]]:
    """
    Find all imports from basicsr (excluding basicsr_tp).

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
                for line_no, line in enumerate(f, start=1):
                    line_stripped = line.strip()

                    # Check for imports from basicsr (but not basicsr_tp)
                    if ("from basicsr" in line or "import basicsr" in line) and "basicsr_tp" not in line:
                        # Ignore comments
                        if not line_stripped.startswith("#"):
                            violations.append((py_file, line_no, line.rstrip()))
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
