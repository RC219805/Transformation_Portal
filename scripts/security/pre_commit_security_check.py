#!/usr/bin/env python3
"""
Pre-Commit Security Check
==========================
Blocks commits containing security-sensitive artifacts or malicious patterns.

This script prevents:
1. Bidirectional Unicode characters (Trojan Source attacks)
2. Sensitive files (.bash_history, .env, .pem, keys, etc.)
3. Large binary files above threshold
4. Output directories and artifacts

Usage:
    # As pre-commit hook
    python scripts/security/pre_commit_security_check.py

    # Manual check on specific files
    python scripts/security/pre_commit_security_check.py file1.py file2.sh

    # Check all staged files
    git diff --cached --name-only | xargs python scripts/security/pre_commit_security_check.py

Exit codes:
    0: All checks passed
    1: Security violations found

Author: Transformation Portal Security Team
Version: 1.0.0
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Bidirectional Unicode control characters that can be used in Trojan Source attacks
BIDI_CHARS = {
    "\u202a": "LEFT-TO-RIGHT EMBEDDING (LRE)",
    "\u202b": "RIGHT-TO-LEFT EMBEDDING (RLE)",
    "\u202c": "POP DIRECTIONAL FORMATTING (PDF)",
    "\u202d": "LEFT-TO-RIGHT OVERRIDE (LRO)",
    "\u202e": "RIGHT-TO-LEFT OVERRIDE (RLO)",
    "\u2066": "LEFT-TO-RIGHT ISOLATE (LRI)",
    "\u2067": "RIGHT-TO-LEFT ISOLATE (RLI)",
    "\u2068": "FIRST STRONG ISOLATE (FSI)",
    "\u2069": "POP DIRECTIONAL ISOLATE (PDI)",
}

# Sensitive file patterns that should never be committed
SENSITIVE_PATTERNS = [
    # Shell history files
    r"\.bash_history$",
    r"\.zsh_history$",
    r"\.sh_history$",
    r"\.history$",
    # Credentials and keys
    r"\.pem$",
    r"\.key$",
    r"\.p12$",
    r"\.pfx$",
    r"\.jks$",
    r"\.keystore$",
    r"^id_rsa$",
    r"^id_dsa$",
    r"^id_ecdsa$",
    r"^id_ed25519$",
    # Environment and config files with secrets
    r"\.env$",
    r"\.env\.local$",
    r"\.env\.production$",
    # AWS and cloud credentials
    r"\.aws/credentials$",
    r"\.aws/config$",
    # Build artifacts
    r"^PKG-INFO$",
    r"^MANIFEST$",
]

# Output directory patterns that shouldn't be committed
OUTPUT_DIR_PATTERNS = [
    r"output_.*/",
    r".*_outputs/",
    r"phase\d+_task\d+_outputs/",
    r"sweep_runs/",
    r"benchmarks_.*/",
]

# File extensions for code/config that should be checked for bidi chars
CODE_EXTENSIONS = {
    ".py",
    ".sh",
    ".bash",
    ".zsh",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".cfg",
    ".conf",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".java",
    ".kt",
    ".rb",
    ".php",
}

# Maximum file size (in bytes) for non-LFS files (5MB default)
MAX_FILE_SIZE = 5 * 1024 * 1024


class SecurityViolation:
    """Represents a security violation found during pre-commit check."""

    def __init__(self, filepath: str, violation_type: str, details: str, line_number: int = None):
        self.filepath = filepath
        self.violation_type = violation_type
        self.details = details
        self.line_number = line_number

    def __str__(self) -> str:
        location = f"{self.filepath}:{self.line_number}" if self.line_number else self.filepath
        return f"[{self.violation_type}] {location}: {self.details}"


def check_bidi_unicode(filepath: Path) -> List[SecurityViolation]:
    """Check for bidirectional Unicode characters in code/config files."""
    violations = []

    if filepath.suffix not in CODE_EXTENSIONS:
        return violations

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                for char, name in BIDI_CHARS.items():
                    if char in line:
                        violations.append(
                            SecurityViolation(str(filepath), "BIDI_UNICODE", f"Contains {name} character", line_num)
                        )
    except (OSError, UnicodeDecodeError) as e:
        # Skip binary files or files we can't read
        pass

    return violations


def check_sensitive_file(filepath: Path) -> List[SecurityViolation]:
    """Check if file matches sensitive file patterns."""
    violations = []

    filename = filepath.name
    path_str = str(filepath)

    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, filename) or re.search(pattern, path_str):
            violations.append(SecurityViolation(str(filepath), "SENSITIVE_FILE", f"Matches sensitive file pattern: {pattern}"))

    return violations


def check_output_directory(filepath: Path) -> List[SecurityViolation]:
    """Check if file is in an output directory that shouldn't be committed."""
    violations = []

    path_str = str(filepath)

    for pattern in OUTPUT_DIR_PATTERNS:
        if re.search(pattern, path_str):
            violations.append(SecurityViolation(str(filepath), "OUTPUT_ARTIFACT", f"File in output directory: {pattern}"))

    return violations


def check_file_size(filepath: Path) -> List[SecurityViolation]:
    """Check if file is too large for git (should use LFS)."""
    violations = []

    try:
        size = filepath.stat().st_size
        if size > MAX_FILE_SIZE:
            size_mb = size / (1024 * 1024)
            violations.append(
                SecurityViolation(
                    str(filepath),
                    "LARGE_FILE",
                    f"File size {size_mb:.1f}MB exceeds {MAX_FILE_SIZE / (1024 * 1024)}MB threshold. Use Git LFS.",
                )
            )
    except OSError:
        pass

    return violations


def check_file(filepath: Path) -> List[SecurityViolation]:
    """Run all security checks on a file."""
    violations = []

    if not filepath.exists():
        return violations

    if filepath.is_dir():
        return violations

    # Run all checks
    violations.extend(check_sensitive_file(filepath))
    violations.extend(check_output_directory(filepath))
    violations.extend(check_file_size(filepath))
    violations.extend(check_bidi_unicode(filepath))

    return violations


def get_staged_files() -> List[Path]:
    """Get list of files staged for commit."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"], capture_output=True, text=True, check=True
        )
        files = [Path(f.strip()) for f in result.stdout.splitlines() if f.strip()]
        return files
    except subprocess.CalledProcessError:
        return []


def main(files: List[str] = None) -> int:
    """Main entry point for pre-commit security check."""

    # If files provided as arguments, check those; otherwise check staged files
    if files:
        files_to_check = [Path(f) for f in files]
    else:
        files_to_check = get_staged_files()

    if not files_to_check:
        print("✓ No files to check")
        return 0

    all_violations = []

    for filepath in files_to_check:
        violations = check_file(filepath)
        all_violations.extend(violations)

    if all_violations:
        print("❌ SECURITY VIOLATIONS DETECTED")
        print("=" * 80)

        # Group violations by type
        by_type = {}
        for v in all_violations:
            by_type.setdefault(v.violation_type, []).append(v)

        for vtype, violations in sorted(by_type.items()):
            print(f"\n{vtype} ({len(violations)} violations):")
            for v in violations:
                print(f"  {v}")

        print("\n" + "=" * 80)
        print(f"Total violations: {len(all_violations)}")
        print("\nREMEDIATION:")
        print("1. Remove sensitive files from commit")
        print("2. Add sensitive files to .gitignore")
        print("3. Use Git LFS for large files")
        print("4. Remove bidirectional Unicode characters from code")
        print("5. Move output artifacts to proper storage location")

        return 1

    print(f"✓ All security checks passed ({len(files_to_check)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else None))
