#!/usr/bin/env python3
"""Pre-commit Security Check - Bidirectional Unicode detection.

This script checks files for bidirectional Unicode characters that could
be used for Trojan Source attacks.
"""

import sys


def check_file(filepath):
    """Check a single file for bidirectional Unicode characters."""
    # Bidirectional override characters
    bidi_chars = [
        "\u202a",  # LEFT-TO-RIGHT EMBEDDING
        "\u202b",  # RIGHT-TO-LEFT EMBEDDING
        "\u202c",  # POP DIRECTIONAL FORMATTING
        "\u202d",  # LEFT-TO-RIGHT OVERRIDE
        "\u202e",  # RIGHT-TO-LEFT OVERRIDE
        "\u2066",  # LEFT-TO-RIGHT ISOLATE
        "\u2067",  # RIGHT-TO-LEFT ISOLATE
        "\u2068",  # FIRST STRONG ISOLATE
        "\u2069",  # POP DIRECTIONAL ISOLATE
    ]

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            for char in bidi_chars:
                if char in content:
                    return False, f"Bidirectional Unicode found: {repr(char)}"
        return True, None
    except Exception as e:
        # Ignore files we can't read
        return True, None


def main():
    """Check all provided files for security issues."""
    if len(sys.argv) < 2:
        print("✅ No files to check")
        return 0

    files_to_check = sys.argv[1:]
    issues_found = []

    for filepath in files_to_check:
        passed, error = check_file(filepath)
        if not passed:
            issues_found.append((filepath, error))

    if issues_found:
        print("❌ Security issues found:")
        for filepath, error in issues_found:
            print(f"  {filepath}: {error}")
        return 1

    print(f"✅ Checked {len(files_to_check)} files - no security issues found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
