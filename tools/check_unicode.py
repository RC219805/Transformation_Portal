#!/usr/bin/env python3
"""
Pre-commit hook: Check for bidirectional Unicode and format control characters.
Prevents "Trojan Source" style attacks.
"""
import sys
import unicodedata
from pathlib import Path

# Dangerous bidirectional control characters
BIDI_CHARS = {
    "\u202A": "LEFT-TO-RIGHT EMBEDDING",
    "\u202B": "RIGHT-TO-LEFT EMBEDDING",
    "\u202C": "POP DIRECTIONAL FORMATTING",
    "\u202D": "LEFT-TO-RIGHT OVERRIDE",
    "\u202E": "RIGHT-TO-LEFT OVERRIDE",
    "\u2066": "LEFT-TO-RIGHT ISOLATE",
    "\u2067": "RIGHT-TO-LEFT ISOLATE",
    "\u2068": "FIRST STRONG ISOLATE",
    "\u2069": "POP DIRECTIONAL ISOLATE",
}

# Other format control characters (category Cf)
def check_file(filepath: Path) -> list[str]:
    """Check a single file for dangerous Unicode."""
    issues = []
    try:
        content = filepath.read_text(encoding="utf-8")

        for line_num, line in enumerate(content.splitlines(), start=1):
            for col_num, char in enumerate(line, start=1):
                # Check bidirectional overrides
                if char in BIDI_CHARS:
                    issues.append(
                        f"{filepath}:{line_num}:{col_num}: "
                        f"Bidirectional Unicode U+{ord(char):04X} ({BIDI_CHARS[char]})"
                    )

                # Check other format control chars (category Cf)
                elif unicodedata.category(char) == "Cf":
                    name = unicodedata.name(char, "UNKNOWN")
                    issues.append(
                        f"{filepath}:{line_num}:{col_num}: "
                        f"Format control character U+{ord(char):04X} ({name})"
                    )
    except Exception as e:
        issues.append(f"{filepath}: Error reading file: {e}")

    return issues


def main():
    """Check all staged files."""
    # Get staged files from git
    import subprocess

    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Error getting staged files", file=sys.stderr)
        return 1

    files = [
        Path(f)
        for f in result.stdout.strip().split("\n")
        if f and (f.endswith(".py") or f.endswith(".yml") or f.endswith(".yaml") or f.endswith(".md"))
    ]

    if not files:
        return 0

    all_issues = []
    for filepath in files:
        if not filepath.exists():
            continue
        all_issues.extend(check_file(filepath))

    if all_issues:
        print("❌ Found dangerous Unicode characters:", file=sys.stderr)
        for issue in all_issues:
            print(f"  {issue}", file=sys.stderr)
        print("\nThese characters can enable 'Trojan Source' attacks.", file=sys.stderr)
        print("Please remove them before committing.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
