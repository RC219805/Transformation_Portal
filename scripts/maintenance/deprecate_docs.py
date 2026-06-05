#!/usr/bin/env python3
"""Add deprecation notices to duplicate documentation files.

Part of DOC-001: Documentation consolidation.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

# Deprecation date: 30 days from now
deprecation_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

# Duplicates to deprecate: (file, canonical_replacement)
DUPLICATES = [
    (
        "docs/guides/CODEBASE_QUALITY_STANDARDS.md",
        "docs/guides/CODE_QUALITY_STANDARDS.md",
    ),
    (
        "docs/guides/CODE_QUALITY_BASELINE.md",
        "docs/guides/CODE_QUALITY_STANDARDS.md",
    ),
    (
        "docs/guides/CODE_QUALITY_SYSTEM.md",
        "docs/guides/CODE_QUALITY_STANDARDS.md",
    ),
    (
        "docs/guides/QUALITY_CONTROL_SYSTEM.md",
        "docs/guides/CODE_QUALITY_STANDARDS.md",
    ),
    (
        "docs/architecture/ARCHITECTURE_PHILOSOPHY.md",
        "docs/architecture/ARCHITECTURE.md",
    ),
]

DEPRECATION_TEMPLATE = """> ⚠️ **DEPRECATED**
>
> This document has been superseded by [{canonical_name}]({relative_link}).
> Please use that document instead. This file will be removed on {date}.

"""


def add_deprecation_notice(filepath: str, canonical: str):
    """Add deprecation notice to top of file with correct relative link."""
    path = Path(filepath)
    if not path.exists():
        print(f"Skip {filepath} (not found)")
        return

    content = path.read_text(encoding="utf-8")

    # Check if already deprecated
    if "DEPRECATED" in content[:200]:
        print(f"Skip {filepath} (already deprecated)")
        return

    # Compute relative path from deprecated doc to canonical
    deprecated_dir = path.parent
    canonical_path = Path(canonical)
    relative_path = os.path.relpath(canonical_path, start=deprecated_dir)
    # Normalize to forward slashes for markdown
    relative_link = relative_path.replace(os.sep, "/")
    canonical_name = canonical_path.name

    # Add notice
    notice = DEPRECATION_TEMPLATE.format(
        canonical_name=canonical_name,
        relative_link=relative_link,
        date=deprecation_date,
    )
    new_content = notice + content
    path.write_text(new_content, encoding="utf-8")
    print(f"✓ Deprecated {filepath} → {relative_link}")


def main() -> None:
    """Add deprecation notices for the configured duplicate docs."""
    for filepath, canonical in DUPLICATES:
        add_deprecation_notice(filepath, canonical)

    print(f"\nDeprecated {len(DUPLICATES)} files")
    print(f"Removal scheduled for: {deprecation_date}")


if __name__ == "__main__":
    main()
