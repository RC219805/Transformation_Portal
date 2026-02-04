#!/usr/bin/env python3
"""Add deprecation notices to duplicate documentation files.

Part of DOC-001: Documentation consolidation.
"""

from pathlib import Path
from datetime import datetime, timedelta

# Deprecation date: 30 days from now
deprecation_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

# Duplicates to deprecate: (file, canonical_replacement)
DUPLICATES = [
    ("docs/CODEBASE_QUALITY_STANDARDS.md", "CODE_QUALITY_STANDARDS.md"),
    ("docs/CODE_QUALITY_BASELINE.md", "CODE_QUALITY_STANDARDS.md"),
    ("docs/CODE_QUALITY_SYSTEM.md", "CODE_QUALITY_STANDARDS.md"),
    ("docs/QUALITY_CONTROL_SYSTEM.md", "CODE_QUALITY_STANDARDS.md"),
    ("docs/ARCHITECTURAL_CONTEXT_INTEGRATION.md", "ARCHITECTURE.md"),
]

DEPRECATION_TEMPLATE = """> ⚠️ **DEPRECATED**
>
> This document has been superseded by [{canonical}](docs/{canonical}).
> Please use that document instead. This file will be removed on {date}.

"""

def add_deprecation_notice(filepath: str, canonical: str):
    """Add deprecation notice to top of file."""
    path = Path(filepath)
    if not path.exists():
        print(f"Skip {filepath} (not found)")
        return
    
    content = path.read_text()
    
    # Check if already deprecated
    if "DEPRECATED" in content[:200]:
        print(f"Skip {filepath} (already deprecated)")
        return
    
    # Add notice
    notice = DEPRECATION_TEMPLATE.format(
        canonical=canonical,
        date=deprecation_date
    )
    new_content = notice + content
    path.write_text(new_content)
    print(f"✓ Deprecated {filepath} → {canonical}")


if __name__ == "__main__":
    for filepath, canonical in DUPLICATES:
        add_deprecation_notice(filepath, canonical)
    
    print(f"\nDeprecated {len(DUPLICATES)} files")
    print(f"Removal scheduled for: {deprecation_date}")
