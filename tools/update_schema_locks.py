#!/usr/bin/env python3
"""
Deterministic schema lockfile updater for evalsuite contracts.

Rewrites docs/contracts/SCHEMA_LOCKS.sha256 in canonical sorted order.

Locks only:
  docs/schemas/evalsuite/**/*.json
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = REPO_ROOT / "docs" / "schemas" / "evalsuite"
LOCKFILE = REPO_ROOT / "docs" / "contracts" / "SCHEMA_LOCKS.sha256"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not SCHEMA_ROOT.exists():
        raise SystemExit(f"Schema root not found: {SCHEMA_ROOT}")

    schema_files = [p for p in SCHEMA_ROOT.rglob("*.json") if p.is_file()]
    if not schema_files:
        raise SystemExit(f"No schema JSON files found under: {SCHEMA_ROOT}")

    # Canonical ordering by repo-relative path
    schema_files_sorted = sorted(
        schema_files,
        key=lambda p: p.relative_to(REPO_ROOT).as_posix(),
    )

    lines = [
        "# Auto-generated. Do not edit manually.",
        "# Update using: python tools/update_schema_locks.py",
        "# Scope: docs/schemas/evalsuite/**/*.json",
        "",
    ]

    for path in schema_files_sorted:
        rel = path.relative_to(REPO_ROOT).as_posix()
        digest = sha256_file(path)
        lines.append(f"{digest}  {rel}")

    LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    LOCKFILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Updated lockfile: {LOCKFILE}")
    print(f"Locked {len(schema_files_sorted)} schema file(s).")


if __name__ == "__main__":
    main()
