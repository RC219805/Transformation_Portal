#!/usr/bin/env python3
"""
CI guardrail: block known-vulnerable or prohibited packages in requirements.

Aligned with Lux V2 security guidance: avoid basicsr/realesrgan/gfpgan.

Usage:
  python scripts/hardening/check_banned_deps.py
  python scripts/hardening/check_banned_deps.py lux_depth_v2/requirements.txt requirements/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Banned packages (CVE-2024-27763 mitigation)
BANNED_PACKAGES = ("basicsr", "realesrgan", "gfpgan")


def find_requirement_files(paths: list[Path]) -> list[Path]:
    """Find all requirements files in given paths."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob("**/*requirements*.txt")))
        elif p.is_file() and p.name.endswith(".txt"):
            out.append(p)
    return out


def parse_req_lines(path: Path) -> list[str]:
    out: list[str] = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        # strip environment markers; keep left side
        s = s.split(";")[0].strip()
        # remove hashes and extras after spaces
        s = s.split()[0].strip()
        out.append(s.lower())
    return out


def main() -> int:
    banned = {b.lower() for b in BANNED_PACKAGES}

    roots = [Path(p) for p in (sys.argv[1:] or ["lux_depth_v2", "requirements"])]
    files = find_requirement_files(roots)

    if not files:
        print("No requirements files found; skipping.")
        return 0

    violations: list[tuple[Path, str]] = []
    for f in files:
        try:
            for line in parse_req_lines(f):
                pkg = line.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0]
                pkg = pkg.split("[")[0].strip()
                if pkg in banned:
                    violations.append((f, pkg))
        except Exception as e:
            print(f"WARNING: unable to parse {f}: {e}")

    if violations:
        print("BANNED DEPENDENCIES DETECTED:")
        for f, pkg in violations:
            print(f"  - {pkg} in {f}")
        print("\nFix: remove these packages and use lux_depth_v2/requirements-repo.txt as the safe baseline.")
        return 2

    print(f"OK: no banned packages found across {len(files)} requirements files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
