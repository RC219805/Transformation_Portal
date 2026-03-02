#!/usr/bin/env python3
"""Fail when pip-tools cache artifacts are tracked in git."""

from __future__ import annotations

import subprocess
import sys

CACHE_PATHSPEC = "requirements/.pip-tools-cache"


def tracked_cache_files() -> list[str]:
    """Return tracked files under the pip-tools cache directory."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", CACHE_PATHSPEC],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"git ls-files failed ({result.returncode}): {stderr}")

    return sorted(path for path in result.stdout.split("\0") if path)


def main() -> int:
    try:
        tracked = tracked_cache_files()
    except FileNotFoundError:
        print("ERROR: git executable not found", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not tracked:
        print("pip-tools cache guardrail passed: no tracked files under requirements/.pip-tools-cache.")
        return 0

    print("ERROR: tracked pip-tools cache files detected:", file=sys.stderr)
    for path in tracked:
        print(f"  - {path}", file=sys.stderr)
    print(
        "Remediation: remove cache artifacts from git tracking "
        "(for example: git rm --cached -r requirements/.pip-tools-cache).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
