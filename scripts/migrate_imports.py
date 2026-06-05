#!/usr/bin/env python3
"""Compatibility wrapper for scripts.maintenance.migrate_imports."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.maintenance import migrate_imports as _impl
from scripts.maintenance.migrate_imports import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":
    raise SystemExit(_impl.main())
