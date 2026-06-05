#!/usr/bin/env python3
"""Compatibility wrapper for ``scripts.verification.verify_depth_pro``."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verification import verify_depth_pro as _impl
from scripts.verification.verify_depth_pro import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":
    raise SystemExit(0 if _impl.verify_depth_pro() else 1)
