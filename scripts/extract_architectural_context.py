#!/usr/bin/env python3
"""Compatibility wrapper for ``scripts.analysis.extract_architectural_context``."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis import extract_architectural_context as _impl
from scripts.analysis.extract_architectural_context import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":
    raise SystemExit(_impl.main())
