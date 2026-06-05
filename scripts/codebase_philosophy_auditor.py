#!/usr/bin/env python3
"""Compatibility wrapper for ``transformation_portal.analyzers.codebase_philosophy_auditor``."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transformation_portal.analyzers import codebase_philosophy_auditor as _impl
from transformation_portal.analyzers.codebase_philosophy_auditor import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
