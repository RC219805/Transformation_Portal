#!/usr/bin/env python3
"""Compatibility wrapper for ``transformation_portal.analyzers.decision_decay_dashboard``."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transformation_portal.analyzers import decision_decay_dashboard as _impl
from transformation_portal.analyzers.decision_decay_dashboard import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":
    raise SystemExit(_impl.main())
