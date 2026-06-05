#!/usr/bin/env python3
"""Compatibility wrapper for ``scripts.pipelines.run_depth_estimation``."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipelines import run_depth_estimation as _impl
from scripts.pipelines.run_depth_estimation import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":
    raise SystemExit(_impl.main())
