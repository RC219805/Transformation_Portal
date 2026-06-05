#!/usr/bin/env python3
"""Compatibility wrapper for scripts.setup.download_samples."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.setup import download_samples as _impl
from scripts.setup.download_samples import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":
    raise SystemExit(_impl.main())
