#!/usr/bin/env python3
"""Compatibility wrapper for ``scripts.validation.check_image_processing_readiness``."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validation import check_image_processing_readiness as _impl
from scripts.validation.check_image_processing_readiness import *  # noqa: F401,F403
from scripts.validation.check_image_processing_readiness import main as _main


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":
    raise SystemExit(_main())
