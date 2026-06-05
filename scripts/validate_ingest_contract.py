#!/usr/bin/env python3
"""Compatibility wrapper for ``scripts.validation.validate_ingest_contract``."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validation import validate_ingest_contract as _impl
from scripts.validation.validate_ingest_contract import *  # noqa: F401,F403
from scripts.validation.validate_ingest_contract import main as _main


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":
    raise SystemExit(_main())
