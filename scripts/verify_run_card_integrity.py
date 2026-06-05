#!/usr/bin/env python3
"""Compatibility wrapper for ``scripts.verification.verify_run_card_integrity``."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verification import verify_run_card_integrity as _impl
from scripts.verification.verify_run_card_integrity import *  # noqa: F401,F403
from scripts.verification.verify_run_card_integrity import main as _main


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
