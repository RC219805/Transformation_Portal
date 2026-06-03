#!/usr/bin/env python3
"""Compatibility wrapper for ``scripts.pipelines.run_aerial_enhancement``."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipelines.run_aerial_enhancement import *  # noqa: F401,F403
from scripts.pipelines.run_aerial_enhancement import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
