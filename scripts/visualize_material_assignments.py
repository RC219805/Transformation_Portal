#!/usr/bin/env python3
"""Compatibility wrapper for ``scripts.utilities.visualize_material_assignments``."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utilities.visualize_material_assignments import *  # noqa: F401,F403
from scripts.utilities.visualize_material_assignments import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
