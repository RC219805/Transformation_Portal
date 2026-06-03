#!/usr/bin/env python3
"""Compatibility wrapper for ``transformation_portal.perceptual.synthetic_viewer``."""

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transformation_portal.perceptual.synthetic_viewer import *  # noqa: F401,F403
