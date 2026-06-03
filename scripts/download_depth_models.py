#!/usr/bin/env python3
"""Compatibility wrapper for ``scripts.setup.download_depth_models``."""

from scripts.setup.download_depth_models import *  # noqa: F401,F403
from scripts.setup.download_depth_models import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
