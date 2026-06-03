#!/usr/bin/env python3
"""Compatibility wrapper for ``scripts.setup.install_models``."""

from scripts.setup.install_models import *  # noqa: F401,F403
from scripts.setup.install_models import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
