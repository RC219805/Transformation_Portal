#!/usr/bin/env python3
"""Compatibility wrapper for ``scripts.setup.install_models_auto``."""

from scripts.setup.install_models_auto import *  # noqa: F401,F403
from scripts.setup.install_models_auto import main as _main


if __name__ == "__main__":
    _main()
