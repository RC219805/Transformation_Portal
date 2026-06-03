#!/usr/bin/env python3
"""Compatibility wrapper for ``scripts.pipelines.run_aerial_enhancement``."""

from scripts.pipelines.run_aerial_enhancement import *  # noqa: F401,F403
from scripts.pipelines.run_aerial_enhancement import main as _main


if __name__ == "__main__":
    _main()
