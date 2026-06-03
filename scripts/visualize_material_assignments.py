#!/usr/bin/env python3
"""Compatibility wrapper for ``scripts.utilities.visualize_material_assignments``."""

from scripts.utilities.visualize_material_assignments import *  # noqa: F401,F403
from scripts.utilities.visualize_material_assignments import main as _main


if __name__ == "__main__":
    _main()
