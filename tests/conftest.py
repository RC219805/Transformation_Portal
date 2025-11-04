#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pytest configuration for Transformation Portal tests.

This configuration ensures proper Python path setup for importing
the transformation_portal package in tests.
"""
import sys
from pathlib import Path

# Add src/ to Python path for package imports
# This allows: from transformation_portal.utils import ...
_repo_root = Path(__file__).parent.parent
_src_path = _repo_root / "src"

if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))
