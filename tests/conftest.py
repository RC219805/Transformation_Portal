#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pytest configuration for Transformation Portal tests.

NOTE: This file previously contained sys.path manipulation to add src/
to the Python path. This approach violates PR 162 guidelines for proper
package development practices.

PROPER SETUP:
  Option 1 (Recommended): Install package in editable mode
    pip install -e .

  Option 2: Set PYTHONPATH environment variable
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
    # or for this test run only:
    PYTHONPATH="$(pwd)/src" pytest

Tests will automatically find the package if installed via pip install -e .
or if PYTHONPATH is set correctly.
"""
