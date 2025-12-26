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

import sys
from pathlib import Path
import pytest

# Add lux_depth_v2 peer module to Python path for test discovery
# This ensures lux_depth_v2 can be imported during tests without requiring
# it to be installed as a package
repo_root = Path(__file__).parent.parent
lux_depth_v2_path = repo_root / "lux_depth_v2"
if lux_depth_v2_path.exists() and str(lux_depth_v2_path) not in sys.path:
    sys.path.insert(0, str(lux_depth_v2_path.parent))


def pytest_addoption(parser):
    """Add custom command-line options (idempotent to avoid duplicate registration)."""
    try:
        parser.addoption(
            "--slow",
            action="store_true",
            default=False,
            help="Run tests marked as slow (stress tests, long-running operations)",
        )
    except ValueError:
        # Option already registered (happens when both root and lux_depth_v2 conftests are loaded)
        pass


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --slow flag is provided."""
    # Use getattr to safely check for --slow option (may not be registered in all pytest roots)
    run_slow = getattr(config.option, "slow", False)
    if run_slow:
        # --slow flag provided: run all tests including slow ones
        return
    
    # Skip slow tests by default
    skip_slow = pytest.mark.skip(reason="need --slow option to run")
    for item in items:
        if item.get_closest_marker("slow"):
            item.add_marker(skip_slow)
