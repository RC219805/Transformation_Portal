"""Pytest configuration for Transformation Portal test suite.

This conftest.py ensures that the src directory is in the Python path
so that tests can import directly from the transformation_portal package
without requiring the package to be installed.
"""
import sys
from pathlib import Path

# Add src directory to Python path for test imports
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
