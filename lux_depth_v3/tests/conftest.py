"""Pytest configuration for lux_depth_v3 tests.

Adds the grandparent directory to sys.path to enable imports without installation.
The package structure is:
  Transformation_Portal-main/
    lux_depth_v3/
      __init__.py
      enhance/
      tests/
"""

import sys
from pathlib import Path

# Add grandparent directory (Transformation_Portal-main) to sys.path
# This allows "from lux_depth_v3.enhance..." imports to work
repo_parent = Path(__file__).parent.parent.parent
if str(repo_parent) not in sys.path:
    sys.path.insert(0, str(repo_parent))
