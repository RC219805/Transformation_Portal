#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible CLI wrapper for depth_tools.

This thin wrapper preserves the ability to invoke depth tools directly from
the repository root as ``python depth_tools.py``. The real implementation
now lives in ``src/transformation_portal/depth/tools.py``.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add src directory to path
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    # Import and run the main CLI from the package
    from transformation_portal.depth.tools import main
    raise SystemExit(main())
