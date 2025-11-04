#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible CLI wrapper for luxury_video_master_grader.

This thin wrapper preserves the ability to invoke the grader directly from
the repository root as ``python luxury_video_master_grader.py``. The real
implementation now lives in ``src/transformation_portal/processors/luxury_video_master_grader.py``.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add src directory to path
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    # Import and run the main CLI from the package
    from transformation_portal.processors.luxury_video_master_grader import main
    raise SystemExit(main())
