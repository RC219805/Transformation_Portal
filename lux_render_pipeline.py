#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible CLI wrapper for lux_render_pipeline.

This thin wrapper preserves the ability to invoke the pipeline directly from
the repository root as ``python lux_render_pipeline.py``. The real
implementation now lives in ``src/transformation_portal/pipelines/lux_render_pipeline.py``.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add src directory to path
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    # Import and run the main CLI from the package
    from transformation_portal.pipelines.lux_render_pipeline import main
    raise SystemExit(main())
