#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible CLI wrapper for lux_render_pipeline.

This thin wrapper preserves the ability to invoke the pipeline directly from
the repository root as ``python lux_render_pipeline.py``. The real
implementation now lives in ``src/transformation_portal/pipelines/lux_render_pipeline.py``.
"""

if __name__ == "__main__":
    # Import and run the main CLI from the package
    # NOTE: Requires package installation: pip install -e .
    from transformation_portal.pipelines.lux_render_pipeline import main
    raise SystemExit(main())
