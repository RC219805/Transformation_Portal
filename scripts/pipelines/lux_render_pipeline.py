#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible wrapper for lux_render_pipeline.

This wrapper preserves backward compatibility for both CLI invocation and
module imports. The real implementation now lives in
``src/transformation_portal/pipelines/lux_render_pipeline.py``.

For module imports, all functions are re-exported from the new location.
For CLI usage, run:
    .venv/bin/python scripts/pipelines/lux_render_pipeline.py [options]

The wrapper bootstraps the repo-local ``src/`` package root for raw checkouts.
"""
import sys
from pathlib import Path

# Add src/ to path if package not installed (for development/testing)
_src_root = Path(__file__).resolve().parents[2] / "src"
if _src_root.exists():
    _src_path = str(_src_root)
    if _src_path not in sys.path:
        sys.path.insert(0, _src_path)

# Re-export all public functions for backward compatibility
from transformation_portal.pipelines.lux_render_pipeline import apply_material_response_finishing, main  # noqa: E402

# Make linting happy - these are intentionally re-exported
__all__ = [
    "apply_material_response_finishing",
    "main",
]

if __name__ == "__main__":
    # Import and run the main CLI from the package
    raise SystemExit(main())
