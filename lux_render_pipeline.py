#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible wrapper for lux_render_pipeline.

This wrapper preserves backward compatibility for both CLI invocation and
module imports. The real implementation now lives in
``src/transformation_portal/pipelines/lux_render_pipeline.py``.

For module imports, all functions are re-exported from the new location.
For CLI usage, run: python lux_render_pipeline.py [options]

NOTE: Requires package installation (pip install -e .) or running from
repository root with src/ in Python path.
"""
import sys
from pathlib import Path

# Add src/ to path if package not installed (for development/testing)
_repo_root = Path(__file__).parent
if (_repo_root / "src").exists():
    _src_path = str(_repo_root / "src")
    if _src_path not in sys.path:
        sys.path.insert(0, _src_path)

# Re-export all public functions for backward compatibility
from transformation_portal.pipelines.lux_render_pipeline import (  # noqa: E402
    apply_material_response_finishing,
    main,
)

# Make linting happy - these are intentionally re-exported
__all__ = [
    "apply_material_response_finishing",
    "main",
]

if __name__ == "__main__":
    # Import and run the main CLI from the package
    raise SystemExit(main())
