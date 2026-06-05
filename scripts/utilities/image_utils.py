"""Backward-compatible wrapper for image_utils.

This thin wrapper preserves the ability to import image utilities directly from
the repository root. The real implementation now lives in
``src/transformation_portal/utils/image_utils.py``.

NOTE: Requires package installation: pip install -e .
"""

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import and re-export all functions from the package
from transformation_portal.utils.image_utils import load_image, load_image_rgb, np_to_pil, pil_to_np, save_image  # noqa: E402

__all__ = [
    "load_image",
    "save_image",
    "pil_to_np",
    "np_to_pil",
    "load_image_rgb",
]
