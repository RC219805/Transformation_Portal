"""Backward-compatible wrapper for image_utils.

This thin wrapper preserves the ability to import image utilities directly from
the repository root. The real implementation now lives in
``src/transformation_portal/utils/image_utils.py``.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import and re-export all functions from the package
from transformation_portal.utils.image_utils import (
    load_image,
    save_image,
    pil_to_np,
    np_to_pil,
    load_image_rgb,
)

__all__ = [
    'load_image',
    'save_image',
    'pil_to_np',
    'np_to_pil',
    'load_image_rgb',
]
