"""Preprocessing utilities for image normalization.

STUB IMPLEMENTATION - Critical functions to enable package imports.
Full implementation pending.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np


def normalize_exif_orientation(input_path: Path, output_path: Path):
    """Normalize EXIF orientation by rotating image to upright position.

    STUB: Not implemented.

    Args:
        input_path: Input image path
        output_path: Output image path (normalized)

    Raises:
        NotImplementedError: This is a stub implementation
    """
    raise NotImplementedError(
        "normalize_exif_orientation() is a stub - full implementation pending. "
        "This module was created to enable package imports."
    )


def validate_depth_image_alignment(
    image_path: Path,
    depth_path: Path
) -> bool:
    """Validate that depth map and image have matching dimensions.

    STUB: Not implemented.

    Args:
        image_path: Path to image
        depth_path: Path to depth map

    Returns:
        True if dimensions match, False otherwise

    Raises:
        NotImplementedError: This is a stub implementation
    """
    raise NotImplementedError(
        "validate_depth_image_alignment() is a stub - full implementation pending. "
        "This module was created to enable package imports."
    )
