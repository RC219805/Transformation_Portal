"""
Image Statistics & Autotuning Helpers.

Analyzes raw image tensors to determine optimal export formats.
detects HDR content, transparency, and frequency detail.
"""

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np


@dataclass
class ImageStats:
    """Statistical profile of an image tensor."""

    min_val: float
    max_val: float
    mean_val: float
    has_alpha: bool
    is_hdr: bool
    bit_depth_hint: int  # 8, 16, or 32
    dynamic_range_stops: float


def compute_image_stats(image: np.ndarray, sample_stride: int = 10) -> ImageStats:
    """
    Analyze image content to derive storage requirements.

    Args:
        image: Numpy array (H, W, C) or (C, H, W).
        sample_stride: Stride for faster statistics on massive images.
    """
    # Ensure (H, W, C) layout for analysis
    if image.ndim == 3 and image.shape[0] <= 4:
        # Likely (C, H, W), transpose
        img_view = image.transpose(1, 2, 0)
    else:
        img_view = image

    # Subsample for speed
    sample = img_view[::sample_stride, ::sample_stride, :]

    # 1. Check Transparency
    has_alpha = False
    if sample.shape[-1] == 4:
        # Check if alpha channel is actually used (not all 1.0 or 255)
        alpha = sample[:, :, 3]
        if image.dtype == np.uint8:
            has_alpha = np.any(alpha < 255)
        else:
            has_alpha = np.any(alpha < 1.0 - 1e-4)

    # 2. Check Dynamic Range & HDR
    min_v = float(sample.min())
    max_v = float(sample.max())
    mean_v = float(sample.mean())

    is_hdr = False
    bit_depth = 8

    if image.dtype == np.float32 or image.dtype == np.float16:
        # If values exceed 1.0, it's definitely HDR
        if max_v > 1.0:
            is_hdr = True
            bit_depth = 32
        # If values are very precise/small, suggests linear data
        elif max_v <= 1.0 and image.dtype == np.float32:
            bit_depth = 16  # Suggest 16-bit for precision
    elif image.dtype == np.uint16:
        bit_depth = 16

    # Estimate stops (log2 dynamic range)
    # Avoid log(0)
    safe_min = max(min_v, 1e-6)
    stops = np.log2(max_v / safe_min) if max_v > safe_min else 0.0

    return ImageStats(
        min_val=min_v,
        max_val=max_v,
        mean_val=mean_v,
        has_alpha=has_alpha,
        is_hdr=is_hdr,
        bit_depth_hint=bit_depth,
        dynamic_range_stops=stops,
    )
