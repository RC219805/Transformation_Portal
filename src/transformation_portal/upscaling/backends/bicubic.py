"""Bicubic upscaler backend (always available, no ML dependencies).

Golden Path implementation - commercial-safe, fast, deterministic.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BicubicUpscaler:
    """Bicubic upscaling backend.

    Uses OpenCV's high-quality bicubic interpolation.
    Always available (no ML dependencies), fast, commercial-safe.

    Performance:
        - ~100-200 images/hour for 4K upscaling to 8K
        - Memory: ~50MB per image
        - Quality: Good for 2x scaling, acceptable for 4x

    Usage:
        >>> upscaler = BicubicUpscaler()
        >>> image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        >>> upscaled = upscaler.upscale(image, scale_factor=2.0)
        >>> upscaled.shape
        (2000, 2000, 3)
    """

    # Backend ID for registry (class-level constant)
    BACKEND_ID = "bicubic"

    # Class-level metadata (for registry introspection without instantiation)
    REQUIRES_ML = False

    @property
    def name(self) -> str:
        """Backend name."""
        return self.BACKEND_ID

    @property
    def requires_ml(self) -> bool:
        """No ML dependencies required."""
        return False

    def upscale(
        self,
        image: np.ndarray,
        scale_factor: float,
    ) -> np.ndarray:
        """Upscale using bicubic interpolation.

        Args:
            image: Input image (H, W, 3), RGB uint8 or float32.
            scale_factor: Upscaling factor (1.0-4.0).

        Returns:
            Upscaled image (RGB) with same dtype as input.

        Raises:
            ValueError: If scale_factor is invalid or image has invalid shape/values.
        """
        # Validate input shape first (fail fast on fundamentally invalid inputs)
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image array (H, W, C), got {image.ndim}D")
        if image.shape[2] != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {image.shape[2]}")

        # Validate dimensions are non-zero
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Image dimensions must be positive, got {h}x{w}")

        # Validate dtype and values (data integrity before operation constraints)
        if image.dtype not in (np.uint8, np.float32):
            raise ValueError(f"Expected dtype uint8 or float32, got {image.dtype}")

        if not np.all(np.isfinite(image)):
            raise ValueError("Image contains non-finite values (NaN or Inf)")

        # Validate scale factor (operation-specific constraint checked last)
        if scale_factor < 1.0 or scale_factor > 4.0:
            raise ValueError(f"scale_factor must be in [1.0, 4.0], got {scale_factor}")

        # Use rounding instead of truncation to match expectations for fractional scales
        new_h = max(1, int(round(h * scale_factor)))
        new_w = max(1, int(round(w * scale_factor)))

        # Use OpenCV for high-quality bicubic interpolation
        # cv2.INTER_CUBIC preserves precision for both uint8 and float32
        # Note: cv2.resize() is channel-agnostic (no BGR conversion needed)
        input_dtype = image.dtype

        # Resize directly (OpenCV resize works on RGB without conversion)
        upscaled = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC,  # pylint: disable=no-member
        )

        # Return with dtype preservation (float32 clipped to [0,1], uint8 as-is)
        if input_dtype == np.float32:
            return np.clip(upscaled, 0.0, 1.0)
        return upscaled
