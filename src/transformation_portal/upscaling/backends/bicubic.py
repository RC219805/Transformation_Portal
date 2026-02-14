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
        # Validate scale factor
        if scale_factor < 1.0 or scale_factor > 4.0:
            raise ValueError(f"scale_factor must be in [1.0, 4.0], got {scale_factor}")

        # Validate input shape
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image array (H, W, C), got {image.ndim}D")
        if image.shape[2] != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {image.shape[2]}")

        # Validate dimensions are non-zero
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Image dimensions must be positive, got {h}x{w}")

        # Validate dtype and values
        if image.dtype not in (np.uint8, np.float32):
            raise ValueError(f"Expected dtype uint8 or float32, got {image.dtype}")

        if not np.all(np.isfinite(image)):
            raise ValueError("Image contains non-finite values (NaN or Inf)")

        h, w = image.shape[:2]
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)

        # Use OpenCV for high-quality bicubic interpolation
        # cv2.INTER_CUBIC preserves precision for both uint8 and float32
        # Note: OpenCV uses BGR, so we need to convert RGB→BGR→RGB
        input_dtype = image.dtype

        if input_dtype == np.float32:
            # For float32, work directly without dtype conversion (preserves 16-bit quality)
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            upscaled_bgr = cv2.resize(
                image_bgr,
                (new_w, new_h),
                interpolation=cv2.INTER_CUBIC,
            )
            # Convert BGR back to RGB
            upscaled = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
            # Clip to valid range
            return np.clip(upscaled, 0.0, 1.0)
        else:
            # For uint8, standard path
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            upscaled_bgr = cv2.resize(
                image_bgr,
                (new_w, new_h),
                interpolation=cv2.INTER_CUBIC,
            )
            return cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
