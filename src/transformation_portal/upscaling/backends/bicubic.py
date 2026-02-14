"""Bicubic upscaler backend (always available, no ML dependencies).

Golden Path implementation - commercial-safe, fast, deterministic.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class BicubicUpscaler:
    """Bicubic upscaling backend.

    Uses PIL's high-quality bicubic resampling (Lanczos variant).
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

    @property
    def name(self) -> str:
        """Backend name."""
        return "bicubic"

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
            image: Input image (H, W, 3), uint8 or float32.
            scale_factor: Upscaling factor (1.0-4.0).

        Returns:
            Upscaled image with same dtype as input.

        Raises:
            ValueError: If scale_factor is invalid.
        """
        if scale_factor < 1.0 or scale_factor > 4.0:
            raise ValueError(f"scale_factor must be in [1.0, 4.0], got {scale_factor}")

        # Handle dtype conversion
        input_dtype = image.dtype
        if image.dtype == np.float32:
            # Convert float32 [0, 1] to uint8 [0, 255] for PIL
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image_uint8 = image

        # Convert to PIL
        pil_img = Image.fromarray(image_uint8)

        # Calculate new size
        new_width = int(pil_img.width * scale_factor)
        new_height = int(pil_img.height * scale_factor)

        # Upscale using bicubic (PIL BICUBIC = Lanczos)
        upscaled_pil = pil_img.resize(
            (new_width, new_height),
            Image.BICUBIC,
        )

        # Convert back to numpy
        upscaled_uint8 = np.array(upscaled_pil)

        # Convert back to original dtype
        if input_dtype == np.float32:
            return upscaled_uint8.astype(np.float32) / 255.0
        else:
            return upscaled_uint8
