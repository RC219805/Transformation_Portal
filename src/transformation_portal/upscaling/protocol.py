"""Upscaler backend protocol.

Defines the contract for all upscaling backends with consistent input/output types.
Follows the pattern established by DepthBackend protocol (ADR-019).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class UpscalerBackend(Protocol):
    """Protocol for upscaler backends.

    All upscalers must implement this interface to work with the registry.
    """

    @property
    def name(self) -> str:
        """Backend name (e.g., 'bicubic', 'realesrgan').

        Returns:
            Backend identifier string.
        """
        ...

    @property
    def requires_ml(self) -> bool:
        """Whether backend requires ML dependencies.

        Returns:
            True if backend needs ML libraries (torch, basicsr, etc.).
        """
        ...

    def upscale(
        self,
        image: np.ndarray,
        scale_factor: float,
    ) -> np.ndarray:
        """Upscale image by scale_factor.

        Args:
            image: Input image as numpy array (H, W, 3).
                   Can be uint8 [0-255] or float32 [0-1].
                   Note: float32 inputs are expected to be normalized to [0, 1],
                   but out-of-range values will be clipped during processing.
            scale_factor: Upscaling factor (1.0-4.0).
                         For Real-ESRGAN: 2.0 or 4.0 recommended.

        Returns:
            Upscaled image as numpy array (H*scale, W*scale, 3).
            Output dtype matches input dtype.

        Raises:
            ValueError: If scale_factor is out of valid range.
            RuntimeError: If upscaling fails.
        """
        ...
