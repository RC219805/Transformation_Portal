"""Inference engine for Depth Anything V3.

STUB IMPLEMENTATION - Critical types to enable package imports.
Full implementation pending.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from .config import DA3Config


@dataclass
class DepthResult:
    """Result from depth inference."""
    depth_map: np.ndarray
    original_image: np.ndarray
    metadata: Dict[str, Any]

    @property
    def depth(self) -> np.ndarray:
        """Alias for depth_map to support both naming conventions."""
        return self.depth_map


class DA3InferenceEngine:
    """Inference engine for Depth Anything V3 models.

    STUB IMPLEMENTATION - Full implementation pending.
    """

    def __init__(
        self,
        config: DA3Config,
        commercial_use: bool = True,
        validate_license_strict: bool = False
    ):
        """Initialize inference engine.

        Args:
            config: DA3 configuration
            commercial_use: Whether commercial use is enabled
            validate_license_strict: Whether to strictly validate license
        """
        self.config = config
        self.commercial_use = commercial_use
        self.validate_license_strict = validate_license_strict

    def predict(self, image: np.ndarray) -> DepthResult:
        """Run depth inference on an image (alias for infer).

        STUB: Not implemented.

        Args:
            image: Input image as numpy array

        Returns:
            DepthResult with depth map and metadata

        Raises:
            NotImplementedError: This is a stub implementation
        """
        raise NotImplementedError(
            "DA3InferenceEngine.predict() is a stub - full implementation pending. "
            "This module was created to enable package imports."
        )

    def infer(self, image: np.ndarray) -> DepthResult:
        """Run depth inference on an image.

        STUB: Not implemented.

        Args:
            image: Input image as numpy array

        Returns:
            DepthResult with depth map and metadata

        Raises:
            NotImplementedError: This is a stub implementation
        """
        raise NotImplementedError(
            "DA3InferenceEngine.infer() is a stub - full implementation pending. "
            "This module was created to enable package imports."
        )

    def infer_from_path(self, image_path: Path) -> DepthResult:
        """Run depth inference on an image file.

        STUB: Not implemented.

        Args:
            image_path: Path to input image

        Returns:
            DepthResult with depth map and metadata

        Raises:
            NotImplementedError: This is a stub implementation
        """
        raise NotImplementedError(
            "DA3InferenceEngine.infer_from_path() is a stub - full implementation pending. "
            "This module was created to enable package imports."
        )
