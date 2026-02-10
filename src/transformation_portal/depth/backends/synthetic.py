"""Synthetic depth backend for testing and CI environments.

This backend provides deterministic, fast depth estimation without ML dependencies.
It uses luminance-based depth approximation: brighter pixels = closer depth.

Design rationale:
- No torch/transformers/ML dependencies required
- Deterministic output for testing
- Fast enough for CI/performance tests
- Provides valid depth maps that allow depth-aware pipeline stages to execute

Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from PIL import Image

from .protocol import DepthResult, LicenseType

if TYPE_CHECKING:
    from ...lux_depth_v3.config import EnhanceConfig

logger = logging.getLogger(__name__)

__version__ = "1.0.0"


class SyntheticDepthBackend:
    """Synthetic depth backend using luminance approximation.

    This backend converts RGB images to grayscale and uses luminance as a proxy
    for depth. It's intended for testing environments where ML dependencies are
    not available or when deterministic depth maps are needed.

    Attributes:
        name: Backend identifier
        license_type: Commercial (no restrictions)
        requires_checkpoint: False (no external files needed)
    """

    name = "synthetic"
    license_type = LicenseType.COMMERCIAL
    requires_checkpoint = False

    def __init__(self, config: Optional[EnhanceConfig] = None) -> None:
        """Initialize synthetic backend.

        Args:
            config: Optional configuration (unused, for compatibility)
        """
        self._config = config
        logger.debug("SyntheticDepthBackend initialized (no ML dependencies required)")

    def ensure_available(self) -> None:
        """Check backend availability.

        Synthetic backend is always available (no external dependencies).
        """
        # Always available - no ML deps required
        pass

    @classmethod
    def required_packages(cls) -> list[str]:
        """Return required packages for this backend.

        Returns:
            Empty list (no additional dependencies beyond numpy/PIL)
        """
        return []

    def compute(
        self,
        image: Union[Image.Image, np.ndarray],
        device: Optional[str] = None,
    ) -> DepthResult:
        """Compute synthetic depth map from luminance.

        Args:
            image: Input PIL Image or numpy array (H, W, 3) RGB
            device: Ignored (synthetic backend is device-agnostic)

        Returns:
            DepthResult with synthetic depth map
        """
        # Convert input to PIL Image if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
            original_array = image
        else:
            pil_image = image
            original_array = np.asarray(pil_image, dtype=np.uint8)

        # Ensure RGB
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
            original_array = np.asarray(pil_image, dtype=np.uint8)

        # Convert to grayscale (luminance)
        gray_img = pil_image.convert("L")
        gray_array = np.asarray(gray_img, dtype=np.float32)

        # Normalize to [0, 1] range
        # Brighter pixels = closer (smaller depth value)
        # Invert so darker = farther (larger depth value)
        depth_map = 1.0 - (gray_array / 255.0)

        metadata = {
            "backend": self.name,
            "synthetic": True,
            "method": "luminance",
            "version": __version__,
        }

        return DepthResult(
            depth_map=depth_map,
            original_image=original_array,
            metadata=metadata,
            depth_units="relative",
            backend_id=self.name,
            device=device or "cpu",
            dtype="float32",
            input_size=(original_array.shape[0], original_array.shape[1]),
        )

    def get_cache_key(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Generate deterministic cache key.

        Args:
            image: Input image

        Returns:
            SHA256 hash of image content + backend version
        """
        # Convert to numpy array for hashing
        if isinstance(image, Image.Image):
            img_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        else:
            img_array = image

        # Hash image content
        content_hash = hashlib.sha256(img_array.tobytes()).hexdigest()[:16]

        return f"synthetic-v{__version__}-{content_hash}"

    def __repr__(self) -> str:
        return f"SyntheticDepthBackend(name={self.name!r})"
