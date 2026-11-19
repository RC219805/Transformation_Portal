"""
Depth Processor Adapter - Compatibility Wrapper

This module provides a DepthProcessor adapter that wraps ArchitecturalDepthPipeline
to maintain backward compatibility with unified pipeline imports.

The unified pipeline expects:
    from .depth.depth_processor import DepthProcessor

But the actual implementation is in:
    from .depth.pipeline import ArchitecturalDepthPipeline
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from PIL import Image

from .pipeline import ArchitecturalDepthPipeline
from .utils import load_image

logger = logging.getLogger(__name__)

# Depth backend that serves each supported DepthProcessor device. ``pytorch_cpu`` is the
# PyTorch backend; it places the model on the requested device ("cpu" or "cuda").
_DEVICE_BACKENDS = {
    "cpu": "pytorch_cpu",
    "cuda": "pytorch_cpu",
    "mps": "coreml",
}


class DepthProcessor:
    """
    Adapter for ArchitecturalDepthPipeline to provide DepthProcessor interface.

    This class wraps ArchitecturalDepthPipeline and provides a simplified
    interface for depth estimation suitable for the unified pipeline.

    Example:
        >>> processor = DepthProcessor(device='cuda')
        >>> depth_map = processor.estimate_depth(image)
    """

    def __init__(self, device: str = "cpu", config_path: Optional[Union[str, Path]] = None):
        """
        Initialize depth processor.

        Args:
            device: Device for processing ('cpu', 'cuda', 'mps'). Ignored when ``config_path`` is
                given: the config file's ``depth_model`` section then selects backend and device.
            config_path: Optional path to config file. If None, uses defaults.

        Raises:
            ValueError: If ``device`` is not one of the supported devices (default config only)
        """
        self.device = device

        if config_path is None:
            self._pipeline = ArchitecturalDepthPipeline(self._create_default_config(device))
        else:
            # Load the YAML and build the depth model once.
            self._pipeline = ArchitecturalDepthPipeline.from_config(config_path)

        logger.info(f"DepthProcessor initialized with device: {device}")

    def _create_default_config(self, device: str) -> Dict:
        """
        Create default configuration for depth processing.

        Args:
            device: Target device

        Returns:
            Default configuration dictionary

        Raises:
            ValueError: If ``device`` is not one of the supported devices
        """
        if device not in _DEVICE_BACKENDS:
            raise ValueError(f"Unsupported DepthProcessor device {device!r}; expected one of {sorted(_DEVICE_BACKENDS)}")
        backend = _DEVICE_BACKENDS[device]
        depth_model: Dict = {
            # The variant the adapter has always loaded (the unrecognised "vitl" fell back to it).
            "variant": "small",
            "backend": backend,
        }
        if backend != "coreml":
            # CoreML runs on its own device; the PyTorch backend honours the requested device.
            depth_model["device"] = device
        return {
            "depth_model": depth_model,
            "cache": {
                "enabled": True,
                "max_cache_size_gb": 2.0,
            },
            "processing": {
                "depth_aware_denoise": {"enabled": False},
                "zone_tone_mapping": {"enabled": False},
                "atmospheric_effects": {"enabled": False},
                "depth_guided_filters": {"enabled": False},
            },
        }

    def estimate_depth(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Estimate depth map from image.

        Args:
            image: Input image (PIL Image or numpy array)

        Returns:
            Depth map as numpy array (H, W), normalized to [0, 1]
            where 0 is far and 1 is near
        """
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            img_array = np.array(image).astype(np.float32) / 255.0
        else:
            img_array = image

        # Ensure normalized
        if img_array.max() > 1.0:
            img_array = img_array / 255.0

        # Use pipeline's depth model
        depth_result = self._pipeline.cache.get_or_compute(
            img_array, lambda: self._pipeline.depth_model.estimate_depth(img_array)
        )

        return depth_result["depth"]

    def process(self, image: Union[Image.Image, np.ndarray], return_depth: bool = False) -> Union[np.ndarray, tuple]:
        """
        Process image with depth-aware enhancements (if configured).

        Args:
            image: Input image
            return_depth: If True, return (processed_image, depth_map)

        Returns:
            Processed image, or (processed_image, depth_map) if return_depth=True
        """
        # For now, just return the depth map since the unified pipeline
        # handles the actual processing stages separately
        depth_map = self.estimate_depth(image)

        if return_depth:
            return image, depth_map
        else:
            return depth_map

    def clear_cache(self) -> None:
        """Clear the depth estimation cache."""
        self._pipeline.cache.clear()
        logger.info("Depth cache cleared")

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache hit/miss counts
        """
        stats = self._pipeline.cache.get_stats()
        return {
            "hits": stats["hits"],
            "misses": stats["misses"],
        }


__all__ = ["DepthProcessor"]
