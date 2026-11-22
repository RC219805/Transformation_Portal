"""
Base class for depth-aware processors.

Provides common functionality for processors that support
runtime configuration overrides.
"""

from typing import Optional

import numpy as np


def split_alpha(image: np.ndarray) -> "tuple[np.ndarray, Optional[np.ndarray]]":
    """Split an HxWx4 image into its RGB channels and its HxWx1 alpha channel.

    Depth processors operate on colour only; alpha is coverage, not light, and
    must pass through untouched. Any other layout is returned unchanged with
    ``None`` for alpha.
    """
    if image.ndim == 3 and image.shape[-1] == 4:
        return image[..., :3], image[..., 3:]
    return image, None


def merge_alpha(rgb: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    """Re-attach an alpha channel produced by :func:`split_alpha`, in ``rgb``'s dtype."""
    if alpha is None:
        return rgb
    return np.concatenate([rgb, alpha.astype(rgb.dtype, copy=False)], axis=-1)


class DepthProcessorMixin:
    """
    Mixin for depth processors with config override support.

    Provides a standard __call__ implementation that temporarily
    overrides parameters when a config dict is provided.

    Subclasses must implement:
    - process(image, depth) method
    - _get_config_params() method returning dict of overridable params
    """

    def process(self, image: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Process image with depth information.

        Must be implemented by subclass.

        Args:
            image: Input image array
            depth: Depth map array

        Returns:
            Processed image array
        """
        raise NotImplementedError("Subclass must implement process()")

    def _get_config_params(self) -> dict:
        """
        Get dictionary of parameters that can be overridden via config.

        Must be implemented by subclass.

        Returns:
            Dictionary mapping parameter names to current values
        """
        raise NotImplementedError("Subclass must implement _get_config_params()")

    def _apply_config_override(self, key: str, value):
        """
        Apply a configuration override.

        Can be overridden by subclass to handle special cases.
        Default implementation uses setattr.

        Args:
            key: Parameter name
            value: Parameter value
        """
        if hasattr(self, key):
            setattr(self, key, value)

    def __call__(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        config: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Callable interface for pipeline integration.

        Args:
            image: Input image
            depth: Depth map
            config: Optional configuration override

        Returns:
            Processed image
        """
        if config:
            # Temporarily override parameters. Snapshot every attribute the
            # override may touch (the declared config params plus any other
            # existing attribute named in ``config``) and restore them even
            # when ``process`` or an override raises, so one call can never
            # leak its overrides into later calls.
            old_params = self._get_config_params()
            for key in config:
                if key not in old_params and hasattr(self, key):
                    old_params[key] = getattr(self, key)

            try:
                for key, value in config.items():
                    self._apply_config_override(key, value)

                return self.process(image, depth)
            finally:
                for key, value in old_params.items():
                    setattr(self, key, value)
        else:
            return self.process(image, depth)
