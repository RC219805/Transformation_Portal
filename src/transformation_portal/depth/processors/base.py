"""
Base class for depth-aware processors.

Provides common functionality for processors that support
runtime configuration overrides.
"""

from typing import Optional
import numpy as np


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
            # Temporarily override parameters
            old_params = self._get_config_params()

            for key, value in config.items():
                self._apply_config_override(key, value)

            result = self.process(image, depth)

            # Restore parameters
            for key, value in old_params.items():
                setattr(self, key, value)

            return result
        else:
            return self.process(image, depth)
