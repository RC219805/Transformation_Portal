"""Example custom depth model plugin.

This demonstrates how to create a custom depth estimation plugin
for the Transformation Portal.
"""

from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from transformation_portal.plugins import (
    DepthModelPlugin,
    PluginMetadata,
    PluginType,
    cached_execution,
    measure_performance,
    plugin,
)


@plugin(
    name="simple_depth_model",
    plugin_type=PluginType.DEPTH_MODEL,
    version="1.0.0",
    description="Simple example depth model using gradient-based estimation",
    author="Transformation Portal Team",
    license="MIT",
    tags=["example", "simple", "gradient-based"]
)
class SimpleDepthModel(DepthModelPlugin):
    """Simple depth model example using gradient-based estimation.

    This is a simplified example for demonstration purposes. Real depth models
    would use neural networks (Depth Anything V2, ZoeDepth, etc.).

    Example:
        >>> from transformation_portal.plugins import get_global_registry
        >>>
        >>> registry = get_global_registry()
        >>> model = registry.get_plugin(
        ...     'depth_model',
        ...     'simple_depth_model',
        ...     initialize=True
        ... )
        >>>
        >>> depth_map = model.estimate_depth(image)
    """

    def _create_metadata(self) -> PluginMetadata:
        """Create plugin metadata."""
        # Access the class attribute set by @plugin decorator
        if hasattr(self, '_decorator_metadata'):
            return self._decorator_metadata

        # Fallback if decorator wasn't used
        return PluginMetadata(
            name="simple_depth_model",
            version="1.0.0",
            plugin_type=PluginType.DEPTH_MODEL,
            description="Simple example depth model using gradient-based estimation",
            author="Transformation Portal Team",
            license="MIT",
            tags=["example", "simple", "gradient-based"],
        )

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the depth model.

        Args:
            config: Optional configuration dictionary
                - normalize: Whether to normalize depth maps (default: True)
                - invert: Invert depth values (default: False)
        """
        self._config = config or {}
        self.normalize = self._config.get('normalize', True)
        self.invert = self._config.get('invert', False)

        # Mark as initialized
        self._initialized = True

        print(f"[SimpleDepthModel] Initialized with config: {self._config}")

    @measure_performance
    def estimate_depth(self, image: Any, **kwargs) -> np.ndarray:
        """Estimate depth map from image.

        This simple implementation uses gradient magnitude as a proxy for depth.
        Real implementations would use trained neural networks.

        Args:
            image: Input image (PIL Image, numpy array, or file path)
            **kwargs: Additional parameters (unused in this simple example)

        Returns:
            Depth map as numpy array (height, width) with values 0-1

        Raises:
            RuntimeError: If plugin not initialized
        """
        if not self._initialized:
            raise RuntimeError("Plugin not initialized. Call initialize() first.")

        # Convert input to numpy array
        img_array = self._to_numpy(image)

        # Convert to grayscale if color
        if len(img_array.shape) == 3:
            # Simple RGB to grayscale
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Compute gradient-based depth estimation
        # This is a simplified proxy - real depth models use neural networks
        from scipy.ndimage import sobel

        # Compute gradients
        gradient_x = sobel(gray, axis=0)
        gradient_y = sobel(gray, axis=1)

        # Gradient magnitude as depth proxy
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        # Normalize if requested
        if self.normalize:
            depth_map = self._normalize_depth(gradient_magnitude)
        else:
            depth_map = gradient_magnitude

        # Invert if requested (far = 1, near = 0)
        if self.invert:
            depth_map = 1.0 - depth_map

        return depth_map.astype(np.float32)

    def _to_numpy(self, image: Any) -> np.ndarray:
        """Convert various image formats to numpy array.

        Args:
            image: PIL Image, numpy array, or file path

        Returns:
            Numpy array
        """
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, Image.Image):
            return np.array(image)
        elif isinstance(image, str):
            # Load from file path
            pil_image = Image.open(image)
            return np.array(pil_image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _normalize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Normalize depth map to 0-1 range.

        Args:
            depth_map: Raw depth values

        Returns:
            Normalized depth map
        """
        min_val = depth_map.min()
        max_val = depth_map.max()

        if max_val - min_val < 1e-8:
            # Avoid division by zero
            return np.zeros_like(depth_map)

        return (depth_map - min_val) / (max_val - min_val)

    def cleanup(self) -> None:
        """Clean up resources."""
        print("[SimpleDepthModel] Cleaning up resources")
        self._initialized = False
        self._config = {}


# Example usage
if __name__ == "__main__":
    # This demonstrates using the plugin directly
    from PIL import Image

    # Create plugin instance
    model = SimpleDepthModel()

    # Initialize
    model.initialize(config={'normalize': True, 'invert': False})

    # Load test image
    test_image = Image.new('RGB', (512, 512), color='white')

    # Estimate depth
    depth_map = model.estimate_depth(test_image)

    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")

    # Cleanup
    model.cleanup()
