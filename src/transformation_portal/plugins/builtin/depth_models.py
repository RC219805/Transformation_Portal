"""Built-in depth model plugins demonstrating DepthModelPlugin pattern.

These plugins serve as reference implementations for creating
custom depth estimation plugins.

Note: These are demonstration implementations using simple algorithms.
For production use, integrate proper ML models like DepthAnything, MiDaS, etc.
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image, ImageFilter

from ..decorators import measure_performance, plugin
from ..interface import DepthModelPlugin, PluginMetadata, PluginType

logger = logging.getLogger(__name__)


@plugin(
    name="edge_depth_estimator",
    plugin_type=PluginType.DEPTH_MODEL,
    version="1.0.0",
    description="Simple edge-based depth estimation (demonstration only)",
    author="Transformation Portal",
    tags=["depth", "edge", "demo", "builtin"],
    auto_register=False,
)
class EdgeDepthEstimator(DepthModelPlugin):
    """Edge-based depth estimation plugin.

    A demonstration implementation showing the DepthModelPlugin pattern.
    Uses edge detection as a proxy for depth (edges = foreground).

    This is NOT a production-quality depth estimator. It serves as:
    - A reference implementation of DepthModelPlugin
    - A fallback when ML models are unavailable
    - A testing/development placeholder

    For production, use proper depth models like:
    - DepthAnything v2
    - MiDaS
    - ZoeDepth

    Example:
        >>> estimator = EdgeDepthEstimator()
        >>> estimator.initialize({"edge_threshold": 50})
        >>> depth_map = estimator.estimate_depth(image)
    """

    def __init__(self):
        """Initialize edge depth estimator."""
        super().__init__()
        self._edge_threshold: int = 30
        self._blur_radius: float = 2.0
        self._normalize: bool = True
        self._invert: bool = False

    def _create_metadata(self) -> PluginMetadata:
        """Create plugin metadata."""
        if hasattr(self, '_decorator_metadata'):
            return self._decorator_metadata

        return PluginMetadata(
            name="edge_depth_estimator",
            version="1.0.0",
            plugin_type=PluginType.DEPTH_MODEL,
            description="Simple edge-based depth estimation (demonstration only)",
            author="Transformation Portal",
            tags=["depth", "edge", "demo", "builtin"],
        )

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with configuration.

        Args:
            config: Configuration with optional keys:
                - edge_threshold: Threshold for edge detection (default: 30)
                - blur_radius: Gaussian blur radius for smoothing (default: 2.0)
                - normalize: Normalize output to 0-1 range (default: True)
                - invert: Invert depth values (default: False)
        """
        if config:
            self._edge_threshold = config.get("edge_threshold", self._edge_threshold)
            self._blur_radius = config.get("blur_radius", self._blur_radius)
            self._normalize = config.get("normalize", self._normalize)
            self._invert = config.get("invert", self._invert)
            self._config = config

        self._initialized = True
        logger.info(f"EdgeDepthEstimator initialized: "
                    f"threshold={self._edge_threshold}, blur={self._blur_radius}")

    @measure_performance
    def estimate_depth(
        self,
        image: Union[Image.Image, np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """Estimate depth from image using edge detection.

        Args:
            image: PIL Image or numpy array
            **kwargs: Optional overrides:
                - edge_threshold: Override threshold
                - normalize: Override normalization setting

        Returns:
            Depth map as numpy array (float32, 0-1 if normalized)
        """
        if not self._initialized:
            raise RuntimeError("Estimator not initialized. Call initialize() first.")

        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Convert to grayscale
        gray = pil_image.convert('L')

        # Apply edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)

        # Apply threshold
        threshold = kwargs.get("edge_threshold", self._edge_threshold)
        edges_array = np.array(edges, dtype=np.float32)
        edges_array = np.where(edges_array > threshold, edges_array, 0)

        # Smooth edges to create depth gradient
        if self._blur_radius > 0:
            edges_pil = Image.fromarray(edges_array.astype(np.uint8))
            smoothed = edges_pil.filter(
                ImageFilter.GaussianBlur(self._blur_radius)
            )
            depth_map = np.array(smoothed, dtype=np.float32)
        else:
            depth_map = edges_array

        # Create distance transform effect
        # Areas near edges are "closer" (foreground)
        depth_map = self._create_distance_field(depth_map)

        # Normalize
        if kwargs.get("normalize", self._normalize):
            min_val, max_val = depth_map.min(), depth_map.max()
            if max_val > min_val:
                depth_map = (depth_map - min_val) / (max_val - min_val)

        # Invert if configured
        if kwargs.get("invert", self._invert):
            depth_map = 1.0 - depth_map

        return depth_map.astype(np.float32)

    def _create_distance_field(self, edge_map: np.ndarray) -> np.ndarray:
        """Create a distance field from edge map.

        Areas closer to strong edges are considered "nearer" in depth.
        """
        from scipy import ndimage

        # Threshold to binary
        binary = edge_map > 0

        # Calculate distance from edges
        distance = ndimage.distance_transform_edt(~binary)

        # Normalize and invert (closer to edge = smaller depth value = closer)
        max_dist = distance.max()
        if max_dist > 0:
            distance = distance / max_dist

        # Combine with original edge strength
        combined = 1.0 - distance * 0.7

        return combined

    def validate(self) -> bool:
        """Validate estimator state."""
        return self._initialized and self._edge_threshold >= 0

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        self._edge_threshold = 30
        self._blur_radius = 2.0
        logger.debug("EdgeDepthEstimator cleaned up")


@plugin(
    name="gradient_depth_estimator",
    plugin_type=PluginType.DEPTH_MODEL,
    version="1.0.0",
    description="Gradient-based depth estimation using image gradients",
    author="Transformation Portal",
    tags=["depth", "gradient", "demo", "builtin"],
    auto_register=False,
)
class GradientDepthEstimator(DepthModelPlugin):
    """Gradient-based depth estimation plugin.

    Uses Sobel gradients to estimate depth. Areas with strong
    gradients are assumed to be at depth boundaries.

    Another demonstration implementation showing alternative
    depth estimation approaches.
    """

    def __init__(self):
        """Initialize gradient depth estimator."""
        super().__init__()
        self._sobel_size: int = 3
        self._smooth_iterations: int = 5

    def _create_metadata(self) -> PluginMetadata:
        """Create plugin metadata."""
        if hasattr(self, '_decorator_metadata'):
            return self._decorator_metadata

        return PluginMetadata(
            name="gradient_depth_estimator",
            version="1.0.0",
            plugin_type=PluginType.DEPTH_MODEL,
            description="Gradient-based depth estimation using image gradients",
            author="Transformation Portal",
            tags=["depth", "gradient", "demo", "builtin"],
        )

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with configuration.

        Args:
            config: Configuration with:
                - sobel_size: Sobel kernel size (default: 3)
                - smooth_iterations: Smoothing passes (default: 5)
        """
        if config:
            self._sobel_size = config.get("sobel_size", self._sobel_size)
            self._smooth_iterations = config.get("smooth_iterations", self._smooth_iterations)
            self._config = config

        self._initialized = True
        logger.info(f"GradientDepthEstimator initialized")

    @measure_performance
    def estimate_depth(
        self,
        image: Union[Image.Image, np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """Estimate depth using gradient analysis.

        Args:
            image: Input image
            **kwargs: Overrides

        Returns:
            Depth map as float32 array
        """
        if not self._initialized:
            raise RuntimeError("Estimator not initialized")

        from scipy import ndimage

        # Convert to array
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert('L'), dtype=np.float32)
        else:
            if len(image.shape) == 3:
                # Convert RGB to grayscale
                img_array = np.mean(image, axis=2).astype(np.float32)
            else:
                img_array = image.astype(np.float32)

        # Calculate Sobel gradients
        sobel_x = ndimage.sobel(img_array, axis=1)
        sobel_y = ndimage.sobel(img_array, axis=0)

        # Gradient magnitude
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        # Smooth to create continuous depth map
        depth_map = gradient_mag
        for _ in range(self._smooth_iterations):
            depth_map = ndimage.gaussian_filter(depth_map, sigma=2)

        # Normalize to 0-1
        min_val, max_val = depth_map.min(), depth_map.max()
        if max_val > min_val:
            depth_map = (depth_map - min_val) / (max_val - min_val)

        return depth_map.astype(np.float32)

    def validate(self) -> bool:
        """Validate state."""
        return self._initialized

    def cleanup(self) -> None:
        """Clean up."""
        super().cleanup()


@plugin(
    name="placeholder_depth_model",
    plugin_type=PluginType.DEPTH_MODEL,
    version="1.0.0",
    description="Placeholder depth model returning uniform depth",
    author="Transformation Portal",
    tags=["depth", "placeholder", "builtin"],
    deprecated=True,
    replacement="edge_depth_estimator",
    auto_register=False,
)
class PlaceholderDepthModel(DepthModelPlugin):
    """Placeholder depth model for testing.

    Returns a uniform depth map. Useful for:
    - Testing pipeline integration
    - Fallback when no model is available
    - Unit testing

    Marked as deprecated to demonstrate deprecation handling.
    """

    def __init__(self):
        """Initialize placeholder."""
        super().__init__()
        self._default_depth: float = 0.5

    def _create_metadata(self) -> PluginMetadata:
        """Create metadata with deprecation flag."""
        if hasattr(self, '_decorator_metadata'):
            return self._decorator_metadata

        return PluginMetadata(
            name="placeholder_depth_model",
            version="1.0.0",
            plugin_type=PluginType.DEPTH_MODEL,
            description="Placeholder depth model returning uniform depth",
            author="Transformation Portal",
            deprecated=True,
            replacement="edge_depth_estimator",
        )

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with default depth value."""
        if config:
            self._default_depth = config.get("default_depth", self._default_depth)
            self._config = config

        self._initialized = True
        logger.warning("Using PlaceholderDepthModel - this should be replaced "
                       "with a proper depth estimation model")

    def estimate_depth(
        self,
        image: Union[Image.Image, np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """Return uniform depth map.

        Args:
            image: Input image (used for dimensions only)
            **kwargs: Ignored

        Returns:
            Uniform depth map
        """
        if not self._initialized:
            raise RuntimeError("Not initialized")

        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            height, width = image.shape[:2]

        return np.full(
            (height, width),
            self._default_depth,
            dtype=np.float32
        )

    def validate(self) -> bool:
        """Validate state."""
        return self._initialized

    def cleanup(self) -> None:
        """Clean up."""
        super().cleanup()
