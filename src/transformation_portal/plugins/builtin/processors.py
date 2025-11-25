"""Built-in processor plugins demonstrating ProcessorPlugin pattern.

These plugins serve as reference implementations for creating
custom processor plugins.
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image, ImageFilter

from ..decorators import measure_performance, plugin
from ..interface import PluginMetadata, PluginType, ProcessorPlugin

logger = logging.getLogger(__name__)


@plugin(
    name="gaussian_blur_processor",
    plugin_type=PluginType.PROCESSOR,
    version="1.0.0",
    description="Apply Gaussian blur to images with configurable radius",
    author="Transformation Portal",
    tags=["blur", "filter", "builtin"],
    auto_register=False,  # Don't auto-register, let manager handle it
)
class GaussianBlurProcessor(ProcessorPlugin):
    """Gaussian blur processor plugin.

    A reference implementation demonstrating:
    - ProcessorPlugin interface
    - Configuration handling
    - Input type handling (PIL Image, numpy array)
    - Proper initialization and cleanup

    Example:
        >>> processor = GaussianBlurProcessor()
        >>> processor.initialize({"radius": 3.0})
        >>> blurred = processor.process(image)
    """

    def __init__(self):
        """Initialize the processor."""
        super().__init__()
        self._radius: float = 2.0
        self._preserve_alpha: bool = True

    def _create_metadata(self) -> PluginMetadata:
        """Create plugin metadata."""
        # Use decorator metadata if available
        if hasattr(self, '_decorator_metadata'):
            return self._decorator_metadata

        return PluginMetadata(
            name="gaussian_blur_processor",
            version="1.0.0",
            plugin_type=PluginType.PROCESSOR,
            description="Apply Gaussian blur to images with configurable radius",
            author="Transformation Portal",
            tags=["blur", "filter", "builtin"],
        )

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize processor with configuration.

        Args:
            config: Configuration with optional keys:
                - radius: Blur radius (default: 2.0)
                - preserve_alpha: Keep alpha channel (default: True)
        """
        if config:
            self._radius = config.get("radius", self._radius)
            self._preserve_alpha = config.get("preserve_alpha", self._preserve_alpha)
            self._config = config

        # Validate radius
        if self._radius < 0:
            raise ValueError("Blur radius must be non-negative")

        self._initialized = True
        logger.info(f"GaussianBlurProcessor initialized with radius={self._radius}")

    @measure_performance
    def process(
        self,
        input_data: Union[Image.Image, np.ndarray],
        **kwargs
    ) -> Union[Image.Image, np.ndarray]:
        """Apply Gaussian blur to input image.

        Args:
            input_data: PIL Image or numpy array
            **kwargs: Optional overrides:
                - radius: Override configured radius

        Returns:
            Blurred image in same format as input
        """
        if not self._initialized:
            raise RuntimeError("Processor not initialized. Call initialize() first.")

        # Get effective radius (allow override)
        radius = kwargs.get("radius", self._radius)

        # Handle numpy array input
        if isinstance(input_data, np.ndarray):
            pil_image = Image.fromarray(input_data)
            result = self._apply_blur(pil_image, radius)
            return np.array(result)

        # Handle PIL Image
        return self._apply_blur(input_data, radius)

    def _apply_blur(self, image: Image.Image, radius: float) -> Image.Image:
        """Apply Gaussian blur to PIL Image.

        Args:
            image: Input PIL Image
            radius: Blur radius

        Returns:
            Blurred image
        """
        # Handle alpha channel separately if needed
        if self._preserve_alpha and image.mode == 'RGBA':
            # Split channels
            r, g, b, a = image.split()

            # Blur RGB channels
            rgb = Image.merge('RGB', (r, g, b))
            rgb_blurred = rgb.filter(ImageFilter.GaussianBlur(radius))

            # Recombine with original alpha
            r2, g2, b2 = rgb_blurred.split()
            return Image.merge('RGBA', (r2, g2, b2, a))

        return image.filter(ImageFilter.GaussianBlur(radius))

    def validate(self) -> bool:
        """Validate processor state."""
        return self._initialized and self._radius >= 0

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        self._radius = 2.0
        logger.debug("GaussianBlurProcessor cleaned up")

    def get_config_schema(self) -> Dict[str, Any]:
        """Get JSON schema for configuration.

        Returns:
            JSON Schema dictionary
        """
        return {
            "type": "object",
            "properties": {
                "radius": {
                    "type": "number",
                    "minimum": 0,
                    "default": 2.0,
                    "description": "Gaussian blur radius in pixels",
                },
                "preserve_alpha": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preserve alpha channel during blur",
                },
            },
        }


@plugin(
    name="resize_processor",
    plugin_type=PluginType.PROCESSOR,
    version="1.0.0",
    description="Resize images with multiple resampling algorithms",
    author="Transformation Portal",
    tags=["resize", "scale", "builtin"],
    auto_register=False,
)
class ResizeProcessor(ProcessorPlugin):
    """Image resize processor plugin.

    Demonstrates:
    - Multiple algorithm support
    - Dimension calculation
    - Aspect ratio handling
    """

    RESAMPLING_METHODS = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }

    def __init__(self):
        """Initialize resize processor."""
        super().__init__()
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._scale: Optional[float] = None
        self._method: str = "lanczos"
        self._maintain_aspect: bool = True

    def _create_metadata(self) -> PluginMetadata:
        """Create plugin metadata."""
        if hasattr(self, '_decorator_metadata'):
            return self._decorator_metadata

        return PluginMetadata(
            name="resize_processor",
            version="1.0.0",
            plugin_type=PluginType.PROCESSOR,
            description="Resize images with multiple resampling algorithms",
            author="Transformation Portal",
            tags=["resize", "scale", "builtin"],
        )

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with resize configuration.

        Args:
            config: Configuration with keys:
                - width: Target width (optional)
                - height: Target height (optional)
                - scale: Scale factor (optional, alternative to width/height)
                - method: Resampling method (nearest/bilinear/bicubic/lanczos)
                - maintain_aspect: Maintain aspect ratio (default: True)
        """
        if config:
            self._width = config.get("width")
            self._height = config.get("height")
            self._scale = config.get("scale")
            self._method = config.get("method", "lanczos")
            self._maintain_aspect = config.get("maintain_aspect", True)
            self._config = config

        if self._method not in self.RESAMPLING_METHODS:
            raise ValueError(f"Unknown resampling method: {self._method}")

        self._initialized = True

    def process(
        self,
        input_data: Union[Image.Image, np.ndarray],
        **kwargs
    ) -> Union[Image.Image, np.ndarray]:
        """Resize input image.

        Args:
            input_data: PIL Image or numpy array
            **kwargs: Override configuration

        Returns:
            Resized image
        """
        if not self._initialized:
            raise RuntimeError("Processor not initialized")

        # Handle numpy input
        is_numpy = isinstance(input_data, np.ndarray)
        if is_numpy:
            image = Image.fromarray(input_data)
        else:
            image = input_data

        # Calculate target dimensions
        target_size = self._calculate_target_size(
            image.size,
            kwargs.get("width", self._width),
            kwargs.get("height", self._height),
            kwargs.get("scale", self._scale),
        )

        # Resize
        method = self.RESAMPLING_METHODS[kwargs.get("method", self._method)]
        resized = image.resize(target_size, method)

        return np.array(resized) if is_numpy else resized

    def _calculate_target_size(
        self,
        original_size: tuple,
        width: Optional[int],
        height: Optional[int],
        scale: Optional[float],
    ) -> tuple:
        """Calculate target dimensions."""
        orig_w, orig_h = original_size

        if scale:
            return (int(orig_w * scale), int(orig_h * scale))

        if width and height and not self._maintain_aspect:
            return (width, height)

        if width and self._maintain_aspect:
            ratio = width / orig_w
            return (width, int(orig_h * ratio))

        if height and self._maintain_aspect:
            ratio = height / orig_h
            return (int(orig_w * ratio), height)

        if width and height:
            # Maintain aspect, fit within bounds
            ratio = min(width / orig_w, height / orig_h)
            return (int(orig_w * ratio), int(orig_h * ratio))

        # No resize specified, return original
        return original_size

    def validate(self) -> bool:
        """Validate configuration."""
        return (
            self._initialized and
            self._method in self.RESAMPLING_METHODS
        )

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        self._width = None
        self._height = None
        self._scale = None
