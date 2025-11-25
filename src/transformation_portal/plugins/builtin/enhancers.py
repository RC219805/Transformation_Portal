"""Built-in enhancer plugins demonstrating EnhancerPlugin pattern.

These plugins serve as reference implementations for creating
custom enhancement plugins.
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ..decorators import measure_performance, plugin
from ..interface import EnhancerPlugin, PluginMetadata, PluginType

logger = logging.getLogger(__name__)


@plugin(
    name="contrast_enhancer",
    plugin_type=PluginType.ENHANCER,
    version="1.0.0",
    description="Enhance image contrast with adjustable strength",
    author="Transformation Portal",
    tags=["contrast", "enhancement", "builtin"],
    auto_register=False,
)
class ContrastEnhancer(EnhancerPlugin):
    """Contrast enhancement plugin.

    A reference implementation demonstrating:
    - EnhancerPlugin interface with strength parameter
    - Proper range handling
    - Configuration validation

    Example:
        >>> enhancer = ContrastEnhancer()
        >>> enhancer.initialize({"base_factor": 1.5})
        >>> enhanced = enhancer.enhance(image, strength=0.8)
    """

    def __init__(self):
        """Initialize contrast enhancer."""
        super().__init__()
        self._base_factor: float = 1.5  # 1.0 = no change, >1 = more contrast
        self._min_factor: float = 0.5
        self._max_factor: float = 3.0

    def _create_metadata(self) -> PluginMetadata:
        """Create plugin metadata."""
        if hasattr(self, '_decorator_metadata'):
            return self._decorator_metadata

        return PluginMetadata(
            name="contrast_enhancer",
            version="1.0.0",
            plugin_type=PluginType.ENHANCER,
            description="Enhance image contrast with adjustable strength",
            author="Transformation Portal",
            tags=["contrast", "enhancement", "builtin"],
        )

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with configuration.

        Args:
            config: Configuration with optional keys:
                - base_factor: Base contrast factor (default: 1.5)
                - min_factor: Minimum allowed factor (default: 0.5)
                - max_factor: Maximum allowed factor (default: 3.0)
        """
        if config:
            self._base_factor = config.get("base_factor", self._base_factor)
            self._min_factor = config.get("min_factor", self._min_factor)
            self._max_factor = config.get("max_factor", self._max_factor)
            self._config = config

        # Validate
        if self._min_factor > self._max_factor:
            raise ValueError("min_factor cannot be greater than max_factor")

        self._initialized = True
        logger.info(f"ContrastEnhancer initialized with base_factor={self._base_factor}")

    @measure_performance
    def enhance(
        self,
        image: Union[Image.Image, np.ndarray],
        strength: float = 1.0,
        **kwargs
    ) -> Union[Image.Image, np.ndarray]:
        """Enhance image contrast.

        Args:
            image: PIL Image or numpy array
            strength: Enhancement strength 0.0-1.0 (0=no change, 1=full effect)
            **kwargs: Additional options

        Returns:
            Enhanced image
        """
        if not self._initialized:
            raise RuntimeError("Enhancer not initialized")

        strength = max(0.0, min(1.0, strength))

        # Handle numpy input
        is_numpy = isinstance(image, np.ndarray)
        if is_numpy:
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Calculate effective factor
        # strength=0 -> factor=1.0 (no change)
        # strength=1 -> factor=base_factor
        factor = 1.0 + (self._base_factor - 1.0) * strength
        factor = max(self._min_factor, min(self._max_factor, factor))

        # Apply contrast enhancement
        if pil_image.mode == 'RGBA':
            # Handle alpha separately
            rgb = pil_image.convert('RGB')
            enhanced_rgb = ImageEnhance.Contrast(rgb).enhance(factor)
            enhanced = enhanced_rgb.convert('RGBA')
            enhanced.putalpha(pil_image.split()[3])
        else:
            enhanced = ImageEnhance.Contrast(pil_image).enhance(factor)

        return np.array(enhanced) if is_numpy else enhanced

    def validate(self) -> bool:
        """Validate enhancer state."""
        return (
            self._initialized and
            self._min_factor <= self._base_factor <= self._max_factor
        )

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        self._base_factor = 1.5


@plugin(
    name="sharpen_enhancer",
    plugin_type=PluginType.ENHANCER,
    version="1.0.0",
    description="Sharpen images using unsharp masking",
    author="Transformation Portal",
    tags=["sharpen", "enhancement", "builtin"],
    auto_register=False,
)
class SharpenEnhancer(EnhancerPlugin):
    """Sharpening enhancement plugin using unsharp mask.

    Demonstrates:
    - Unsharp mask technique
    - Multi-parameter control
    - Threshold-based selective sharpening
    """

    def __init__(self):
        """Initialize sharpen enhancer."""
        super().__init__()
        self._radius: float = 1.0
        self._percent: int = 150
        self._threshold: int = 3

    def _create_metadata(self) -> PluginMetadata:
        """Create plugin metadata."""
        if hasattr(self, '_decorator_metadata'):
            return self._decorator_metadata

        return PluginMetadata(
            name="sharpen_enhancer",
            version="1.0.0",
            plugin_type=PluginType.ENHANCER,
            description="Sharpen images using unsharp masking",
            author="Transformation Portal",
            tags=["sharpen", "enhancement", "builtin"],
        )

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with unsharp mask parameters.

        Args:
            config: Configuration with optional keys:
                - radius: Blur radius for mask (default: 1.0)
                - percent: Sharpening strength percentage (default: 150)
                - threshold: Edge threshold for selective sharpening (default: 3)
        """
        if config:
            self._radius = config.get("radius", self._radius)
            self._percent = config.get("percent", self._percent)
            self._threshold = config.get("threshold", self._threshold)
            self._config = config

        self._initialized = True
        logger.info(f"SharpenEnhancer initialized: radius={self._radius}, "
                    f"percent={self._percent}, threshold={self._threshold}")

    @measure_performance
    def enhance(
        self,
        image: Union[Image.Image, np.ndarray],
        strength: float = 1.0,
        **kwargs
    ) -> Union[Image.Image, np.ndarray]:
        """Sharpen image using unsharp mask.

        Args:
            image: PIL Image or numpy array
            strength: Enhancement strength 0.0-1.0
            **kwargs: Override parameters

        Returns:
            Sharpened image
        """
        if not self._initialized:
            raise RuntimeError("Enhancer not initialized")

        strength = max(0.0, min(1.0, strength))

        # Handle numpy input
        is_numpy = isinstance(image, np.ndarray)
        if is_numpy:
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Scale percent by strength
        effective_percent = int(self._percent * strength)

        if effective_percent == 0:
            return image  # No change

        # Get parameters with possible overrides
        radius = kwargs.get("radius", self._radius)
        threshold = kwargs.get("threshold", self._threshold)

        # Apply unsharp mask
        sharpened = pil_image.filter(
            ImageFilter.UnsharpMask(
                radius=radius,
                percent=effective_percent,
                threshold=threshold
            )
        )

        return np.array(sharpened) if is_numpy else sharpened

    def validate(self) -> bool:
        """Validate state."""
        return (
            self._initialized and
            self._radius >= 0 and
            self._percent >= 0 and
            self._threshold >= 0
        )

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        self._radius = 1.0
        self._percent = 150
        self._threshold = 3


@plugin(
    name="brightness_enhancer",
    plugin_type=PluginType.ENHANCER,
    version="1.0.0",
    description="Adjust image brightness",
    author="Transformation Portal",
    tags=["brightness", "enhancement", "builtin"],
    auto_register=False,
)
class BrightnessEnhancer(EnhancerPlugin):
    """Brightness adjustment plugin."""

    def __init__(self):
        """Initialize brightness enhancer."""
        super().__init__()
        self._target_factor: float = 1.2  # >1 = brighter

    def _create_metadata(self) -> PluginMetadata:
        """Create plugin metadata."""
        if hasattr(self, '_decorator_metadata'):
            return self._decorator_metadata

        return PluginMetadata(
            name="brightness_enhancer",
            version="1.0.0",
            plugin_type=PluginType.ENHANCER,
            description="Adjust image brightness",
            author="Transformation Portal",
            tags=["brightness", "enhancement", "builtin"],
        )

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with target brightness.

        Args:
            config: Configuration with:
                - target_factor: Target brightness factor (default: 1.2)
        """
        if config:
            self._target_factor = config.get("target_factor", self._target_factor)
            self._config = config

        self._initialized = True

    def enhance(
        self,
        image: Union[Image.Image, np.ndarray],
        strength: float = 1.0,
        **kwargs
    ) -> Union[Image.Image, np.ndarray]:
        """Adjust image brightness.

        Args:
            image: PIL Image or numpy array
            strength: Effect strength 0.0-1.0
            **kwargs: Additional options

        Returns:
            Brightness-adjusted image
        """
        if not self._initialized:
            raise RuntimeError("Enhancer not initialized")

        strength = max(0.0, min(1.0, strength))

        is_numpy = isinstance(image, np.ndarray)
        if is_numpy:
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Interpolate between 1.0 (no change) and target
        factor = 1.0 + (self._target_factor - 1.0) * strength

        # Apply
        enhanced = ImageEnhance.Brightness(pil_image).enhance(factor)

        return np.array(enhanced) if is_numpy else enhanced

    def validate(self) -> bool:
        """Validate state."""
        return self._initialized and self._target_factor > 0

    def cleanup(self) -> None:
        """Clean up."""
        super().cleanup()
        self._target_factor = 1.2
