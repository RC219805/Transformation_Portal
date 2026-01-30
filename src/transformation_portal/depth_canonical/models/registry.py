"""Model registry for managing depth estimation models.

This module provides a centralized registry for Depth Anything V2 and V3 models.
Phase 2: Full model loading with lazy initialization and caching.
"""

import logging
from typing import Optional, Dict, Any

from ..config import ModelVariant, DeviceType

logger = logging.getLogger(__name__)


class DepthEstimationModel:
    """Abstract interface for depth estimation models."""

    def estimate(
        self,
        image: Any,
        output_size: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """Estimate depth from image.

        Args:
            image: Input image (PIL Image, numpy array, or path)
            output_size: Optional output size (height, width)

        Returns:
            Dictionary with:
                - 'depth': Normalized depth map [0, 1] as numpy array
                - 'depth_raw': Raw depth predictions
                - 'metadata': Model information and timing
        """
        raise NotImplementedError


class ModelRegistry:
    """Registry for managing depth estimation models.

    Phase 2: Full model loading with lazy initialization and caching.

    Features:
    - Lazy loading (models loaded on first use)
    - Model caching (avoid repeated loading)
    - Device auto-detection (CoreML/ANE, CUDA, MPS, CPU)
    - Support for DA2 and DA3 variants

    Example:
        >>> registry = ModelRegistry()
        >>> model = registry.get_model(
        ...     variant=ModelVariant.DA3_METRIC_LARGE,
        ...     device=DeviceType.MPS
        ... )
        >>> result = model.estimate(image)
        >>> depth_map = result['depth']  # Normalized [0, 1]
    """

    def __init__(self):
        """Initialize the model registry."""
        self._models: Dict[str, DepthEstimationModel] = {}
        self._supported_variants = {
            ModelVariant.DA3_METRIC_LARGE,
            ModelVariant.DA3_METRIC_BASE,
            ModelVariant.DA3_METRIC_SMALL,
            ModelVariant.DA2_LARGE,
            ModelVariant.DA2_BASE,
        }

    def get_model(
        self,
        variant: ModelVariant,
        device: Optional[DeviceType] = None,
        dtype: str = "float32"
    ) -> DepthEstimationModel:
        """Get or load a depth estimation model.

        Args:
            variant: Model variant to load
            device: Device to load model on (auto-detected if None)
            dtype: Data type for model inference

        Returns:
            Model instance ready for inference

        Raises:
            ValueError: If variant is not supported
            ImportError: If required dependencies not available
        """
        if variant not in self._supported_variants:
            raise ValueError(
                f"Unsupported model variant: {variant}. "
                f"Supported: {self._supported_variants}"
            )

        # Auto-detect device if not specified
        if device is None:
            device = self._auto_detect_device()

        # Generate cache key
        model_key = f"{variant.value}_{device.value}_{dtype}"

        # Return cached model if available
        if model_key in self._models:
            logger.debug("Using cached model: %s", model_key)
            return self._models[model_key]

        # Load new model
        logger.info("Loading model: %s on device: %s", variant.value, device.value)
        model = self._load_model(variant, device, dtype)

        # Cache the model
        self._models[model_key] = model

        return model

    def _auto_detect_device(self) -> DeviceType:
        """Auto-detect optimal device for current hardware.

        Priority:
        1. CoreML/ANE (Apple Silicon M-series)
        2. CUDA (NVIDIA GPU)
        3. MPS (Apple Silicon GPU)
        4. CPU (fallback)

        Returns:
            Optimal device type
        """
        try:
            import torch

            # Check for Apple Neural Engine via CoreML
            if torch.backends.mps.is_available():
                try:
                    import coremltools  # noqa: F401
                    # CoreML available, prefer for best performance
                    return DeviceType.COREML
                except ImportError:
                    # Fall back to MPS (Apple Silicon GPU)
                    return DeviceType.MPS

            # Check for CUDA
            if torch.cuda.is_available():
                return DeviceType.CUDA

        except ImportError:
            pass

        # CPU fallback
        return DeviceType.CPU

    def _load_model(
        self,
        variant: ModelVariant,
        device: DeviceType,
        dtype: str
    ) -> DepthEstimationModel:
        """Load a depth estimation model based on variant.

        Args:
            variant: Model variant to load
            device: Target device
            dtype: Data type

        Returns:
            Loaded model instance

        Raises:
            ImportError: If required dependencies not available
        """
        # DA3 models
        if variant in {
            ModelVariant.DA3_METRIC_LARGE,
            ModelVariant.DA3_METRIC_BASE,
            ModelVariant.DA3_METRIC_SMALL
        }:
            return self._load_da3_model(variant, device, dtype)

        # DA2 models
        if variant in {ModelVariant.DA2_LARGE, ModelVariant.DA2_BASE}:
            return self._load_da2_model(variant, device, dtype)

        raise ValueError(f"Unknown variant: {variant}")

    def _load_da3_model(
        self,
        variant: ModelVariant,
        device: DeviceType,
        dtype: str
    ) -> DepthEstimationModel:
        """Load Depth Anything V3 model.

        Args:
            variant: DA3 variant
            device: Target device
            dtype: Data type

        Returns:
            DA3 model wrapper
        """
        from .da3_wrapper import DA3ModelWrapper

        # Map variant to model ID
        model_id_map = {
            ModelVariant.DA3_METRIC_LARGE: "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large",
            ModelVariant.DA3_METRIC_BASE: "depth-anything/Depth-Anything-V2-Metric-Hypersim-Base",
            ModelVariant.DA3_METRIC_SMALL: "depth-anything/Depth-Anything-V2-Metric-Hypersim-Small",
        }

        model_id = model_id_map.get(variant)
        if not model_id:
            raise ValueError(f"No model ID mapping for {variant}")

        return DA3ModelWrapper(
            model_id=model_id,
            device=device,
            dtype=dtype
        )

    def _load_da2_model(
        self,
        variant: ModelVariant,
        device: DeviceType,
        dtype: str
    ) -> DepthEstimationModel:
        """Load Depth Anything V2 model.

        Args:
            variant: DA2 variant
            device: Target device
            dtype: Data type

        Returns:
            DA2 model wrapper
        """
        from .da2_wrapper import DA2ModelWrapper

        # Map variant to model ID
        model_id_map = {
            ModelVariant.DA2_LARGE: "depth-anything/Depth-Anything-V2-Large-hf",
            ModelVariant.DA2_BASE: "depth-anything/Depth-Anything-V2-Base-hf",
        }

        model_id = model_id_map.get(variant)
        if not model_id:
            raise ValueError(f"No model ID mapping for {variant}")

        return DA2ModelWrapper(
            model_id=model_id,
            device=device,
            dtype=dtype
        )

    def is_variant_supported(self, variant: ModelVariant) -> bool:
        """Check if a model variant is supported.

        Args:
            variant: Model variant to check

        Returns:
            True if variant is supported
        """
        return variant in self._supported_variants

    def clear_cache(self) -> None:
        """Clear cached models to free memory."""
        logger.info("Clearing model cache (%d models)", len(self._models))
        self._models.clear()
