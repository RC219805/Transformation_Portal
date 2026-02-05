"""Depth Anything V3 backend adapter for unified backend registry.

Wraps existing DA3InferenceEngine to provide DepthBackend interface with
consistent contract and license governance.

See ADR-019 for architectural rationale.
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


class DA3Backend:
    """Depth Anything V3 backend adapter implementing DepthBackend protocol.

    Wraps DA3InferenceEngine for use with DepthBackendRegistry.
    Provides relative depth (0-1 normalized) using transformers pipeline.

    Attributes:
        name: Backend identifier ("da3").
        license_type: COMMERCIAL (MIT license).
        requires_checkpoint: False (models auto-downloaded from HuggingFace).

    License:
        MIT License - unrestricted commercial and non-commercial use.
        See: https://github.com/DepthAnything/Depth-Anything-V3

    Example:
        >>> from transformation_portal.depth.backends import DepthBackendRegistry
        >>> from transformation_portal.lux_depth_v3 import EnhanceConfig
        >>>
        >>> config = EnhanceConfig(depth_device="mps")
        >>> registry = DepthBackendRegistry()
        >>> backend = registry.get_backend("da3", config)
        >>> result = backend.compute(image)
        >>> print(f"Relative depth, shape: {result.depth_map.shape}")
    """

    # Backend protocol attributes
    name = "da3"
    license_type = LicenseType.COMMERCIAL
    requires_checkpoint = False

    def __init__(self, config: Optional["EnhanceConfig"] = None):
        """Initialize DA3 backend.

        Args:
            config: EnhanceConfig with depth backend settings.
                If None, uses defaults (CPU device).
        """
        self._config = config
        self._engine = None
        self._device = self._resolve_device(config)
        self._model_variant = self._resolve_model_variant(config)

    def _resolve_device(self, config: Optional["EnhanceConfig"]) -> str:
        """Resolve device from config or auto-detect."""
        if config is not None:
            device = getattr(config, "depth_device", None)
            if device:
                return device

        # Auto-detect device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
        except ImportError:
            logger.debug("PyTorch not installed; falling back to CPU for DA3Backend.")

        return "cpu"

    def _resolve_model_variant(self, config: Optional["EnhanceConfig"]):
        """Resolve model variant from config."""
        if config is not None:
            variant = getattr(config, "model_variant", None)
            if variant:
                return variant

        # Default to METRIC_LARGE
        from ...lux_depth_v3.config import ModelVariant

        return ModelVariant.METRIC_LARGE

    def ensure_available(self) -> None:
        """Ensure DA3 dependencies are available.

        DA3 is always available if transformers and torch are installed.
        Models are auto-downloaded from HuggingFace Hub on first use.

        Raises:
            ImportError: If required packages not installed.
        """
        # Check transformers
        try:
            import transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                "transformers package not installed.\n\n"
                "Install with:\n"
                "  pip install transformers\n\n"
                "See: https://huggingface.co/docs/transformers"
            )

        # Check torch
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "torch package not installed.\n\n"
                "Install with:\n"
                "  pip install torch\n\n"
                "See: https://pytorch.org/get-started/locally/"
            )

        logger.debug("DA3 backend dependencies available")

    def compute(
        self,
        image: Union[Image.Image, np.ndarray],
        device: Optional[str] = None,
    ) -> DepthResult:
        """Estimate relative depth from image.

        Args:
            image: Input image as PIL Image or numpy array (H, W, 3).
            device: Optional device override (cpu, cuda, mps).
                    Note: Device override requires engine reload.

        Returns:
            DepthResult with relative depth (0-1 normalized).

        Raises:
            RuntimeError: If inference fails.
        """
        # Ensure dependencies available
        self.ensure_available()

        # Lazy-load engine
        use_device = device or self._device
        if self._engine is None or device is not None:
            self._load_engine(use_device)

        # Convert image to format expected by DA3InferenceEngine
        if isinstance(image, np.ndarray):
            # DA3InferenceEngine.predict() accepts PIL.Image (after PR #841)
            image_pil = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))
            image_array = image
        else:
            image_pil = image.convert("RGB")
            image_array = np.array(image_pil)

        # Run inference
        result = self._engine.predict(image_pil)

        # Convert to unified DepthResult
        return DepthResult(
            depth_map=result.depth_map.astype(np.float32),
            original_image=image_array,
            metadata=result.metadata,
            depth_units="relative",  # DA3 produces relative depth (0-1)
            focal_length_px=None,  # DA3 doesn't estimate focal length
            field_of_view_deg=None,
            backend_id=self.name,
            device=use_device,
            dtype="float32",
            input_size=(image_array.shape[0], image_array.shape[1]),
        )

    def get_cache_key(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Generate deterministic cache key for this image.

        Cache key includes:
        - Image content hash
        - Model variant
        - Device
        - Backend version

        Args:
            image: Input image.

        Returns:
            Cache key string.
        """
        # Hash image content
        if isinstance(image, np.ndarray):
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]
        else:
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]

        # Model identifier
        model_name = self._model_variant.value.name if hasattr(self._model_variant, "value") else "metric_large"

        return f"da3_{model_name}_{image_hash}_{self._device}_v1"

    def _load_engine(self, device: str) -> None:
        """Lazy-load DA3InferenceEngine."""
        from ...lux_depth_v3.config import DA3Config, DeviceConfig
        from ...lux_depth_v3.inference import DA3InferenceEngine

        # Build DA3Config
        device_config = DeviceConfig(device=device)
        da3_config = DA3Config(model_variant=self._model_variant, device=device_config)

        # Initialize engine
        commercial_use = not getattr(self._config, "non_commercial_ok", False) if self._config else True
        self._engine = DA3InferenceEngine(
            config=da3_config,
            commercial_use=commercial_use,
            validate_license_strict=False,  # DA3 has no license restrictions
        )

        logger.info(f"Loaded DA3 backend: model={self._model_variant.value.name} device={device}")
