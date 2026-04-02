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

from ...core.ml_dependency_health import (
    _installed_version,
    detect_transformers_torch_version_issue,
    ensure_dependency_importable,
)
from ...core.platform_matrix import CURRENT_PLATFORM
from .protocol import DepthResult, LicenseType

if TYPE_CHECKING:
    from ...lux_depth_v3.config import EnhanceConfig, ModelVariant
    from ...lux_depth_v3.inference import DA3InferenceEngine

logger = logging.getLogger(__name__)


class DA3Backend:
    """Depth Anything V3 backend adapter implementing DepthBackend protocol.

    Wraps DA3InferenceEngine for use with DepthBackendRegistry.
    Produces normalized relative depth (0-1) for downstream compatibility,
    while preserving source unit semantics in metadata.

    Attributes:
        name: Backend identifier ("da3").
        license_type: COMMERCIAL (MIT license).
        requires_checkpoint: False (models auto-downloaded from HuggingFace).

    License:
        MIT License - unrestricted commercial and non-commercial use.
        See: https://github.com/DepthAnything/Depth-Anything-V3

    Example:
        >>> from transformation_portal.depth.backends import (
        ...     DepthBackendRegistry
        ... )
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
        self._engine: Optional[DA3InferenceEngine] = None
        self._device = self._resolve_device(config)
        self._model_variant = self._resolve_model_variant(config)

    def _resolve_device(self, config: Optional["EnhanceConfig"]) -> str:
        """Resolve device from config or auto-detect."""
        if config is not None:
            device = getattr(config, "depth_device", None)
            if device:
                return device

        # Backend construction should not import torch just to probe accelerators.
        # The orchestrator passes an explicit depth_device; CPU is the safe default
        # for ad-hoc or test instantiation in partially provisioned environments.
        return "cpu"

    def _resolve_model_variant(self, config: Optional["EnhanceConfig"]) -> "ModelVariant":
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
        transformers_version = _installed_version("transformers")
        if transformers_version is None:
            raise ImportError(
                "transformers package not installed.\n\n"
                "Install with:\n"
                "  pip install transformers\n\n"
                "See: https://huggingface.co/docs/transformers"
            )

        torch_version = _installed_version("torch")
        if torch_version is None:
            raise ImportError(
                "torch package not installed.\n\n"
                "Install with:\n"
                "  pip install torch\n\n"
                "See: https://pytorch.org/get-started/locally/"
            )

        ensure_dependency_importable("transformers")
        ensure_dependency_importable("torch")

        runtime_issue = detect_transformers_torch_version_issue(torch_version, transformers_version)
        if runtime_issue:
            raise ImportError(runtime_issue)

        logger.debug("DA3 backend dependencies available")

    @classmethod
    def required_packages(cls) -> list[str]:
        """Return required import module names for DA3 backend.

        DA3 requires transformers (HuggingFace Transformers)
        for model inference.
        torch is handled by the APEX runner and not listed here.

        Returns:
            ["transformers"]
        """
        return ["transformers"]

    def _infer_source_depth_units(self, metadata: dict) -> str:
        """Infer source depth unit semantics from metadata."""
        resolved_model_id = str(metadata.get("resolved_model_id", "")).lower()
        requested_model_id = str(metadata.get("requested_model_id", "")).lower()
        model_hint = resolved_model_id or requested_model_id

        depth_tokens = ("metric", "da3nested", "nested-giant")
        if any(token in model_hint for token in depth_tokens):
            return "meters"
        return "relative"

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
            DepthResult with relative depth (0-1 normalized). Metadata includes
            source/output unit semantics for contract-level clarity.

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
            if image.max() <= 1.0:
                arr = (image * 255).astype(np.uint8)
            else:
                arr = image.astype(np.uint8)
            image_pil = Image.fromarray(arr)
            image_array = image
        else:
            image_pil = image.convert("RGB")
            image_array = np.array(image_pil)

        # Run inference
        if self._engine is None:
            raise RuntimeError("DA3 engine failed to initialize")
        result = self._engine.predict(image_pil)

        source_depth_units = self._infer_source_depth_units(result.metadata)
        normalized_metadata = dict(result.metadata)
        normalized_metadata["source_depth_units"] = source_depth_units
        normalized_metadata["output_depth_units"] = "relative"
        normalized_metadata["output_normalization"] = "minmax_0_1_per_image"

        warnings = []
        if source_depth_units == "meters":
            warnings.append("source metric depth normalized to" " relative [0,1] for unified" " pipeline output")

        # Convert to unified DepthResult
        return DepthResult(
            depth_map=result.depth_map.astype(np.float32),
            original_image=image_array,
            metadata=normalized_metadata,
            depth_units="relative",  # DA3 produces relative depth (0-1)
            focal_length_px=None,  # DA3 doesn't estimate focal length
            field_of_view_deg=None,
            backend_id=self.name,
            device=use_device,
            dtype="float32",
            input_size=(image_array.shape[0], image_array.shape[1]),
            warnings=warnings,
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
        if hasattr(self._model_variant, "value"):
            model_name = self._model_variant.value.name
        else:
            model_name = "metric_large"

        return f"da3_{model_name}_{image_hash}_{self._device}_v1"

    def _load_engine(self, device: str) -> None:
        """Lazy-load DA3InferenceEngine."""
        from ...lux_depth_v3.config import DA3Config, DeviceConfig
        from ...lux_depth_v3.inference import DA3InferenceEngine

        # Build DA3Config
        use_coreml = bool(
            self._config is not None
            and getattr(self._config, "use_coreml_backend", False)
            and CURRENT_PLATFORM is not None
            and CURRENT_PLATFORM.is_apple_silicon
        )
        device_config = DeviceConfig(device=device, use_coreml=use_coreml)
        da3_config = DA3Config(
            model_variant=self._model_variant,
            device=device_config,
        )

        # Initialize engine
        if self._config:
            commercial_use = not getattr(self._config, "non_commercial_ok", False)
        else:
            commercial_use = True
        self._engine = DA3InferenceEngine(
            config=da3_config,
            commercial_use=commercial_use,
            validate_license_strict=False,  # DA3 has no license restrictions
        )

        logger.info(
            "Loaded DA3 backend: model=%s device=%s",
            self._model_variant.value.name,
            device,
        )
