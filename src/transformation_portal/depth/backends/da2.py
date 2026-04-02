"""Depth Anything V2 backend adapter for unified backend registry.

Provides a lightweight DA2 fallback backend used by orchestrator runtime
fallback logic when primary backends fail operationally.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from PIL import Image

from ...core.ml_dependency_health import (
    OPTIONAL_IMPORT_EXCEPTIONS,
    _installed_version,
    detect_transformers_torch_version_issue,
    ensure_dependency_importable,
)
from .protocol import DepthResult, LicenseType

if TYPE_CHECKING:
    from ...depth.models.depth_anything_v2 import DepthAnythingV2Model  # noqa: F401
    from ...lux_depth_v3.config import EnhanceConfig

logger = logging.getLogger(__name__)


class DA2Backend:
    """Depth Anything V2 backend implementing DepthBackend protocol."""

    name = "da2"
    license_type = LicenseType.COMMERCIAL
    requires_checkpoint = False

    def __init__(self, config: Optional["EnhanceConfig"] = None):
        self._config = config
        self._device = self._resolve_device(config)
        self._model: Optional[DepthAnythingV2Model] = None

    def _resolve_device(self, config: Optional["EnhanceConfig"]) -> str:
        requested: Optional[str] = None
        if config is not None:
            requested = getattr(config, "depth_device", None)
            if isinstance(requested, str):
                requested = requested.lower()
        if requested == "cpu":
            return "cpu"
        if requested == "cuda":
            logger.warning(
                "Requested DA2 device=cuda" " but DA2 adapter only supports" " cpu/mps; falling back to cpu.",
            )
            return "cpu"
        if requested is None:
            return "cpu"

        try:
            import torch

            if requested == "mps":
                if torch.backends.mps.is_available():
                    return "mps"
                logger.warning(
                    "Requested DA2 device=mps" " but MPS is unavailable;" " falling back to cpu.",
                )
                return "cpu"
            if requested == "cpu":
                return "cpu"

            if torch.backends.mps.is_available():
                return "mps"
        except OPTIONAL_IMPORT_EXCEPTIONS:
            logger.debug("PyTorch not installed; DA2 defaults to CPU.")

        if requested in {"mps", "cuda"}:
            return "cpu"
        return "cpu"

    @classmethod
    def required_packages(cls) -> list[str]:
        """Return required modules beyond torch for DA2."""
        return ["transformers"]

    def ensure_available(self) -> None:
        """Ensure DA2 dependencies are present and version-compatible."""
        transformers_version = _installed_version("transformers")
        if transformers_version is None:
            raise ImportError(
                "transformers package not installed" " for DA2 backend.",
            )

        torch_version = _installed_version("torch")
        if torch_version is None:
            raise ImportError(
                "torch package not installed" " for DA2 backend.",
            )

        ensure_dependency_importable("transformers")
        ensure_dependency_importable("torch")

        runtime_issue = detect_transformers_torch_version_issue(torch_version, transformers_version)
        if runtime_issue:
            raise ImportError(runtime_issue)

    def _load_model(self) -> None:
        if self._model is not None:
            return

        from ...depth.models.depth_anything_v2 import DepthAnythingV2Model as DepthAnythingV2Model  # noqa: F811
        from ...depth.models.depth_anything_v2 import ModelBackend, ModelVariant

        if self._device == "mps":
            backend = ModelBackend.PYTORCH_MPS
            model_device = "mps"
        else:
            backend = ModelBackend.PYTORCH_CPU
            model_device = "cpu"

        self._model = DepthAnythingV2Model(
            variant=ModelVariant.SMALL,
            backend=backend,
            device=model_device,
        )
        logger.info(
            "Loaded DA2 backend:" " variant=SMALL device=%s",
            model_device,
        )

    def compute(
        self,
        image: Union[Image.Image, np.ndarray],
        device: Optional[str] = None,
    ) -> DepthResult:
        """Estimate relative depth using Depth Anything V2."""
        self.ensure_available()

        if device is not None:
            requested_device = str(device).lower()
            if requested_device not in {"cpu", "mps", "cuda"}:
                logger.warning(
                    "Unknown DA2 override" " device=%s;" " falling back to cpu.",
                    requested_device,
                )
                requested_device = "cpu"
            if requested_device == "cuda":
                logger.warning(
                    "Requested DA2 override" " device=cuda but only" " cpu/mps are supported;" " using cpu.",
                )
                requested_device = "cpu"
            if requested_device != self._device:
                self._device = requested_device
                self._model = None

        self._load_model()
        assert self._model is not None

        if isinstance(image, np.ndarray):
            image_np = image
            if image_np.max() <= 1.0:
                image_pil = Image.fromarray(
                    (np.clip(image_np, 0, 1) * 255).astype(np.uint8),
                )
            else:
                image_pil = Image.fromarray(image_np.astype(np.uint8))
        else:
            image_pil = image.convert("RGB")
            image_np = np.array(image_pil)

        estimate = self._model.estimate_depth(image_pil)
        depth = np.asarray(estimate.get("depth"), dtype=np.float32)
        metadata = dict(estimate.get("metadata") or {})
        metadata["source_depth_units"] = "relative"
        metadata["output_depth_units"] = "relative"
        metadata["output_normalization"] = "native_relative_0_1"

        return DepthResult(
            depth_map=depth,
            original_image=image_np,
            metadata=metadata,
            depth_units="relative",
            focal_length_px=None,
            field_of_view_deg=None,
            backend_id=self.name,
            device=self._device,
            dtype="float32",
            input_size=(image_np.shape[0], image_np.shape[1]),
            warnings=[],
        )

    def get_cache_key(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Generate deterministic cache key for DA2."""
        if isinstance(image, np.ndarray):
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]
        else:
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]
        return f"da2_small_{image_hash}_{self._device}_v1"
