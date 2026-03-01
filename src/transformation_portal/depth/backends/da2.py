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

from .protocol import DepthResult, LicenseType

if TYPE_CHECKING:
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
        self._model = None

    def _resolve_device(self, config: Optional["EnhanceConfig"]) -> str:
        if config is not None:
            requested = getattr(config, "depth_device", None)
            if requested:
                return requested

        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            logger.debug("PyTorch not installed; DA2 defaults to CPU.")

        return "cpu"

    @classmethod
    def required_packages(cls) -> list[str]:
        """Return required modules beyond torch for DA2."""
        return ["transformers"]

    def ensure_available(self) -> None:
        """Ensure DA2 dependencies are importable."""
        try:
            import transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError("transformers package not installed for DA2 backend.") from exc

        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError("torch package not installed for DA2 backend.") from exc

    def _load_model(self) -> None:
        if self._model is not None:
            return

        from ...depth.models.depth_anything_v2 import DepthAnythingV2Model, ModelBackend, ModelVariant

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
        logger.info("Loaded DA2 backend: variant=SMALL device=%s", model_device)

    def compute(
        self,
        image: Union[Image.Image, np.ndarray],
        device: Optional[str] = None,
    ) -> DepthResult:
        """Estimate relative depth using Depth Anything V2."""
        self.ensure_available()

        if device is not None and device != self._device:
            self._device = device
            self._model = None

        self._load_model()

        if isinstance(image, np.ndarray):
            image_np = image
            if image_np.max() <= 1.0:
                image_pil = Image.fromarray((np.clip(image_np, 0, 1) * 255).astype(np.uint8))
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
