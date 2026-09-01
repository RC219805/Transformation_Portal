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
    detect_transformers_torch_runtime_issue,
    detect_transformers_torch_version_issue,
    ensure_dependency_importable,
)
from .protocol import DepthResult, LicenseType

if TYPE_CHECKING:
    from ...depth.models.depth_anything_v2 import DepthAnythingV2Model  # noqa: F401
    from ...lux_depth_v3.config import EnhanceConfig
    from ...lux_depth_v3.execution_lifecycle import BackendCandidateAuthority

logger = logging.getLogger(__name__)


class DA2Backend:
    """Depth Anything V2 backend implementing DepthBackend protocol."""

    name = "da2"
    license_type = LicenseType.COMMERCIAL
    requires_checkpoint = False

    def __init__(
        self,
        config: Optional["EnhanceConfig"] = None,
        *,
        candidate_authority: Optional["BackendCandidateAuthority"] = None,
        canonical_plan_bytes: Optional[bytes] = None,
    ):
        if (candidate_authority is None) != (canonical_plan_bytes is None):
            raise ValueError("candidate_authority and canonical_plan_bytes must be provided together")
        carried_contract = candidate_authority.model_contract if candidate_authority is not None else None
        if candidate_authority is not None:
            if (
                candidate_authority.backend_id != self.name
                or carried_contract is None
                or carried_contract.backend_id != self.name
            ):
                raise ValueError("Canonical DA2 authority does not select a DA2 model contract")
            if carried_contract.model.canonical_key != "da2_small":
                raise ValueError("Canonical DA2 authority does not select the supported Small model")
            if type(canonical_plan_bytes) is not bytes or not canonical_plan_bytes:
                raise ValueError("Canonical DA2 authority requires non-empty immutable plan bytes")
        self._config = config
        self._candidate_authority = candidate_authority
        self._canonical_plan_bytes = canonical_plan_bytes
        self._model_revision = carried_contract.model.revision if carried_contract is not None else None
        self._device = self._resolve_device(config, candidate_authority)
        self._model: Optional[DepthAnythingV2Model] = None

    def _resolve_device(
        self,
        config: Optional["EnhanceConfig"],
        candidate_authority: Optional["BackendCandidateAuthority"] = None,
    ) -> str:
        """Resolve device from config, defaulting to CPU.

        This method does NOT auto-detect accelerators (MPS/CUDA) because
        importing torch here would load libomp.dylib on macOS. If a depth
        backend subprocess (running in a separate venv) later loads its
        own libomp, the process aborts with "OMP: Error #15".

        The orchestrator passes an explicit depth_device, so production
        workflows always get the correct device. CPU is the safe default
        for ad-hoc or test instantiation. Device validation happens at
        compute() time when torch is actually needed.
        """
        if candidate_authority is not None:
            carried = str(candidate_authority.device or "").strip().lower()
            if carried and carried != "auto":
                if carried == "cuda":
                    raise ValueError("Canonical DA2 authority cannot select unsupported CUDA execution")
                if carried not in {"cpu", "mps"}:
                    raise ValueError(f"Canonical DA2 authority selects unsupported device {carried!r}")
                return carried

        if config is not None:
            requested = getattr(config, "depth_device", None)
            if isinstance(requested, str):
                requested = requested.lower()
                if requested == "cuda":
                    logger.warning(
                        "Requested DA2 device=cuda" " but DA2 adapter only supports" " cpu/mps; using cpu.",
                    )
                    return "cpu"
                if requested in {"cpu", "mps"}:
                    return requested

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

        transformers_module = ensure_dependency_importable("transformers")
        torch_module = ensure_dependency_importable("torch")

        runtime_issue = detect_transformers_torch_runtime_issue(torch_module, transformers_module)
        if runtime_issue:
            raise ImportError(runtime_issue)

        runtime_issue = detect_transformers_torch_version_issue(torch_version, transformers_version)
        if runtime_issue:
            raise ImportError(runtime_issue)

    def _load_model(self) -> None:
        if self._model is not None:
            return

        from ...depth.models.depth_anything_v2 import DepthAnythingV2Model as DepthAnythingV2Model  # noqa: F811
        from ...depth.models.depth_anything_v2 import ModelBackend, ModelVariant

        # Validate MPS availability at compute time (torch is now safe to import)
        if self._device == "mps":
            try:
                import torch

                mps_available = bool(torch.backends.mps.is_available())
            except OPTIONAL_IMPORT_EXCEPTIONS as exc:
                if self._candidate_authority is not None:
                    raise RuntimeError(
                        "Canonical DA2 candidate planned device='mps', but MPS availability could not be verified"
                    ) from exc
                logger.warning("PyTorch not available; DA2 falling back to CPU.")
                self._device = "cpu"
            else:
                if not mps_available:
                    if self._candidate_authority is not None:
                        raise RuntimeError("Canonical DA2 candidate planned device='mps', but MPS is unavailable")
                    logger.warning(
                        "Requested DA2 device=mps" " but MPS is unavailable;" " falling back to cpu.",
                    )
                    self._device = "cpu"

        if self._device == "mps":
            backend = ModelBackend.PYTORCH_MPS
            model_device = "mps"
        else:
            backend = ModelBackend.PYTORCH_CPU
            model_device = "cpu"

        if self._model_revision is not None:
            self._model = DepthAnythingV2Model(
                variant=ModelVariant.SMALL,
                backend=backend,
                device=model_device,
                model_revision=self._model_revision,
            )
        else:
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
            if self._candidate_authority is not None and requested_device != self._device:
                raise ValueError(f"DA2 device override {requested_device!r} disagrees with carried authority {self._device!r}")
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
        if self._candidate_authority is not None:
            metadata["execution_authority"] = {
                "plan_fingerprint_sha256": self._candidate_authority.plan_fingerprint_sha256,
                "candidate_id": self._candidate_authority.candidate_id,
                "model_backend_id": self._candidate_authority.constituent_backend_id,
                "executed_backend_id": self.name,
            }

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
