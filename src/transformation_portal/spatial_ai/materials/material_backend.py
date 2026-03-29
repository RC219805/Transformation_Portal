"""Material backend for NVDIFFREC/MaterialGAN integration (Phase 2.2).

This module provides a unified interface to neural PBR texture generation,
with support for NVDIFFREC (preferred) and MaterialGAN (alternative).

Architecture:
- Lazy model loading (only load when needed)
- Automatic device detection (CUDA > MPS > CPU)
- Fallback to heuristic if GPU unavailable
- HuggingFace integration for model downloads

Licensing:
- NVDIFFREC: NVIDIA Source Code License (non-commercial/research only)
  SEE: https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt
- MaterialGAN: CC BY-NC 4.0 (research only)

IMPORTANT: NVDIFFREC is NOT BSD-3-Clause. The NVIDIA Source Code License
restricts use to non-commercial research and evaluation purposes only.
For commercial use, contact NVIDIA for a commercial license.

Performance:
- 1024x1024 PBR generation: <10s on GPU (RTX 3090)
- Memory: ~2-4GB VRAM
"""

import inspect
from typing import Any, Dict, Literal, Optional, cast

import numpy as np

from transformation_portal.spatial_ai.materials.contracts import (
    AvailabilityState,
    BackendDecision,
    MaterialGenerationConfig,
    MaterialInput,
    MaterialProperties,
    PBRGenerationMetadata,
    PBRTextures,
)
from transformation_portal.spatial_ai.materials.heuristic_fallback import HeuristicFallback

_MATERIAL_GENERATION_CONFIG_FIELDS = frozenset(inspect.signature(MaterialGenerationConfig).parameters)


class BackendResolutionWarning(UserWarning):
    """Warn when a requested materials backend resolves to a different executor."""


class MaterialBackend:
    """Unified backend for neural PBR texture generation.

    Supports multiple backends:
    - "pbr_fusion": PBRFusion diffusion model (Apache 2.0, commercial OK)
    - "nvdiffrec": NVIDIA Differentiable Rendering (NVIDIA Source Code License, research only)
    - "material_gan": MaterialGAN (CC BY-NC 4.0, research only)
    - "heuristic": CPU fallback (no ML dependencies)

    WARNING: nvdiffrec is NOT commercially licensed. The NVIDIA Source Code License
    restricts use to non-commercial research/evaluation. See NVDiffRecBackend for
    tier enforcement.

    Models are lazy-loaded on first use to minimize import time.
    """

    def __init__(
        self,
        backend: Literal["pbr_fusion", "nvdiffrec", "material_gan", "heuristic"] = "heuristic",
        device: Literal["cuda", "mps", "cpu"] = "cuda",
        model_repo_id: Optional[str] = None,
        model_revision: Optional[str] = None,
        generation_config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Initialize material backend.

        Args:
            backend: Backend to use ("pbr_fusion", "nvdiffrec", "material_gan", "heuristic").
            device: Compute device ("cuda", "mps", "cpu").
            model_repo_id: HuggingFace model repo ID (e.g., "nvidia/nvdiffrec").
            model_revision: HuggingFace commit SHA for reproducibility.
            generation_config_overrides: Optional mapping of attempted
                ``MaterialGenerationConfig`` field overrides at the instance
                level. ``None`` values are filtered out and ignored when
                stored. When building a generation config, only keys that
                correspond to valid ``MaterialGenerationConfig`` parameters are
                applied; any other keys are ignored. Overrides are merged using
                the following precedence (lowest to highest): 1) backend/
                constructor defaults, 2) instance-level
                ``generation_config_overrides``, 3) per-call overrides passed
                to ``_build_generation_config()``.
        """
        self.backend = backend
        self.device = device
        self.model_repo_id = model_repo_id
        self.model_revision = model_revision
        self.generation_config_overrides = {
            key: value for key, value in (generation_config_overrides or {}).items() if value is not None
        }

        # Lazy-loaded model instance
        self._model = None
        self._model_loaded = False

        # Heuristic fallback (always available)
        self._heuristic = HeuristicFallback()
        self._bilateral_filter_available = self._is_bilateral_filter_available()

    def clone_for_device(self, device: str) -> "MaterialBackend":
        """Create an equivalent backend bound to a new execution device."""
        overrides = dict(self.generation_config_overrides)
        overrides["device"] = device
        return MaterialBackend(
            backend=self.backend,
            device=cast(Literal["cuda", "mps", "cpu"], device),
            model_repo_id=self.model_repo_id,
            model_revision=self.model_revision,
            generation_config_overrides=overrides,
        )

    def _build_generation_config(
        self,
        *,
        device: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> MaterialGenerationConfig:
        """Build MaterialGenerationConfig from backend defaults plus overrides."""
        raw = {
            "backend": self.backend,
            "device": device or self.device,
            "model_repo_id": self.model_repo_id,
            "model_revision": self.model_revision,
        }
        raw.update(self.generation_config_overrides)
        raw.update({key: value for key, value in (overrides or {}).items() if value is not None})

        filtered: Dict[str, Any] = {key: value for key, value in raw.items() if key in _MATERIAL_GENERATION_CONFIG_FIELDS}
        return MaterialGenerationConfig(**filtered)

    @staticmethod
    def _is_bilateral_filter_available() -> bool:
        """Return True if OpenCV bilateral filtering is available."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            return False
        return True

    def generate(self, mat_input: MaterialInput) -> PBRTextures:
        """Generate PBR textures from MaterialInput contract.

        Args:
            mat_input: Validated material input contract.

        Returns:
            PBRTextures contract with all texture maps.

        Raises:
            ValueError: If input contract violated.
            RuntimeError: If generation fails.
        """
        # Contract is already validated in MaterialInput.__post_init__

        # Create config if not using default
        config = self._build_generation_config()

        # Call generate_pbr_textures - returns PBRTextures now
        return self.generate_pbr_textures(
            rgb=mat_input.image,
            mask=mat_input.mask,
            depth=mat_input.depth,
            material_hint=mat_input.material_hint,
            config=config,
        )

    def generate_pbr_textures(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        material_hint: Optional[str] = None,
        config: Optional[MaterialGenerationConfig] = None,
    ) -> PBRTextures:
        """Generate PBR textures for input image.

        Args:
            rgb: Linear RGB image (H, W, 3) float32.
            mask: Optional segmentation mask (H, W) bool.
            depth: Optional depth map (H, W) float32.
            material_hint: Optional material category hint.
            config: Optional generation configuration.

        Returns:
            PBRTextures contract with all texture maps and metadata.
        """
        # Use default config if not provided
        if config is None:
            config = self._build_generation_config()

        decision = self._resolve_backend_decision()

        if decision.requested_backend != decision.executed_backend:
            self._warn_backend_resolution(decision)

        # Route to appropriate backend (returns metadata tuple now)
        result = None
        if decision.executed_backend == "pbr_fusion":
            result = self._generate_pbr_fusion(rgb, mask, depth, material_hint, config, decision)
        elif decision.executed_backend == "nvdiffrec":
            result = self._generate_nvdiffrec(rgb, mask, depth, material_hint, config, decision)
        elif decision.executed_backend == "material_gan":
            result = self._generate_material_gan(rgb, mask, depth, material_hint, config, decision)
        elif decision.executed_backend == "heuristic":
            result = self._generate_heuristic(rgb, mask, depth, material_hint, config, decision)
        else:
            raise ValueError(f"Unknown backend: {decision.executed_backend}")

        # Validate/unpack result - handle both old (7-tuple) and new (8-tuple with metadata) formats
        if not isinstance(result, tuple):
            raise TypeError(
                f"Material backend '{self.backend}' returned {type(result)!r}, " "expected a tuple of 7 or 8 elements."
            )

        result_len = len(result)
        if result_len == 8:
            albedo, normal, roughness, metallic, ao, height, properties, metadata = result
        elif result_len == 7:
            # Fallback for backends that don't provide metadata yet
            albedo, normal, roughness, metallic, ao, height, properties = result
            metadata = None
        else:
            raise ValueError(
                f"Material backend '{self.backend}' returned {result_len} values; "
                "expected 7 (without metadata) or 8 (with metadata)."
            )

        # Wrap into PBRTextures contract
        return PBRTextures(
            albedo=albedo,
            normal=normal,
            roughness=roughness,
            metallic=metallic,
            ambient_occlusion=ao,
            height=height,
            properties=properties,
            metadata=metadata,
        )

    def _generate_heuristic(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray],
        depth: Optional[np.ndarray],
        material_hint: Optional[str],
        config: MaterialGenerationConfig,
        backend_decision: BackendDecision,
    ) -> tuple:
        """Generate textures using heuristic fallback."""
        albedo, normal, roughness, metallic, ao, height = self._heuristic.generate_pbr_textures(
            rgb=rgb,
            mask=mask,
            depth=depth,
            material_hint=material_hint,
            normal_strength=config.normal_strength,
            ao_intensity=config.ao_intensity,
        )

        # Compute aggregated properties
        if mask is not None:
            active_region = mask
        else:
            active_region = np.ones(roughness.shape, dtype=bool)

        properties = MaterialProperties(
            roughness_mean=float(np.mean(roughness[active_region])),
            metallic_mean=float(np.mean(metallic[active_region])),
            ao_strength=float(np.mean(1.0 - ao[active_region])),
            normal_strength=config.normal_strength,
        )

        # Phase 5F: Add generation metadata for reproducibility
        metadata = PBRGenerationMetadata(
            backend="heuristic_v5.0.0",
            normal_scale=config.normal_strength,
            ao_blend_ratio="0.7_concavity_0.3_variance",
            bilateral_enabled=self._bilateral_filter_available,
            material_hint=material_hint,
            depth_used=(depth is not None),
            backend_decision=backend_decision,
        )

        return albedo, normal, roughness, metallic, ao, height, properties, metadata

    def _generate_pbr_fusion(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray],
        depth: Optional[np.ndarray],
        material_hint: Optional[str],
        config: MaterialGenerationConfig,
        backend_decision: BackendDecision,
    ) -> tuple:
        """Generate textures using PBRFusion diffusion model.

        PBRFusion is a state-of-the-art (2026) diffusion-based model for PBR
        texture generation and upscaling. It produces high-quality albedo,
        normal, roughness, and height maps.

        License: Apache 2.0 (commercial use OK)
        Model: NightRaven109/PBRFusion4-RTXREMIX-Portable
        Requirements: ComfyUI + custom nodes OR direct PyTorch integration

        Implementation Status:
        ----------------------
        This is a PLACEHOLDER implementation pending ComfyUI integration.

        Two integration paths are documented:
        1. ComfyUI subprocess (easier, 32GB portable package)
        2. Direct PyTorch (cleaner, requires extracting models)

        For now, we fall back to the enhanced heuristic backend.
        To enable PBRFusion:
        - Install ComfyUI with PBRFusion nodes
        - Set PBRFUSION_PATH environment variable
        - See docs/guides/MATERIAL_PBR_GUIDE.md

        Args:
            rgb: Linear RGB image (H, W, 3) float32.
            mask: Optional segmentation mask (H, W) bool.
            depth: Optional depth map (H, W) float32.
            material_hint: Optional material category hint.
            config: Generation configuration.

        Returns:
            Tuple of (albedo, normal, roughness, metallic, ao, height, properties, metadata).
            Current fallback path always returns metadata via heuristic backend.

        Note:
            Falls back to heuristic if PBRFusion not installed.
        """
        import os

        # Check if PBRFusion is available
        pbrfusion_path = os.getenv("PBRFUSION_PATH")

        if pbrfusion_path and os.path.exists(pbrfusion_path):
            # Phase 5B roadmap item: ComfyUI subprocess integration
            # Implementation steps when ready:
            # 1. Write rgb to temp file
            # 2. Spawn ComfyUI with PBRFusion workflow
            # 3. Parse output PBR maps
            # 4. Return as tuple
            backend_decision = BackendDecision(
                requested_backend="pbr_fusion",
                executed_backend="heuristic",
                availability_state=AvailabilityState.RUNTIME_MISSING,
                fallback_reason=("PBRFusion runtime path exists, but direct ComfyUI integration is not implemented yet."),
                required_inputs=[],
                required_runtime=["comfyui_pbrfusion_workflow"],
            )
            self._warn_backend_resolution(backend_decision)
        else:
            backend_decision = BackendDecision(
                requested_backend="pbr_fusion",
                executed_backend="heuristic",
                availability_state=AvailabilityState.RUNTIME_MISSING,
                fallback_reason=(
                    "PBRFusion runtime is not installed or PBRFUSION_PATH is not set. "
                    "Install ComfyUI + PBRFusion nodes to enable it."
                ),
                required_inputs=[],
                required_runtime=["PBRFUSION_PATH", "comfyui_pbrfusion_workflow"],
            )
            self._warn_backend_resolution(backend_decision)

        # Fallback to enhanced heuristic (Phase 5C)
        return self._generate_heuristic(rgb, mask, depth, material_hint, config, backend_decision)

    def _generate_nvdiffrec(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray],
        depth: Optional[np.ndarray],
        material_hint: Optional[str],
        config: MaterialGenerationConfig,
        backend_decision: BackendDecision,
    ) -> tuple:
        """Generate textures using NVDIFFREC.

        This materials API does not satisfy the multiview input contract
        required by NVDIFFREC, so execution is currently routed elsewhere.
        """
        return self._generate_heuristic(rgb, mask, depth, material_hint, config, backend_decision)

    def _generate_material_gan(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray],
        depth: Optional[np.ndarray],
        material_hint: Optional[str],
        config: MaterialGenerationConfig,
        backend_decision: BackendDecision,
    ) -> tuple:
        """Generate textures using MaterialGAN.

        MaterialGAN is a research-only backend (CC BY-NC 4.0).

        This materials API does not satisfy the richer capture contract
        required by MaterialGAN, so execution is currently routed elsewhere.
        """
        return self._generate_heuristic(rgb, mask, depth, material_hint, config, backend_decision)

    def _resolve_backend_decision(self) -> BackendDecision:
        """Resolve requested backend to the backend actually executable in this API."""
        if self.backend == "heuristic":
            return BackendDecision(
                requested_backend="heuristic",
                executed_backend="heuristic",
                availability_state=AvailabilityState.AVAILABLE,
                fallback_reason=None,
                required_inputs=[],
                required_runtime=[],
            )

        if self.backend == "nvdiffrec":
            return BackendDecision(
                requested_backend="nvdiffrec",
                executed_backend="heuristic",
                availability_state=AvailabilityState.INPUT_CONTRACT_MISMATCH,
                fallback_reason=(
                    "NVDIFFREC requires a multiview capture bundle with camera poses "
                    "and lighting context; MaterialBackend only exposes single-image input."
                ),
                required_inputs=["multi_view_images", "camera_poses", "lighting_context"],
                required_runtime=["cuda", "nvdiffrast", "pinned_nvdiffrec_revision"],
            )

        if self.backend == "material_gan":
            return BackendDecision(
                requested_backend="material_gan",
                executed_backend="heuristic",
                availability_state=AvailabilityState.INPUT_CONTRACT_MISMATCH,
                fallback_reason=(
                    "MaterialGAN expects multi-image capture evidence with lighting "
                    "variation metadata; MaterialBackend only exposes single-image input."
                ),
                required_inputs=["multi_lighting_images", "capture_metadata_json"],
                required_runtime=["materialgan_runtime", "checkpoint_weights"],
            )

        if self.backend == "pbr_fusion":
            return BackendDecision(
                requested_backend="pbr_fusion",
                executed_backend="pbr_fusion",
                availability_state=AvailabilityState.AVAILABLE,
                fallback_reason=None,
                required_inputs=[],
                required_runtime=["PBRFUSION_PATH", "comfyui_pbrfusion_workflow"],
            )

        raise ValueError(f"Unknown backend: {self.backend}")

    @staticmethod
    def _warn_backend_resolution(decision: BackendDecision) -> None:
        """Emit a typed warning for backend fallback or unavailability."""
        import warnings

        if decision.fallback_reason:
            warnings.warn(decision.fallback_reason, BackendResolutionWarning, stacklevel=2)

    def unload_model(self) -> None:
        """Unload model from memory to free resources."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False

            # Clear CUDA cache if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
