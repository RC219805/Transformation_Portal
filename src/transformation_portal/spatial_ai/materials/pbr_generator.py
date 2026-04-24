"""PBR texture generator orchestrator (Phase 2.2).

High-level API for PBR texture generation with contract validation,
integration with Phase 2.1 segmentation, and multi-segment support.

Example:
    >>> from transformation_portal.spatial_ai.materials import PBRGenerator
    >>> generator = PBRGenerator(backend="nvdiffrec", device="cuda")
    >>> result = generator.generate(
    ...     image=linear_rgb,
    ...     gamma=1.0,
    ...     mask=segmentation_mask,
    ...     depth=depth_map,
    ... )
    >>> assert result.albedo.shape == linear_rgb.shape
"""

from typing import List, Optional

import numpy as np

from transformation_portal.spatial_ai.materials.contracts import (
    MaterialGenerationConfig,
    MaterialInput,
    MaterialProperties,
    PBRTextures,
)
from transformation_portal.spatial_ai.materials.material_backend import MaterialBackend


class PBRGenerator:
    """High-level PBR texture generator with contract validation.

    This orchestrator:
    - Validates inputs via MaterialInput contract
    - Routes to appropriate backend (NVDIFFREC/MaterialGAN/heuristic)
    - Validates outputs via PBRTextures contract
    - Supports batch processing of multiple segments
    - Integrates with Phase 2.1 segmentation masks
    """

    def __init__(
        self,
        backend: str = "heuristic",
        device: str = "cuda",
        model_repo_id: Optional[str] = None,
        model_revision: Optional[str] = None,
    ):
        """Initialize PBR generator.

        Args:
            backend: Backend to use ("nvdiffrec", "material_gan", "heuristic").
            device: Compute device ("cuda", "mps", "cpu").
            model_repo_id: Optional HuggingFace model repo ID.
            model_revision: Optional HuggingFace commit SHA.
        """
        self.backend_engine = MaterialBackend(
            backend=backend,
            device=device,
            model_repo_id=model_repo_id,
            model_revision=model_revision,
        )

    def generate(
        self,
        image: np.ndarray,
        gamma: float,
        mask: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        material_hint: Optional[str] = None,
        config: Optional[MaterialGenerationConfig] = None,
    ) -> PBRTextures:
        """Generate PBR textures for input image.

        Args:
            image: Linear RGB image (H, W, 3) float32.
            gamma: Gamma value (must be 1.0).
            mask: Optional segmentation mask (H, W) bool.
            depth: Optional depth map (H, W) float32.
            material_hint: Optional material category.
            config: Optional generation configuration.

        Returns:
            PBRTextures object with validated outputs.
        """
        # Validate input via contract
        mat_input = MaterialInput(
            image=image,
            gamma=gamma,
            mask=mask,
            depth=depth,
            material_hint=material_hint,
        )

        # Generate textures - returns PBRTextures directly now
        pbr_textures = self.backend_engine.generate_pbr_textures(
            rgb=mat_input.image,
            mask=mat_input.mask,
            depth=mat_input.depth,
            material_hint=mat_input.material_hint,
            config=config,
        )

        return pbr_textures

    def generate_batch(
        self,
        image: np.ndarray,
        gamma: float,
        masks: List[np.ndarray],
        depth: Optional[np.ndarray] = None,
        material_hints: Optional[List[str]] = None,
        config: Optional[MaterialGenerationConfig] = None,
    ) -> List[PBRTextures]:
        """Generate PBR textures for multiple segments.

        This is useful for processing multiple objects/materials from Phase 2.1
        segmentation results.

        Args:
            image: Linear RGB image (H, W, 3) float32.
            gamma: Gamma value (must be 1.0).
            masks: List of N segmentation masks (H, W) bool.
            depth: Optional depth map (H, W) float32.
            material_hints: Optional list of N material categories.
            config: Optional generation configuration.

        Returns:
            List of N PBRTextures objects, one per segment.
        """
        config = config or self.backend_engine._build_generation_config()
        decision = self.backend_engine.resolve_backend_decision(config.backend)
        if decision.executed_backend == "heuristic" and not (
            config.strict_backend and decision.requested_backend != decision.executed_backend
        ):
            return self._generate_batch_heuristic_shared(
                image=image,
                gamma=gamma,
                masks=masks,
                depth=depth,
                material_hints=material_hints,
                config=config,
            )

        results = []

        # Process each segment
        for i, mask in enumerate(masks):
            # Get material hint for this segment
            hint = material_hints[i] if material_hints else None

            # Generate PBR textures for this segment
            pbr = self.generate(
                image=image,
                gamma=gamma,
                mask=mask,
                depth=depth,
                material_hint=hint,
                config=config,
            )

            results.append(pbr)

        return results

    def _generate_batch_heuristic_shared(
        self,
        *,
        image: np.ndarray,
        gamma: float,
        masks: List[np.ndarray],
        depth: Optional[np.ndarray],
        material_hints: Optional[List[str]],
        config: MaterialGenerationConfig,
    ) -> List[PBRTextures]:
        """Generate heuristic PBR textures by sharing full-image intermediates.

        The heuristic backend computes full-image albedo/normal/roughness/AO
        before applying each mask, so one unmasked generation per material hint
        is equivalent to repeated per-mask generation and avoids duplicate
        bilateral/Sobel/variance passes.
        """
        MaterialInput(image=image, gamma=gamma, depth=depth)
        base_by_hint: dict[Optional[str], PBRTextures] = {}
        results: List[PBRTextures] = []

        for idx, mask in enumerate(masks):
            hint = material_hints[idx] if material_hints else None
            MaterialInput(image=image, gamma=gamma, mask=mask, depth=depth, material_hint=hint)
            if hint not in base_by_hint:
                base_by_hint[hint] = self.backend_engine.generate_pbr_textures(
                    rgb=image,
                    mask=None,
                    depth=depth,
                    material_hint=hint,
                    config=config,
                )
            base = base_by_hint[hint]
            mask_bool = np.asarray(mask, dtype=bool)

            albedo = base.albedo.copy()
            normal = base.normal.copy()
            roughness = base.roughness.copy()
            metallic = base.metallic.copy()
            ao = base.ambient_occlusion.copy()
            height = base.height.copy() if base.height is not None else None

            albedo[~mask_bool] = 0.0
            normal[~mask_bool] = [0.0, 0.0, 1.0]
            roughness[~mask_bool] = 0.5
            metallic[~mask_bool] = 0.0
            ao[~mask_bool] = 1.0
            if height is not None:
                height[~mask_bool] = 0.5

            properties = MaterialProperties(
                roughness_mean=float(np.mean(roughness[mask_bool])),
                metallic_mean=float(np.mean(metallic[mask_bool])),
                ao_strength=float(np.mean(1.0 - ao[mask_bool])),
                normal_strength=config.normal_strength,
            )
            results.append(
                PBRTextures(
                    albedo=albedo,
                    normal=normal,
                    roughness=roughness,
                    metallic=metallic,
                    ambient_occlusion=ao,
                    height=height,
                    properties=properties,
                    metadata=base.metadata,
                )
            )

        return results

    def unload_model(self):
        """Unload model from memory."""
        self.backend_engine.unload_model()
