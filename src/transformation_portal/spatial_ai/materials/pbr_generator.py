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

        # Generate textures
        albedo, normal, roughness, metallic, ao, height, properties = self.backend_engine.generate_pbr_textures(
            rgb=mat_input.image,
            mask=mat_input.mask,
            depth=mat_input.depth,
            material_hint=mat_input.material_hint,
            config=config,
        )

        # Validate output via contract
        pbr_textures = PBRTextures(
            albedo=albedo,
            normal=normal,
            roughness=roughness,
            metallic=metallic,
            ambient_occlusion=ao,
            height=height,
            properties=properties,
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

    def unload_model(self):
        """Unload model from memory."""
        self.backend_engine.unload_model()
