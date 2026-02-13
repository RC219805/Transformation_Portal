"""Materials module for Spatial AI Foundation (Phase 2.2).

This module provides PBR texture generation for architectural visualization,
with support for neural backends (NVDIFFREC, MaterialGAN) and CPU fallback.

Architecture (ADR-027):
- Isolated from lux_depth_v3 (ADR-023 compliance)
- Contract-driven design (input/output validation)
- Integration with Phase 2.1 segmentation
- Lazy model loading for performance

Components:
- contracts: MaterialInput, PBRTextures, MaterialProperties, MaterialGenerationConfig
- material_backend: NVDIFFREC/MaterialGAN wrapper with HF integration
- pbr_generator: High-level orchestrator with contract validation
- heuristic_fallback: CPU-based classical image processing fallback

Example:
    >>> from transformation_portal.spatial_ai.materials import PBRGenerator
    >>> generator = PBRGenerator(backend="nvdiffrec", device="cuda")
    >>> result = generator.generate(linear_rgb, gamma=1.0, mask=seg_mask)
    >>> assert result.albedo.shape == linear_rgb.shape
    >>> assert result.normal.dtype == np.float32
"""

from transformation_portal.spatial_ai.materials.contracts import (
    MaterialGenerationConfig,
    MaterialInput,
    MaterialProperties,
    PBRTextures,
)
from transformation_portal.spatial_ai.materials.heuristic_fallback import HeuristicFallback
from transformation_portal.spatial_ai.materials.material_backend import MaterialBackend
from transformation_portal.spatial_ai.materials.pbr_generator import PBRGenerator

__all__ = [
    "MaterialInput",
    "PBRTextures",
    "MaterialProperties",
    "MaterialGenerationConfig",
    "MaterialBackend",
    "PBRGenerator",
    "HeuristicFallback",
]
