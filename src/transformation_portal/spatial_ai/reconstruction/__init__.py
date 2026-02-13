"""3D Reconstruction module (Phase 2.3) - Gaussian Splatting integration.

This module provides depth-guided 3D scene reconstruction using:
- 3D Gaussian Splatting (Inria research license)
- Multi-view geometry
- Integration with depth maps (Phase 1)
- Integration with segmentation masks (Phase 2.1)
- Integration with PBR materials (Phase 2.2)

Key Features:
- Novel view synthesis
- Mesh export (PLY, OBJ)
- Geometric validation (RMSE < 2% target)
- Tier restriction enforcement (research-only)

License Warning:
    This module uses Inria 3D Gaussian Splatting which requires
    research tier (apex_research or higher). Commercial use requires
    separate license agreement with Inria.

Public API:
    - CameraParams: Camera parameter specification
    - GaussianSplat: 3D Gaussian representation
    - ReconstructionInput: Multi-view input contract
    - Scene3D: Complete 3D scene
    - GaussianBackend: 3DGS reconstruction engine
    - SceneBuilder: High-level scene construction
    - MeshExporter: Export to PLY/OBJ formats
    - GeometricValidator: Quality validation
    - LicenseRestrictionError: License violation exception

Usage:
    >>> from transformation_portal.spatial_ai.reconstruction import (
    ...     SceneBuilder,
    ...     MeshExporter,
    ...     GeometricValidator,
    ... )
    >>>
    >>> # Build 3D scene from multi-view images
    >>> builder = SceneBuilder(tier="apex_research")
    >>> scene = builder.build_from_images(
    ...     image_paths=["view1.png", "view2.png", "view3.png"],
    ...     cameras=cameras,
    ...     depth_maps=depth_maps,
    ... )
    >>>
    >>> # Validate quality
    >>> validator = GeometricValidator()
    >>> results = validator.validate_scene(scene)
    >>> print(f"Quality: {results['quality_grade']}, RMSE: {results['rmse']:.4f}")
    >>>
    >>> # Export to mesh
    >>> exporter = MeshExporter()
    >>> exporter.export_ply(scene, "scene.ply", include_attributes=True)

Architecture (ADR-027):
    - SpatialCaptureV1 contract (gamma=1.0)
    - Tier-based license enforcement
    - Integration with spatial_ai phases
    - Mock implementation for testing (verified revision required for production)

Performance Targets:
    - 3-view reconstruction: <30s on GPU
    - RMSE: <2% for production quality
    - Memory: <6GB VRAM for typical scenes
"""

from .contracts import CameraParams, GaussianSplat, LicenseRestrictionError, ReconstructionInput, Scene3D
from .gaussian_backend import GaussianBackend
from .geometric_validator import GeometricValidator
from .mesh_exporter import MeshExporter
from .scene_builder import SceneBuilder

__all__ = [
    # Core contracts
    "CameraParams",
    "GaussianSplat",
    "ReconstructionInput",
    "Scene3D",
    # Backend and builders
    "GaussianBackend",
    "SceneBuilder",
    # Export and validation
    "MeshExporter",
    "GeometricValidator",
    # Exceptions
    "LicenseRestrictionError",
]
