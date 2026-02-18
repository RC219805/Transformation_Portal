"""Backend protocol for PBR texture generation (Phase 5F).

Defines the standard interface that all PBR generation backends must implement.
This enables clean integration with Gaussian Splatting, SuGaR, and other 3D
rendering systems in Phase 6+.

Protocol design principles:
- Backend-agnostic: Works with heuristic, PBRFusion, NVDIFFREC, MaterialGAN
- Type-safe: Mypy-compatible protocol
- Minimal coupling: No dependencies on specific model implementations
- Forward-compatible: Extensible for future backends
"""

from typing import Optional, Protocol

import numpy as np

from transformation_portal.spatial_ai.materials.contracts import MaterialGenerationConfig, PBRGenerationMetadata, PBRTextures


class PBRBackendProtocol(Protocol):
    """Protocol for PBR texture generation backends.

    All backends (heuristic, PBRFusion, NVDIFFREC, MaterialGAN) must implement
    this interface for compatibility with the spatial AI pipeline.

    Contract:
        - Input: Linear RGB (gamma=1.0) + optional depth + material hint
        - Output: PBRTextures with 6 texture maps + metadata
        - Determinism: Same inputs → same outputs (when possible)
        - Performance: <5s/MP for production backends
        - Memory: <2GB VRAM/RAM for single-image generation

    Usage:
        >>> backend: PBRBackendProtocol = MaterialBackend(backend="heuristic")
        >>> result = backend.generate(rgb, depth=depth, material_hint="wood")
        >>> assert result.metadata.backend == "heuristic_v5.0.0"
    """

    def generate(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        material_hint: Optional[str] = None,
        config: Optional[MaterialGenerationConfig] = None,
    ) -> PBRTextures:
        """Generate PBR textures from input image.

        Args:
            rgb: Linear RGB image (H, W, 3) float32, values in [0, ∞).
                Gamma must be 1.0 (linear space, not sRGB).
            mask: Optional segmentation mask (H, W) bool.
                If provided, only generate PBR for masked region.
            depth: Optional depth map (H, W) float32, normalized [0, 1].
                0=far, 1=near. Improves normal/AO quality.
            material_hint: Optional material category (e.g., "wood", "metal").
                Guides roughness/metallic generation.
            config: Optional generation configuration.
                If None, uses backend defaults.

        Returns:
            PBRTextures contract with all texture maps:
                - albedo: Base color (H, W, 3) float32 in [0, 1]
                - normal: Surface normals (H, W, 3) float32 in [-1, 1], normalized
                - roughness: Surface roughness (H, W) float32 in [0, 1]
                - metallic: Metallic property (H, W) float32 in [0, 1]
                - ambient_occlusion: AO (H, W) float32 in [0, 1]
                - height: Optional height map (H, W) float32 in [0, 1]
                - properties: Aggregated material properties
                - metadata: Generation metadata (backend version, parameters)

        Raises:
            ValueError: If input validation fails (wrong shape, dtype, gamma).
            RuntimeError: If backend unavailable (e.g., GPU model not installed).

        Contract Invariants:
            - All output arrays are float32
            - All arrays share same spatial dimensions (H, W)
            - Value ranges strictly enforced (see PBRTextures docstring)
            - Metadata always populated (for reproducibility)
        """
        ...


class PBRBackendFactory(Protocol):
    """Factory protocol for creating PBR backends.

    Enables dependency injection and testing. Useful for:
        - Mocking backends in tests
        - Dynamic backend selection based on hardware
        - Plugin architecture for custom backends

    Usage:
        >>> factory: PBRBackendFactory = MaterialBackendFactory()
        >>> backend = factory.create("heuristic", device="cpu")
    """

    def create(
        self,
        backend_type: str,
        device: str = "cpu",
        model_repo_id: Optional[str] = None,
        model_revision: Optional[str] = None,
    ) -> PBRBackendProtocol:
        """Create a PBR backend instance.

        Args:
            backend_type: Backend identifier ("heuristic", "pbr_fusion", etc.).
            device: Compute device ("cpu", "cuda", "mps").
            model_repo_id: Optional HuggingFace repo for neural backends.
            model_revision: Optional commit SHA for reproducibility.

        Returns:
            PBRBackendProtocol-compliant backend instance.

        Raises:
            ValueError: If backend_type unknown.
            RuntimeError: If backend dependencies unavailable.
        """
        ...


# Type alias for convenience
PBRBackend = PBRBackendProtocol


def validate_backend_protocol(backend: PBRBackendProtocol) -> bool:
    """Validate that a backend conforms to PBRBackendProtocol.

    Useful for runtime checks during plugin registration or testing.

    Args:
        backend: Backend instance to validate.

    Returns:
        True if backend conforms to protocol.

    Raises:
        TypeError: If backend missing required methods.
    """
    # Check for required method
    if not hasattr(backend, "generate"):
        raise TypeError(f"Backend {type(backend).__name__} missing 'generate' method")

    # Check method signature (basic validation)
    import inspect

    sig = inspect.signature(backend.generate)
    required_params = {"rgb"}
    actual_params = set(sig.parameters.keys())

    if not required_params.issubset(actual_params):
        missing = required_params - actual_params
        raise TypeError(f"Backend {type(backend).__name__}.generate() missing parameters: {missing}")

    return True


# Phase 6+ Integration Notes:
#
# Gaussian Splatting Integration:
#   - PBRTextures.albedo → splat base color
#   - PBRTextures.normal → splat surface orientation
#   - PBRTextures.roughness → splat BRDF parameter
#   - PBRTextures.metallic → splat material type
#   - PBRTextures.ambient_occlusion → splat lighting occlusion
#
# SuGaR 3D Mesh Integration:
#   - PBRTextures → UV-mapped texture channels
#   - Normal maps → baked into geometry or tangent space
#   - Height maps → displacement mapping
#
# Multi-view Consistency:
#   - Same material_hint across views for consistent PBR
#   - Depth-aware PBR reduces multi-view flickering
#   - Metadata tracks per-view generation parameters
