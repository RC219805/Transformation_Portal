"""Data contracts for materials module (Phase 2.2).

Contract validation ensures:
- Gamma=1.0 enforcement (linear RGB only)
- Float32 dtype for texture maps
- Value ranges for PBR channels ([0,1] or normalized)
- Consistent spatial dimensions across all maps
- Valid material properties

Architecture (ADR-027):
- SpatialCaptureV1 contract alignment (gamma=1.0)
- Explicit shape/dtype validation
- Runtime contract enforcement
- Integration with Phase 2.1 segmentation masks
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

import numpy as np

from transformation_portal.attestation.materials_policy import VALID_MATERIAL_BACKENDS


@dataclass
class MaterialInput:
    """Input contract for PBR texture generation.

    Attributes:
        image: Linear RGB image (H, W, 3) float32, values in [0, ∞).
        gamma: Gamma value (must be 1.0 for linear).
        mask: Optional segmentation mask (H, W) bool.
            If provided, only generate textures for masked region.
        depth: Optional depth map (H, W) float32.
            Depth in meters, used for geometry-aware material optimization.
        material_hint: Optional material category hint.
            One of: "wood", "stone", "metal", "glass", "fabric", "concrete", "leather", "ceramic".
            Used to guide texture generation if available.
    """

    image: np.ndarray
    gamma: float
    mask: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    material_hint: Optional[str] = None

    def __post_init__(self):
        """Validate input contract."""
        # Gamma enforcement (SpatialCaptureV1 contract)
        if abs(self.gamma - 1.0) > 1e-6:
            raise ValueError(
                f"Material generation requires gamma=1.0 (linear RGB), got {self.gamma}. "
                "This violates the SpatialCaptureV1 contract."
            )

        # Dtype enforcement
        if self.image.dtype != np.float32:
            raise ValueError(f"Image must be float32, got {self.image.dtype}. " "Convert to linear float32 first.")

        # Shape validation
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"Image must be (H, W, 3), got shape {self.image.shape}")

        H, W = self.image.shape[:2]

        # Mask validation
        if self.mask is not None:
            if self.mask.dtype != bool:
                raise ValueError(f"Mask must be bool dtype, got {self.mask.dtype}")
            if self.mask.shape != (H, W):
                raise ValueError(f"Mask shape {self.mask.shape} must match image spatial dims ({H}, {W})")

        # Depth validation
        if self.depth is not None:
            if self.depth.dtype != np.float32:
                raise ValueError(f"Depth must be float32, got {self.depth.dtype}")
            if self.depth.shape != (H, W):
                raise ValueError(f"Depth shape {self.depth.shape} must match image spatial dims ({H}, {W})")
            if np.any(self.depth < 0):
                raise ValueError("Depth values must be non-negative")

        # Material hint validation
        VALID_MATERIALS = {"wood", "stone", "metal", "glass", "fabric", "concrete", "leather", "ceramic"}
        if self.material_hint is not None and self.material_hint not in VALID_MATERIALS:
            raise ValueError(f"Material hint must be one of {VALID_MATERIALS}, got '{self.material_hint}'")


@dataclass
class MaterialProperties:
    """Physical material properties for PBR rendering.

    Attributes:
        roughness_mean: Average surface roughness [0, 1] (0=smooth, 1=rough).
        metallic_mean: Average metallic value [0, 1] (0=dielectric, 1=metal).
        ao_strength: Ambient occlusion strength [0, 1].
        normal_strength: Normal map intensity multiplier [0, 2].
        specular_intensity: Specular reflection intensity [0, 1].
        subsurface_scattering: Subsurface scattering factor [0, 1].
            Used for materials like marble, wax, skin.
    """

    roughness_mean: float
    metallic_mean: float
    ao_strength: float
    normal_strength: float = 1.0
    specular_intensity: float = 0.5
    subsurface_scattering: float = 0.0

    def __post_init__(self):
        """Validate material properties."""
        # Roughness in [0, 1]
        if not 0.0 <= self.roughness_mean <= 1.0:
            raise ValueError(f"Roughness must be in [0, 1], got {self.roughness_mean}")

        # Metallic in [0, 1]
        if not 0.0 <= self.metallic_mean <= 1.0:
            raise ValueError(f"Metallic must be in [0, 1], got {self.metallic_mean}")

        # AO strength in [0, 1]
        if not 0.0 <= self.ao_strength <= 1.0:
            raise ValueError(f"AO strength must be in [0, 1], got {self.ao_strength}")

        # Normal strength in [0, 2]
        if not 0.0 <= self.normal_strength <= 2.0:
            raise ValueError(f"Normal strength must be in [0, 2], got {self.normal_strength}")

        # Specular intensity in [0, 1]
        if not 0.0 <= self.specular_intensity <= 1.0:
            raise ValueError(f"Specular intensity must be in [0, 1], got {self.specular_intensity}")

        # Subsurface scattering in [0, 1]
        if not 0.0 <= self.subsurface_scattering <= 1.0:
            raise ValueError(f"Subsurface scattering must be in [0, 1], got {self.subsurface_scattering}")


class AvailabilityState(str, Enum):
    """Explicit availability state for a requested materials backend."""

    AVAILABLE = "available"
    INPUT_CONTRACT_MISMATCH = "input_contract_mismatch"
    RUNTIME_MISSING = "runtime_missing"
    INTEGRATION_MISSING = "integration_missing"
    LICENSE_GATED = "license_gated"
    ATTESTATION_INCOMPLETE = "attestation_incomplete"


@dataclass
class BackendDecision:
    """Describe how a requested materials backend resolved at runtime."""

    requested_backend: str
    executed_backend: str
    availability_state: AvailabilityState
    fallback_reason: Optional[str]
    required_inputs: list[str]
    required_runtime: list[str]

    def to_dict(self) -> dict:
        """Convert decision metadata to a JSON-serializable dictionary."""
        return {
            "requested_backend": self.requested_backend,
            "executed_backend": self.executed_backend,
            "availability_state": self.availability_state.value,
            "fallback_reason": self.fallback_reason,
            "required_inputs": list(self.required_inputs),
            "required_runtime": list(self.required_runtime),
        }


@dataclass
class PBRGenerationMetadata:
    """Metadata for PBR generation reproducibility.

    Attributes:
        backend: Backend identifier with version (e.g., "heuristic_v5.0.0").
        normal_scale: Normal map scale factor applied.
        ao_blend_ratio: AO blend configuration (e.g., "0.7_concavity_0.3_variance").
        bilateral_enabled: Whether bilateral filtering was used for albedo.
        material_hint: Optional material hint used during generation.
        depth_used: Whether depth map was provided and used.
        backend_decision: Explicit requested-vs-executed backend resolution.
    """

    backend: str
    normal_scale: float
    ao_blend_ratio: str
    bilateral_enabled: bool
    material_hint: Optional[str] = None
    depth_used: bool = False
    backend_decision: Optional[BackendDecision] = None
    timing_ms: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert metadata to dictionary for serialization."""
        return {
            "backend": self.backend,
            "normal_scale": self.normal_scale,
            "ao_blend_ratio": self.ao_blend_ratio,
            "bilateral_enabled": self.bilateral_enabled,
            "material_hint": self.material_hint,
            "depth_used": self.depth_used,
            "backend_decision": (self.backend_decision.to_dict() if self.backend_decision is not None else None),
            "timing_ms": dict(self.timing_ms or {}),
        }


@dataclass
class PBRTextures:
    """Output contract for PBR texture generation.

    All textures are linear (gamma=1.0) and share the same spatial dimensions.

    Attributes:
        albedo: Base color map (H, W, 3) float32, RGB in [0, 1].
            Pure color without lighting/shadows.
        normal: Normal map (H, W, 3) float32, XYZ in [-1, 1] normalized.
            Surface orientation in tangent space.
        roughness: Roughness map (H, W) float32, values in [0, 1].
            0=perfectly smooth (mirror), 1=completely rough (matte).
        metallic: Metallic map (H, W) float32, values in [0, 1].
            0=dielectric (wood, plastic), 1=conductor (metal).
        ambient_occlusion: AO map (H, W) float32, values in [0, 1].
            1=fully lit, 0=fully occluded. Represents cavity darkening.
        height: Optional height/displacement map (H, W) float32, values in [0, 1].
            Used for parallax occlusion mapping.
        properties: Aggregated material properties.
        metadata: Generation metadata for reproducibility (Phase 5F).
    """

    albedo: np.ndarray
    normal: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    ambient_occlusion: np.ndarray
    height: Optional[np.ndarray] = None
    properties: Optional[MaterialProperties] = None
    metadata: Optional[PBRGenerationMetadata] = None

    def __post_init__(self):
        """Validate output contract."""
        # Check albedo
        if self.albedo.dtype != np.float32:
            raise ValueError(f"Albedo must be float32, got {self.albedo.dtype}")
        if self.albedo.ndim != 3 or self.albedo.shape[2] != 3:
            raise ValueError(f"Albedo must be (H, W, 3), got {self.albedo.shape}")
        if np.any((self.albedo < 0) | (self.albedo > 1)):
            raise ValueError(f"Albedo must be in [0, 1], got range [{self.albedo.min()}, {self.albedo.max()}]")

        H, W = self.albedo.shape[:2]

        # Check normal map
        if self.normal.dtype != np.float32:
            raise ValueError(f"Normal must be float32, got {self.normal.dtype}")
        if self.normal.shape != (H, W, 3):
            raise ValueError(f"Normal shape {self.normal.shape} must match albedo spatial dims ({H}, {W}, 3)")
        if np.any((self.normal < -1) | (self.normal > 1)):
            raise ValueError(f"Normal must be in [-1, 1], got range [{self.normal.min()}, {self.normal.max()}]")

        # Check roughness
        if self.roughness.dtype != np.float32:
            raise ValueError(f"Roughness must be float32, got {self.roughness.dtype}")
        if self.roughness.shape != (H, W):
            raise ValueError(f"Roughness shape {self.roughness.shape} must match albedo spatial dims ({H}, {W})")
        if np.any((self.roughness < 0) | (self.roughness > 1)):
            raise ValueError(f"Roughness must be in [0, 1], got range [{self.roughness.min()}, {self.roughness.max()}]")

        # Check metallic
        if self.metallic.dtype != np.float32:
            raise ValueError(f"Metallic must be float32, got {self.metallic.dtype}")
        if self.metallic.shape != (H, W):
            raise ValueError(f"Metallic shape {self.metallic.shape} must match albedo spatial dims ({H}, {W})")
        if np.any((self.metallic < 0) | (self.metallic > 1)):
            raise ValueError(f"Metallic must be in [0, 1], got range [{self.metallic.min()}, {self.metallic.max()}]")

        # Check AO
        if self.ambient_occlusion.dtype != np.float32:
            raise ValueError(f"AO must be float32, got {self.ambient_occlusion.dtype}")
        if self.ambient_occlusion.shape != (H, W):
            raise ValueError(f"AO shape {self.ambient_occlusion.shape} must match albedo spatial dims ({H}, {W})")
        if np.any((self.ambient_occlusion < 0) | (self.ambient_occlusion > 1)):
            raise ValueError(
                f"AO must be in [0, 1], got range [{self.ambient_occlusion.min()}, {self.ambient_occlusion.max()}]"
            )

        # Check height if provided
        if self.height is not None:
            if self.height.dtype != np.float32:
                raise ValueError(f"Height must be float32, got {self.height.dtype}")
            if self.height.shape != (H, W):
                raise ValueError(f"Height shape {self.height.shape} must match albedo spatial dims ({H}, {W})")
            if np.any((self.height < 0) | (self.height > 1)):
                raise ValueError(f"Height must be in [0, 1], got range [{self.height.min()}, {self.height.max()}]")


@dataclass
class MaterialGenerationConfig:
    """Configuration for PBR texture generation.

    Attributes:
        backend: Backend engine ("pbr_fusion", "nvdiffrec", "material_gan", or "heuristic").
        resolution: Target texture resolution (power of 2: 512, 1024, 2048, 4096).
        optimize_iterations: Number of optimization iterations (10-500).
            More iterations = better quality but slower.
        use_depth: Whether to use depth for geometry-aware optimization.
        normal_strength: Normal map intensity multiplier [0, 2].
        ao_intensity: AO darkness multiplier [0, 1].
        device: Compute device ("cuda", "mps", "cpu").
        strict_backend: If True, fail instead of falling back when the requested
            backend cannot execute under the current runtime/input contract.
    """

    backend: Literal["pbr_fusion", "nvdiffrec", "material_gan", "heuristic"]
    resolution: int = 1024
    optimize_iterations: int = 100
    use_depth: bool = True
    normal_strength: float = 1.0
    ao_intensity: float = 0.7
    device: Literal["cuda", "mps", "cpu"] = "cuda"
    strict_backend: bool = False

    def __post_init__(self):
        """Validate config."""
        if self.backend not in VALID_MATERIAL_BACKENDS:
            raise ValueError(f"Backend must be one of {VALID_MATERIAL_BACKENDS}, got '{self.backend}'")

        # Resolution must be power of 2
        if self.resolution not in [512, 1024, 2048, 4096]:
            raise ValueError(f"Resolution must be 512/1024/2048/4096, got {self.resolution}")

        # Iterations must be positive
        if self.optimize_iterations <= 0:
            raise ValueError(f"Iterations must be positive, got {self.optimize_iterations}")

        # Normal strength in [0, 2]
        if not 0.0 <= self.normal_strength <= 2.0:
            raise ValueError(f"Normal strength must be in [0, 2], got {self.normal_strength}")

        # AO intensity in [0, 1]
        if not 0.0 <= self.ao_intensity <= 1.0:
            raise ValueError(f"AO intensity must be in [0, 1], got {self.ao_intensity}")

        if not isinstance(self.strict_backend, bool):
            raise ValueError(f"strict_backend must be bool, got {type(self.strict_backend).__name__}")
