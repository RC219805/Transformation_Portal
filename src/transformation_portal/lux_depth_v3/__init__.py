"""Lux Depth V3 Pipeline - Public API.

This module defines the stable public surface for the lux_depth_v3 package.
Import from this module rather than internal submodules to ensure API stability.

v3.0 adds:
- DepthArtifact: Universal spatial currency for depth processing
- DepthModel Protocol: Unified interface for swappable depth backends
- BackendRole: Role-based backend routing (DRAFT, PRODUCTION, VIDEO, AUDIT)
- LicenseTier: License-aware routing (COMMERCIAL, NON_COMMERCIAL, EXPERIMENTAL)

Example:
    >>> from pathlib import Path
    >>> from transformation_portal.lux_depth_v3 import EnhanceOrchestrator, EnhanceConfig
    >>> config = EnhanceConfig()
    >>> orchestrator = EnhanceOrchestrator(config, output_root=Path("./output"))

v3.0 Contract Example:
    >>> from transformation_portal.lux_depth_v3.contracts import DepthArtifact, LicenseTier
    >>> artifact = DepthArtifact(
    ...     depth_map=depth_array,
    ...     provenance=DepthProvenance(
    ...         model_id="depth-anything/DA3-Large",
    ...         license_tier=LicenseTier.COMMERCIAL,
    ...     )
    ... )
"""

# Configuration
from .config import DA3Config, DeviceConfig, EnhanceConfig, ModelVariant, PostprocessingConfig, Preset

# v3.0 Contracts (Universal Depth Currency)
from .contracts import CameraIntrinsics, DepthArtifact, DepthArtifactWriter, DepthProvenance, LicenseTier
from .inference import DA3InferenceEngine, DepthResult

# Orchestration
from .orchestrator import EnhanceOrchestrator

# PBR processor (NEW - standalone PBR generation)
from .pbr import PBRConfig

# PBR presets
from .pbr_presets import (
    FABRIC_OPTIMIZED,
    FAST_PREVIEW,
    GLASS_OPTIMIZED,
    METAL_OPTIMIZED,
    PREMIUM_QUALITY,
    STANDARD_QUALITY,
    STONE_OPTIMIZED,
    WOOD_OPTIMIZED,
    get_preset,
    list_presets,
)
from .pbr_processor import PBRProcessor

# Core processing
from .postprocessing import Postprocessor

# v3.0 Protocols (Depth Model Interface)
from .protocols import BackendCapability, BackendInfo, BackendRole, DepthModel, DepthModelRegistry

__all__ = [
    # Orchestration
    "EnhanceOrchestrator",
    # Configuration
    "DA3Config",
    "ModelVariant",
    "Preset",
    "EnhanceConfig",
    "PostprocessingConfig",
    "DeviceConfig",
    # Core processing
    "Postprocessor",
    "DA3InferenceEngine",
    "DepthResult",
    # PBR presets
    "STANDARD_QUALITY",
    "PREMIUM_QUALITY",
    "FAST_PREVIEW",
    "WOOD_OPTIMIZED",
    "METAL_OPTIMIZED",
    "GLASS_OPTIMIZED",
    "STONE_OPTIMIZED",
    "FABRIC_OPTIMIZED",
    "get_preset",
    "list_presets",
    # PBR processor
    "PBRConfig",
    "PBRProcessor",
    # v3.0 Contracts
    "DepthArtifact",
    "DepthProvenance",
    "CameraIntrinsics",
    "DepthArtifactWriter",
    "LicenseTier",
    # v3.0 Protocols
    "DepthModel",
    "BackendRole",
    "BackendCapability",
    "BackendInfo",
    "DepthModelRegistry",
]

__version__ = "3.0.0"  # Lux Depth Engine v3.0
