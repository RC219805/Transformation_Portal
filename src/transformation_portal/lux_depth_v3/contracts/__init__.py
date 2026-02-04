"""Contracts module for lux_depth_v3.

This module defines the stable data contracts that serve as the universal
currency for depth processing across all pipeline stages.

The contracts enforce:
- Schema versioning for forward/backward compatibility
- Immutable provenance for audit and compliance
- Type safety with strict validation
"""

from .depth_artifact import CameraIntrinsics, DepthArtifact, DepthArtifactWriter, DepthProvenance, LicenseTier

__all__ = [
    "DepthArtifact",
    "DepthProvenance",
    "CameraIntrinsics",
    "DepthArtifactWriter",
    "LicenseTier",
]
