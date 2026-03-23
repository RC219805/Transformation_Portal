"""Core geometry contracts and utilities.

This module provides neutral camera and reconstruction contracts that can be
used across pipelines without creating hard coupling between lux_depth_v3
and spatial_ai modules.

Key types:
- CoreCameraParams: Simple pinhole camera intrinsics with source provenance
- MultiViewReconstructionRequest: Neutral multi-view reconstruction request

These contracts serve as a cross-pipeline boundary layer per ADR-042.
"""

from __future__ import annotations

from .camera_params import CoreCameraParams
from .multiview_request import MultiViewReconstructionRequest

__all__ = [
    "CoreCameraParams",
    "MultiViewReconstructionRequest",
]
