"""Core camera parameter contract for cross-pipeline use.

Provides a neutral camera representation that is:
- Simple (pinhole model, intrinsics only)
- Source-aware (tracks explicit/exif/synthetic origin)
- Frozen (immutable for safety in reconstruction)

This is the boundary layer contract for camera data flowing between
lux_depth_v3 scene groups and spatial_ai reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, get_args

CameraSource = Literal["explicit", "exif", "synthetic"]

# Valid camera sources extracted from the Literal type for runtime validation
VALID_CAMERA_SOURCES = get_args(CameraSource)


@dataclass(frozen=True)
class CoreCameraParams:
    """Core camera parameters for multi-view reconstruction.

    A simple pinhole camera model with explicit provenance tracking.
    This is the neutral contract used between pipelines.

    Attributes:
        fx: Focal length X in pixels.
        fy: Focal length Y in pixels.
        cx: Principal point X in pixels.
        cy: Principal point Y in pixels.
        width: Image width in pixels.
        height: Image height in pixels.
        source: Origin of camera parameters ("explicit", "exif", "synthetic").
        image_path: Optional associated image path (for traceability).

    Note:
        This is a minimal camera model (pinhole without distortion).
        For full reconstruction with extrinsics, the spatial_ai.reconstruction
        module's CameraParams should be used.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: CameraSource = "synthetic"
    image_path: Optional[Path] = None

    def __post_init__(self) -> None:
        """Validate camera parameters."""
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"Focal lengths must be positive: fx={self.fx}, fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Image dimensions must be positive: {self.width}x{self.height}")
        if self.source not in VALID_CAMERA_SOURCES:
            raise ValueError(f"Camera source must be one of {VALID_CAMERA_SOURCES}, got '{self.source}'")

    @property
    def is_verified(self) -> bool:
        """Check if camera parameters come from a verified source.

        Returns:
            True if source is 'explicit' or 'exif' (not synthetic).
        """
        return self.source in ("explicit", "exif")

    def to_intrinsics_tuple(self) -> tuple[float, float, float, float]:
        """Return intrinsics as (fx, fy, cx, cy) tuple."""
        return (self.fx, self.fy, self.cx, self.cy)
