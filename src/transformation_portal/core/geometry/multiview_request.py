"""Neutral multi-view reconstruction request contract.

Provides a pipeline-neutral request object for multi-view reconstruction
that can be created by callers (e.g., lux_depth_v3 orchestration) and
consumed by spatial_ai without hard coupling.

This contract enforces:
- Multi-view requirements (>= 2 views)
- Camera/image alignment
- Camera source validation (fail-closed on synthetic by default)
- Research tier requirement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .camera_params import CoreCameraParams


class CameraValidationError(ValueError):
    """Raised when camera validation fails for reconstruction.

    This indicates that camera provenance requirements are not met.
    """

    pass


@dataclass(frozen=True)
class MultiViewReconstructionRequest:
    """Neutral contract for multi-view reconstruction requests.

    This is the boundary layer between orchestration (e.g., lux_depth_v3)
    and reconstruction execution (spatial_ai). Callers create this request
    after validating reconstruction eligibility; the reconstruction pipeline
    re-validates internally for defense-in-depth.

    Attributes:
        cameras: List of camera parameters, one per view.
        image_paths: Optional list of image file paths (for file-based input).
        images: Optional list of image arrays (for in-memory input).
            Must be (H, W, 3) float32 in linear RGB.
        depth_maps: Optional depth priors (H, W) float32 from Phase 1.
        masks: Optional segmentation masks (H, W) bool from Phase 2.1.
        material_maps: Optional PBR texture maps from Phase 2.2.
            Each dict may contain: "albedo", "roughness", "metallic", "normal".
        camera_sources: Explicit list of camera source types for validation.
            Populated from cameras[i].source if not provided.
        tier: License tier for reconstruction backend.
            Must be one of: "apex_research", "apex_research_ultra", "experimental".
        gamma: Gamma value for images. Must be 1.0 (linear RGB).
        allow_synthetic_cameras: Override to allow synthetic cameras.
            Default False - reconstruction fails if any camera is synthetic.
        optimization_seed: Optional seed for deterministic optimization.

    Contract Rules:
        - At least 2 views required
        - Either image_paths or images must be provided (not both)
        - Camera count must match view count
        - All images must have matching spatial dimensions
        - Gamma must be 1.0 (linear RGB)
        - Tier must be research-only
    """

    cameras: List[CoreCameraParams]
    image_paths: Optional[List[Path]] = None
    images: Optional[List[np.ndarray]] = None
    depth_maps: Optional[List[np.ndarray]] = None
    masks: Optional[List[np.ndarray]] = None
    material_maps: Optional[List[Dict[str, np.ndarray]]] = None
    camera_sources: List[str] = field(default_factory=list)
    tier: str = "apex_research"
    gamma: float = 1.0
    allow_synthetic_cameras: bool = False
    optimization_seed: Optional[int] = None

    # Valid tiers (research-only due to Inria 3DGS license)
    VALID_TIERS: tuple = ("apex_research", "apex_research_ultra", "experimental")

    def __post_init__(self) -> None:
        """Validate multi-view reconstruction request contract."""
        # Populate camera_sources from cameras if not explicitly provided
        if not self.camera_sources:
            sources = [cam.source for cam in self.cameras]
            # Use object.__setattr__ since frozen=True
            object.__setattr__(self, "camera_sources", sources)

        # Tier validation
        if self.tier not in self.VALID_TIERS:
            raise ValueError(
                f"Reconstruction requires research tier {self.VALID_TIERS}, "
                f"got '{self.tier}'. This is required by Inria 3DGS license."
            )

        # Gamma enforcement
        if abs(self.gamma - 1.0) > 1e-6:
            raise ValueError(
                f"Reconstruction requires gamma=1.0 (linear RGB), got {self.gamma}. " "Input images must be pre-linearized."
            )

        # Input validation: either paths or arrays (check BEFORE view count)
        if self.image_paths is None and self.images is None:
            raise ValueError("Either image_paths or images must be provided")
        if self.image_paths is not None and self.images is not None:
            raise ValueError("Provide either image_paths or images, not both")

        # View count validation (now safe since we have inputs)
        num_views = self.num_views
        if num_views < 2:
            raise ValueError(f"Reconstruction requires at least 2 views, got {num_views}")

        # Camera count alignment
        if len(self.cameras) != num_views:
            raise ValueError(f"Camera count ({len(self.cameras)}) must match " f"view count ({num_views})")

        # Camera source validation
        self._validate_camera_sources()

        # Images validation (if provided as arrays)
        if self.images is not None:
            self._validate_image_arrays()

        # Depth maps validation
        if self.depth_maps is not None:
            self._validate_depth_maps()

        # Masks validation
        if self.masks is not None:
            self._validate_masks()

    @property
    def num_views(self) -> int:
        """Number of views in the reconstruction request."""
        if self.image_paths is not None:
            return len(self.image_paths)
        if self.images is not None:
            return len(self.images)
        return 0

    @property
    def has_depth_priors(self) -> bool:
        """Whether depth priors are available."""
        return self.depth_maps is not None and len(self.depth_maps) > 0

    @property
    def has_segmentation(self) -> bool:
        """Whether segmentation masks are available."""
        return self.masks is not None and len(self.masks) > 0

    @property
    def has_materials(self) -> bool:
        """Whether PBR material maps are available."""
        return self.material_maps is not None and len(self.material_maps) > 0

    @property
    def all_cameras_verified(self) -> bool:
        """Check if all cameras have verified sources (explicit or exif)."""
        return all(cam.is_verified for cam in self.cameras)

    @property
    def has_synthetic_cameras(self) -> bool:
        """Check if any camera has synthetic source."""
        return any(cam.source == "synthetic" for cam in self.cameras)

    def _validate_camera_sources(self) -> None:
        """Validate camera sources according to policy.

        Raises:
            CameraValidationError: If synthetic cameras present and not allowed.
        """
        if self.has_synthetic_cameras and not self.allow_synthetic_cameras:
            synthetic_indices = [i for i, cam in enumerate(self.cameras) if cam.source == "synthetic"]
            raise CameraValidationError(
                f"Reconstruction requires verified cameras (explicit or exif). "
                f"Synthetic cameras found at indices: {synthetic_indices}. "
                f"To override this check, set allow_synthetic_cameras=True "
                f"(experimental/research use only)."
            )

    def _validate_image_arrays(self) -> None:
        """Validate image array formats."""
        if self.images is None:
            return

        for i, img in enumerate(self.images):
            if img.dtype != np.float32:
                raise ValueError(f"Image {i} must be float32, got {img.dtype}")
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Image {i} must be (H, W, 3), got shape {img.shape}")

    def _validate_depth_maps(self) -> None:
        """Validate depth map formats and alignment."""
        if self.depth_maps is None:
            return

        num_views = self.num_views
        if len(self.depth_maps) != num_views:
            raise ValueError(f"Depth map count ({len(self.depth_maps)}) must match " f"view count ({num_views})")

        for i, depth in enumerate(self.depth_maps):
            if depth.dtype != np.float32:
                raise ValueError(f"Depth map {i} must be float32, got {depth.dtype}")

    def _validate_masks(self) -> None:
        """Validate mask formats and alignment."""
        if self.masks is None:
            return

        num_views = self.num_views
        if len(self.masks) != num_views:
            raise ValueError(f"Mask count ({len(self.masks)}) must match " f"view count ({num_views})")

        for i, mask in enumerate(self.masks):
            if mask.dtype != bool:
                raise ValueError(f"Mask {i} must be bool dtype, got {mask.dtype}")

    def get_camera_source_summary(self) -> Dict[str, int]:
        """Get summary of camera sources.

        Returns:
            Dict with counts per source type, e.g. {"explicit": 2, "exif": 1}.
        """
        summary: Dict[str, int] = {}
        for cam in self.cameras:
            summary[cam.source] = summary.get(cam.source, 0) + 1
        return summary

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert request metadata to a serializable dict for provenance.

        Returns:
            Dict with request parameters (excluding large arrays).
        """
        return {
            "num_views": self.num_views,
            "tier": self.tier,
            "gamma": self.gamma,
            "camera_source_summary": self.get_camera_source_summary(),
            "has_depth_priors": self.has_depth_priors,
            "has_segmentation": self.has_segmentation,
            "has_materials": self.has_materials,
            "allow_synthetic_cameras": self.allow_synthetic_cameras,
            "all_cameras_verified": self.all_cameras_verified,
            "optimization_seed": self.optimization_seed,
        }
