"""Data contracts for 3D reconstruction module (Phase 2.3).

Contract validation ensures:
- Gamma=1.0 enforcement (linear RGB only)
- Float32 dtype for spatial data
- Valid camera parameters (K matrix, extrinsics)
- Quaternion normalization for rotations
- Proper value ranges for Gaussian properties
- Tier restriction enforcement (research license)

Architecture (ADR-027):
- SpatialCaptureV1 contract alignment (gamma=1.0)
- Explicit shape/dtype validation
- Runtime contract enforcement
- Integration with Phases 2.1 (segmentation) and 2.2 (materials)
- License tier enforcement for Inria 3DGS

3D Gaussian Splatting License:
- Inria research license (non-commercial)
- Requires tier: apex_research, apex_research_ultra, or experimental
- Commercial use prohibited without explicit license
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np


class LicenseRestrictionError(Exception):
    """Raised when 3D reconstruction license requirements are not met.

    3D Gaussian Splatting uses Inria research license (non-commercial).
    Tier enforcement prevents accidental commercial use.
    """

    pass


@dataclass
class CameraParams:
    """Camera parameters for multi-view reconstruction.

    Attributes:
        intrinsics: Camera intrinsic matrix (3x3) in format:
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
            where fx/fy are focal lengths and cx/cy are principal points.
        extrinsics: Camera extrinsic matrix (4x4) in format:
            [[r11, r12, r13, tx],
             [r21, r22, r23, ty],
             [r31, r32, r33, tz],
             [0,   0,   0,   1]]
            where R (3x3) is rotation and t (3x1) is translation.
        width: Image width in pixels.
        height: Image height in pixels.
        distortion: Optional distortion coefficients [k1, k2, p1, p2, k3].
        camera_id: Optional camera identifier for multi-camera setups.
    """

    intrinsics: np.ndarray
    extrinsics: np.ndarray
    width: int
    height: int
    distortion: Optional[np.ndarray] = None
    camera_id: Optional[str] = None

    def __post_init__(self):
        """Validate camera parameters."""
        # Intrinsics validation
        if self.intrinsics.shape != (3, 3):
            raise ValueError(f"Intrinsics must be (3, 3), got shape {self.intrinsics.shape}")
        if self.intrinsics.dtype != np.float32:
            raise ValueError(f"Intrinsics must be float32, got {self.intrinsics.dtype}")
        if abs(self.intrinsics[2, 2] - 1.0) > 1e-6:
            raise ValueError("Intrinsics[2,2] must be 1.0 for homogeneous coordinates")

        # Extrinsics validation
        if self.extrinsics.shape != (4, 4):
            raise ValueError(f"Extrinsics must be (4, 4), got shape {self.extrinsics.shape}")
        if self.extrinsics.dtype != np.float32:
            raise ValueError(f"Extrinsics must be float32, got {self.extrinsics.dtype}")
        if not np.allclose(self.extrinsics[3, :], [0, 0, 0, 1]):
            raise ValueError("Extrinsics bottom row must be [0, 0, 0, 1]")

        # Dimension validation
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Image dimensions must be positive, got {self.width}x{self.height}")

        # Distortion validation
        if self.distortion is not None:
            if self.distortion.shape[0] not in [4, 5, 8]:
                raise ValueError(f"Distortion must have 4, 5, or 8 coefficients, got {self.distortion.shape[0]}")


@dataclass
class GaussianSplat:
    """3D Gaussian Splatting representation.

    Attributes:
        positions: 3D positions (N, 3) in world coordinates [x, y, z].
        colors: RGB colors (N, 3) in linear space, values in [0, 1].
        scales: Scale factors (N, 3) for Gaussian covariance [sx, sy, sz].
        rotations: Quaternions (N, 4) as [w, x, y, z] (normalized).
        opacities: Opacity values (N, 1) in [0, 1].
        sh_coefficients: Optional spherical harmonics (N, SH_dim, 3).
            For view-dependent appearance. SH_dim = 1 (DC only) to 16 (3rd order).
        metadata: Additional metadata (optimization stats, convergence, etc.).
    """

    positions: np.ndarray
    colors: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacities: np.ndarray
    sh_coefficients: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate Gaussian splat data."""
        N = self.positions.shape[0]

        # Positions validation
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"Positions must be (N, 3), got shape {self.positions.shape}")
        if self.positions.dtype != np.float32:
            raise ValueError(f"Positions must be float32, got {self.positions.dtype}")

        # Colors validation
        if self.colors.shape != (N, 3):
            raise ValueError(f"Colors must be ({N}, 3), got shape {self.colors.shape}")
        if self.colors.dtype != np.float32:
            raise ValueError(f"Colors must be float32, got {self.colors.dtype}")
        if np.any(self.colors < 0) or np.any(self.colors > 1):
            raise ValueError("Colors must be in [0, 1] (linear RGB)")

        # Scales validation
        if self.scales.shape != (N, 3):
            raise ValueError(f"Scales must be ({N}, 3), got shape {self.scales.shape}")
        if self.scales.dtype != np.float32:
            raise ValueError(f"Scales must be float32, got {self.scales.dtype}")
        if np.any(self.scales <= 0):
            raise ValueError("Scales must be positive")

        # Rotations validation
        if self.rotations.shape != (N, 4):
            raise ValueError(f"Rotations must be ({N}, 4), got shape {self.rotations.shape}")
        if self.rotations.dtype != np.float32:
            raise ValueError(f"Rotations must be float32, got {self.rotations.dtype}")
        # Check quaternion normalization
        norms = np.linalg.norm(self.rotations, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("Rotation quaternions must be normalized (unit length)")

        # Opacities validation
        if self.opacities.shape != (N, 1):
            raise ValueError(f"Opacities must be ({N}, 1), got shape {self.opacities.shape}")
        if self.opacities.dtype != np.float32:
            raise ValueError(f"Opacities must be float32, got {self.opacities.dtype}")
        if np.any(self.opacities < 0) or np.any(self.opacities > 1):
            raise ValueError("Opacities must be in [0, 1]")

        # Spherical harmonics validation
        if self.sh_coefficients is not None:
            if self.sh_coefficients.ndim != 3:
                raise ValueError(f"SH coefficients must be (N, SH_dim, 3), got shape {self.sh_coefficients.shape}")
            if self.sh_coefficients.shape[0] != N:
                raise ValueError(f"SH coefficients must have {N} entries, got {self.sh_coefficients.shape[0]}")
            if self.sh_coefficients.shape[2] != 3:
                raise ValueError(f"SH coefficients must have 3 color channels, got {self.sh_coefficients.shape[2]}")

    @property
    def num_gaussians(self) -> int:
        """Number of Gaussian primitives."""
        return self.positions.shape[0]


@dataclass
class ReconstructionInput:
    """Input contract for 3D scene reconstruction.

    Attributes:
        images: List of multi-view images (H, W, 3) float32, linear RGB.
        gamma: Gamma value (must be 1.0 for linear).
        cameras: Camera parameters for each view.
        depth_maps: Optional depth priors (H, W) float32 from Phase 1.
        masks: Optional segmentation masks (H, W) bool from Phase 2.1.
        material_maps: Optional PBR textures from Phase 2.2.
            Dict with keys: "albedo", "roughness", "metallic", "normal".
        tier: Tier restriction for license enforcement.
            Must be "apex_research", "apex_research_ultra", or "experimental".
    """

    images: List[np.ndarray]
    gamma: float
    cameras: List[CameraParams]
    depth_maps: Optional[List[np.ndarray]] = None
    masks: Optional[List[np.ndarray]] = None
    material_maps: Optional[List[Dict[str, np.ndarray]]] = None
    tier: str = "apex_research"

    def __post_init__(self):
        """Validate reconstruction input contract."""
        # Gamma enforcement (SpatialCaptureV1 contract)
        if abs(self.gamma - 1.0) > 1e-6:
            raise ValueError(
                f"Reconstruction requires gamma=1.0 (linear RGB), got {self.gamma}. "
                "This violates the SpatialCaptureV1 contract."
            )

        # Tier restriction enforcement (Inria license)
        VALID_TIERS = ["apex_research", "apex_research_ultra", "experimental"]
        if self.tier not in VALID_TIERS:
            raise LicenseRestrictionError(
                f"3D Gaussian Splatting requires research tier ({', '.join(VALID_TIERS)}) "
                f"due to Inria research license (non-commercial). Got tier: '{self.tier}'. "
                "See: https://github.com/graphdeco-inria/gaussian-splatting for license details."
            )

        # Multi-view validation
        if len(self.images) < 2:
            raise ValueError(f"Reconstruction requires at least 2 views, got {len(self.images)}")

        if len(self.cameras) != len(self.images):
            raise ValueError(f"Number of cameras ({len(self.cameras)}) must match " f"number of images ({len(self.images)})")

        # Images validation
        for i, img in enumerate(self.images):
            if img.dtype != np.float32:
                raise ValueError(f"Image {i} must be float32, got {img.dtype}")
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Image {i} must be (H, W, 3), got shape {img.shape}")

        # Depth maps validation
        if self.depth_maps is not None:
            if len(self.depth_maps) != len(self.images):
                raise ValueError(
                    f"Number of depth maps ({len(self.depth_maps)}) must match " f"number of images ({len(self.images)})"
                )
            for i, (depth, img) in enumerate(zip(self.depth_maps, self.images)):
                if depth.dtype != np.float32:
                    raise ValueError(f"Depth map {i} must be float32, got {depth.dtype}")
                if depth.shape != img.shape[:2]:
                    raise ValueError(f"Depth map {i} shape {depth.shape} must match " f"image spatial dims {img.shape[:2]}")

        # Masks validation
        if self.masks is not None:
            if len(self.masks) != len(self.images):
                raise ValueError(f"Number of masks ({len(self.masks)}) must match " f"number of images ({len(self.images)})")
            for i, (mask, img) in enumerate(zip(self.masks, self.images)):
                if mask.dtype != bool:
                    raise ValueError(f"Mask {i} must be bool dtype, got {mask.dtype}")
                if mask.shape != img.shape[:2]:
                    raise ValueError(f"Mask {i} shape {mask.shape} must match " f"image spatial dims {img.shape[:2]}")

        # Material maps validation
        if self.material_maps is not None:
            if len(self.material_maps) != len(self.images):
                raise ValueError(
                    f"Number of material maps ({len(self.material_maps)}) must match " f"number of images ({len(self.images)})"
                )

    @property
    def num_views(self) -> int:
        """Number of input views."""
        return len(self.images)


@dataclass
class Scene3D:
    """Complete 3D scene representation.

    Attributes:
        splats: Gaussian splatting representation.
        cameras: Camera parameters for all training views.
        rmse: Root mean square error (geometric validation).
            Target: < 0.02 (2% error) for production quality.
        iteration: Optimization iteration count.
        convergence: Convergence status.
            - "converged": RMSE below threshold
            - "max_iterations": Stopped at iteration limit
            - "diverged": Optimization failed
        metadata: Additional scene metadata (optimization stats, timing, etc.).
    """

    splats: GaussianSplat
    cameras: List[CameraParams]
    rmse: float
    iteration: int
    convergence: Literal["converged", "max_iterations", "diverged"]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate scene data."""
        # RMSE validation
        if self.rmse < 0:
            raise ValueError(f"RMSE must be non-negative, got {self.rmse}")

        # Iteration validation
        if self.iteration < 0:
            raise ValueError(f"Iteration must be non-negative, got {self.iteration}")

        # Camera validation
        if len(self.cameras) < 2:
            raise ValueError(f"Scene requires at least 2 camera views, got {len(self.cameras)}")

    @property
    def is_converged(self) -> bool:
        """Check if optimization converged."""
        return self.convergence == "converged"

    @property
    def quality_score(self) -> float:
        """Quality score based on RMSE (0-100, higher is better).

        Score mapping:
        - RMSE < 0.01: 95-100 (excellent)
        - RMSE < 0.02: 85-95 (good)
        - RMSE < 0.05: 70-85 (acceptable)
        - RMSE >= 0.05: 0-70 (poor)
        """
        if self.rmse < 0.01:
            return 95 + (0.01 - self.rmse) * 500  # 95-100
        elif self.rmse < 0.02:
            return 85 + (0.02 - self.rmse) * 1000  # 85-95
        elif self.rmse < 0.05:
            return 70 + (0.05 - self.rmse) * 500  # 70-85
        else:
            return max(0, 70 - (self.rmse - 0.05) * 200)  # 0-70
