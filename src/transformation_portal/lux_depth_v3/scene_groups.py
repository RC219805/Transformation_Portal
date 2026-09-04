"""Scene-group utilities for deterministic multi-view reconstruction grouping.

ADR-042 Implementation Status:
- Phase A (Complete): SceneGroup with scene_id and images
- Phase B (This commit): Adding cameras field and reconstruction eligibility

Camera Source Precedence (ADR-042):
1. Explicit metadata file (manifest/sidecar)
2. EXIF-derived intrinsics/extrinsics where available
3. Synthetic defaults

Reconstruction Feature Gate:
- Controlled by `enable_reconstruction` config flag
- Default: False (per-image behavior preserved)
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CameraParams:
    """Camera parameters for reconstruction (Phase B, ADR-042).

    Attributes:
        image_path: Associated image path (one-to-one with SceneGroup.images).
        fx: Focal length X in pixels.
        fy: Focal length Y in pixels.
        cx: Principal point X in pixels.
        cy: Principal point Y in pixels.
        width: Image width in pixels.
        height: Image height in pixels.
        source: Origin of camera parameters ("explicit", "exif", "synthetic").

    Note:
        This is a minimal camera model (pinhole without distortion).
        For advanced reconstruction, extend with distortion coefficients
        and extrinsic parameters (rotation, translation).
    """

    image_path: Path
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: str = "synthetic"  # "explicit" | "exif" | "synthetic"

    def __post_init__(self) -> None:
        """Validate camera parameters."""
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"Focal lengths must be positive: fx={self.fx}, fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Image dimensions must be positive: {self.width}x{self.height}")
        if self.source not in ("explicit", "exif", "synthetic"):
            raise ValueError(f"Camera source must be 'explicit', 'exif', or 'synthetic', " f"got '{self.source}'")


@dataclass(frozen=True)
class SceneGroup:
    """Logical grouping of images that belong to the same reconstruction scene.

    ADR-042 Phase B Contract:
    - scene_id: Deterministic SHA1-based identifier (12 hex chars)
    - images: Ordered, immutable tuple of image paths
    - cameras: Optional camera tuple aligned one-to-one with images (Phase B)

    Reconstruction Eligibility (when enable_reconstruction=True):
    - len(images) >= 2
    - cameras is not None
    - len(cameras) == len(images)

    If eligibility is not met, per-image pipeline behavior continues.
    """

    scene_id: str
    images: Tuple[Path, ...]
    cameras: Optional[Tuple[CameraParams, ...]] = None

    def __post_init__(self) -> None:
        """Validate SceneGroup invariants."""
        if not self.scene_id:
            raise ValueError("scene_id cannot be empty")
        if not self.images:
            raise ValueError("images cannot be empty")
        # Validate camera alignment if cameras provided
        if self.cameras is not None:
            if len(self.cameras) != len(self.images):
                raise ValueError(
                    f"cameras must align with images: " f"got {len(self.cameras)} cameras for {len(self.images)} images"
                )
            # Ensure each camera is attached to the correct image.
            # We compare both the raw Paths and their non-strict resolved forms
            # to be robust to minor relative-path differences while staying
            # deterministic and filesystem-independent.
            # Note: resolve(strict=False) follows symlinks when they exist on disk.
            # Symlinked paths may compare as equal if they resolve to the same target.
            for idx, (image_path, camera) in enumerate(zip(self.images, self.cameras)):
                cam_image_path = camera.image_path
                if image_path == cam_image_path:
                    continue
                image_norm = image_path.resolve(strict=False)
                cam_norm = cam_image_path.resolve(strict=False)
                if image_norm != cam_norm:
                    raise ValueError(
                        "Camera image_path mismatch at index "
                        f"{idx}: images[{idx}]={image_path!s}, "
                        f"camera.image_path={cam_image_path!s}"
                    )

    def is_reconstruction_eligible(self) -> bool:
        """Check if scene meets reconstruction eligibility requirements.

        Returns:
            True if scene can be used for multi-view reconstruction.

        Eligibility requirements (ADR-042):
        - At least 2 images
        - Cameras resolved (not None)
        - Camera count matches image count
        """
        if len(self.images) < 2:
            return False
        if self.cameras is None:
            return False
        if len(self.cameras) != len(self.images):
            return False
        return True

    @property
    def num_images(self) -> int:
        """Number of images in this scene group."""
        return len(self.images)

    @property
    def has_cameras(self) -> bool:
        """Whether camera metadata is available."""
        return self.cameras is not None


def _normalize_relative_path(path: Path, dataset_root: Path) -> str:
    """Normalize path relative to dataset root with stable cross-platform formatting."""
    root_resolved = dataset_root.resolve()
    path_resolved = path.resolve()
    try:
        rel = path_resolved.relative_to(root_resolved)
    except ValueError:
        # Fall back to absolute normalized path when input is outside dataset root.
        rel = path_resolved
    return rel.as_posix().lower()


def normalize_relative_path(path: Path, dataset_root: Path) -> str:
    """Public wrapper for stable path normalization used across scene modules."""
    return _normalize_relative_path(path, dataset_root)


def lexical_relative_path(path: Path, dataset_root: Path) -> str:
    """Return an exact-case relative path without consulting the filesystem.

    Prepared execution has already canonicalized and confined its input paths.
    Re-resolving those paths after snapshot capture could instead follow a
    replacement filesystem namespace, so this helper deliberately performs
    lexical normalization only and requires containment.
    """

    candidate = Path(path)
    root = Path(dataset_root)
    if not candidate.is_absolute() or not root.is_absolute():
        raise ValueError("Canonical scene paths must be absolute")
    normalized_candidate = Path(os.path.abspath(os.fspath(candidate)))
    normalized_root = Path(os.path.abspath(os.fspath(root)))
    if candidate != normalized_candidate or root != normalized_root:
        raise ValueError("Canonical scene paths must be lexically normalized")
    try:
        relative = normalized_candidate.relative_to(normalized_root)
    except ValueError as exc:
        raise ValueError("Canonical scene image path must be contained by its dataset root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("Canonical scene image path must be a non-empty contained relative path")
    return relative.as_posix()


def _compute_scene_id(images: Tuple[Path, ...], dataset_root: Path) -> str:
    """Compute deterministic scene id from normalized relative image paths."""
    normalized = [normalize_relative_path(p, dataset_root) for p in images]
    payload = "\n".join(sorted(normalized)).encode("utf-8")
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:12]


def compute_scene_id(images: Tuple[Path, ...], dataset_root: Path) -> str:
    """Public wrapper for deterministic scene identifier derivation."""
    return _compute_scene_id(images, dataset_root)


def _group_key(path: Path, dataset_root: Path) -> str:
    """Group key for parent-directory grouping mode."""
    normalized = normalize_relative_path(path, dataset_root)
    if "/" not in normalized:
        return ""
    return normalized.rsplit("/", 1)[0]


def generate_synthetic_camera(
    image_path: Path,
    width: int,
    height: int,
    fov_degrees: float = 60.0,
) -> CameraParams:
    """Generate synthetic camera parameters as fallback.

    Args:
        image_path: Path to the image file.
        width: Image width in pixels.
        height: Image height in pixels.
        fov_degrees: Horizontal field of view in degrees (default: 60).

    Returns:
        CameraParams with synthetic intrinsics.

    Raises:
        ValueError: If fov_degrees is not in (0, 180) or width/height not positive.

    Note:
        This is a fallback when no explicit or EXIF camera data is available.
        The synthetic camera uses a centered principal point and focal length
        derived from the field of view.
    """
    # Validate inputs up front for clear error messages
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive: width={width}, height={height}")
    if not (0 < fov_degrees < 180):
        raise ValueError(f"FOV must be in (0, 180) degrees, got {fov_degrees}")

    # Compute focal length from FOV: fx = width / (2 * tan(fov/2))
    fov_rad = math.radians(fov_degrees)
    fx = width / (2 * math.tan(fov_rad / 2))
    fy = fx  # Square pixels

    # Centered principal point
    cx = width / 2
    cy = height / 2

    return CameraParams(
        image_path=image_path,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        source="synthetic",
    )


def build_scene_groups(
    images: Sequence[Path],
    dataset_root: Path,
    grouping_mode: str = "single",
    cameras: Optional[Sequence[CameraParams]] = None,
) -> List[SceneGroup]:
    """Build deterministic scene groups from image paths.

    Args:
        images: Sequence of image paths.
        dataset_root: Root directory for relative path normalization.
        grouping_mode: Grouping strategy ("single" or "parent_dir").
        cameras: Optional camera parameters aligned with images.

    Returns:
        List of SceneGroup instances.

    Modes:
    - single: each image forms its own scene (default, inert behavior)
    - parent_dir: group images by normalized relative parent directory

    Note:
        When cameras are provided, they must align one-to-one with images.
        After grouping, each SceneGroup receives the corresponding cameras
        for its subset of images.
    """
    mode = grouping_mode.strip().lower()
    image_list = [Path(img) for img in images]

    # Build image-to-camera mapping if cameras provided
    camera_map: dict = {}
    if cameras is not None:
        if len(cameras) != len(images):
            raise ValueError(f"cameras must align with images: " f"got {len(cameras)} cameras for {len(images)} images")
        for img, cam in zip(image_list, cameras):
            camera_map[img.resolve()] = cam

    def _get_cameras_for_group(group_images: Tuple[Path, ...]) -> Optional[Tuple[CameraParams, ...]]:
        """Extract cameras for a group's images from the camera map."""
        if not camera_map:
            return None
        try:
            return tuple(camera_map[img.resolve()] for img in group_images)
        except KeyError:
            # Some images in group don't have cameras
            return None

    if mode == "single":
        groups: List[SceneGroup] = []
        for img in image_list:
            group_images = (img,)
            group_cameras = _get_cameras_for_group(group_images)
            groups.append(
                SceneGroup(
                    scene_id=compute_scene_id(group_images, dataset_root),
                    images=group_images,
                    cameras=group_cameras,
                )
            )
        return groups

    if mode == "parent_dir":
        # Stable ordering regardless of caller order.
        sorted_images = sorted(image_list, key=lambda p: normalize_relative_path(p, dataset_root))
        parent_groups: List[SceneGroup] = []
        for _, grouped_iter in groupby(sorted_images, key=lambda p: _group_key(p, dataset_root)):
            grouped_images = tuple(grouped_iter)
            group_cameras = _get_cameras_for_group(grouped_images)
            parent_groups.append(
                SceneGroup(
                    scene_id=compute_scene_id(grouped_images, dataset_root),
                    images=grouped_images,
                    cameras=group_cameras,
                )
            )
        return parent_groups

    raise ValueError(f"Unknown grouping_mode '{grouping_mode}'. Expected one of: single, parent_dir")
