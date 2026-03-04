"""Validated scene-level reconstruction context contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple

import numpy as np

from .scene_groups import SceneGroup, compute_scene_id

if TYPE_CHECKING:
    from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams

CameraSource = Literal["sidecar", "exif", "synthetic", "sfm"]
CameraConfidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class CameraProvenance:
    """Origin and trust metadata for a camera calibration entry."""

    source: CameraSource
    confidence: CameraConfidence
    file: Optional[str] = None


@dataclass(frozen=True)
class CameraWithProvenance:
    """Camera parameters paired with provenance metadata."""

    params: "CameraParams"
    provenance: CameraProvenance


@dataclass(frozen=True)
class SceneContext:
    """Validated immutable context passed into reconstruction.

    This centralizes scene-level invariants so reconstruction receives a single,
    aligned contract rather than loose parameters.
    """

    scene_id: str
    dataset_root: Path
    images: Tuple[Path, ...]
    cameras: Tuple[CameraWithProvenance, ...]
    segmentation_masks: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        scene: SceneGroup,
        dataset_root: Path,
        cameras: Tuple[CameraWithProvenance, ...],
        segmentation_masks: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SceneContext":
        """Build and validate SceneContext from scene-group + camera inputs."""
        images = tuple(scene.images)

        if len(images) < 2:
            raise ValueError(f"Reconstruction requires >=2 images, got {len(images)} for scene {scene.scene_id}")

        if len(cameras) != len(images):
            raise ValueError(
                "Camera/image alignment mismatch for scene " f"{scene.scene_id}: cameras={len(cameras)} images={len(images)}"
            )

        expected_scene_id = compute_scene_id(images, dataset_root)
        if scene.scene_id != expected_scene_id:
            raise ValueError(
                f"Scene ID mismatch for reconstruction context: got '{scene.scene_id}', expected '{expected_scene_id}'"
            )

        return cls(
            scene_id=scene.scene_id,
            dataset_root=dataset_root,
            images=images,
            cameras=tuple(cameras),
            segmentation_masks=segmentation_masks,
            metadata=dict(metadata) if metadata else {},
        )
