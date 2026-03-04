"""Camera metadata loading for scene-level reconstruction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams

from .scene_context import CameraProvenance, CameraWithProvenance
from .scene_groups import SceneGroup, normalize_relative_path

logger = logging.getLogger(__name__)

SCENE_CAMERA_SCHEMA = "tp.scene_cameras.v1"


def _normalize_sidecar_image(path_str: str) -> str:
    """Normalize sidecar image entries for stable comparisons."""
    return Path(path_str).as_posix().lower().lstrip("./")


def _camera_from_dict(camera_data: dict) -> CameraParams:
    """Construct CameraParams from sidecar camera entry."""
    intrinsics = np.asarray(camera_data["intrinsics"], dtype=np.float32)
    extrinsics = np.asarray(camera_data["extrinsics"], dtype=np.float32)
    width = int(camera_data["width"])
    height = int(camera_data["height"])

    distortion = camera_data.get("distortion")
    if distortion is not None:
        distortion = np.asarray(distortion, dtype=np.float32)

    camera_id = camera_data.get("camera_id")
    if camera_id is not None:
        camera_id = str(camera_id)

    return CameraParams(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        width=width,
        height=height,
        distortion=distortion,
        camera_id=camera_id,
    )


def load_sidecar_payload(sidecar_path: Optional[Path]) -> Optional[dict]:
    """Load and validate sidecar root payload from disk."""
    if sidecar_path is None:
        return None
    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load camera sidecar %s: %s", sidecar_path, exc)
        return None

    if not isinstance(payload, dict):
        logger.warning("Invalid camera sidecar format (root must be object): %s", sidecar_path)
        return None

    if payload.get("schema") != SCENE_CAMERA_SCHEMA:
        logger.warning(
            "Unsupported camera sidecar schema in %s: expected %s, got %r",
            sidecar_path,
            SCENE_CAMERA_SCHEMA,
            payload.get("schema"),
        )
        return None

    scenes = payload.get("scenes")
    if not isinstance(scenes, dict):
        logger.warning("Invalid camera sidecar: 'scenes' must be an object (%s)", sidecar_path)
        return None

    return payload


def load_scene_cameras(
    scene: SceneGroup,
    dataset_root: Path,
    sidecar_path: Optional[Path],
    sidecar_payload: Optional[dict] = None,
) -> Optional[Tuple[CameraWithProvenance, ...]]:
    """Load cameras for a scene from explicit sidecar metadata.

    Returns None when sidecar is missing, invalid, or does not contain
    a valid camera bundle for the requested scene.
    """
    if sidecar_path is None:
        logger.debug("No camera sidecar path configured; scene reconstruction disabled for %s", scene.scene_id)
        return None

    payload = sidecar_payload if sidecar_payload is not None else load_sidecar_payload(sidecar_path)
    if payload is None:
        return None
    scenes = payload["scenes"]

    scene_entry = scenes.get(scene.scene_id)
    if not isinstance(scene_entry, dict):
        logger.debug("No camera entry for scene_id=%s in %s", scene.scene_id, sidecar_path)
        return None

    entry_images = scene_entry.get("images")
    entry_cameras = scene_entry.get("cameras")
    if not isinstance(entry_images, list) or not isinstance(entry_cameras, list):
        logger.warning("Invalid camera scene entry for %s: requires 'images' and 'cameras' arrays", scene.scene_id)
        return None

    normalized_scene_images = [normalize_relative_path(p, dataset_root).lstrip("./") for p in scene.images]
    normalized_entry_images = [_normalize_sidecar_image(str(p)) for p in entry_images]
    if normalized_entry_images != normalized_scene_images:
        logger.warning(
            "Camera sidecar image ordering mismatch for scene %s: sidecar_order=%s, required_order=%s",
            scene.scene_id,
            normalized_entry_images,
            normalized_scene_images,
        )
        return None

    if len(entry_cameras) != len(scene.images):
        logger.warning(
            "Camera sidecar count mismatch for scene %s: cameras=%d images=%d",
            scene.scene_id,
            len(entry_cameras),
            len(scene.images),
        )
        return None

    try:
        camera_params = tuple(_camera_from_dict(camera_data) for camera_data in entry_cameras)
    except Exception as exc:
        logger.warning("Invalid camera parameters for scene %s: %s", scene.scene_id, exc)
        return None

    source_file = str(sidecar_path.resolve())
    return tuple(
        CameraWithProvenance(
            params=params,
            provenance=CameraProvenance(
                source="sidecar",
                confidence="high",
                file=source_file,
            ),
        )
        for params in camera_params
    )
