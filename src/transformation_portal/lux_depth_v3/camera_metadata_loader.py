"""Camera metadata loading for scene-level reconstruction."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams  # noqa: E501

from .execution_plan_adapter import LuxExecutionPlanAuthorityError
from .scene_context import CameraProvenance, CameraWithProvenance
from .scene_groups import SceneGroup, normalize_relative_path

logger = logging.getLogger(__name__)

SCENE_CAMERA_SCHEMA = "tp.scene_cameras.v1"
SCENE_CAMERA_SIDECAR_MAX_BYTES = 10 * 1024 * 1024


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


def _sidecar_failure(
    message: str,
    *,
    sidecar_path: Optional[Path],
    expected_sha256: Optional[str],
    exc: Optional[BaseException] = None,
) -> None:
    """Fail closed for plan-authorized reads and warn for legacy reads."""

    rendered = f"{message}: {sidecar_path}"
    if expected_sha256 is not None:
        raise LuxExecutionPlanAuthorityError(rendered) from exc
    if exc is None:
        logger.warning("%s", rendered)
    else:
        logger.warning("%s: %s", rendered, exc)


def load_sidecar_payload(
    sidecar_path: Optional[Path],
    *,
    expected_sha256: Optional[str] = None,
) -> Optional[dict]:
    """Load and validate one camera sidecar byte snapshot.

    ``expected_sha256`` designates an execution-plan-authorized read. Such a
    read is bounded, hashes the exact bytes later decoded as JSON, and raises
    on every authority or validation failure. Legacy callers that omit the
    digest retain the historical warning-and-``None`` behavior.
    """

    if sidecar_path is None:
        if expected_sha256 is not None:
            _sidecar_failure(
                "Authoritative camera sidecar path is missing",
                sidecar_path=sidecar_path,
                expected_sha256=expected_sha256,
            )
        return None

    normalized_expected: Optional[str] = None
    if expected_sha256 is not None:
        normalized_expected = expected_sha256.strip().lower()
        if len(normalized_expected) != 64 or any(character not in "0123456789abcdef" for character in normalized_expected):
            _sidecar_failure(
                "Authoritative camera sidecar SHA-256 is invalid",
                sidecar_path=sidecar_path,
                expected_sha256=expected_sha256,
            )

    try:
        if normalized_expected is None:
            with sidecar_path.open("rb") as handle:
                sidecar_bytes = handle.read()
        else:
            open_flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
            file_descriptor = os.open(sidecar_path, open_flags)
            with os.fdopen(file_descriptor, "rb") as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    _sidecar_failure(
                        "Authoritative camera sidecar is not a regular file",
                        sidecar_path=sidecar_path,
                        expected_sha256=expected_sha256,
                    )
                sidecar_bytes = handle.read(SCENE_CAMERA_SIDECAR_MAX_BYTES + 1)
    except OSError as exc:
        _sidecar_failure(
            "Failed to read camera sidecar",
            sidecar_path=sidecar_path,
            expected_sha256=expected_sha256,
            exc=exc,
        )
        return None

    if normalized_expected is not None and len(sidecar_bytes) > SCENE_CAMERA_SIDECAR_MAX_BYTES:
        _sidecar_failure(
            f"Camera sidecar exceeds {SCENE_CAMERA_SIDECAR_MAX_BYTES} bytes",
            sidecar_path=sidecar_path,
            expected_sha256=expected_sha256,
        )
        return None

    if normalized_expected is not None:
        actual_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()
        if actual_sha256 != normalized_expected:
            _sidecar_failure(
                "Camera sidecar SHA-256 does not match the authoritative execution plan",
                sidecar_path=sidecar_path,
                expected_sha256=expected_sha256,
            )

    try:
        payload = json.loads(sidecar_bytes.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        _sidecar_failure(
            "Failed to decode camera sidecar JSON",
            sidecar_path=sidecar_path,
            expected_sha256=expected_sha256,
            exc=exc,
        )
        return None

    if not isinstance(payload, dict):
        _sidecar_failure(
            "Invalid camera sidecar format (root must be object)",
            sidecar_path=sidecar_path,
            expected_sha256=expected_sha256,
        )
        return None

    if payload.get("schema") != SCENE_CAMERA_SCHEMA:
        _sidecar_failure(
            f"Unsupported camera sidecar schema (expected {SCENE_CAMERA_SCHEMA}, got {payload.get('schema')!r})",
            sidecar_path=sidecar_path,
            expected_sha256=expected_sha256,
        )
        return None

    scenes = payload.get("scenes")
    if not isinstance(scenes, dict):
        _sidecar_failure(
            "Invalid camera sidecar ('scenes' must be an object)",
            sidecar_path=sidecar_path,
            expected_sha256=expected_sha256,
        )
        return None

    return payload


def load_scene_cameras(
    scene: SceneGroup,
    dataset_root: Path,
    sidecar_path: Optional[Path],
    sidecar_payload: Optional[dict] = None,
    sidecar_source_file: Optional[str] = None,
) -> Optional[Tuple[CameraWithProvenance, ...]]:
    """Load cameras for a scene from explicit sidecar metadata.

    Returns None when sidecar is missing, invalid, or does not contain
    a valid camera bundle for the requested scene.
    """
    if sidecar_path is None:
        logger.debug(
            "No camera sidecar path" " configured; scene" " reconstruction disabled" " for %s",
            scene.scene_id,
        )
        return None

    payload = sidecar_payload if sidecar_payload is not None else load_sidecar_payload(sidecar_path)
    if payload is None:
        return None
    scenes = payload["scenes"]

    scene_entry = scenes.get(scene.scene_id)
    if not isinstance(scene_entry, dict):
        logger.debug(
            "No camera entry for" " scene_id=%s in %s",
            scene.scene_id,
            sidecar_path,
        )
        return None

    entry_images = scene_entry.get("images")
    entry_cameras = scene_entry.get("cameras")
    if not isinstance(entry_images, list) or not isinstance(
        entry_cameras,
        list,
    ):
        logger.warning(
            "Invalid camera scene entry" " for %s: requires 'images'" " and 'cameras' arrays",
            scene.scene_id,
        )
        return None

    normalized_scene_images = [
        normalize_relative_path(
            p,
            dataset_root,
        ).lstrip("./")
        for p in scene.images
    ]
    normalized_entry_images = [_normalize_sidecar_image(str(p)) for p in entry_images]
    if normalized_entry_images != normalized_scene_images:
        logger.warning(
            "Camera sidecar image ordering" " mismatch for scene %s:" " sidecar_order=%s," " required_order=%s",
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
        logger.warning(
            "Invalid camera parameters" " for scene %s: %s",
            scene.scene_id,
            exc,
        )
        return None

    source_file = sidecar_source_file if sidecar_source_file is not None else str(sidecar_path.resolve())
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
