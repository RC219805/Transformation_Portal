from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from transformation_portal.lux_depth_v3.scene_context import CameraProvenance, CameraWithProvenance
from transformation_portal.lux_depth_v3.scene_groups import SceneGroup
from transformation_portal.lux_depth_v3.scene_preflight import (
    preflight_artifact_path,
    validate_scene_preflight,
    write_scene_preflight_artifact,
)
from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams


def _camera(tx: float) -> CameraWithProvenance:
    intrinsics = np.array(
        [[1000.0, 0.0, 32.0], [0.0, 1000.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[0, 3] = tx
    return CameraWithProvenance(
        params=CameraParams(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            width=64,
            height=64,
        ),
        provenance=CameraProvenance(
            source="sidecar",
            confidence="high",
            file="/tmp/scene_cameras.json",
        ),
    )


def test_validate_scene_preflight_returns_valid_for_normal_pair():
    scene = SceneGroup(scene_id="scene-a", images=(Path("a.jpg"), Path("b.jpg")))
    result = validate_scene_preflight(scene=scene, cameras=(_camera(0.0), _camera(0.2)))

    assert result.valid is True
    assert result.reason is None
    assert result.checks["view_count"] == "pass"
    assert result.checks["baseline"] == "pass"
    assert result.checks["fov_overlap"] == "pass"


def test_validate_scene_preflight_rejects_zero_baseline():
    scene = SceneGroup(scene_id="scene-a", images=(Path("a.jpg"), Path("b.jpg")))
    result = validate_scene_preflight(scene=scene, cameras=(_camera(0.0), _camera(0.0)))

    assert result.valid is False
    assert result.reason == "baseline_too_small"
    assert result.checks["baseline"] == "fail"


def test_validate_scene_preflight_rejects_single_view():
    scene = SceneGroup(scene_id="scene-a", images=(Path("a.jpg"),))
    result = validate_scene_preflight(scene=scene, cameras=(_camera(0.0),))

    assert result.valid is False
    assert result.reason == "view_count"


def test_write_scene_preflight_artifact_is_deterministic(tmp_path: Path):
    scene_id = "abc123"
    result = validate_scene_preflight(
        scene=SceneGroup(scene_id=scene_id, images=(Path("a.jpg"), Path("b.jpg"))),
        cameras=(_camera(0.0), _camera(0.2)),
    )

    artifact_path = write_scene_preflight_artifact(
        scene_id=scene_id,
        result=result,
        output_dir=tmp_path / "reconstruction",
    )

    assert artifact_path == preflight_artifact_path(scene_id=scene_id, output_dir=tmp_path / "reconstruction")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "tp.scene_preflight.v1"
    assert payload["scene_id"] == scene_id
    assert payload["valid"] is True
