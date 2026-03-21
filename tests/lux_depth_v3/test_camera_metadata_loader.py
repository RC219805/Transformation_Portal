from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from transformation_portal.lux_depth_v3.camera_metadata_loader import SCENE_CAMERA_SCHEMA, load_scene_cameras
from transformation_portal.lux_depth_v3.scene_groups import SceneGroup
import pytest



pytestmark = pytest.mark.unit

def _camera_payload(width: int, height: int) -> dict:
    return {
        "intrinsics": [[1000.0, 0.0, width / 2.0], [0.0, 1000.0, height / 2.0], [0.0, 0.0, 1.0]],
        "extrinsics": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "width": width,
        "height": height,
    }


def test_load_scene_cameras_from_explicit_sidecar(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    image_a = dataset_root / "scene_a" / "img1.jpg"
    image_b = dataset_root / "scene_a" / "img2.jpg"
    image_a.parent.mkdir(parents=True, exist_ok=True)
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")

    sidecar_path = tmp_path / "cameras.json"
    sidecar_payload = {
        "schema": SCENE_CAMERA_SCHEMA,
        "scenes": {
            "scene-a": {
                "images": ["scene_a/img1.jpg", "scene_a/img2.jpg"],
                "cameras": [_camera_payload(1280, 720), _camera_payload(1280, 720)],
            }
        },
    }
    sidecar_path.write_text(json.dumps(sidecar_payload), encoding="utf-8")

    scene = SceneGroup(scene_id="scene-a", images=(image_a, image_b))
    cameras = load_scene_cameras(scene=scene, dataset_root=dataset_root, sidecar_path=sidecar_path)

    assert cameras is not None
    assert len(cameras) == 2
    assert cameras[0].params.intrinsics.shape == (3, 3)
    assert cameras[0].params.extrinsics.shape == (4, 4)
    assert cameras[0].params.intrinsics.dtype == np.float32
    assert cameras[0].params.extrinsics.dtype == np.float32
    assert cameras[0].provenance.source == "sidecar"
    assert cameras[0].provenance.confidence == "high"
    assert cameras[0].provenance.file == str(sidecar_path.resolve())


def test_load_scene_cameras_returns_none_without_sidecar(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    image = dataset_root / "scene_a" / "img1.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"a")

    scene = SceneGroup(scene_id="scene-a", images=(image,))
    assert load_scene_cameras(scene=scene, dataset_root=dataset_root, sidecar_path=None) is None


def test_load_scene_cameras_returns_none_on_image_mismatch(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    image_a = dataset_root / "scene_a" / "img1.jpg"
    image_b = dataset_root / "scene_a" / "img2.jpg"
    image_a.parent.mkdir(parents=True, exist_ok=True)
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")

    sidecar_path = tmp_path / "cameras.json"
    sidecar_payload = {
        "schema": SCENE_CAMERA_SCHEMA,
        "scenes": {
            "scene-a": {
                "images": ["scene_a/img2.jpg", "scene_a/img1.jpg"],  # wrong order
                "cameras": [_camera_payload(1280, 720), _camera_payload(1280, 720)],
            }
        },
    }
    sidecar_path.write_text(json.dumps(sidecar_payload), encoding="utf-8")

    scene = SceneGroup(scene_id="scene-a", images=(image_a, image_b))
    assert load_scene_cameras(scene=scene, dataset_root=dataset_root, sidecar_path=sidecar_path) is None


def test_load_scene_cameras_recovers_after_sidecar_created(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    image_a = dataset_root / "scene_a" / "img1.jpg"
    image_b = dataset_root / "scene_a" / "img2.jpg"
    image_a.parent.mkdir(parents=True, exist_ok=True)
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")

    sidecar_path = tmp_path / "cameras.json"
    scene = SceneGroup(scene_id="scene-a", images=(image_a, image_b))

    assert load_scene_cameras(scene=scene, dataset_root=dataset_root, sidecar_path=sidecar_path) is None

    sidecar_payload = {
        "schema": SCENE_CAMERA_SCHEMA,
        "scenes": {
            "scene-a": {
                "images": ["scene_a/img1.jpg", "scene_a/img2.jpg"],
                "cameras": [_camera_payload(1280, 720), _camera_payload(1280, 720)],
            }
        },
    }
    sidecar_path.write_text(json.dumps(sidecar_payload), encoding="utf-8")

    cameras = load_scene_cameras(scene=scene, dataset_root=dataset_root, sidecar_path=sidecar_path)
    assert cameras is not None
    assert len(cameras) == 2
