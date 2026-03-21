from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.reconstruction_manifest import (

pytestmark = pytest.mark.unit

    build_reconstruction_manifest,
    load_reconstruction_manifest,
    manifest_image_paths,
    reconstruction_manifest_path,
    write_reconstruction_manifest,
)
from transformation_portal.lux_depth_v3.scene_context import CameraProvenance, CameraWithProvenance, SceneContext
from transformation_portal.lux_depth_v3.scene_groups import SceneGroup, compute_scene_id
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


def _context(tmp_path: Path) -> SceneContext:
    dataset_root = tmp_path / "input"
    first = dataset_root / "scene_a" / "view_1.jpg"
    second = dataset_root / "scene_a" / "view_2.jpg"
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    images = (first, second)
    scene = SceneGroup(scene_id=compute_scene_id(images, dataset_root), images=images)
    return SceneContext.build(
        scene=scene,
        dataset_root=dataset_root,
        cameras=(_camera(0.0), _camera(0.2)),
        metadata={"grouping_mode": "parent_dir"},
    )


def test_reconstruction_manifest_roundtrip_and_hash_verification(tmp_path: Path):
    context = _context(tmp_path)
    manifest = build_reconstruction_manifest(
        context=context,
        iterations=321,
        tier="apex_research",
    )
    path = write_reconstruction_manifest(manifest=manifest, output_dir=tmp_path / "out")
    loaded = load_reconstruction_manifest(manifest_path=path)

    assert path == reconstruction_manifest_path(scene_id=context.scene_id, output_dir=tmp_path / "out")
    assert loaded.scene_id == context.scene_id
    assert loaded.reconstruction_parameters["iterations"] == 321
    assert loaded.reconstruction_parameters["tier"] == "apex_research"
    assert len(loaded.images) == 2
    assert len(loaded.image_hashes) == 2
    assert manifest_image_paths(loaded)[0].exists()


def test_reconstruction_manifest_rejects_image_hash_drift(tmp_path: Path):
    context = _context(tmp_path)
    manifest = build_reconstruction_manifest(
        context=context,
        iterations=100,
        tier="apex_research",
    )
    path = write_reconstruction_manifest(manifest=manifest, output_dir=tmp_path / "out")

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["image_hashes"][0] = "sha256:deadbeef"
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="image hash mismatch"):
        load_reconstruction_manifest(manifest_path=path)
