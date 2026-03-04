from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.manifest import compute_file_sha256
from transformation_portal.lux_depth_v3.scene_context import CameraProvenance, CameraWithProvenance, SceneContext
from transformation_portal.lux_depth_v3.scene_groups import SceneGroup, compute_scene_id
from transformation_portal.lux_depth_v3.scene_integrity import (
    build_scene_manifest,
    compute_scene_fingerprint,
    verify_scene_integrity,
)
from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams


def _camera(tx: float, sidecar_path: Path) -> CameraWithProvenance:
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
            file=str(sidecar_path),
        ),
    )


def _context(tmp_path: Path, sidecar_path: Path) -> SceneContext:
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
        cameras=(_camera(0.0, sidecar_path), _camera(0.2, sidecar_path)),
        metadata={"grouping_mode": "parent_dir"},
    )


def test_scene_integrity_build_verify_and_fingerprint(tmp_path: Path):
    output_root = tmp_path / "output"
    segmentation_artifact = output_root / "segmentation" / "scene_a_masks.npz"
    segmentation_artifact.parent.mkdir(parents=True, exist_ok=True)
    segmentation_artifact.write_bytes(b"segmentation")

    sidecar_path = tmp_path / "scene_cameras.json"
    sidecar_path.write_text('{"schema":"tp.scene_cameras.v1","scenes":{}}', encoding="utf-8")

    context = _context(tmp_path, sidecar_path)
    scene_manifest = build_scene_manifest(
        context=context,
        output_root=output_root,
        segmentation_artifact_paths=(segmentation_artifact,),
        camera_sidecar_path=sidecar_path,
    )

    relative_seg_path = "segmentation/scene_a_masks.npz"
    artifact_index = {
        relative_seg_path: {
            "sha256": compute_file_sha256(segmentation_artifact),
        }
    }
    verify_scene_integrity(scene_manifest, artifact_index=artifact_index)

    config = {
        "iterations": 321,
        "tier": "apex_research",
        "grouping_mode": "parent_dir",
    }
    fingerprint_1 = compute_scene_fingerprint(
        scene_manifest=scene_manifest,
        artifact_index=artifact_index,
        reconstruction_config=config,
    )
    fingerprint_2 = compute_scene_fingerprint(
        scene_manifest=scene_manifest,
        artifact_index=artifact_index,
        reconstruction_config=config,
    )

    assert len(fingerprint_1) == 64
    assert fingerprint_1 == fingerprint_2


def test_scene_integrity_rejects_segmentation_hash_drift(tmp_path: Path):
    output_root = tmp_path / "output"
    segmentation_artifact = output_root / "segmentation" / "scene_a_masks.npz"
    segmentation_artifact.parent.mkdir(parents=True, exist_ok=True)
    segmentation_artifact.write_bytes(b"segmentation")

    sidecar_path = tmp_path / "scene_cameras.json"
    sidecar_path.write_text('{"schema":"tp.scene_cameras.v1","scenes":{}}', encoding="utf-8")

    context = _context(tmp_path, sidecar_path)
    scene_manifest = build_scene_manifest(
        context=context,
        output_root=output_root,
        segmentation_artifact_paths=(segmentation_artifact,),
        camera_sidecar_path=sidecar_path,
    )

    segmentation_artifact.write_bytes(b"drift")

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        verify_scene_integrity(scene_manifest)


def test_scene_fingerprint_rejects_missing_artifact_index_entries(tmp_path: Path):
    output_root = tmp_path / "output"
    segmentation_artifact = output_root / "segmentation" / "scene_a_masks.npz"
    segmentation_artifact.parent.mkdir(parents=True, exist_ok=True)
    segmentation_artifact.write_bytes(b"segmentation")

    sidecar_path = tmp_path / "scene_cameras.json"
    sidecar_path.write_text('{"schema":"tp.scene_cameras.v1","scenes":{}}', encoding="utf-8")

    context = _context(tmp_path, sidecar_path)
    scene_manifest = build_scene_manifest(
        context=context,
        output_root=output_root,
        segmentation_artifact_paths=(segmentation_artifact,),
        camera_sidecar_path=sidecar_path,
    )

    with pytest.raises(RuntimeError, match="missing artifact index entry"):
        compute_scene_fingerprint(
            scene_manifest=scene_manifest,
            artifact_index={},
            reconstruction_config={"iterations": 1000, "tier": "apex_research"},
        )
