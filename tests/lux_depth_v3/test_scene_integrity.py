from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.manifest import compute_file_sha256
from transformation_portal.lux_depth_v3.scene_context import CameraProvenance, CameraWithProvenance, SceneContext
from transformation_portal.lux_depth_v3.scene_groups import SceneGroup, compute_scene_id
from transformation_portal.lux_depth_v3.scene_integrity import (
    build_dataset_triage_report,
    build_scene_manifest,
    check_camera_geometry_sanity,
    compute_scene_fingerprint,
    normalize_camera_poses,
    verify_scene_integrity,
)
from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams

pytestmark = pytest.mark.unit


def _camera(
    tx: float,
    sidecar_path: Path,
    *,
    rotation: np.ndarray | None = None,
    ty: float = 0.0,
    tz: float = 0.0,
) -> CameraWithProvenance:
    intrinsics = np.array(
        [[1000.0, 0.0, 32.0], [0.0, 1000.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    extrinsics = np.eye(4, dtype=np.float32)
    if rotation is not None:
        extrinsics[:3, :3] = rotation
    extrinsics[0, 3] = tx
    extrinsics[1, 3] = ty
    extrinsics[2, 3] = tz
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


def test_scene_fingerprint_ignores_non_identity_manifest_fields(tmp_path: Path):
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
    reconstruction_config = {
        "iterations": 321,
        "tier": "apex_research",
        "grouping_mode": "parent_dir",
    }
    baseline_fp = compute_scene_fingerprint(
        scene_manifest=scene_manifest,
        artifact_index=artifact_index,
        reconstruction_config=reconstruction_config,
    )

    mutated_manifest = copy.deepcopy(scene_manifest)
    mutated_manifest["grouping_mode"] = "single"  # non-identity field (already covered by reconstruction_config)
    mutated_manifest["debug_note"] = "irrelevant metadata"
    mutated_manifest["camera_sidecar"]["path"] = str(Path(mutated_manifest["camera_sidecar"]["path"])) + ".alt"
    mutated_fp = compute_scene_fingerprint(
        scene_manifest=mutated_manifest,
        artifact_index=artifact_index,
        reconstruction_config=reconstruction_config,
    )

    assert baseline_fp == mutated_fp


def test_check_camera_geometry_sanity_accepts_connected_overlap_graph(tmp_path: Path):
    sidecar_path = tmp_path / "scene_cameras.json"
    sidecar_path.write_text('{"schema":"tp.scene_cameras.v1","scenes":{}}', encoding="utf-8")
    cameras = (_camera(0.0, sidecar_path), _camera(0.2, sidecar_path), _camera(0.4, sidecar_path))

    health = check_camera_geometry_sanity(cameras)
    assert health["largest_component"] == 3
    assert health["num_components"] == 1
    assert 0.0 <= float(health["risk_score"]) <= 1.0
    assert float(health["pair_overlap_fraction"]) >= 0.3


def test_check_camera_geometry_sanity_rejects_disconnected_overlap_graph(tmp_path: Path):
    sidecar_path = tmp_path / "scene_cameras.json"
    sidecar_path.write_text('{"schema":"tp.scene_cameras.v1","scenes":{}}', encoding="utf-8")
    first = _camera(0.0, sidecar_path)
    second = _camera(0.2, sidecar_path)
    third = _camera(0.4, sidecar_path)

    backward_extrinsics = np.eye(4, dtype=np.float32)
    backward_extrinsics[:3, :3] = np.diag([-1.0, 1.0, -1.0])
    backward_extrinsics[0, 3] = 0.8
    disconnected = CameraWithProvenance(
        params=CameraParams(
            intrinsics=third.params.intrinsics.copy(),
            extrinsics=backward_extrinsics,
            width=third.params.width,
            height=third.params.height,
        ),
        provenance=third.provenance,
    )

    with pytest.raises(ValueError, match="overlap graph disconnected|frustum overlap"):
        check_camera_geometry_sanity((first, second, disconnected))


def test_check_camera_geometry_sanity_uses_camera_centers_from_extrinsics(tmp_path: Path):
    sidecar_path = tmp_path / "scene_cameras.json"
    sidecar_path.write_text('{"schema":"tp.scene_cameras.v1","scenes":{}}', encoding="utf-8")
    rotation_90_z = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    cameras = (
        _camera(1.0, sidecar_path),
        _camera(1.0, sidecar_path, rotation=rotation_90_z),
    )

    health = check_camera_geometry_sanity(cameras)
    assert health["camera_count"] == 2
    assert float(health["pair_overlap_fraction"]) > 0.0


def test_normalize_camera_poses_centers_and_scales_baseline(tmp_path: Path):
    sidecar_path = tmp_path / "scene_cameras.json"
    sidecar_path.write_text('{"schema":"tp.scene_cameras.v1","scenes":{}}', encoding="utf-8")
    cameras = (_camera(2.0, sidecar_path), _camera(6.0, sidecar_path))

    normalized, metadata = normalize_camera_poses(cameras)

    first_tx = float(normalized[0].params.extrinsics[0, 3])
    second_tx = float(normalized[1].params.extrinsics[0, 3])
    assert pytest.approx(first_tx, rel=1e-6) == -0.5
    assert pytest.approx(second_tx, rel=1e-6) == 0.5
    assert metadata["method"] == "centered_median_baseline"
    assert pytest.approx(float(metadata["scale"]), rel=1e-6) == 0.25
    assert pytest.approx(float(metadata["median_baseline"]), rel=1e-6) == 4.0


def test_normalize_camera_poses_respects_rotated_extrinsics(tmp_path: Path):
    sidecar_path = tmp_path / "scene_cameras.json"
    sidecar_path.write_text('{"schema":"tp.scene_cameras.v1","scenes":{}}', encoding="utf-8")
    rotation_90_z = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    cameras = (
        _camera(1.0, sidecar_path),
        _camera(1.0, sidecar_path, rotation=rotation_90_z),
    )

    normalized, metadata = normalize_camera_poses(cameras)

    def _camera_center(camera: CameraWithProvenance) -> np.ndarray:
        rotation = np.asarray(camera.params.extrinsics[:3, :3], dtype=np.float32)
        translation = np.asarray(camera.params.extrinsics[:3, 3], dtype=np.float32)
        return -rotation.T @ translation

    centers = [_camera_center(camera) for camera in normalized]
    baseline = float(np.linalg.norm(centers[0] - centers[1]))
    assert pytest.approx(baseline, rel=1e-6) == 1.0
    assert metadata["method"] == "centered_median_baseline"


def test_build_scene_manifest_includes_dataset_health_hash(tmp_path: Path):
    output_root = tmp_path / "output"
    segmentation_artifact = output_root / "segmentation" / "scene_a_masks.npz"
    segmentation_artifact.parent.mkdir(parents=True, exist_ok=True)
    segmentation_artifact.write_bytes(b"segmentation")

    sidecar_path = tmp_path / "scene_cameras.json"
    sidecar_path.write_text('{"schema":"tp.scene_cameras.v1","scenes":{}}', encoding="utf-8")
    context = _context(tmp_path, sidecar_path)
    health = check_camera_geometry_sanity(context.cameras)
    context = SceneContext.build(
        scene=SceneGroup(scene_id=context.scene_id, images=context.images),
        dataset_root=context.dataset_root,
        cameras=context.cameras,
        metadata={"grouping_mode": "parent_dir", "dataset_health": health},
    )

    scene_manifest = build_scene_manifest(
        context=context,
        output_root=output_root,
        segmentation_artifact_paths=(segmentation_artifact,),
        camera_sidecar_path=sidecar_path,
    )

    assert "dataset_health" in scene_manifest
    assert "dataset_health_hash" in scene_manifest
    assert len(str(scene_manifest["dataset_health_hash"])) == 64


def test_build_dataset_triage_report_includes_actionable_guidance():
    report = build_dataset_triage_report(
        "scene-123",
        {
            "camera_count": 12,
            "largest_component": 5,
            "average_overlap": 0.08,
            "weak_edges": 9,
        },
    )

    assert report.startswith("Scene scene-123 dataset triage:")
    assert "Capture additional bridging images" in report
    assert "Increase view overlap" in report
    assert "Many weak connections" in report
