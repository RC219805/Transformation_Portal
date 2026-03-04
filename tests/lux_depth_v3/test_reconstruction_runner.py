from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.reconstruction_runner import (
    diagnostics_artifact_path,
    manifest_artifact_path,
    run_scene_reconstruction,
    write_scene_debug_bundle,
)
from transformation_portal.lux_depth_v3.scene_context import CameraProvenance, CameraWithProvenance, SceneContext
from transformation_portal.lux_depth_v3.scene_groups import SceneGroup, compute_scene_id
from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams, GaussianSplat, Scene3D

pytestmark = pytest.mark.ml


def _camera_with_provenance(tx: float) -> CameraWithProvenance:
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


def _scene_from_cameras(cameras: list[CameraParams]) -> Scene3D:
    splats = GaussianSplat(
        positions=np.array([[2.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float32),
        colors=np.full((2, 3), 0.5, dtype=np.float32),
        scales=np.ones((2, 3), dtype=np.float32),
        rotations=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
        opacities=np.full((2, 1), 0.5, dtype=np.float32),
        metadata={},
    )
    return Scene3D(
        splats=splats,
        cameras=cameras,
        rmse=0.01,
        iteration=12,
        convergence="converged",
        metadata={},
    )


def _context_with_cameras(tmp_path: Path, cameras: tuple[CameraWithProvenance, ...]) -> SceneContext:
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
        cameras=cameras,
        metadata={"grouping_mode": "parent_dir"},
    )


def test_run_scene_reconstruction_normalizes_scale_and_writes_metadata(tmp_path: Path):
    context = _context_with_cameras(
        tmp_path,
        cameras=(
            _camera_with_provenance(2.0),
            _camera_with_provenance(6.0),
        ),
    )
    reconstructed_scene = _scene_from_cameras(
        cameras=[_camera_with_provenance(2.0).params, _camera_with_provenance(6.0).params]
    )

    class FakeSceneBuilder:
        def __init__(self, tier: str):
            self.tier = tier

        def build_from_images(self, **kwargs):  # noqa: ANN003
            _ = kwargs
            return reconstructed_scene

    with patch("transformation_portal.spatial_ai.reconstruction.scene_builder.SceneBuilder", FakeSceneBuilder):
        report_path = run_scene_reconstruction(
            context=context,
            output_dir=tmp_path / "out",
            iterations=123,
            tier="apex_research",
            scene_fingerprint="f" * 64,
            run_card_merkle_root="e" * 64,
        )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_path = manifest_artifact_path(scene_id=context.scene_id, output_dir=tmp_path / "out")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics_path = diagnostics_artifact_path(scene_id=context.scene_id, output_dir=tmp_path / "out")
    diagnostics_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))

    assert manifest_path.exists()
    assert payload["manifest_path"] == str(manifest_path)
    assert manifest_payload["schema"] == "tp.reconstruction_manifest.v1"
    assert manifest_payload["scene_id"] == context.scene_id
    assert len(manifest_payload["images"]) == 2
    assert len(manifest_payload["image_hashes"]) == 2
    assert diagnostics_path.exists()
    assert payload["diagnostics_path"] == str(diagnostics_path)
    assert payload["scene_scale"]["method"] == "median_baseline"
    assert payload["scene_fingerprint"] == "f" * 64
    assert payload["run_card_merkle_root"] == "e" * 64
    assert pytest.approx(payload["scene_scale"]["scale_factor"], rel=1e-6) == 0.25
    assert pytest.approx(payload["scene_scale"]["baseline_before"], rel=1e-6) == 4.0
    assert pytest.approx(payload["scene_scale"]["baseline_after"], rel=1e-6) == 1.0
    assert 0.1 < payload["scene_scale"]["baseline_after"] < 10.0
    assert diagnostics_payload["schema"] == "tp.scene_diagnostics.v1"
    assert diagnostics_payload["inputs"]["image_count"] == 2
    assert diagnostics_payload["inputs"]["grouping_mode"] == "parent_dir"
    assert diagnostics_payload["cameras"]["count"] == 2
    assert pytest.approx(diagnostics_payload["cameras"]["baseline_median"], rel=1e-6) == 1.0
    assert diagnostics_payload["geometry"]["point_count"] == 2
    assert diagnostics_payload["scale"]["method"] == "median_baseline"
    assert pytest.approx(diagnostics_payload["scale"]["scale_factor"], rel=1e-6) == 0.25

    assert pytest.approx(float(reconstructed_scene.splats.positions[0, 0]), rel=1e-6) == 0.5
    assert pytest.approx(float(reconstructed_scene.cameras[0].extrinsics[0, 3]), rel=1e-6) == 0.5
    assert pytest.approx(float(reconstructed_scene.cameras[1].extrinsics[0, 3]), rel=1e-6) == 1.5


def test_run_scene_reconstruction_raises_on_degenerate_camera_baseline(tmp_path: Path):
    context = _context_with_cameras(
        tmp_path,
        cameras=(
            _camera_with_provenance(1.0),
            _camera_with_provenance(1.0),
        ),
    )
    reconstructed_scene = _scene_from_cameras(
        cameras=[_camera_with_provenance(1.0).params, _camera_with_provenance(1.0).params]
    )

    class FakeSceneBuilder:
        def __init__(self, tier: str):
            self.tier = tier

        def build_from_images(self, **kwargs):  # noqa: ANN003
            _ = kwargs
            return reconstructed_scene

    with (
        patch("transformation_portal.spatial_ai.reconstruction.scene_builder.SceneBuilder", FakeSceneBuilder),
        pytest.raises(ValueError, match="median camera baseline"),
    ):
        run_scene_reconstruction(
            context=context,
            output_dir=tmp_path / "out",
            iterations=50,
            tier="apex_research",
        )


def test_write_scene_debug_bundle_writes_manifest_cameras_and_inputs(tmp_path: Path):
    sidecar_path = tmp_path / "scene_cameras.json"
    sidecar_path.write_text("{}", encoding="utf-8")
    context = _context_with_cameras(
        tmp_path,
        cameras=(
            _camera_with_provenance(2.0),
            _camera_with_provenance(6.0),
        ),
    )

    segmentation_artifact = tmp_path / "segmentation" / "scene_a_masks.npz"
    segmentation_artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(segmentation_artifact, wall=np.ones((64, 64), dtype=np.float32))

    scene_manifest = {
        "schema": "tp.scene_manifest.v1",
        "scene_id": context.scene_id,
        "images": [{"path": str(path), "sha256": "dummy"} for path in context.images],
        "cameras": [{"signature": "abc123"} for _ in context.cameras],
        "segmentation_artifacts": [{"path": str(segmentation_artifact), "sha256": "dummy"}],
    }
    debug_paths = write_scene_debug_bundle(
        context=context,
        segmentation_artifact_paths=(segmentation_artifact, segmentation_artifact),
        scene_manifest=scene_manifest,
        output_dir=tmp_path / "out",
    )

    assert debug_paths["scene_manifest_path"].exists()
    assert debug_paths["cameras_path"].exists()
    assert (tmp_path / "out" / "debug" / "inputs" / context.images[0].name).exists()
