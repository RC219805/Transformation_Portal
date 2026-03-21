from __future__ import annotations

import json

import numpy as np

from transformation_portal.lux_depth_v3.reconstruction_manifest import ReconstructionManifest
from transformation_portal.lux_depth_v3.reconstruction_runner import reprojection_percentiles, write_reconstruction_diagnostics
from transformation_portal.lux_depth_v3.scene_context import CameraProvenance, CameraWithProvenance
from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams, GaussianSplat, Scene3D
import pytest



pytestmark = pytest.mark.unit

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
            camera_id=f"cam_{int(tx * 10):02d}",
        ),
        provenance=CameraProvenance(source="sidecar", confidence="high", file="/tmp/scene_cameras.json"),
    )


def _scene_and_manifest() -> tuple[Scene3D, ReconstructionManifest]:
    first = _camera_with_provenance(0.0)
    second = _camera_with_provenance(0.1)
    scene = Scene3D(
        splats=GaussianSplat(
            positions=np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 3.0]], dtype=np.float32),
            colors=np.full((2, 3), 0.5, dtype=np.float32),
            scales=np.ones((2, 3), dtype=np.float32),
            rotations=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
            opacities=np.full((2, 1), 0.5, dtype=np.float32),
            metadata={},
        ),
        cameras=[first.params, second.params],
        rmse=0.25,
        iteration=10,
        convergence="converged",
        metadata={},
    )
    manifest = ReconstructionManifest(
        scene_id="scene_a",
        dataset_root="/tmp/input",
        images=("scene_a/view_1.jpg", "scene_a/view_2.jpg"),
        image_hashes=("sha256:" + ("a" * 64), "sha256:" + ("b" * 64)),
        cameras=(first, second),
        reconstruction_parameters={"iterations": 10, "tier": "apex_research"},
    )
    return scene, manifest


def test_reprojection_percentiles():
    populated = reprojection_percentiles(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert populated["p50"] == 2.5
    assert populated["p95"] is not None and populated["p95"] > populated["p50"]
    assert populated["p99"] is not None and populated["p99"] >= populated["p95"]

    empty = reprojection_percentiles(np.array([], dtype=np.float32))
    assert empty == {"p50": None, "p95": None, "p99": None}


def test_write_reconstruction_diagnostics_is_deterministic(tmp_path):
    scene, manifest = _scene_and_manifest()
    path_1 = write_reconstruction_diagnostics(
        scene=scene,
        manifest=manifest,
        output_dir=tmp_path / "out",
        scene_fingerprint="f" * 64,
    )
    payload_1 = json.loads(path_1.read_text(encoding="utf-8"))
    bytes_1 = path_1.read_bytes()

    path_2 = write_reconstruction_diagnostics(
        scene=scene,
        manifest=manifest,
        output_dir=tmp_path / "out",
        scene_fingerprint="f" * 64,
    )
    payload_2 = json.loads(path_2.read_text(encoding="utf-8"))
    bytes_2 = path_2.read_bytes()

    assert payload_1["schema"] == "tp.reconstruction_diagnostics.v1"
    assert payload_1["scene_fingerprint"] == "f" * 64
    assert payload_1["camera_count"] == 2
    assert len(payload_1["cameras"]) == 2
    assert "reprojection_p95" in payload_1["cameras"][0]
    assert "reprojection_p99" in payload_1["cameras"][0]
    assert bytes_1 == bytes_2
    assert payload_1 == payload_2
