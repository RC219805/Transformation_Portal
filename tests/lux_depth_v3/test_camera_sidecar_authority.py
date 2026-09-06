"""Prepared reconstruction must consume the exact planned camera sidecar."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from transformation_portal.core.execution_plan import ExecutionPlanError
from transformation_portal.lux_depth_v3.camera_metadata_loader import (
    SCENE_CAMERA_SIDECAR_MAX_BYTES,
    load_scene_cameras,
    load_sidecar_payload,
)
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
from transformation_portal.lux_depth_v3.execution_plan_adapter import LuxExecutionPlanAuthorityError
from transformation_portal.lux_depth_v3.input_manager import ImageInput
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
from transformation_portal.lux_depth_v3.scene_groups import SceneGroup, build_scene_groups

pytestmark = pytest.mark.unit


def _sidecar_bytes(marker: str) -> bytes:
    return json.dumps(
        {
            "schema": "tp.scene_cameras.v1",
            "scenes": {},
            "marker": marker,
        },
        sort_keys=True,
    ).encode("utf-8")


def _prepare_reconstruction(
    tmp_path: Path,
    sidecar_path: Path,
) -> tuple[EnhanceOrchestrator, Path]:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    image = input_root / "scene.jpg"
    image.write_bytes(b"plan-input")
    prepared = prepare_lux_execution(
        EnhanceConfig(
            depth_backend="synthetic",
            allow_synthetic_fallback=True,
            enable_v2=False,
            enable_reconstruction=True,
            grouping_mode="parent_dir",
            cameras_sidecar_path=str(sidecar_path),
            non_commercial_ok=True,
            accept_research_tools_license=True,
        ),
        input_root,
        [image],
    )
    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator.config = prepared.runtime_config
    orchestrator._prepared_execution = prepared
    return orchestrator, input_root


def _run_empty_reconstruction(
    orchestrator: EnhanceOrchestrator,
    input_root: Path,
) -> None:
    orchestrator._run_scene_reconstruction_stage(
        scene_groups=[],
        results=[],
        dataset_root=input_root,
    )


def test_prepared_reconstruction_accepts_unchanged_camera_sidecar(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "cameras.json"
    sidecar_path.write_bytes(_sidecar_bytes("planned"))
    orchestrator, input_root = _prepare_reconstruction(tmp_path, sidecar_path)

    _run_empty_reconstruction(orchestrator, input_root)


def test_prepared_reconstruction_preserves_optional_absent_sidecar(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    image = input_root / "scene.jpg"
    image.write_bytes(b"plan-input")
    prepared = prepare_lux_execution(
        EnhanceConfig(
            depth_backend="synthetic",
            allow_synthetic_fallback=True,
            enable_v2=False,
            enable_reconstruction=True,
            grouping_mode="parent_dir",
            cameras_sidecar_path=None,
            non_commercial_ok=True,
            accept_research_tools_license=True,
        ),
        input_root,
        [image],
    )
    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator.config = prepared.runtime_config
    orchestrator._prepared_execution = prepared

    _run_empty_reconstruction(orchestrator, input_root)


def test_prepared_reconstruction_freezes_relative_sidecar_against_worker_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation_cwd = tmp_path / "preparation"
    preparation_cwd.mkdir()
    monkeypatch.chdir(preparation_cwd)
    relative_sidecar = Path("cameras.json")
    relative_sidecar.write_bytes(_sidecar_bytes("planned"))

    orchestrator, input_root = _prepare_reconstruction(tmp_path, relative_sidecar)
    expected_path = str((preparation_cwd / relative_sidecar).absolute())
    assert orchestrator.config.cameras_sidecar_path == expected_path

    worker_cwd = tmp_path / "worker"
    worker_cwd.mkdir()
    monkeypatch.chdir(worker_cwd)
    _run_empty_reconstruction(orchestrator, input_root)


def test_authoritative_preparation_rejects_oversized_sidecar_but_legacy_loader_remains_compatible(
    tmp_path: Path,
) -> None:
    sidecar_path = tmp_path / "oversized-cameras.json"
    sidecar_path.write_bytes(_sidecar_bytes("x" * SCENE_CAMERA_SIDECAR_MAX_BYTES))

    legacy_payload = load_sidecar_payload(sidecar_path)
    assert legacy_payload is not None
    assert len(legacy_payload["marker"]) == SCENE_CAMERA_SIDECAR_MAX_BYTES
    with pytest.raises(ExecutionPlanError, match="exceeds"):
        _prepare_reconstruction(tmp_path, sidecar_path)


@pytest.mark.parametrize("change", ["mutate", "replace", "missing", "non_regular"])
def test_prepared_reconstruction_rejects_camera_sidecar_changed_after_prepare(
    tmp_path: Path,
    change: str,
) -> None:
    sidecar_path = tmp_path / "cameras.json"
    sidecar_path.write_bytes(_sidecar_bytes("planned"))
    orchestrator, input_root = _prepare_reconstruction(tmp_path, sidecar_path)

    if change == "mutate":
        sidecar_path.write_bytes(_sidecar_bytes("mutated"))
    elif change == "replace":
        replacement = tmp_path / "replacement.json"
        replacement.write_bytes(_sidecar_bytes("replacement"))
        os.replace(replacement, sidecar_path)
    elif change == "missing":
        sidecar_path.unlink()
    else:
        sidecar_path.unlink()
        sidecar_path.mkdir()

    with pytest.raises(
        LuxExecutionPlanAuthorityError,
        match="camera sidecar|Camera sidecar",
    ):
        _run_empty_reconstruction(orchestrator, input_root)


def test_prepared_reconstruction_rejects_camera_sidecar_symlink_retarget(
    tmp_path: Path,
) -> None:
    planned_target = tmp_path / "planned.json"
    planned_target.write_bytes(_sidecar_bytes("planned"))
    replacement_target = tmp_path / "replacement.json"
    replacement_target.write_bytes(_sidecar_bytes("replacement"))
    sidecar_path = tmp_path / "cameras.json"
    sidecar_path.symlink_to(planned_target)
    orchestrator, input_root = _prepare_reconstruction(tmp_path, sidecar_path)

    sidecar_path.unlink()
    sidecar_path.symlink_to(replacement_target)

    with pytest.raises(
        LuxExecutionPlanAuthorityError,
        match="SHA-256 does not match",
    ):
        _run_empty_reconstruction(orchestrator, input_root)


def test_verified_sidecar_provenance_does_not_reresolve_retargeted_symlink(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    image = dataset_root / "scene" / "view.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    camera = {
        "intrinsics": [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
        "extrinsics": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "width": 1,
        "height": 1,
    }
    planned_bytes = json.dumps(
        {
            "schema": "tp.scene_cameras.v1",
            "scenes": {
                "scene": {
                    "images": ["scene/view.jpg"],
                    "cameras": [camera],
                }
            },
        },
        sort_keys=True,
    ).encode("utf-8")
    planned_target = tmp_path / "planned.json"
    planned_target.write_bytes(planned_bytes)
    replacement_target = tmp_path / "replacement.json"
    replacement_target.write_bytes(_sidecar_bytes("replacement"))
    sidecar_path = tmp_path / "cameras.json"
    sidecar_path.symlink_to(planned_target)
    lexical_source = str(sidecar_path.absolute())
    payload = load_sidecar_payload(
        sidecar_path,
        expected_sha256=hashlib.sha256(planned_bytes).hexdigest(),
    )
    assert payload is not None

    sidecar_path.unlink()
    sidecar_path.symlink_to(replacement_target)
    cameras = load_scene_cameras(
        scene=SceneGroup(scene_id="scene", images=(image,)),
        dataset_root=dataset_root,
        sidecar_path=sidecar_path,
        sidecar_payload=payload,
        sidecar_source_file=lexical_source,
    )

    assert cameras is not None
    assert cameras[0].provenance.file == lexical_source
    assert cameras[0].provenance.file != str(replacement_target.resolve())


def test_prepared_reconstruction_carries_verified_sidecar_digest_across_build_race(tmp_path: Path) -> None:
    from PIL import Image

    import transformation_portal.lux_depth_v3.orchestrator as orchestrator_module

    input_root = tmp_path / "inputs"
    scene_root = input_root / "scene"
    scene_root.mkdir(parents=True)
    images = (scene_root / "view_1.png", scene_root / "view_2.png")
    for index, image_path in enumerate(images):
        Image.new("RGB", (64, 64), color=(index + 1, index + 2, index + 3)).save(image_path)
    scene = build_scene_groups(images, dataset_root=input_root, grouping_mode="parent_dir")[0]

    camera = {
        "intrinsics": [[1000.0, 0.0, 32.0], [0.0, 1000.0, 32.0], [0.0, 0.0, 1.0]],
        "extrinsics": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "width": 64,
        "height": 64,
    }
    second_camera = json.loads(json.dumps(camera))
    second_camera["extrinsics"][0][3] = 0.1
    sidecar_payload = {
        "schema": "tp.scene_cameras.v1",
        "scenes": {
            scene.scene_id: {
                "images": [path.relative_to(input_root).as_posix() for path in scene.images],
                "cameras": [camera, second_camera],
            }
        },
    }
    planned_bytes = json.dumps(sidecar_payload, sort_keys=True).encode("utf-8")
    planned_digest = hashlib.sha256(planned_bytes).hexdigest()
    replacement_bytes = _sidecar_bytes("replacement")
    sidecar_path = tmp_path / "cameras.json"
    sidecar_path.write_bytes(planned_bytes)

    prepared = prepare_lux_execution(
        EnhanceConfig(
            depth_backend="synthetic",
            allow_synthetic_fallback=True,
            enable_v2=False,
            enable_reconstruction=True,
            grouping_mode="parent_dir",
            cameras_sidecar_path=str(sidecar_path),
            non_commercial_ok=True,
            accept_research_tools_license=True,
        ),
        input_root,
        images,
    )
    orchestrator = EnhanceOrchestrator.from_prepared(prepared, tmp_path / "output")
    image_inputs = [ImageInput(path) for path in scene.images]
    results = [{"status": "ok", "image": str(path)} for path in scene.images]
    real_load_scene_cameras = orchestrator_module.load_scene_cameras
    real_build_scene_manifest = orchestrator_module.build_scene_manifest
    built_manifests: list[dict] = []

    def load_then_replace(*args, **kwargs):
        loaded = real_load_scene_cameras(*args, **kwargs)
        assert loaded is not None
        sidecar_path.write_bytes(replacement_bytes)
        return loaded

    def build_then_restore(*args, **kwargs):
        assert sidecar_path.read_bytes() == replacement_bytes
        manifest = real_build_scene_manifest(*args, **kwargs)
        built_manifests.append(manifest)
        sidecar_path.write_bytes(planned_bytes)
        return manifest

    def run_reconstruction(**kwargs):
        report_path = kwargs["output_dir"] / f"{kwargs['context'].scene_id}_reconstruction_report.json"
        report_path.write_text("{}", encoding="utf-8")
        return report_path

    orchestrator.run_scene_reconstruction_fn = run_reconstruction
    orchestrator._prepare_prepared_batch_input_snapshots(image_inputs)
    try:
        with (
            patch.object(orchestrator_module, "load_scene_cameras", side_effect=load_then_replace),
            patch.object(orchestrator_module, "build_scene_manifest", side_effect=build_then_restore),
        ):
            orchestrator._run_scene_reconstruction_stage(
                scene_groups=[scene],
                results=results,
                dataset_root=input_root,
            )
    finally:
        orchestrator._cleanup_prepared_batch_input_snapshots()

    assert len(built_manifests) == 1
    assert built_manifests[0]["camera_sidecar"]["sha256"] == planned_digest
    scene_manifest_path = Path(results[0]["reconstruction_scene_manifest_path"])
    written_manifest = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
    assert written_manifest["camera_sidecar"]["sha256"] == planned_digest
