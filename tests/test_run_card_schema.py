"""Run card schema and integrity helper tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
from transformation_portal.lux_depth_v3.orchestrator import (
    ApexStrictGateError,
    EnhanceOrchestrator,
    _build_artifact_index,
    _compute_artifact_merkle_root,
    _run_card_schema_path,
    _v2_log_filename,
    _validate_run_card_backend_semantics,
    _validate_run_card_payload,
)


def _valid_run_card_payload() -> dict:
    return {
        "batch_id": "2026-02-28_120000",
        "start_time": "2026-02-28T12:00:00Z",
        "end_time": "2026-02-28T12:01:00Z",
        "config_fingerprint": {
            "model_variant": "METRIC_LARGE",
            "depth_quantization": "u16",
            "depth_device": "cpu",
            "preset": "premium",
            "preset_requested": "premium",
            "preset_resolved": "premium",
            "backend_requested": "da3",
            "backend_resolved": "da3",
            "device_requested": "cpu",
            "device_resolved": "cpu",
            "quality_tier": "premium",
            "raw_ingest_profile": "tp.raw_ingest.deterministic_v1",
            "raw_ingest_settings_hash": "e" * 64,
            "strict_inputs": False,
            "strict_segmentation": False,
            "apex_strict_mode": False,
            "v2_preset": "premium",
            "v2_device": "cpu",
            "v2_upscaler_backend": "realesrgan",
            "hash_algorithm": "sha256",
            "canonical_json": (
                '{"apex_strict_mode":false,"backend_requested":"da3","backend_resolved":"da3",'
                '"depth_device":"cpu","depth_quantization":"u16","device_requested":"cpu","device_resolved":"cpu",'
                '"model_variant":"METRIC_LARGE","preset":"premium","preset_requested":"premium","preset_resolved":"premium",'
                '"quality_tier":"premium","strict_inputs":false,"strict_segmentation":false,'
                '"v2_device":"cpu","v2_preset":"premium","v2_upscaler_backend":"realesrgan"}'
            ),
            "sha256": "a" * 64,
        },
        "backend_selection": {
            "requested": "da3",
            "resolved": "da3",
            "device": "cpu",
            "model_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        },
        "backend_summary": {
            "requested_backend": "da3",
            "primary_backend": "da3",
            "final_backends_used": ["da3"],
            "fallback_images": 0,
            "semantic_fallback_images": 0,
            "operational_fallback_images": 0,
        },
        "environment": {
            "python_version": "3.11.9",
            "platform": "macOS-26.3-arm64-arm-64bit",
            "machine": "arm64",
        },
        "git_revision": {
            "v2": "b" * 40,
            "v3": "b" * 40,
        },
        "runtime_stats": {
            "count": 1,
            "total": 2.5,
            "mean": 2.5,
            "min": 2.5,
            "max": 2.5,
            "median": 2.5,
        },
        "outliers": [],
        "total_images": 1,
        "success_count": 1,
        "error_count": 0,
        "artifact_index": [
            {
                "artifact_type": "combined_manifest",
                "path": "manifests/a_combined.json",
                "relative_path": "manifests/a_combined.json",
                "size_bytes": 100,
                "sha256": "c" * 64,
            }
        ],
        "artifact_merkle_root": "d" * 64,
    }


def test_run_card_schema_validates_payload():
    pytest.importorskip("jsonschema")
    payload = _valid_run_card_payload()

    _validate_run_card_payload(payload, _run_card_schema_path())
    _validate_run_card_backend_semantics(payload)


def test_run_card_schema_rejects_invalid_merkle_root():
    pytest.importorskip("jsonschema")
    payload = _valid_run_card_payload()
    payload["artifact_merkle_root"] = "not-a-digest"

    with pytest.raises(RuntimeError, match="artifact_merkle_root"):
        _validate_run_card_payload(payload, _run_card_schema_path())


def test_run_card_backend_semantics_rejects_mismatch_without_wrapper():
    payload = _valid_run_card_payload()
    payload["backend_selection"]["resolved"] = "depth_pro"

    with pytest.raises(RuntimeError, match="backend_summary.final_backends_used\\[0\\]"):
        _validate_run_card_backend_semantics(payload)


def test_run_card_backend_semantics_accepts_wrapper_metadata():
    payload = _valid_run_card_payload()
    payload["backend_selection"]["logical_backend"] = "depth_pro"
    payload["backend_selection"]["resolved_engine"] = "da3"

    _validate_run_card_backend_semantics(payload)


def test_run_card_schema_rejects_empty_artifact_index():
    pytest.importorskip("jsonschema")
    payload = _valid_run_card_payload()
    payload["artifact_index"] = []

    with pytest.raises(RuntimeError, match="artifact_index"):
        _validate_run_card_payload(payload, _run_card_schema_path())


def test_build_artifact_index_is_deterministic(tmp_path: Path):
    output_root = tmp_path / "output"
    first = output_root / "manifests" / "alpha.json"
    second = output_root / "depth" / "beta_depth.png"
    first.parent.mkdir(parents=True, exist_ok=True)
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    artifact_index = _build_artifact_index(output_root, [second, first, first])

    assert [entry["relative_path"] for entry in artifact_index] == [
        "depth/beta_depth.png",
        "manifests/alpha.json",
    ]
    assert artifact_index[0]["sha256"] == hashlib.sha256(b"second").hexdigest()
    assert artifact_index[1]["sha256"] == hashlib.sha256(b"first").hexdigest()


def test_compute_artifact_merkle_root_is_deterministic():
    artifact_index = [
        {"relative_path": "b", "sha256": "b" * 64},
        {"relative_path": "a", "sha256": "a" * 64},
    ]

    root1 = _compute_artifact_merkle_root(artifact_index)
    root2 = _compute_artifact_merkle_root(list(reversed(artifact_index)))

    assert root1 == root2


def test_v2_log_filename_is_batch_scoped_and_deterministic():
    assert _v2_log_filename("image_key", "2026-03-01_154123") == "v2_image_key__2026-03-01_154123.log"
    assert _v2_log_filename("image_key", "2026-03-01_154124") == "v2_image_key__2026-03-01_154124.log"
    assert _v2_log_filename("image_key") == "v2_image_key.log"


def test_collect_run_card_artifacts_includes_v2_deliverables(tmp_path: Path):
    output_root = tmp_path / "output"
    manifests_dir = output_root / "manifests"
    v2_dir = output_root / "v2"
    logs_dir = output_root / "logs"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifests_dir / "image_01_combined.json"
    manifest_path.write_text("{}", encoding="utf-8")
    batch_manifest = manifests_dir / "batch_2026-03-01_154123.json"
    batch_manifest.write_text("{}", encoding="utf-8")

    v2_output = v2_dir / "image_01_materials_v3_enhanced.tif"
    v2_output.write_bytes(b"output")
    v2_report = v2_dir / "image_01_materials_v3_enhanced_report.json"
    v2_report.write_text("{}", encoding="utf-8")
    stale_output = v2_dir / "image_01_stale.tif"
    stale_output.write_bytes(b"stale")
    stale_report = v2_dir / "image_01_stale_report.json"
    stale_report.write_text("{}", encoding="utf-8")

    scoped_log = logs_dir / "v2_image_01__2026-03-01_154123.log"
    scoped_log.write_text("scoped", encoding="utf-8")
    legacy_log = logs_dir / "v2_image_01.log"
    legacy_log.write_text("legacy", encoding="utf-8")

    orch = object.__new__(EnhanceOrchestrator)
    orch.manifests_dir = manifests_dir
    orch.logs_dir = logs_dir
    orch.v2_dir = v2_dir

    result = {
        "status": "ok",
        "manifest": str(manifest_path),
        "v2_output_path": str(v2_output),
        "v2_report_path": str(v2_report),
    }
    artifact_paths = orch._collect_run_card_artifact_paths([result], batch_manifest_path=batch_manifest)
    artifact_index = _build_artifact_index(output_root, artifact_paths)
    relative_paths = {entry["relative_path"] for entry in artifact_index}

    assert "v2/image_01_materials_v3_enhanced.tif" in relative_paths
    assert "v2/image_01_materials_v3_enhanced_report.json" in relative_paths
    assert "logs/v2_image_01__2026-03-01_154123.log" in relative_paths
    assert "v2/image_01_stale.tif" not in relative_paths
    assert "v2/image_01_stale_report.json" not in relative_paths
    assert "logs/v2_image_01.log" not in relative_paths


def test_collect_run_card_artifacts_includes_segmentation_mask_artifact(tmp_path: Path):
    from transformation_portal.lux_depth_v3.manifest import CombinedManifest, MaterialsV3Metadata

    output_root = tmp_path / "output"
    manifests_dir = output_root / "manifests"
    logs_dir = output_root / "logs"
    v2_dir = output_root / "v2"
    segmentation_dir = output_root / "segmentation"

    manifests_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    mask_artifact = segmentation_dir / "image_01_materials_v3_masks.npz"
    np.savez_compressed(mask_artifact, glass=np.ones((2, 2), dtype=np.float32))

    manifest_path = manifests_dir / "image_01_combined.json"
    CombinedManifest(
        materials_v3=MaterialsV3Metadata(
            enabled=True,
            segmentation_metadata={
                "mask_artifact_path": str(mask_artifact),
                "mask_artifact_format": "npz",
            },
        )
    ).save(manifest_path)

    orch = object.__new__(EnhanceOrchestrator)
    orch.manifests_dir = manifests_dir
    orch.logs_dir = logs_dir
    orch.v2_dir = v2_dir

    result = {
        "status": "ok",
        "manifest": str(manifest_path),
    }
    artifact_paths = orch._collect_run_card_artifact_paths([result])
    artifact_index = _build_artifact_index(output_root, artifact_paths)
    artifacts_by_path = {entry["relative_path"]: entry for entry in artifact_index}

    assert "segmentation/image_01_materials_v3_masks.npz" in artifacts_by_path
    assert artifacts_by_path["segmentation/image_01_materials_v3_masks.npz"]["artifact_type"] == "segmentation_mask_npz"


def test_collect_run_card_artifacts_includes_reconstruction_report(tmp_path: Path):
    output_root = tmp_path / "output"
    manifests_dir = output_root / "manifests"
    logs_dir = output_root / "logs"
    v2_dir = output_root / "v2"
    reconstruction_dir = output_root / "reconstruction"

    manifests_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    reconstruction_report = reconstruction_dir / "abc123_reconstruction_report.json"
    reconstruction_report.write_text("{}", encoding="utf-8")

    orch = object.__new__(EnhanceOrchestrator)
    orch.manifests_dir = manifests_dir
    orch.logs_dir = logs_dir
    orch.v2_dir = v2_dir

    result = {
        "status": "ok",
        "reconstruction_report_path": str(reconstruction_report),
    }
    artifact_paths = orch._collect_run_card_artifact_paths([result])
    artifact_index = _build_artifact_index(output_root, artifact_paths)
    artifacts_by_path = {entry["relative_path"]: entry for entry in artifact_index}

    assert "reconstruction/abc123_reconstruction_report.json" in artifacts_by_path
    assert artifacts_by_path["reconstruction/abc123_reconstruction_report.json"]["artifact_type"] == "reconstruction_report"


def test_collect_run_card_artifacts_includes_reconstruction_diagnostics(tmp_path: Path):
    output_root = tmp_path / "output"
    manifests_dir = output_root / "manifests"
    logs_dir = output_root / "logs"
    v2_dir = output_root / "v2"
    reconstruction_dir = output_root / "reconstruction"

    manifests_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_artifact = reconstruction_dir / "abc123_reconstruction_diagnostics.json"
    diagnostics_artifact.write_text("{}", encoding="utf-8")

    orch = object.__new__(EnhanceOrchestrator)
    orch.manifests_dir = manifests_dir
    orch.logs_dir = logs_dir
    orch.v2_dir = v2_dir

    result = {
        "status": "ok",
        "reconstruction_diagnostics_path": str(diagnostics_artifact),
    }
    artifact_paths = orch._collect_run_card_artifact_paths([result])
    artifact_index = _build_artifact_index(output_root, artifact_paths)
    artifacts_by_path = {entry["relative_path"]: entry for entry in artifact_index}

    assert "reconstruction/abc123_reconstruction_diagnostics.json" in artifacts_by_path
    assert (
        artifacts_by_path["reconstruction/abc123_reconstruction_diagnostics.json"]["artifact_type"]
        == "reconstruction_diagnostics"
    )


def test_collect_run_card_artifacts_includes_reconstruction_preflight(tmp_path: Path):
    output_root = tmp_path / "output"
    manifests_dir = output_root / "manifests"
    logs_dir = output_root / "logs"
    v2_dir = output_root / "v2"
    reconstruction_dir = output_root / "reconstruction"

    manifests_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    preflight_artifact = reconstruction_dir / "abc123_preflight.json"
    preflight_artifact.write_text("{}", encoding="utf-8")

    orch = object.__new__(EnhanceOrchestrator)
    orch.manifests_dir = manifests_dir
    orch.logs_dir = logs_dir
    orch.v2_dir = v2_dir

    result = {
        "status": "ok",
        "reconstruction_preflight_path": str(preflight_artifact),
    }
    artifact_paths = orch._collect_run_card_artifact_paths([result])
    artifact_index = _build_artifact_index(output_root, artifact_paths)
    artifacts_by_path = {entry["relative_path"]: entry for entry in artifact_index}

    assert "reconstruction/abc123_preflight.json" in artifacts_by_path
    assert artifacts_by_path["reconstruction/abc123_preflight.json"]["artifact_type"] == "reconstruction_preflight_json"


def test_collect_run_card_artifacts_includes_reconstruction_manifest(tmp_path: Path):
    output_root = tmp_path / "output"
    manifests_dir = output_root / "manifests"
    logs_dir = output_root / "logs"
    v2_dir = output_root / "v2"
    reconstruction_dir = output_root / "reconstruction"

    manifests_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    manifest_artifact = reconstruction_dir / "abc123_manifest.json"
    manifest_artifact.write_text("{}", encoding="utf-8")

    orch = object.__new__(EnhanceOrchestrator)
    orch.manifests_dir = manifests_dir
    orch.logs_dir = logs_dir
    orch.v2_dir = v2_dir

    result = {
        "status": "ok",
        "reconstruction_manifest_path": str(manifest_artifact),
    }
    artifact_paths = orch._collect_run_card_artifact_paths([result])
    artifact_index = _build_artifact_index(output_root, artifact_paths)
    artifacts_by_path = {entry["relative_path"]: entry for entry in artifact_index}

    assert "reconstruction/abc123_manifest.json" in artifacts_by_path
    assert artifacts_by_path["reconstruction/abc123_manifest.json"]["artifact_type"] == "reconstruction_manifest_json"


def test_collect_run_card_artifacts_includes_reconstruction_scene_manifest(tmp_path: Path):
    output_root = tmp_path / "output"
    manifests_dir = output_root / "manifests"
    logs_dir = output_root / "logs"
    v2_dir = output_root / "v2"
    reconstruction_dir = output_root / "reconstruction"

    manifests_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    scene_manifest_artifact = reconstruction_dir / "abc123_scene_manifest.json"
    scene_manifest_artifact.write_text("{}", encoding="utf-8")

    orch = object.__new__(EnhanceOrchestrator)
    orch.manifests_dir = manifests_dir
    orch.logs_dir = logs_dir
    orch.v2_dir = v2_dir

    result = {
        "status": "ok",
        "reconstruction_scene_manifest_path": str(scene_manifest_artifact),
    }
    artifact_paths = orch._collect_run_card_artifact_paths([result])
    artifact_index = _build_artifact_index(output_root, artifact_paths)
    artifacts_by_path = {entry["relative_path"]: entry for entry in artifact_index}

    assert "reconstruction/abc123_scene_manifest.json" in artifacts_by_path
    assert artifacts_by_path["reconstruction/abc123_scene_manifest.json"]["artifact_type"] == "reconstruction_scene_manifest"


def test_collect_run_card_artifacts_includes_reconstruction_debug_bundle_artifacts(tmp_path: Path):
    output_root = tmp_path / "output"
    manifests_dir = output_root / "manifests"
    logs_dir = output_root / "logs"
    v2_dir = output_root / "v2"
    reconstruction_debug_dir = output_root / "reconstruction" / "abc123" / "debug"

    manifests_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_debug_dir.mkdir(parents=True, exist_ok=True)

    debug_scene_manifest = reconstruction_debug_dir / "scene_manifest.json"
    debug_scene_manifest.write_text("{}", encoding="utf-8")
    debug_cameras = reconstruction_debug_dir / "cameras.json"
    debug_cameras.write_text("[]", encoding="utf-8")
    debug_preview = reconstruction_debug_dir / "reprojection_preview.png"
    debug_preview.write_bytes(b"png")

    orch = object.__new__(EnhanceOrchestrator)
    orch.manifests_dir = manifests_dir
    orch.logs_dir = logs_dir
    orch.v2_dir = v2_dir

    result = {
        "status": "ok",
        "reconstruction_debug_manifest_path": str(debug_scene_manifest),
        "reconstruction_debug_cameras_path": str(debug_cameras),
        "reconstruction_debug_preview_path": str(debug_preview),
    }
    artifact_paths = orch._collect_run_card_artifact_paths([result])
    artifact_index = _build_artifact_index(output_root, artifact_paths)
    artifacts_by_path = {entry["relative_path"]: entry for entry in artifact_index}

    assert "reconstruction/abc123/debug/scene_manifest.json" in artifacts_by_path
    assert "reconstruction/abc123/debug/cameras.json" in artifacts_by_path
    assert "reconstruction/abc123/debug/reprojection_preview.png" in artifacts_by_path
    assert (
        artifacts_by_path["reconstruction/abc123/debug/scene_manifest.json"]["artifact_type"]
        == "reconstruction_debug_scene_manifest_json"
    )
    assert (
        artifacts_by_path["reconstruction/abc123/debug/cameras.json"]["artifact_type"] == "reconstruction_debug_cameras_json"
    )
    assert (
        artifacts_by_path["reconstruction/abc123/debug/reprojection_preview.png"]["artifact_type"]
        == "reconstruction_debug_preview_png"
    )


def test_apex_v2_depth_handoff_missing_raises_strict_gate(tmp_path: Path):
    depth_path = tmp_path / "depth" / "image_depth.png"
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.write_bytes(b"depth")

    report_path = tmp_path / "v2" / "image_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('{"depth_consumed":false}', encoding="utf-8")

    orch = object.__new__(EnhanceOrchestrator)
    orch._is_apex_tier = lambda: True

    with pytest.raises(ApexStrictGateError, match="APEX_V2_DEPTH_HANDOFF_MISSING"):
        orch._enforce_v2_depth_handoff(depth_path=depth_path, v2_result=None, v2_report_path=report_path)


def test_extract_v2_depth_handoff_prefers_depth_consumed_field():
    orch = object.__new__(EnhanceOrchestrator)

    status = orch._extract_v2_depth_handoff_status(
        v2_result={"depth_consumed": False, "depth_map": "/tmp/depth.png"},
        v2_report_path=None,
    )

    assert status is False


def test_config_fingerprint_uses_raw_preset_requested_when_enum_unset():
    orch = object.__new__(EnhanceOrchestrator)
    orch.config = EnhanceConfig(
        model_variant=ModelVariant.METRIC_LARGE,
        preset=None,
        preset_requested="premium",
        quality_tier="apex",
        depth_device="cpu",
    )
    orch._backend_metadata = SimpleNamespace(
        requested_backend="depth_pro",
        resolved_backend="depth_pro",
        device="cpu",
    )
    orch._is_apex_tier = lambda: True

    fingerprint = orch._build_run_card_config_fingerprint()

    assert fingerprint["preset_requested"] == "premium"
    assert fingerprint["preset_resolved"] == "quality_tier:apex"
    assert fingerprint["raw_ingest_profile"] == "tp.raw_ingest.deterministic_v1"
    assert len(fingerprint["raw_ingest_settings_hash"]) == 64


def test_resolve_run_card_backend_model_id_prefers_selected_attempt_model():
    orch = object.__new__(EnhanceOrchestrator)
    orch._backend_metadata = SimpleNamespace(
        resolved_backend="da3",
        model_id="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    )
    orch._depth_backend_cache = {}
    orch.config = EnhanceConfig(model_variant=ModelVariant.METRIC_LARGE)

    results = [
        {
            "status": "ok",
            "backend": "da2",
            "model_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            "selected_attempt_index": 2,
            "attempts": [
                {
                    "attempt": 0,
                    "backend": "depth_pro",
                    "status": "failed",
                    "model_id": "apple/ml-depth-pro",
                },
                {
                    "attempt": 1,
                    "backend": "da3",
                    "status": "failed",
                    "model_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                },
                {
                    "attempt": 2,
                    "backend": "da2",
                    "status": "success",
                    "model_id": "depth-anything/Depth-Anything-V2-Small-hf",
                },
            ],
        },
    ]

    model_id = orch._resolve_run_card_backend_model_id(results, "da2")
    assert model_id == "depth-anything/Depth-Anything-V2-Small-hf"


def test_run_card_schema_accepts_backend_model_artifact_fields():
    pytest.importorskip("jsonschema")
    payload = _valid_run_card_payload()
    payload["backend_selection"]["model_artifact_filename"] = "depth_pro.pt"
    payload["backend_selection"]["model_artifact_sha256"] = "f" * 64

    _validate_run_card_payload(payload, _run_card_schema_path())


def test_run_card_schema_rejects_invalid_backend_model_artifact_sha256():
    pytest.importorskip("jsonschema")
    payload = _valid_run_card_payload()
    payload["backend_selection"]["model_artifact_sha256"] = "invalid-sha"

    with pytest.raises(RuntimeError, match="model_artifact_sha256"):
        _validate_run_card_payload(payload, _run_card_schema_path())


def test_resolve_backend_model_id_depth_pro_uses_canonical_identifier():
    orch = object.__new__(EnhanceOrchestrator)
    backend = SimpleNamespace(
        _checkpoint_path=Path("/tmp/depth_pro_custom.pt"),
        model_id="apple/ml-depth-pro:depth_pro_custom.pt",
    )

    model_id = orch._resolve_backend_model_id(
        "depth_pro",
        result_metadata={"model_id": "apple/ml-depth-pro:depth_pro_custom.pt"},
        backend=backend,
    )

    assert model_id == "apple/ml-depth-pro"


def test_resolve_run_card_backend_model_artifact_prefers_selected_attempt():
    orch = object.__new__(EnhanceOrchestrator)
    orch._backend_metadata = SimpleNamespace(
        resolved_backend="depth_pro",
        model_id="apple/ml-depth-pro",
    )
    orch._depth_backend_cache = {}

    results = [
        {
            "status": "ok",
            "backend": "depth_pro",
            "selected_attempt_index": 1,
            "attempts": [
                {
                    "attempt": 0,
                    "backend": "depth_pro",
                    "status": "failed",
                    "model_artifact_filename": "depth_pro_old.pt",
                    "model_artifact_sha256": "a" * 64,
                },
                {
                    "attempt": 1,
                    "backend": "depth_pro",
                    "status": "success",
                    "model_artifact_filename": "depth_pro.pt",
                    "model_artifact_sha256": "B" * 64,
                },
            ],
        },
    ]

    artifact = orch._resolve_run_card_backend_model_artifact(results, "depth_pro")

    assert artifact == {
        "model_artifact_filename": "depth_pro.pt",
        "model_artifact_sha256": "b" * 64,
    }
