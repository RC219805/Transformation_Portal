"""Run card schema and integrity helper tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

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
from transformation_portal.lux_depth_v3.run_card_contract import render_run_card_output_relative_path
from transformation_portal.lux_depth_v3.security import HashMode
from transformation_portal.schemas.run_card import load_run_card_schema


def _valid_run_card_payload() -> dict:
    config_fingerprint = {
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
        "strict_inputs": False,
        "strict_segmentation": False,
        "apex_strict_mode": False,
        "v2_preset": "premium",
        "v2_device": "cpu",
        "v2_upscaler_backend": "realesrgan",
        "depth_pro_python_executable": None,
        "raw_python_executable": None,
        "da3_python_executable": None,
    }
    canonical_json = json.dumps(
        {
            field: config_fingerprint[field]
            for field in (
                "model_variant",
                "depth_quantization",
                "depth_device",
                "preset",
                "v2_preset",
                "v2_device",
                "v2_upscaler_backend",
                "preset_requested",
                "preset_resolved",
                "backend_requested",
                "backend_resolved",
                "device_requested",
                "device_resolved",
                "quality_tier",
                "strict_inputs",
                "strict_segmentation",
                "apex_strict_mode",
                "depth_pro_python_executable",
                "raw_python_executable",
                "da3_python_executable",
            )
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    config_fingerprint["hash_algorithm"] = "sha256"
    config_fingerprint["canonical_json"] = canonical_json
    config_fingerprint["sha256"] = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return {
        "batch_id": "2026-02-28_120000",
        "start_time": "2026-02-28T12:00:00Z",
        "end_time": "2026-02-28T12:01:00Z",
        "config_fingerprint": {
            **config_fingerprint,
            "raw_ingest_profile": "tp.raw_ingest.deterministic_v1",
            "raw_ingest_settings_hash": "e" * 64,
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


def test_packaged_run_card_schemas_match_documented_copies() -> None:
    for version in ("v1", "v2"):
        documented_schema = json.loads(_run_card_schema_path(version).read_text(encoding="utf-8"))
        assert load_run_card_schema(version) == documented_schema


def test_run_card_schema_enforces_datetime_format() -> None:
    pytest.importorskip("jsonschema")
    payload = _valid_run_card_payload()
    payload["start_time"] = "not-a-date-time"

    with pytest.raises(RuntimeError, match="start_time"):
        _validate_run_card_payload(payload, _run_card_schema_path())


def test_run_card_schema_rejects_space_separated_datetime() -> None:
    pytest.importorskip("jsonschema")
    payload = _valid_run_card_payload()
    payload["start_time"] = "2026-02-28 12:00:00Z"

    with pytest.raises(RuntimeError, match="start_time"):
        _validate_run_card_payload(payload, _run_card_schema_path())


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


def test_run_card_output_relative_path_handles_alias_equivalent_roots(tmp_path: Path):
    actual_root = tmp_path / "actual_output"
    actual_manifest = actual_root / "manifests" / "alpha.json"
    actual_manifest.parent.mkdir(parents=True)
    actual_manifest.write_text("{}")

    alias_root = tmp_path / "alias_output"
    alias_root.symlink_to(actual_root, target_is_directory=True)

    relative_path = render_run_card_output_relative_path(str(actual_manifest), alias_root)

    assert relative_path == "manifests/alpha.json"


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


def test_build_run_card_result_summary_uses_cached_segmentation_metadata(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / "image_01_combined.json"

    orch = object.__new__(EnhanceOrchestrator)
    orch.output_root = output_root
    orch._active_run_card_segmentation_metadata = {
        str(manifest_path): {
            "mask_artifact_path": str(output_root / "segmentation" / "image_01_materials_v3_masks.npz"),
            "mask_artifact_format": "npz",
            "tile_size": 1024,
        }
    }

    with patch.object(Path, "read_text", side_effect=AssertionError("manifest reread")):
        summary = orch._build_run_card_result_summary(
            [
                {
                    "image": str(tmp_path / "inputs" / "image_01.png"),
                    "status": "ok",
                    "backend": "da3",
                    "runtime_s": 1.23,
                    "manifest": str(manifest_path),
                }
            ]
        )

    assert summary == [
        {
            "image": "image_01.png",
            "status": "ok",
            "backend": "da3",
            "runtime_s": 1.23,
            "manifest_path": "manifests/image_01_combined.json",
            "error_code": None,
            "error_message": None,
            "error_details": None,
            "segmentation_metadata": {
                "mask_artifact_path": str(output_root / "segmentation" / "image_01_materials_v3_masks.npz"),
                "mask_artifact_format": "npz",
                "tile_size": 1024,
            },
            "quality_gate": None,
            "capability": {
                "requested_backend": "da3",
                "executed_backend": "da3",
                "availability_state": "available",
                "reason": None,
                "synthetic_output": False,
                "stub_mode": False,
                "fallback_executed": False,
                "model_repo_id": None,
                "model_revision": None,
                "asset_bundle_version": None,
            },
        }
    ]


def test_build_run_card_result_summary_projects_quality_gate_and_capability(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    orch = object.__new__(EnhanceOrchestrator)
    orch.output_root = output_root
    orch._active_run_card_segmentation_metadata = {}
    orch._backend_metadata = SimpleNamespace(requested_backend="depth_pro", resolution_reason=None)

    summary = orch._build_run_card_result_summary(
        [
            {
                "image": str(tmp_path / "inputs" / "image_02.png"),
                "status": "ok",
                "backend": "depth_pro",
                "runtime_s": 2.34,
                "manifest": None,
                "fallback_used": False,
                "model_id": "apple/ml-depth-pro",
                "quality_gate": {
                    "passed": True,
                    "failure_codes": [],
                    "warnings": ["APEX_DEPTH_GRADIENT_LOW"],
                    "metrics": {"finite_pct": 1.0},
                    "thresholds": {"finite_pct_min": 0.999},
                    "shape_context": {"native_shape": [64, 64]},
                },
            }
        ]
    )

    assert summary[0]["quality_gate"] == {
        "kind": "apex_depth",
        "passed": True,
        "failure_codes": [],
        "warnings": ["APEX_DEPTH_GRADIENT_LOW"],
        "details": {
            "metrics": {"finite_pct": 1.0},
            "thresholds": {"finite_pct_min": 0.999},
            "shape_context": {"native_shape": [64, 64]},
        },
    }
    assert summary[0]["capability"] == {
        "requested_backend": "depth_pro",
        "executed_backend": "depth_pro",
        "availability_state": "available",
        "reason": None,
        "synthetic_output": False,
        "stub_mode": False,
        "fallback_executed": False,
        "model_repo_id": "apple/ml-depth-pro",
        "model_revision": None,
        "asset_bundle_version": None,
    }


def test_build_run_card_result_summary_keeps_semantic_gate_failure_available(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    orch = object.__new__(EnhanceOrchestrator)
    orch.output_root = output_root
    orch._active_run_card_segmentation_metadata = {}
    orch._backend_metadata = SimpleNamespace(
        requested_backend="da3",
        resolution_reason=None,
    )

    summary = orch._build_run_card_result_summary(
        [
            {
                "image": str(tmp_path / "inputs" / "image_03.png"),
                "status": "error",
                "backend": "da3",
                "runtime_s": 1.11,
                "manifest": None,
                "fallback_used": False,
                "error_code": "APEX_DEPTH_SATURATION_LOW",
                "error": "APEX depth validity gate failed: APEX_DEPTH_SATURATION_LOW",
                "error_details": {
                    "passed": False,
                    "failure_codes": ["APEX_DEPTH_SATURATION_LOW"],
                    "warnings": [],
                    "metrics": {"saturation_low_fraction": 0.031},
                    "thresholds": {"saturation_low_fraction_max": 0.02},
                    "shape_context": {"native_shape": [64, 64]},
                },
                "attempts": [
                    {
                        "backend": "da3",
                        "status": "failed",
                        "failure_kind": "semantic",
                        "error_code": "APEX_DEPTH_SATURATION_LOW",
                        "error_message": "APEX depth validity gate failed: APEX_DEPTH_SATURATION_LOW",
                        "model_id": "depth-anything/Depth-Anything-V2-Small-hf",
                        "model_revision": "rev-semantic",
                    }
                ],
            }
        ]
    )

    assert summary[0]["quality_gate"] == {
        "kind": "apex_depth",
        "passed": False,
        "failure_codes": ["APEX_DEPTH_SATURATION_LOW"],
        "warnings": [],
        "details": {
            "metrics": {"saturation_low_fraction": 0.031},
            "thresholds": {"saturation_low_fraction_max": 0.02},
            "shape_context": {"native_shape": [64, 64]},
        },
    }
    assert summary[0]["capability"] == {
        "requested_backend": "da3",
        "executed_backend": "da3",
        "availability_state": "available",
        "reason": None,
        "synthetic_output": False,
        "stub_mode": False,
        "fallback_executed": False,
        "model_repo_id": "depth-anything/Depth-Anything-V2-Small-hf",
        "model_revision": "rev-semantic",
        "asset_bundle_version": None,
    }


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_run_card_schema_accepts_additive_quality_gate_and_capability(version: str) -> None:
    pytest.importorskip("jsonschema")
    payload = _valid_run_card_payload()
    payload["run_card_version"] = version
    payload["result_summary"] = [
        {
            "image": "image_01.png",
            "status": "ok",
            "backend": "depth_pro",
            "runtime_s": 1.25,
            "quality_gate": {
                "kind": "apex_depth",
                "passed": True,
                "failure_codes": [],
                "warnings": [],
                "details": {
                    "metrics": {"finite_pct": 1.0},
                    "thresholds": {"finite_pct_min": 0.999},
                    "shape_context": {"native_shape": [64, 64]},
                },
            },
            "capability": {
                "requested_backend": "depth_pro",
                "executed_backend": "depth_pro",
                "availability_state": "available",
                "reason": None,
                "synthetic_output": False,
                "stub_mode": False,
                "fallback_executed": False,
                "model_repo_id": "apple/ml-depth-pro",
                "model_revision": "rev-123",
                "asset_bundle_version": None,
            },
        }
    ]
    if version == "v2":
        payload.pop("artifact_merkle_root", None)
        payload["artifact_tree"] = {
            "algorithm": "ct-sha256-v1",
            "leaf_format": "tp.run_card.artifact_leaf.v1",
            "leaf_count": 0,
            "root_sha256": "e" * 64,
            "artifacts": [],
        }

    _validate_run_card_payload(payload, _run_card_schema_path(version))


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_run_card_schema_accepts_additive_model_contract(version: str) -> None:
    payload = _valid_run_card_payload()
    payload["run_card_version"] = version
    if version == "v1":
        payload["artifact_merkle_root"] = "0" * 64
    else:
        payload.pop("artifact_merkle_root", None)
        payload["artifact_tree"] = {
            "algorithm": "ct-sha256-v1",
            "leaf_format": "tp.run_card.artifact_leaf.v1",
            "leaf_count": 0,
            "root_sha256": "1" * 64,
            "artifacts": [],
        }
    payload["model_contract"] = {
        "requested_model_selector": "METRIC_LARGE",
        "canonical_model_key": "da3_research",
        "resolved_repo_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        "resolved_revision": "b2359bdf726fb44ef62acca04d629dcf158053e7",
        "license_id": "cc-by-nc-4.0",
        "usage_class": "non_commercial_only",
        "requires_non_commercial_ok": True,
        "non_commercial_ok": True,
        "backend_kind": "da3_api",
        "accelerator_kind": "none",
        "fallback_chain": [],
        "manifest_schema_version": 1,
    }
    _validate_run_card_payload(payload, _run_card_schema_path(version))


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
    orch.config = SimpleNamespace(model_variant=ModelVariant.METRIC_LARGE)
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


def test_emit_run_card_respects_run_card_include_proofs_flag(tmp_path: Path):
    config = EnhanceConfig(
        model_variant=ModelVariant.METRIC_LARGE,
        model_key="da3-metric",
        run_card_version="v2",
        run_card_include_proofs=False,
    )
    orch = EnhanceOrchestrator(config, tmp_path)

    artifact_path = tmp_path / "depth" / "image_01_depth.png"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"depth")
    missing_schema_path = tmp_path / "missing_run_card.schema.json"

    with (
        patch(
            "transformation_portal.lux_depth_v3.orchestrator.build_artifact_tree", return_value={"artifacts": []}
        ) as build_tree,
        patch.object(orch, "_collect_run_card_artifact_paths", return_value=[artifact_path]),
        patch.object(
            orch,
            "_compute_backend_summary",
            return_value={
                "requested_backend": "da3",
                "primary_backend": "da3",
                "final_backends_used": ["da3"],
                "fallback_images": 0,
                "semantic_fallback_images": 0,
                "operational_fallback_images": 0,
            },
        ),
        patch.object(orch, "_requested_backend_fulfillment_defect", return_value=None),
        patch.object(
            orch,
            "_resolve_run_card_backend_model_artifact",
            return_value={"model_artifact_filename": None, "model_artifact_sha256": None},
        ),
        patch.object(orch, "_resolve_run_card_backend_model_id", return_value="depth-anything/DA3"),
        patch("transformation_portal.lux_depth_v3.orchestrator._run_card_schema_path", return_value=missing_schema_path),
    ):
        orch._emit_run_card(
            batch_id="2026-04-10_120000",
            start_time="2026-04-10T12:00:00Z",
            end_time="2026-04-10T12:05:00Z",
            results=[{"status": "ok"}],
            runtime_stats={"count": 1},
            outliers=[],
        )

    build_tree.assert_called_once()
    assert build_tree.call_args.kwargs["include_proofs"] is False


def test_emit_run_card_skips_legacy_merkle_root_for_v2(tmp_path: Path):
    config = EnhanceConfig(
        model_variant=ModelVariant.METRIC_LARGE,
        model_key="da3-metric",
        run_card_version="v2",
    )
    orch = EnhanceOrchestrator(config, tmp_path)

    artifact_path = tmp_path / "depth" / "image_01_depth.png"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"depth")
    missing_schema_path = tmp_path / "missing_run_card.schema.json"

    with (
        patch(
            "transformation_portal.lux_depth_v3.orchestrator.build_artifact_tree", return_value={"artifacts": []}
        ) as build_tree,
        patch(
            "transformation_portal.lux_depth_v3.orchestrator._compute_artifact_merkle_root",
            side_effect=AssertionError("legacy merkle root should not be computed for v2"),
        ) as merkle_root,
        patch.object(orch, "_collect_run_card_artifact_paths", return_value=[artifact_path]),
        patch.object(
            orch,
            "_compute_backend_summary",
            return_value={
                "requested_backend": "da3",
                "primary_backend": "da3",
                "final_backends_used": ["da3"],
                "fallback_images": 0,
                "semantic_fallback_images": 0,
                "operational_fallback_images": 0,
            },
        ),
        patch.object(orch, "_requested_backend_fulfillment_defect", return_value=None),
        patch.object(
            orch,
            "_resolve_run_card_backend_model_artifact",
            return_value={"model_artifact_filename": None, "model_artifact_sha256": None},
        ),
        patch.object(orch, "_resolve_run_card_backend_model_id", return_value="depth-anything/DA3"),
        patch("transformation_portal.lux_depth_v3.orchestrator._run_card_schema_path", return_value=missing_schema_path),
    ):
        orch._emit_run_card(
            batch_id="2026-04-10_120000",
            start_time="2026-04-10T12:00:00Z",
            end_time="2026-04-10T12:05:00Z",
            results=[{"status": "ok"}],
            runtime_stats={"count": 1},
            outliers=[],
        )

    build_tree.assert_called_once()


def test_build_run_card_inputs_skips_hashing_when_hash_mode_never(tmp_path: Path) -> None:
    config = EnhanceConfig(
        model_variant=ModelVariant.METRIC_LARGE,
        model_key="da3-metric",
        hash_mode=HashMode.NEVER,
    )
    orch = EnhanceOrchestrator(config, tmp_path)
    input_path = tmp_path / "inputs" / "image_01.png"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"pixels")

    with patch("transformation_portal.lux_depth_v3.orchestrator.compute_file_sha256", side_effect=AssertionError("hashing")):
        assert orch._build_run_card_inputs([{"image": str(input_path)}]) == []


def test_build_run_card_inputs_reuses_result_input_sha256(tmp_path: Path) -> None:
    config = EnhanceConfig(model_variant=ModelVariant.METRIC_LARGE, model_key="da3-metric")
    orch = EnhanceOrchestrator(config, tmp_path)
    input_path = tmp_path / "inputs" / "image_01.png"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"pixels")

    with patch("transformation_portal.lux_depth_v3.orchestrator.compute_file_sha256", side_effect=AssertionError("rehash")):
        records = orch._build_run_card_inputs(
            [
                {
                    "image": str(input_path),
                    "input_sha256": "A" * 64,
                }
            ]
        )

    assert records == [
        {
            "path": "image_01.png",
            "sha256": "a" * 64,
            "size_bytes": len(b"pixels"),
        }
    ]


def test_build_run_card_inputs_preserves_per_result_input_sha256(tmp_path: Path) -> None:
    config = EnhanceConfig(model_variant=ModelVariant.METRIC_LARGE, model_key="da3-metric")
    orch = EnhanceOrchestrator(config, tmp_path)
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_a = input_dir / "image_01.png"
    input_b = input_dir / "image_02.png"
    input_a.write_bytes(b"first")
    input_b.write_bytes(b"second")

    with patch(
        "transformation_portal.lux_depth_v3.orchestrator.compute_file_sha256",
        side_effect=AssertionError("rehash"),
    ):
        records = orch._build_run_card_inputs(
            [
                {
                    "image": str(input_a),
                    "input_sha256": "A" * 64,
                },
                {
                    "image": str(input_b),
                    "input_sha256": "B" * 64,
                },
            ]
        )

    assert records == [
        {
            "path": "image_01.png",
            "sha256": "a" * 64,
            "size_bytes": len(b"first"),
        },
        {
            "path": "image_02.png",
            "sha256": "b" * 64,
            "size_bytes": len(b"second"),
        },
    ]


def test_build_run_card_inputs_omits_size_bytes_when_stat_fails(tmp_path: Path) -> None:
    config = EnhanceConfig(model_variant=ModelVariant.METRIC_LARGE, model_key="da3-metric")
    orch = EnhanceOrchestrator(config, tmp_path)
    input_path = tmp_path / "inputs" / "image_01.png"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"pixels")

    original_stat = type(input_path).stat
    stat_calls = 0

    def flaky_stat(path_obj: Path, *args: Any, **kwargs: Any):
        nonlocal stat_calls
        if path_obj == input_path:
            stat_calls += 1
            if stat_calls >= 3:
                raise OSError("stat unavailable")
        return original_stat(path_obj, *args, **kwargs)

    with patch.object(type(input_path), "stat", autospec=True, side_effect=flaky_stat):
        records = orch._build_run_card_inputs(
            [
                {
                    "image": str(input_path),
                    "input_sha256": "A" * 64,
                }
            ]
        )

    assert records == [
        {
            "path": "image_01.png",
            "sha256": "a" * 64,
        }
    ]
