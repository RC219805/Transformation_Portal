"""Tests for scripts/verify_run_card_integrity.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("jsonschema")


def _load_script_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_run_card_payload(module) -> dict:
    artifact_index = [
        {
            "artifact_type": "depth_u16_png",
            "path": "depth/image_01_depth.png",
            "relative_path": "depth/image_01_depth.png",
            "size_bytes": 1024,
            "sha256": "a" * 64,
        },
        {
            "artifact_type": "combined_manifest",
            "path": "manifests/image_01_combined.json",
            "relative_path": "manifests/image_01_combined.json",
            "size_bytes": 2048,
            "sha256": "b" * 64,
        },
    ]
    canonical_json = (
        '{"apex_strict_mode":false,"backend_requested":"da3","backend_resolved":"da3",'
        '"depth_device":"cpu","depth_quantization":"u16","device_requested":"cpu","device_resolved":"cpu",'
        '"model_variant":"METRIC_LARGE","preset":"premium","preset_requested":"premium","preset_resolved":"premium",'
        '"quality_tier":"premium","strict_inputs":false,"strict_segmentation":false,'
        '"v2_device":"cpu","v2_preset":"premium","v2_upscaler_backend":"realesrgan"}'
    )
    fingerprint_sha = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return {
        "batch_id": "2026-02-28_120000",
        "start_time": "2026-02-28T12:00:00Z",
        "end_time": "2026-02-28T12:05:00Z",
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
            "strict_inputs": False,
            "strict_segmentation": False,
            "apex_strict_mode": False,
            "v2_preset": "premium",
            "v2_device": "cpu",
            "v2_upscaler_backend": "realesrgan",
            "hash_algorithm": "sha256",
            "canonical_json": canonical_json,
            "sha256": fingerprint_sha,
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
            "v2": "d" * 40,
            "v3": "d" * 40,
        },
        "runtime_stats": {
            "count": 1,
            "total": 1.0,
            "mean": 1.0,
            "min": 1.0,
            "max": 1.0,
            "median": 1.0,
        },
        "outliers": [],
        "total_images": 1,
        "success_count": 1,
        "error_count": 0,
        "artifact_index": artifact_index,
        "artifact_merkle_root": module.compute_artifact_merkle_root(artifact_index),
    }


def _write_json(path: Path, payload: dict, *, canonical: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if canonical:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def _artifact_entry(*, output_root: Path, file_path: Path, artifact_type: str) -> dict:
    relative = file_path.resolve().relative_to(output_root.resolve()).as_posix()
    data = file_path.read_bytes()
    return {
        "artifact_type": artifact_type,
        "path": relative,
        "relative_path": relative,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def test_verify_run_card_integrity_accepts_valid_payload(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_valid.json"
    payload = _valid_run_card_payload(module)
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert errors == []


def test_verify_run_card_integrity_rejects_schema_violation(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_schema", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_invalid_schema.json"
    payload = _valid_run_card_payload(module)
    payload.pop("runtime_stats")
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("Schema validation failed" in error for error in errors)


def test_verify_run_card_integrity_rejects_missing_sha256(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_sha", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_missing_sha.json"
    payload = _valid_run_card_payload(module)
    payload["artifact_index"][0].pop("sha256")
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any(".sha256 must be a lowercase 64-char hex digest" in error for error in errors)


def test_verify_run_card_integrity_rejects_merkle_mismatch(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_merkle", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_bad_merkle.json"
    payload = _valid_run_card_payload(module)
    payload["artifact_merkle_root"] = "f" * 64
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("artifact_merkle_root mismatch" in error for error in errors)


def test_verify_run_card_integrity_rejects_non_deterministic_ordering(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_order", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_unsorted.json"
    payload = _valid_run_card_payload(module)
    payload["artifact_index"] = list(reversed(payload["artifact_index"]))
    payload["artifact_merkle_root"] = module.compute_artifact_merkle_root(payload["artifact_index"])
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("ordering is non-deterministic" in error for error in errors)


def test_verify_run_card_integrity_detects_canonical_json_drift(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_canonical", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_non_canonical.json"
    payload = _valid_run_card_payload(module)
    _write_json(run_card_path, payload, canonical=False)

    errors = module.verify_run_card_integrity(run_card_path, check_canonical_json=True)
    assert any("canonical serialization drift" in error for error in errors)


def test_verify_run_card_integrity_rejects_backend_semantic_mismatch(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_backend", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_backend_mismatch.json"
    payload = _valid_run_card_payload(module)
    payload["backend_selection"]["resolved"] = "depth_pro"
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("backend_selection.resolved must match backend_summary.final_backends_used[0]" in error for error in errors)


def test_verify_run_card_integrity_accepts_wrapper_semantics(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_wrapper", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_wrapper.json"
    payload = _valid_run_card_payload(module)
    payload["backend_selection"]["logical_backend"] = "depth_pro"
    payload["backend_selection"]["resolved_engine"] = "da3"
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert errors == []


def test_verify_run_card_integrity_rejects_config_fingerprint_hash_mismatch(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_fingerprint", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_fingerprint_mismatch.json"
    payload = _valid_run_card_payload(module)
    payload["config_fingerprint"]["sha256"] = "f" * 64
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("config_fingerprint.sha256 mismatch" in error for error in errors)


def test_verify_run_card_integrity_reports_invalid_json(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_invalid_json", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_invalid.json"
    run_card_path.write_text("{", encoding="utf-8")

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("Invalid JSON" in error for error in errors)


def test_verify_run_card_integrity_validates_reconstruction_scene_manifest(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_scene_manifest", "scripts/verify_run_card_integrity.py")
    output_root = tmp_path / "output"
    run_card_path = output_root / "run_card_valid.json"
    reconstruction_dir = output_root / "reconstruction"
    segmentation_dir = output_root / "segmentation"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    segmentation_artifact = segmentation_dir / "scene_a_masks.npz"
    segmentation_artifact.write_bytes(b"segmentation")
    image_a = output_root / "input" / "scene_a" / "view_1.jpg"
    image_b = output_root / "input" / "scene_a" / "view_2.jpg"
    image_a.parent.mkdir(parents=True, exist_ok=True)
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")

    scene_manifest_path = reconstruction_dir / "scene_a_scene_manifest.json"
    scene_manifest_payload = {
        "schema": "tp.scene_manifest.v1",
        "scene_id": "scene_a",
        "images": [
            {
                "path": str(image_a.resolve()),
                "relative_path": "scene_a/view_1.jpg",
                "sha256": hashlib.sha256(b"a").hexdigest(),
            },
            {
                "path": str(image_b.resolve()),
                "relative_path": "scene_a/view_2.jpg",
                "sha256": hashlib.sha256(b"b").hexdigest(),
            },
        ],
        "cameras": [
            {"signature": "a" * 12, "source": "sidecar", "confidence": "high", "file": None},
            {"signature": "b" * 12, "source": "sidecar", "confidence": "high", "file": None},
        ],
        "segmentation_artifacts": [
            {
                "path": str(segmentation_artifact.resolve()),
                "relative_path": "segmentation/scene_a_masks.npz",
                "sha256": hashlib.sha256(b"segmentation").hexdigest(),
            }
        ],
        "inputs": ["segmentation/scene_a_masks.npz"],
        "input_hashes": {"segmentation/scene_a_masks.npz": hashlib.sha256(b"segmentation").hexdigest()},
    }
    scene_manifest_path.write_text(json.dumps(scene_manifest_payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    payload = _valid_run_card_payload(module)
    artifact_index = sorted(
        [
            _artifact_entry(output_root=output_root, file_path=segmentation_artifact, artifact_type="segmentation_mask_npz"),
            _artifact_entry(
                output_root=output_root, file_path=scene_manifest_path, artifact_type="reconstruction_scene_manifest"
            ),
        ],
        key=lambda entry: entry["relative_path"],
    )
    payload["artifact_index"] = artifact_index
    payload["artifact_merkle_root"] = module.compute_artifact_merkle_root(artifact_index)
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert errors == []


def test_verify_run_card_integrity_rejects_reconstruction_scene_manifest_hash_drift(tmp_path: Path):
    module = _load_script_module(
        "verify_run_card_integrity_script_scene_manifest_drift", "scripts/verify_run_card_integrity.py"
    )
    output_root = tmp_path / "output"
    run_card_path = output_root / "run_card_invalid_scene_manifest.json"
    reconstruction_dir = output_root / "reconstruction"
    segmentation_dir = output_root / "segmentation"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    segmentation_artifact = segmentation_dir / "scene_a_masks.npz"
    segmentation_artifact.write_bytes(b"segmentation")
    image_a = output_root / "input" / "scene_a" / "view_1.jpg"
    image_b = output_root / "input" / "scene_a" / "view_2.jpg"
    image_a.parent.mkdir(parents=True, exist_ok=True)
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")

    scene_manifest_path = reconstruction_dir / "scene_a_scene_manifest.json"
    scene_manifest_payload = {
        "schema": "tp.scene_manifest.v1",
        "scene_id": "scene_a",
        "images": [
            {
                "path": str(image_a.resolve()),
                "relative_path": "scene_a/view_1.jpg",
                "sha256": hashlib.sha256(b"a").hexdigest(),
            },
            {
                "path": str(image_b.resolve()),
                "relative_path": "scene_a/view_2.jpg",
                "sha256": hashlib.sha256(b"b").hexdigest(),
            },
        ],
        "cameras": [
            {"signature": "a" * 12, "source": "sidecar", "confidence": "high", "file": None},
            {"signature": "b" * 12, "source": "sidecar", "confidence": "high", "file": None},
        ],
        "segmentation_artifacts": [
            {
                "path": str(segmentation_artifact.resolve()),
                "relative_path": "segmentation/scene_a_masks.npz",
                "sha256": "0" * 64,
            }
        ],
        "inputs": ["segmentation/scene_a_masks.npz"],
        "input_hashes": {"segmentation/scene_a_masks.npz": "0" * 64},
    }
    scene_manifest_path.write_text(json.dumps(scene_manifest_payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    payload = _valid_run_card_payload(module)
    artifact_index = sorted(
        [
            _artifact_entry(output_root=output_root, file_path=segmentation_artifact, artifact_type="segmentation_mask_npz"),
            _artifact_entry(
                output_root=output_root, file_path=scene_manifest_path, artifact_type="reconstruction_scene_manifest"
            ),
        ],
        key=lambda entry: entry["relative_path"],
    )
    payload["artifact_index"] = artifact_index
    payload["artifact_merkle_root"] = module.compute_artifact_merkle_root(artifact_index)
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("Reconstruction scene manifest validation failed" in error for error in errors)


pytestmark = [
    pytest.mark.unit,
    pytest.mark.regression,
]
