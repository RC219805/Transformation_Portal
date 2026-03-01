"""Tests for scripts/verify_run_card_integrity.py."""

from __future__ import annotations

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
    return {
        "batch_id": "2026-02-28_120000",
        "start_time": "2026-02-28T12:00:00Z",
        "end_time": "2026-02-28T12:05:00Z",
        "config_fingerprint": {
            "model_variant": "METRIC_LARGE",
            "depth_quantization": "u16",
            "depth_device": "cpu",
            "preset": "premium",
            "v2_preset": "premium",
            "v2_device": "cpu",
            "v2_upscaler_backend": "realesrgan",
            "sha256": "c" * 64,
        },
        "backend_selection": {
            "requested": "da3",
            "resolved": "da3",
            "device": "cpu",
            "model_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
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


pytestmark = [
    pytest.mark.unit,
    pytest.mark.regression,
]
