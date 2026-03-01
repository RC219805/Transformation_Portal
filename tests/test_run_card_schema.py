"""Run card schema and integrity helper tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from transformation_portal.lux_depth_v3.orchestrator import (
    _build_artifact_index,
    _compute_artifact_merkle_root,
    _run_card_schema_path,
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
            "v2_preset": "premium",
            "v2_device": "cpu",
            "v2_upscaler_backend": "realesrgan",
            "sha256": "a" * 64,
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


def test_run_card_schema_rejects_invalid_merkle_root():
    pytest.importorskip("jsonschema")
    payload = _valid_run_card_payload()
    payload["artifact_merkle_root"] = "not-a-digest"

    with pytest.raises(RuntimeError, match="artifact_merkle_root"):
        _validate_run_card_payload(payload, _run_card_schema_path())


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
