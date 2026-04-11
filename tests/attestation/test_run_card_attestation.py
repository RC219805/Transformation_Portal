"""Tests for Lux run-card native and DSSE attestation helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from transformation_portal.attestation.dsse import decode_dsse_payload
from transformation_portal.attestation.run_card_detached import (
    bind_run_card_detached_attestation,
    build_run_card_detached_attestation_payload,
    canonical_run_card_attestation_preimage_bytes,
    compute_run_card_attestation_sha256,
    validate_run_card_detached_attestation_surface,
    verify_run_card_attestation_self_hash,
)
from transformation_portal.attestation.run_card_intoto import (
    build_run_card_dsse_envelope,
    build_run_card_statement,
    decode_run_card_statement_from_envelope,
    validate_run_card_statement_binding,
)
from transformation_portal.lux_depth_v3.artifact_tree import build_artifact_tree

pytestmark = pytest.mark.unit


def _run_card_v2() -> tuple[dict[str, object], bytes]:
    artifact_index = [
        {
            "artifact_type": "depth_u16_png",
            "path": "depth/image_01_depth.png",
            "relative_path": "depth/image_01_depth.png",
            "size_bytes": 1024,
            "sha256": "a" * 64,
        },
        {
            "artifact_type": "batch_manifest",
            "path": "manifests/batch_01.json",
            "relative_path": "manifests/batch_01.json",
            "size_bytes": 2048,
            "sha256": "b" * 64,
        },
    ]
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
    canonical_json = json.dumps(config_fingerprint, sort_keys=True, separators=(",", ":"))
    payload = {
        "batch_id": "2026-04-10_120000",
        "start_time": "2026-04-10T12:00:00Z",
        "end_time": "2026-04-10T12:05:00Z",
        "config_fingerprint": {
            **config_fingerprint,
            "hash_algorithm": "sha256",
            "canonical_json": canonical_json,
            "sha256": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
        },
        "backend_selection": {
            "requested": "da3",
            "resolved": "da3",
            "device": "cpu",
            "model_id": "depth-anything/DA3",
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
    }
    payload["artifact_tree"] = build_artifact_tree(artifact_index, include_proofs=True)
    run_card_bytes = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    return payload, run_card_bytes


def test_native_run_card_attestation_binds_to_run_card() -> None:
    run_card_payload, run_card_bytes = _run_card_v2()
    attestation = build_run_card_detached_attestation_payload(
        run_card_payload,
        run_card_bytes=run_card_bytes,
        signature={"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"},
    )

    validate_run_card_detached_attestation_surface(attestation)
    bind_run_card_detached_attestation(attestation, run_card_payload, run_card_bytes=run_card_bytes)
    verify_run_card_attestation_self_hash(attestation)
    assert attestation["attestation_sha256"] == compute_run_card_attestation_sha256(attestation)


def test_native_run_card_attestation_detects_tamper() -> None:
    run_card_payload, run_card_bytes = _run_card_v2()
    attestation = build_run_card_detached_attestation_payload(
        run_card_payload,
        run_card_bytes=run_card_bytes,
        signature={"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"},
    )
    attestation["claims"] = {"tampered": True}

    with pytest.raises(ValueError, match="attestation_sha256 mismatch"):
        verify_run_card_attestation_self_hash(attestation)


def test_dsse_statement_binds_to_run_card() -> None:
    run_card_payload, run_card_bytes = _run_card_v2()
    statement = build_run_card_statement(
        run_card_path=Path("run_card_2026-04-10_120000.json"),
        run_card_payload=run_card_payload,
        run_card_bytes=run_card_bytes,
    )
    envelope = build_run_card_dsse_envelope(
        run_card_path=Path("run_card_2026-04-10_120000.json"),
        run_card_payload=run_card_payload,
        run_card_bytes=run_card_bytes,
        key_id="test",
        signature_bytes=b"fake-signature",
    )

    decoded_statement = decode_run_card_statement_from_envelope(envelope)
    validate_run_card_statement_binding(
        decoded_statement,
        run_card_path=Path("run_card_2026-04-10_120000.json"),
        run_card_payload=run_card_payload,
        run_card_bytes=run_card_bytes,
    )
    assert decoded_statement == statement
    assert json.loads(decode_dsse_payload(envelope).decode("utf-8"))["_type"] == "https://in-toto.io/Statement/v1"


def test_run_card_attestation_preimage_is_deterministic() -> None:
    run_card_payload, run_card_bytes = _run_card_v2()
    reordered_payload = {
        "artifact_tree": run_card_payload["artifact_tree"],
        "batch_id": run_card_payload["batch_id"],
    }
    preimage_a = canonical_run_card_attestation_preimage_bytes(run_card_payload, run_card_bytes=run_card_bytes)
    preimage_b = canonical_run_card_attestation_preimage_bytes(
        {
            **run_card_payload,
            **reordered_payload,
        },
        run_card_bytes=run_card_bytes,
    )
    assert preimage_a == preimage_b
