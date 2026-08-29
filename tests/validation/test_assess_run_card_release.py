"""Tests for run-card release assessment policy gates."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from transformation_portal.attestation.run_card_detached import (
    build_run_card_detached_attestation_payload,
    canonical_run_card_attestation_preimage_bytes,
    compute_run_card_attestation_sha256,
)
from transformation_portal.attestation.run_card_intoto import build_run_card_dsse_envelope, canonical_run_card_statement_bytes
from transformation_portal.lux_depth_v3.artifact_tree import build_artifact_tree
from transformation_portal.lux_depth_v3.validators.release_assessment import assess_run_card_release

pytestmark = pytest.mark.unit


def _write_run_card_v2(tmp_path: Path) -> Path:
    artifact_payloads = {
        "depth/image_01_depth.png": (b"depth-preview-fixture", "depth_u16_png"),
        "manifests/batch_01.json": (b'{"batch_id":"2026-04-10_120000"}\n', "batch_manifest"),
    }
    artifact_index = []
    for relative_path, (content, artifact_type) in artifact_payloads.items():
        artifact_path = tmp_path / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(content)
        artifact_index.append(
            {
                "artifact_type": artifact_type,
                "path": relative_path,
                "relative_path": relative_path,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    fingerprint_fields = {
        "model_variant": "METRIC_LARGE",
        "depth_quantization": "u16",
        "depth_device": "cpu",
        "preset": "premium",
        "v2_preset": "premium",
        "v2_device": "cpu",
        "v2_upscaler_backend": "realesrgan",
        "depth_pro_python_executable": None,
        "raw_python_executable": None,
        "da3_python_executable": None,
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
    }
    canonical_json = json.dumps(fingerprint_fields, sort_keys=True, separators=(",", ":"))
    payload = {
        "batch_id": "2026-04-10_120000",
        "start_time": "2026-04-10T12:00:00Z",
        "end_time": "2026-04-10T12:05:00Z",
        "config_fingerprint": {
            **fingerprint_fields,
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
        "artifact_tree": build_artifact_tree(artifact_index, include_proofs=True),
    }
    run_card_path = tmp_path / "run_card_2026-04-10_120000.json"
    run_card_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return run_card_path


def _write_attestations(run_card_path: Path) -> None:
    run_card_payload = json.loads(run_card_path.read_text(encoding="utf-8"))
    run_card_bytes = run_card_path.read_bytes()
    native_attestation = build_run_card_detached_attestation_payload(
        run_card_payload,
        run_card_bytes=run_card_bytes,
        signature={"algorithm": "openpgp-clearsign", "key_id": "test", "signature": "deadbeef"},
    )
    dsse_statement_bytes = canonical_run_card_statement_bytes(
        run_card_path=run_card_path,
        run_card_payload=run_card_payload,
        run_card_bytes=run_card_bytes,
    )
    dsse_attestation = build_run_card_dsse_envelope(
        run_card_path=run_card_path,
        run_card_payload=run_card_payload,
        run_card_bytes=run_card_bytes,
        key_id="test",
        signature_bytes=dsse_statement_bytes,
    )
    run_card_path.with_suffix(".attestation.native.json").write_text(json.dumps(native_attestation), encoding="utf-8")
    run_card_path.with_suffix(".attestation.dsse.json").write_text(json.dumps(dsse_attestation), encoding="utf-8")
    run_card_path.with_suffix(".attestation.dsse.sigstore.bundle.json").write_text(
        json.dumps({"verificationMaterial": {"tlogEntries": [{"logIndex": 1}]}}),
        encoding="utf-8",
    )


def test_release_assessment_fails_closed_when_required_attestations_are_missing(tmp_path: Path) -> None:
    run_card_path = _write_run_card_v2(tmp_path)

    assessment = assess_run_card_release(
        run_card_path=run_card_path,
        require_native_attestation=True,
        require_dsse_attestation=True,
        require_sigstore_bundle=True,
    )

    assert assessment["status"] == "FAIL"
    assert any(check["status"] == "FAIL" for check in assessment["checks"])


def test_release_assessment_passes_when_required_sidecars_exist(tmp_path: Path) -> None:
    run_card_path = _write_run_card_v2(tmp_path)
    _write_attestations(run_card_path)

    assessment = assess_run_card_release(
        run_card_path=run_card_path,
        require_native_attestation=True,
        require_dsse_attestation=True,
    )

    assert assessment["status"] == "PASS"


def test_release_assessment_requires_rekor_inclusion_when_requested(tmp_path: Path) -> None:
    run_card_path = _write_run_card_v2(tmp_path)
    _write_attestations(run_card_path)
    run_card_path.with_suffix(".attestation.dsse.sigstore.bundle.json").write_text("{}", encoding="utf-8")

    assessment = assess_run_card_release(
        run_card_path=run_card_path,
        require_dsse_attestation=True,
        require_sigstore_bundle=True,
        require_rekor_inclusion=True,
    )

    assert assessment["status"] == "FAIL"
    sigstore_check = next(check for check in assessment["checks"] if check["name"] == "sigstore_bundle")
    assert sigstore_check["status"] == "FAIL"


def test_release_assessment_can_allow_missing_native_attestation_sha(tmp_path: Path) -> None:
    run_card_path = _write_run_card_v2(tmp_path)
    _write_attestations(run_card_path)
    native_path = run_card_path.with_suffix(".attestation.native.json")
    native_attestation = json.loads(native_path.read_text(encoding="utf-8"))
    native_attestation["attestation_sha256"] = None
    native_path.write_text(json.dumps(native_attestation), encoding="utf-8")

    default_assessment = assess_run_card_release(
        run_card_path=run_card_path,
        require_native_attestation=True,
    )
    tolerant_assessment = assess_run_card_release(
        run_card_path=run_card_path,
        require_native_attestation=True,
        allow_missing_attestation_sha=True,
    )

    assert default_assessment["status"] == "FAIL"
    native_check = next(check for check in default_assessment["checks"] if check["name"] == "native_attestation")
    assert any("attestation_sha256" in error for error in native_check["details"]["errors"])
    assert tolerant_assessment["status"] == "PASS"


def test_release_assessment_gpg_verification_binds_native_preimage_and_recorded_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_card_path = _write_run_card_v2(tmp_path)
    _write_attestations(run_card_path)
    run_card_path.with_suffix(".attestation.dsse.json").unlink()
    run_card_path.with_suffix(".attestation.dsse.sigstore.bundle.json").unlink()
    captured: dict[str, object] = {}

    def _verify(signature: str, *, expected_payload: bytes, key_id: str) -> None:
        captured.update(signature=signature, expected_payload=expected_payload, key_id=key_id)

    monkeypatch.setattr(
        "transformation_portal.lux_depth_v3.validators.release_assessment.gpg_verify_clearsign",
        _verify,
    )

    assessment = assess_run_card_release(
        run_card_path=run_card_path,
        require_native_attestation=True,
        verify_gpg=True,
    )

    run_card_payload = json.loads(run_card_path.read_text(encoding="utf-8"))
    assert assessment["status"] == "PASS"
    assert captured == {
        "signature": "deadbeef",
        "expected_payload": canonical_run_card_attestation_preimage_bytes(
            run_card_payload,
            run_card_bytes=run_card_path.read_bytes(),
        ),
        "key_id": "test",
    }


def test_release_assessment_rejects_non_gpg_native_algorithm_when_verification_requested(tmp_path: Path) -> None:
    run_card_path = _write_run_card_v2(tmp_path)
    _write_attestations(run_card_path)
    run_card_path.with_suffix(".attestation.dsse.json").unlink()
    run_card_path.with_suffix(".attestation.dsse.sigstore.bundle.json").unlink()
    native_path = run_card_path.with_suffix(".attestation.native.json")
    native_attestation = json.loads(native_path.read_text(encoding="utf-8"))
    native_attestation["signature"]["algorithm"] = "logical-label"
    native_attestation["attestation_sha256"] = compute_run_card_attestation_sha256(native_attestation)
    native_path.write_text(json.dumps(native_attestation), encoding="utf-8")

    assessment = assess_run_card_release(
        run_card_path=run_card_path,
        require_native_attestation=True,
        verify_gpg=True,
    )

    native_check = next(check for check in assessment["checks"] if check["name"] == "native_attestation")
    assert assessment["status"] == "FAIL"
    assert any("openpgp-clearsign" in error for error in native_check["details"]["errors"])


def test_assess_run_card_release_script_runs_from_source_checkout(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    script_path = repo_root / "scripts" / "validation" / "assess_run_card_release.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        (Path(__file__).resolve().parents[2] / "scripts" / "validation" / "assess_run_card_release.py").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    for package_dir in (
        repo_root / "src" / "transformation_portal",
        repo_root / "src" / "transformation_portal" / "lux_depth_v3",
        repo_root / "src" / "transformation_portal" / "lux_depth_v3" / "validators",
    ):
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
    validator_path = repo_root / "src" / "transformation_portal" / "lux_depth_v3" / "validators" / "release_assessment.py"
    validator_path.write_text(
        (
            "def assess_run_card_release(*, run_card_path, **_kwargs):\n"
            '    return {"status": "PASS", "run_card": str(run_card_path)}\n'
        ),
        encoding="utf-8",
    )
    run_card_path = repo_root / "run_card.json"
    run_card_path.write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script_path), str(run_card_path)],
        cwd=repo_root,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"run_card": str(run_card_path), "status": "PASS"}
