"""Tests for scripts/verify_run_card_integrity.py."""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from transformation_portal.ingest.canonical_json import canonicalize_json, dumps_json
from transformation_portal.lux_depth_v3.artifact_manager import compute_artifact_merkle_root
from transformation_portal.lux_depth_v3.artifact_tree import build_artifact_tree

pytest.importorskip("jsonschema")

_DEFAULT_ARTIFACT_BYTES = {
    "depth/image_01_depth.png": b"depth-preview",
    "manifests/image_01_combined.json": b'{"backend_selection":{}}',
}


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
            "size_bytes": len(_DEFAULT_ARTIFACT_BYTES["depth/image_01_depth.png"]),
            "sha256": hashlib.sha256(_DEFAULT_ARTIFACT_BYTES["depth/image_01_depth.png"]).hexdigest(),
        },
        {
            "artifact_type": "combined_manifest",
            "path": "manifests/image_01_combined.json",
            "relative_path": "manifests/image_01_combined.json",
            "size_bytes": len(_DEFAULT_ARTIFACT_BYTES["manifests/image_01_combined.json"]),
            "sha256": hashlib.sha256(_DEFAULT_ARTIFACT_BYTES["manifests/image_01_combined.json"]).hexdigest(),
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
    fingerprint_sha = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return {
        "batch_id": "2026-02-28_120000",
        "start_time": "2026-02-28T12:00:00Z",
        "end_time": "2026-02-28T12:05:00Z",
        "config_fingerprint": {
            **config_fingerprint,
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
        "artifact_merkle_root": (
            module.compute_artifact_merkle_root(artifact_index)
            if module is not None
            else compute_artifact_merkle_root(artifact_index)
        ),
    }


def _write_json(path: Path, payload: dict, *, canonical: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for artifact in payload.get("artifact_index", []):
        if not isinstance(artifact, dict):
            continue
        relative_path = artifact.get("relative_path")
        if not isinstance(relative_path, str):
            continue
        content = _DEFAULT_ARTIFACT_BYTES.get(relative_path)
        if content is None:
            continue
        artifact_path = path.parent / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if not artifact_path.exists():
            artifact_path.write_bytes(content)
    if canonical:
        path.write_text(
            dumps_json(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def _write_self_attested_run_card(path: Path, payload: dict) -> None:
    integrity = {
        "path": path.name,
        "self_indexing": "excluded_self_hash_cycle",
    }
    payload["run_card_integrity"] = {
        **integrity,
        "canonical_payload_sha256": hashlib.sha256(
            canonicalize_json(
                {
                    **payload,
                    "run_card_integrity": integrity,
                }
            )
        ).hexdigest(),
    }
    _write_json(path, payload)
    (path.with_suffix(".self.json")).write_text(
        json.dumps(
            {
                "run_card_path": path.name,
                "self_indexing": "excluded_self_hash_cycle",
                "final_run_card_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "hash_algorithm": "sha256",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


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


def _valid_run_card_v2_payload() -> dict:
    payload = _valid_run_card_payload(module=None)  # type: ignore[arg-type]
    payload.pop("artifact_merkle_root")
    payload["artifact_tree"] = build_artifact_tree(payload["artifact_index"], include_proofs=True)
    return payload


def test_verify_run_card_integrity_accepts_valid_payload(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_valid.json"
    payload = _valid_run_card_payload(module)
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert errors == []


def test_verify_run_card_integrity_rejects_artifact_hash_drift(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_artifact_drift", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_drift.json"
    payload = _valid_run_card_payload(module)
    _write_json(run_card_path, payload)
    (tmp_path / "depth" / "image_01_depth.png").write_bytes(b"mutated")

    errors = module.verify_run_card_integrity(run_card_path)

    assert any("artifact_index[0].size_bytes mismatch" in error for error in errors)
    assert any("artifact_index[0].sha256 mismatch" in error for error in errors)


def test_verify_run_card_integrity_reports_artifact_hash_read_failure(tmp_path: Path, monkeypatch):
    module = _load_script_module(
        "verify_run_card_integrity_script_artifact_hash_failure", "scripts/verify_run_card_integrity.py"
    )
    run_card_path = tmp_path / "run_card_hash_failure.json"
    payload = _valid_run_card_payload(module)
    _write_json(run_card_path, payload)

    def fail_hash(path: Path):
        return None, f"simulated hash failure for {path}"

    monkeypatch.setitem(module.verify_run_card_integrity.__globals__, "_compute_file_sha256", fail_hash)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("artifact_index[0] file hash failed" in error for error in errors)


def test_verify_run_card_integrity_rejects_non_regular_artifact(tmp_path: Path):
    module = _load_script_module(
        "verify_run_card_integrity_script_non_regular_artifact", "scripts/verify_run_card_integrity.py"
    )
    run_card_path = tmp_path / "run_card_non_regular.json"
    payload = _valid_run_card_payload(module)
    relative_path = payload["artifact_index"][0]["relative_path"]
    artifact_path = run_card_path.parent / relative_path
    artifact_path.parent.mkdir(parents=True)
    artifact_path.mkdir()
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("artifact_index[0] is not a regular file" in error for error in errors)


def test_verify_run_card_integrity_accepts_self_attestation_sidecar(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_self", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_self.json"
    payload = _valid_run_card_payload(module)
    _write_self_attested_run_card(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)

    assert errors == []


def test_verify_run_card_integrity_rejects_self_attestation_hash_drift(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_self_drift", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_self_drift.json"
    payload = _valid_run_card_payload(module)
    _write_self_attested_run_card(run_card_path, payload)
    sidecar_path = run_card_path.with_suffix(".self.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["final_run_card_sha256"] = "f" * 64
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")

    errors = module.verify_run_card_integrity(run_card_path)

    assert any("final_run_card_sha256 mismatch" in error for error in errors)


def test_verify_run_card_integrity_rejects_self_attestation_metadata_mismatch(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_self_metadata", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_self_metadata.json"
    payload = _valid_run_card_payload(module)
    _write_self_attested_run_card(run_card_path, payload)
    sidecar_path = run_card_path.with_suffix(".self.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["run_card_path"] = "other_run_card.json"
    sidecar["self_indexing"] = "included_in_tree"
    sidecar["hash_algorithm"] = "sha512"
    sidecar_path.write_text(
        dumps_json(sidecar, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    errors = module.verify_run_card_integrity(run_card_path)

    assert any("sidecar run_card_path mismatch" in error for error in errors)
    assert any("sidecar self_indexing" in error for error in errors)
    assert any("sidecar hash_algorithm" in error for error in errors)


def test_verify_run_card_integrity_rejects_schema_violation(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_schema", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_invalid_schema.json"
    payload = _valid_run_card_payload(module)
    payload.pop("runtime_stats")
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("Schema validation failed" in error for error in errors)


def test_verify_run_card_integrity_rejects_non_json_numbers(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_non_finite", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_non_finite.json"
    payload = _valid_run_card_payload(module)
    payload["runtime_stats"]["mean"] = float("nan")
    run_card_path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

    errors = module.verify_run_card_integrity(run_card_path)

    assert len(errors) == 1
    assert "forbidden non-finite number NaN" in errors[0]


def test_verify_run_card_integrity_rejects_duplicate_members(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_duplicate_member", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_duplicate_member.json"
    payload = json.dumps(_valid_run_card_payload(module))
    run_card_path.write_text('{"batch_id":"forged",' + payload[1:], encoding="utf-8")

    errors = module.verify_run_card_integrity(run_card_path)

    assert len(errors) == 1
    assert "duplicate JSON member 'batch_id'" in errors[0]


def test_verify_run_card_integrity_bounds_schema_error_rendering(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_bounded_schema", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_oversized_value.json"
    payload = _valid_run_card_payload(module)
    payload["runtime_stats"]["count"] = "x" * 1_000_000
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)

    schema_errors = [error for error in errors if "schema validation failed" in error.casefold()]
    assert schema_errors
    assert max(len(error) for error in schema_errors) <= 1_024


def test_verify_run_card_integrity_rejects_oversized_run_card_before_json_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module("verify_run_card_integrity_run_card_limit", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_oversized.json"
    run_card_path.write_bytes(b"{}" + b" " * 32)
    monkeypatch.setitem(
        module.verify_run_card_integrity.__globals__,
        "_MAX_RUN_CARD_BYTES",
        16,
    )

    errors = module.verify_run_card_integrity(run_card_path)

    assert errors == [f"JSON file exceeds the bounded byte limit of 16: {run_card_path}"]


@pytest.mark.parametrize(
    ("raw_payload", "expected_error"),
    [
        pytest.param('{"value":' + "1" * 5_000 + "}", "Invalid JSON value", id="integer-digit-limit"),
        pytest.param(
            "[" * 20_000 + "0" + "]" * 20_000,
            "nesting exceeds the decoder limit",
            id="nesting-limit",
        ),
    ],
)
def test_verify_run_card_integrity_rejects_decoder_resource_limits(
    tmp_path: Path,
    raw_payload: str,
    expected_error: str,
):
    module = _load_script_module("verify_run_card_integrity_decoder_limit", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_decoder_limit.json"
    run_card_path.write_text(raw_payload, encoding="utf-8")

    errors = module.verify_run_card_integrity(run_card_path)

    assert len(errors) == 1
    assert expected_error in errors[0]
    assert len(errors[0]) <= 1_024


def test_verify_run_card_integrity_rejects_oversized_collection_before_schema_walk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from transformation_portal.lux_depth_v3.validators import run_card_validator

    module = _load_script_module("verify_run_card_integrity_collection_limit", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_oversized_collection.json"
    payload = _valid_run_card_payload(module)
    payload["result_summary"] = [{}] * 4_097
    _write_json(run_card_path, payload)
    monkeypatch.setitem(
        module.verify_run_card_integrity.__globals__,
        "_load_validator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("schema walk must not run")),
    )

    errors = module.verify_run_card_integrity(run_card_path)

    assert errors == ["Run card validation failed: result_summary exceeds the bounded limit of 4096"]
    run_card_validator._load_validator.cache_clear()


def test_verify_run_card_integrity_bounds_every_nested_collection_before_schema_walk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from transformation_portal.lux_depth_v3.validators import run_card_validator

    module = _load_script_module("verify_run_card_integrity_nested_limit", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_oversized_outliers.json"
    payload = _valid_run_card_payload(module)
    payload["outliers"] = [{}] * 4_097
    _write_json(run_card_path, payload)
    monkeypatch.setitem(
        module.verify_run_card_integrity.__globals__,
        "_load_validator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("schema walk must not run")),
    )

    errors = module.verify_run_card_integrity(run_card_path)

    assert errors == ["Run card validation failed: outliers exceeds the bounded limit of 4096"]
    run_card_validator._load_validator.cache_clear()


def test_verify_run_card_integrity_caps_schema_error_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module("verify_run_card_integrity_error_limit", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_many_schema_errors.json"
    payload = _valid_run_card_payload(module)
    _write_json(run_card_path, payload)
    observed: list[int] = []

    class Error:
        def __init__(self, index: int):
            self.path = (index,)
            self.validator = "type"
            self.message = "injected schema failure"

    class Validator:
        @staticmethod
        def iter_errors(_payload):
            for index in range(1_000):
                observed.append(index)
                yield Error(index)

    monkeypatch.setitem(
        module.verify_run_card_integrity.__globals__,
        "_load_validator",
        lambda *_args, **_kwargs: Validator(),
    )

    errors = module.verify_run_card_integrity(run_card_path)

    assert len(observed) == 65
    assert "stopped after the bounded limit of 64 errors" in errors[64]


def test_verify_run_card_integrity_caps_total_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module("verify_run_card_integrity_total_error_limit", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_many_integrity_errors.json"
    payload = _valid_run_card_payload(module)
    payload["artifact_index"] = [None] * 1_000
    _write_json(run_card_path, payload)

    class Validator:
        @staticmethod
        def iter_errors(_payload):
            return iter(())

    monkeypatch.setitem(
        module.verify_run_card_integrity.__globals__,
        "_load_validator",
        lambda *_args, **_kwargs: Validator(),
    )

    errors = module.verify_run_card_integrity(run_card_path)

    assert len(errors) == 65
    assert errors[-1] == "Run card integrity validation stopped after the bounded limit of 64 errors"


def test_verify_run_card_integrity_reports_jsonschema_install_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Missing jsonschema is reported with the governed version/install hint."""
    from transformation_portal.lux_depth_v3.validators import run_card_validator

    module = _load_script_module("verify_run_card_integrity_missing_jsonschema", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_valid.json"
    payload = _valid_run_card_payload(module)
    _write_json(run_card_path, payload)

    original_import = builtins.__import__

    def block_jsonschema_import(name, global_vars=None, local_vars=None, fromlist=(), level=0):
        if name == "jsonschema" or name.startswith("jsonschema."):
            raise ImportError("simulated missing jsonschema")
        return original_import(name, global_vars, local_vars, fromlist, level)

    run_card_validator._load_validator.cache_clear()
    monkeypatch.setattr(builtins, "__import__", block_jsonschema_import)
    errors = module.verify_run_card_integrity(run_card_path)
    run_card_validator._load_validator.cache_clear()

    assert errors == [
        "jsonschema dependency is required for run card schema validation "
        "(jsonschema>=4.21.0,<5); install the core runtime with "
        "`make install-core` or install dependencies from requirements/base.in"
    ]


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


def test_verify_run_card_integrity_accepts_valid_v2_payload(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_v2", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_valid_v2.json"
    payload = _valid_run_card_v2_payload()
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert errors == []


def test_verify_run_card_integrity_rejects_artifact_tree_root_mismatch(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_v2_root", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_bad_v2_root.json"
    payload = _valid_run_card_v2_payload()
    payload["artifact_tree"]["root_sha256"] = "f" * 64
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("artifact_tree.root_sha256 mismatch" in error for error in errors)


def test_verify_run_card_integrity_detects_canonical_json_drift(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_canonical", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_non_canonical.json"
    payload = _valid_run_card_payload(module)
    _write_json(run_card_path, payload, canonical=False)

    errors = module.verify_run_card_integrity(run_card_path, check_canonical_json=True)
    assert any("canonical serialization drift" in error for error in errors)


def test_verify_run_card_integrity_collects_canonical_read_failure(tmp_path: Path, monkeypatch):
    module = _load_script_module("verify_run_card_integrity_script_canonical_read", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card.json"
    payload = _valid_run_card_payload(module)
    _write_json(run_card_path, payload)

    def fail_read_text(path: Path):
        return None, f"Failed to read text file {path}: simulated failure"

    monkeypatch.setitem(module.verify_run_card_integrity.__globals__, "_read_text", fail_read_text)

    errors = module.verify_run_card_integrity(run_card_path, check_canonical_json=True)
    assert any("simulated failure" in error for error in errors)


def test_verify_run_card_integrity_rejects_backend_semantic_mismatch(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_backend", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_backend_mismatch.json"
    payload = _valid_run_card_payload(module)
    payload["backend_selection"]["resolved"] = "depth_pro"
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("backend_selection.resolved must match backend_summary.final_backends_used[0]" in error for error in errors)


def test_verify_run_card_integrity_and_validator_share_final_backend_type_semantics(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_backend_type", "scripts/verify_run_card_integrity.py")
    from transformation_portal.lux_depth_v3.validators import validate_run_card_backend_semantics

    run_card_path = tmp_path / "run_card_backend_type.json"
    payload = _valid_run_card_payload(module)
    payload["backend_summary"]["final_backends_used"] = "da3"
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("backend_summary.final_backends_used must be an array" in error for error in errors)
    with pytest.raises(RuntimeError, match="backend_summary.final_backends_used must be an array"):
        validate_run_card_backend_semantics(payload)


def test_verify_run_card_integrity_accepts_wrapper_semantics(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_wrapper", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_wrapper.json"
    payload = _valid_run_card_payload(module)
    payload["backend_selection"]["logical_backend"] = "depth_pro"
    payload["backend_selection"]["resolved_engine"] = "da3"
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert errors == []


def test_verify_run_card_integrity_rejects_requested_depth_pro_full_fallback(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_depth_pro_fallback", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_depth_pro_fallback.json"
    payload = _valid_run_card_payload(module)
    payload["backend_selection"]["requested"] = "depth_pro"
    payload["backend_selection"]["resolved"] = "da3"
    payload["backend_summary"]["requested_backend"] = "depth_pro"
    payload["backend_summary"]["primary_backend"] = "da3"
    payload["backend_summary"]["final_backends_used"] = ["da3"]
    payload["backend_summary"]["fallback_images"] = 2
    payload["total_images"] = 2
    payload["success_count"] = 2
    payload["error_count"] = 0
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("requested backend 'depth_pro' was not honored" in error for error in errors)


def test_verify_run_card_integrity_accepts_explicit_requested_backend_defect(tmp_path: Path):
    module = _load_script_module(
        "verify_run_card_integrity_script_depth_pro_fallback_audited",
        "scripts/verify_run_card_integrity.py",
    )
    run_card_path = tmp_path / "run_card_depth_pro_fallback_audited.json"
    payload = _valid_run_card_payload(module)
    payload["backend_selection"]["requested"] = "depth_pro"
    payload["backend_selection"]["resolved"] = "da3"
    payload["backend_summary"]["requested_backend"] = "depth_pro"
    payload["backend_summary"]["primary_backend"] = "da3"
    payload["backend_summary"]["final_backends_used"] = ["da3"]
    payload["backend_summary"]["fallback_images"] = 2
    payload["backend_summary"]["requested_backend_status"] = "not_honored"
    payload["backend_summary"]["requested_backend_defect"] = "Depth Pro MPS runtime unavailable."
    payload["total_images"] = 2
    payload["success_count"] = 2
    payload["error_count"] = 0
    _write_json(run_card_path, payload)

    assert module.verify_run_card_integrity(run_card_path) == []


def test_verify_run_card_integrity_and_validator_share_depth_pro_fallback_semantics(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_depth_pro_shared", "scripts/verify_run_card_integrity.py")
    from transformation_portal.lux_depth_v3.validators import validate_run_card_backend_semantics

    run_card_path = tmp_path / "run_card_depth_pro_fallback.json"
    payload = _valid_run_card_payload(module)
    payload["backend_selection"]["requested"] = "depth_pro"
    payload["backend_selection"]["resolved"] = "da3"
    payload["backend_summary"]["requested_backend"] = "depth_pro"
    payload["backend_summary"]["primary_backend"] = "da3"
    payload["backend_summary"]["final_backends_used"] = ["da3"]
    payload["backend_summary"]["fallback_images"] = 1
    payload["total_images"] = 1
    payload["success_count"] = 1
    payload["error_count"] = 0
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("requested backend 'depth_pro' was not honored" in error for error in errors)
    with pytest.raises(RuntimeError, match="requested backend 'depth_pro' was not honored"):
        validate_run_card_backend_semantics(payload)


def test_verify_run_card_integrity_handles_non_integer_success_count_without_crashing(tmp_path: Path):
    module = _load_script_module(
        "verify_run_card_integrity_script_depth_pro_type_guard", "scripts/verify_run_card_integrity.py"
    )
    run_card_path = tmp_path / "run_card_depth_pro_invalid_success_count.json"
    payload = _valid_run_card_payload(module)
    payload["backend_selection"]["requested"] = "depth_pro"
    payload["backend_selection"]["resolved"] = "da3"
    payload["backend_summary"]["requested_backend"] = "depth_pro"
    payload["backend_summary"]["primary_backend"] = "da3"
    payload["backend_summary"]["final_backends_used"] = ["da3"]
    payload["backend_summary"]["fallback_images"] = 1
    payload["total_images"] = 1
    payload["success_count"] = "1"
    payload["error_count"] = 0
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)

    assert errors
    assert any("success_count" in error for error in errors)


def test_verify_run_card_integrity_reports_malformed_override_schema(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_bad_schema", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card.json"
    bad_schema_path = tmp_path / "bad_schema.json"
    payload = _valid_run_card_payload(module)
    _write_json(run_card_path, payload)
    bad_schema_path.write_text(json.dumps({"type": 123}), encoding="utf-8")

    errors = module.verify_run_card_integrity(run_card_path, schema_path=bad_schema_path)
    assert any("Run card schema is invalid" in error for error in errors)


def test_verify_run_card_integrity_rejects_combined_manifest_path_escape(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_combined_escape", "scripts/verify_run_card_integrity.py")
    output_root = tmp_path / "output"
    run_card_path = output_root / "run_card_depth_pro_fallback.json"
    outside_manifest = tmp_path / "outside_manifest.json"
    outside_manifest.write_text(
        json.dumps({"backend_selection": {"resolution_reason": "secret outside reason"}}),
        encoding="utf-8",
    )

    payload = _valid_run_card_payload(module)
    payload["backend_selection"]["requested"] = "depth_pro"
    payload["backend_selection"]["resolved"] = "da3"
    payload["backend_summary"]["requested_backend"] = "depth_pro"
    payload["backend_summary"]["primary_backend"] = "da3"
    payload["backend_summary"]["final_backends_used"] = ["da3"]
    payload["backend_summary"]["fallback_images"] = 2
    payload["success_count"] = 2
    payload["artifact_index"][1]["path"] = "../outside_manifest.json"
    payload["artifact_index"][1]["relative_path"] = "../outside_manifest.json"
    payload["artifact_merkle_root"] = module.compute_artifact_merkle_root(payload["artifact_index"])
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any(
        "artifact_index[1].relative_path artifact relative_path must not contain traversal segments: ../outside_manifest.json"
        in error
        for error in errors
    )
    assert not any("secret outside reason" in error for error in errors)


def test_verify_run_card_integrity_rejects_config_fingerprint_hash_mismatch(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_fingerprint", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_fingerprint_mismatch.json"
    payload = _valid_run_card_payload(module)
    payload["config_fingerprint"]["sha256"] = "f" * 64
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("config_fingerprint.sha256 mismatch" in error for error in errors)


def test_verify_run_card_integrity_accepts_config_fingerprint_with_raw_ingest_fields(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_raw_ingest", "scripts/verify_run_card_integrity.py")
    run_card_path = tmp_path / "run_card_raw_ingest_fingerprint.json"
    payload = _valid_run_card_payload(module)
    config_fingerprint = payload["config_fingerprint"]
    config_fingerprint["raw_ingest_profile"] = "tp.raw_ingest.deterministic_v1"
    config_fingerprint["raw_ingest_settings_hash"] = "e" * 64

    fields = (
        "model_variant",
        "depth_quantization",
        "depth_device",
        "preset",
        "v2_preset",
        "v2_device",
        "v2_upscaler_backend",
        "depth_pro_python_executable",
        "raw_python_executable",
        "da3_python_executable",
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
        "raw_ingest_profile",
        "raw_ingest_settings_hash",
    )
    canonical_json = json.dumps(
        {field: config_fingerprint.get(field) for field in fields},
        sort_keys=True,
        separators=(",", ":"),
    )
    config_fingerprint["canonical_json"] = canonical_json
    config_fingerprint["sha256"] = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert errors == []


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


def test_verify_run_card_integrity_rejects_reconstruction_diagnostics_path_escape(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_diagnostics_escape", "scripts/verify_run_card_integrity.py")
    output_root = tmp_path / "output"
    run_card_path = output_root / "run_card_diagnostics_escape.json"
    outside_diagnostics = tmp_path / "outside_diagnostics.json"
    outside_diagnostics.write_text(
        json.dumps(
            {
                "schema": "tp.reconstruction_diagnostics.v1",
                "camera_count": 0,
                "cameras": [],
            }
        ),
        encoding="utf-8",
    )

    payload = _valid_run_card_payload(module)
    payload["artifact_index"].append(
        {
            "artifact_type": "reconstruction_diagnostics",
            "path": "../outside_diagnostics.json",
            "relative_path": "../outside_diagnostics.json",
            "size_bytes": len(outside_diagnostics.read_bytes()),
            "sha256": hashlib.sha256(outside_diagnostics.read_bytes()).hexdigest(),
        }
    )
    payload["artifact_merkle_root"] = module.compute_artifact_merkle_root(payload["artifact_index"])
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any(
        "artifact relative_path must not contain traversal segments: ../outside_diagnostics.json" in error for error in errors
    )


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


def test_verify_run_card_integrity_validates_reconstruction_diagnostics(tmp_path: Path):
    module = _load_script_module("verify_run_card_integrity_script_diagnostics", "scripts/verify_run_card_integrity.py")
    output_root = tmp_path / "output"
    run_card_path = output_root / "run_card_diagnostics.json"
    reconstruction_dir = output_root / "reconstruction"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_path = reconstruction_dir / "scene_a_reconstruction_diagnostics.json"
    diagnostics_payload = {
        "schema": "tp.reconstruction_diagnostics.v1",
        "scene_id": "scene_a",
        "scene_fingerprint": "f" * 64,
        "camera_count": 2,
        "total_points": 12,
        "global_rmse": 0.25,
        "cameras": [
            {
                "camera_id": "cam_00",
                "points_observed": 6,
                "reprojection_rmse": 0.25,
                "reprojection_max": 0.25,
                "reprojection_p50": 0.25,
                "reprojection_p95": 0.25,
                "reprojection_p99": 0.25,
            },
            {
                "camera_id": "cam_01",
                "points_observed": 6,
                "reprojection_rmse": 0.25,
                "reprojection_max": 0.25,
                "reprojection_p50": 0.25,
                "reprojection_p95": 0.25,
                "reprojection_p99": 0.25,
            },
        ],
    }
    diagnostics_path.write_text(json.dumps(diagnostics_payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    payload = _valid_run_card_payload(module)
    artifact_index = sorted(
        [
            _artifact_entry(output_root=output_root, file_path=diagnostics_path, artifact_type="reconstruction_diagnostics"),
        ],
        key=lambda entry: entry["relative_path"],
    )
    payload["artifact_index"] = artifact_index
    payload["artifact_merkle_root"] = module.compute_artifact_merkle_root(artifact_index)
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert errors == []


def test_verify_run_card_integrity_rejects_reconstruction_diagnostics_missing_percentiles(tmp_path: Path):
    module = _load_script_module(
        "verify_run_card_integrity_script_diagnostics_missing", "scripts/verify_run_card_integrity.py"
    )
    output_root = tmp_path / "output"
    run_card_path = output_root / "run_card_diagnostics_invalid.json"
    reconstruction_dir = output_root / "reconstruction"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_path = reconstruction_dir / "scene_a_reconstruction_diagnostics.json"
    diagnostics_payload = {
        "schema": "tp.reconstruction_diagnostics.v1",
        "scene_id": "scene_a",
        "scene_fingerprint": "f" * 64,
        "camera_count": 1,
        "total_points": 10,
        "global_rmse": 0.2,
        "cameras": [
            {
                "camera_id": "cam_00",
                "points_observed": 10,
                "reprojection_rmse": 0.2,
                "reprojection_max": 0.2,
                "reprojection_p50": 0.2,
            }
        ],
    }
    diagnostics_path.write_text(json.dumps(diagnostics_payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    payload = _valid_run_card_payload(module)
    artifact_index = sorted(
        [
            _artifact_entry(output_root=output_root, file_path=diagnostics_path, artifact_type="reconstruction_diagnostics"),
        ],
        key=lambda entry: entry["relative_path"],
    )
    payload["artifact_index"] = artifact_index
    payload["artifact_merkle_root"] = module.compute_artifact_merkle_root(artifact_index)
    _write_json(run_card_path, payload)

    errors = module.verify_run_card_integrity(run_card_path)
    assert any("missing reprojection_p95" in error or "missing reprojection_p99" in error for error in errors)


pytestmark = [
    pytest.mark.unit,
    pytest.mark.regression,
]
