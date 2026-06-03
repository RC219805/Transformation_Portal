"""Core unit coverage for ADR-030 determinism harness scaffold."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest

from transformation_portal.determinism import cli as determinism_cli
from transformation_portal.determinism.cas import (
    atomic_commit_dir,
    build_artifact_manifest,
    sha256_file,
    stage_dir,
    write_json,
)
from transformation_portal.determinism.cli import _load_tensor_from_cas, _parse_artifact_id
from transformation_portal.determinism.jcs import dumps, sha256_hex_of_canonical_json
from transformation_portal.determinism.policy import Adr030PolicyV1, load_policy
from transformation_portal.determinism.tensor import compute_artifact_id
from transformation_portal.determinism.trace import parse_traceparent
from transformation_portal.determinism.verify import compute_metrics, verify_against_policy

pytestmark = [pytest.mark.unit]


def _sample_policy(**overrides: object) -> Adr030PolicyV1:
    values = {
        "verification_policy_version": "adr030-v1",
        "float_model": "IEEE754_binary32_rn_even",
        "float32_eps_exp": -23,
        "pixel_parity_multiplier": 41,
        "mae_threshold": 5e-7,
        "rmse_threshold": 5e-7,
        "nan_policy": "fail_closed",
        "inf_policy": "fail_closed",
        "subnormal_policy": "preserve",
        "ftz_daz_policy": "fail_closed",
        "reduction_mode": "single_thread_float64_c_order",
        "matrix_backend": "explicit_f32_no_blas",
        "certified_tensor_role": "xyz_d50_linear_fp32",
        "policy_source": "policy/adr030_v1.json",
    }
    values.update(overrides)
    return Adr030PolicyV1(**values)


def test_jcs_canonical_output_and_number_normalization():
    payload = {"b": 1.0, "a": 1e-7, "c": -0.0}
    assert dumps(payload) == '{"a":1e-7,"b":1,"c":0}'


def test_jcs_rejects_non_finite_numbers():
    with pytest.raises(ValueError, match="Non-finite"):
        dumps({"x": float("inf")})


def test_compute_artifact_id_is_stable_across_dtype_and_layout():
    base = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    be_fortran = np.asarray(base, dtype=">f4", order="F")

    h1 = compute_artifact_id("xyz_d50_linear_fp32", base)
    h2 = compute_artifact_id("xyz_d50_linear_fp32", be_fortran)
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_compute_artifact_id_rejects_non_lowercase_role():
    tensor = np.zeros((1, 1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="ASCII lowercase"):
        compute_artifact_id("XYZ_D50_LINEAR_FP32", tensor)


def test_parse_artifact_id_normalizes_hex_case():
    upper = "sha256:" + ("AB" * 32)
    normalized = _parse_artifact_id(upper)
    assert normalized == ("ab" * 32)


def test_parse_artifact_id_error_includes_received_value():
    with pytest.raises(ValueError, match="got:"):
        _parse_artifact_id("sha256:not-valid-hex")


def test_load_tensor_from_cas_reports_corrupt_npy_with_context(tmp_path: Path):
    cas_root = tmp_path / "cas"
    tensor_hash = "a" * 64
    artifact_id = f"sha256:{tensor_hash}"
    artifact_dir = cas_root / "sha256" / tensor_hash
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "output_tensor.npy").write_bytes(b"not-a-valid-npy")

    with pytest.raises(RuntimeError, match=f"Failed to load tensor npy payload for artifact {artifact_id}"):
        _load_tensor_from_cas(cas_root, artifact_id)


def test_load_tensor_from_cas_reports_missing_shape_with_context(tmp_path: Path):
    cas_root = tmp_path / "cas"
    tensor_hash = "b" * 64
    artifact_id = f"sha256:{tensor_hash}"
    artifact_dir = cas_root / "sha256" / tensor_hash
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "tensor_meta.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "output_tensor.bin").write_bytes(b"")

    with pytest.raises(KeyError, match="Missing 'shape' in tensor metadata"):
        _load_tensor_from_cas(cas_root, artifact_id)


def test_camera_native_linear_missing_optional_deps_points_to_raw_runtime(monkeypatch, tmp_path: Path):
    input_path = tmp_path / "input.dng"
    input_path.write_bytes(b"raw")
    original_import = builtins.__import__
    target_module = "transformation_portal.spatial_ai.ingest.phase2_camera_native_linear"

    def fake_import(name, globals_=None, locals_=None, fromlist=(), level=0):  # noqa: ANN001
        if name == target_module:
            raise ModuleNotFoundError("No module named 'rawpy'")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as excinfo:
        determinism_cli.run(input_path=input_path, contract="camera_native_linear", json_out=False)

    message = str(excinfo.value)
    assert "./scripts/setup/install_raw_runtime.sh" in message
    assert "requirements/README.md" in message
    assert "make install-ml" not in message


def test_compute_metrics_known_values():
    reference = np.zeros((1, 1, 3), dtype=np.float32)
    candidate = np.array([[[1e-6, -2e-6, 3e-6]]], dtype=np.float32)
    max_abs, mae, rmse = compute_metrics(reference, candidate)

    assert max_abs == pytest.approx(3e-6)
    assert mae == pytest.approx(2e-6)
    assert rmse == pytest.approx(np.sqrt((1.0 + 4.0 + 9.0) / 3.0) * 1e-6)


def test_verify_against_policy_pass_and_fail_paths():
    policy = _sample_policy()
    reference = np.zeros((2, 2, 3), dtype=np.float32)
    candidate_pass = np.zeros((2, 2, 3), dtype=np.float32)
    candidate_fail = np.full((2, 2, 3), 1e-4, dtype=np.float32)

    result_pass = verify_against_policy(reference, candidate_pass, policy)
    result_fail = verify_against_policy(reference, candidate_fail, policy)

    assert result_pass.status == "pass"
    assert result_fail.status == "fail"
    assert policy.max_abs_diff_bound <= 5e-6


def test_parse_traceparent_round_trip_and_validation():
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    ctx = parse_traceparent(traceparent)
    assert ctx.traceparent == traceparent

    with pytest.raises(ValueError, match="Invalid trace_id"):
        parse_traceparent("00-00000000000000000000000000000000-00f067aa0ba902b7-01")


def test_load_policy_hash_and_missing_key_validation(tmp_path: Path):
    policy_path = Path("policy/adr030_v1.json")
    raw = json.loads(policy_path.read_text(encoding="utf-8"))
    policy, policy_hash = load_policy(policy_path)

    assert policy_hash == sha256_hex_of_canonical_json(raw)
    assert policy.pixel_parity_multiplier == 41
    assert policy.max_abs_diff_bound <= 5e-6

    bad = tmp_path / "bad_policy.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(KeyError, match="Policy missing required key"):
        load_policy(bad)


def test_cas_atomic_commit_and_manifest(tmp_path: Path):
    cas_root = tmp_path / "cas"
    staging = stage_dir(cas_root, "staging-")
    digest = write_json(staging / "meta.json", {"b": 2, "a": 1})
    assert digest == sha256_hex_of_canonical_json({"b": 2, "a": 1})
    assert digest == sha256_file(staging / "meta.json")

    final_dir = cas_root / "sha256" / "artifact_01"
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    atomic_commit_dir(staging, final_dir)

    manifest = build_artifact_manifest(final_dir)
    assert manifest["meta.json"] == sha256_file(final_dir / "meta.json")

    staging2 = stage_dir(cas_root, "staging-")
    (staging2 / "tmp.txt").write_text("tmp", encoding="utf-8")
    existing = cas_root / "sha256" / "artifact_existing"
    existing.mkdir(parents=True, exist_ok=True)
    (existing / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        atomic_commit_dir(staging2, existing)
