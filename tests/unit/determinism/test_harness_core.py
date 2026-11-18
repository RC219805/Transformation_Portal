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
from transformation_portal.determinism.tensor import compute_artifact_id, write_tensor_bin, write_tensor_npy
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


@pytest.mark.parametrize("number", [0.5, 1e-5, 1e-7, 123456.789, 1e21, 1e22, -2.5e-9, 0.1 + 0.2])
def test_jcs_serializes_float_subclasses_like_plain_floats(number: float):
    """np.float64 is a float; its NumPy 2 repr must not leak into the canonical text."""
    canonical = dumps({"m": np.float64(number)})

    assert canonical == dumps({"m": number})
    assert json.loads(canonical) == {"m": number}
    assert sha256_hex_of_canonical_json({"m": np.float64(number)}) == sha256_hex_of_canonical_json({"m": number})


def test_jcs_rejects_non_finite_float_subclass():
    with pytest.raises(ValueError, match="Non-finite"):
        dumps({"x": np.float64("nan")})


@pytest.mark.parametrize("payload", [{1: "a"}, {"a": 1, 2: "b"}, {None: 1}, {("a",): 1}])
def test_jcs_rejects_non_string_keys_with_type_error(payload: dict):
    with pytest.raises(TypeError, match="string object keys"):
        dumps(payload)


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


_TENSOR_ROLE = "xyz_d50_linear_fp32"


def _store_artifact(cas_root: Path, tensor: np.ndarray, *, with_npy: bool = True) -> str:
    """Write a CAS artifact directory the way ``determinism run`` commits it."""
    artifact_id = compute_artifact_id(_TENSOR_ROLE, tensor)
    artifact_dir = cas_root / "sha256" / artifact_id.split(":", 1)[1]
    artifact_dir.mkdir(parents=True)
    write_tensor_bin(artifact_dir / "output_tensor.bin", tensor)
    if with_npy:
        write_tensor_npy(artifact_dir / "output_tensor.npy", tensor)
    meta = {"tensor_role": _TENSOR_ROLE, "dtype": "float32", "order": "C", "shape": list(tensor.shape)}
    (artifact_dir / "tensor_meta.json").write_text(json.dumps({**meta, "artifact_id": artifact_id}), encoding="utf-8")
    return artifact_id


@pytest.mark.parametrize("with_npy", [True, False], ids=["npy", "bin"])
def test_load_tensor_from_cas_returns_the_payload_named_by_the_id(tmp_path: Path, with_npy: bool):
    tensor = np.linspace(0.0, 1.0, num=24, dtype=np.float32).reshape((2, 4, 3))
    artifact_id = _store_artifact(tmp_path, tensor, with_npy=with_npy)

    loaded = _load_tensor_from_cas(tmp_path, artifact_id)

    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, tensor)


@pytest.mark.parametrize("with_npy", [True, False], ids=["npy", "bin"])
def test_load_tensor_from_cas_rejects_a_payload_that_does_not_hash_to_its_id(tmp_path: Path, with_npy: bool):
    tensor = np.zeros((2, 2, 3), dtype=np.float32)
    other = np.full((2, 2, 3), 0.75, dtype=np.float32)
    artifact_id = _store_artifact(tmp_path, tensor, with_npy=with_npy)
    artifact_dir = tmp_path / "sha256" / artifact_id.split(":", 1)[1]
    if with_npy:
        write_tensor_npy(artifact_dir / "output_tensor.npy", other)
    else:
        write_tensor_bin(artifact_dir / "output_tensor.bin", other)

    with pytest.raises(ValueError, match="does not match its artifact id"):
        _load_tensor_from_cas(tmp_path, artifact_id)


def test_load_tensor_from_cas_rejects_a_payload_stored_under_another_role(tmp_path: Path):
    tensor = np.zeros((2, 2, 3), dtype=np.float32)
    artifact_id = _store_artifact(tmp_path, tensor)
    meta_path = tmp_path / "sha256" / artifact_id.split(":", 1)[1] / "tensor_meta.json"
    meta_path.write_text(json.dumps({"tensor_role": "some_other_role", "shape": [2, 2, 3]}), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its artifact id"):
        _load_tensor_from_cas(tmp_path, artifact_id)


def test_load_tensor_from_cas_requires_the_recorded_tensor_role(tmp_path: Path):
    tensor = np.zeros((2, 2, 3), dtype=np.float32)
    artifact_id = _store_artifact(tmp_path, tensor)
    meta_path = tmp_path / "sha256" / artifact_id.split(":", 1)[1] / "tensor_meta.json"

    meta_path.write_text(json.dumps({"shape": [2, 2, 3]}), encoding="utf-8")
    with pytest.raises(KeyError, match="Missing 'tensor_role' in tensor metadata"):
        _load_tensor_from_cas(tmp_path, artifact_id)

    meta_path.unlink()
    with pytest.raises(FileNotFoundError, match="No tensor metadata found"):
        _load_tensor_from_cas(tmp_path, artifact_id)


def test_verify_command_fails_when_the_candidate_payload_was_replaced(tmp_path: Path):
    """A candidate whose payload was swapped for the baseline's must not verify as itself."""
    from typer.testing import CliRunner

    baseline = np.zeros((2, 2, 3), dtype=np.float32)
    candidate = np.full((2, 2, 3), 0.75, dtype=np.float32)
    baseline_id = _store_artifact(tmp_path, baseline)
    candidate_id = _store_artifact(tmp_path, candidate)
    arguments = ["verify", "--baseline", baseline_id, "--artifact", candidate_id, "--cas-root", str(tmp_path)]
    arguments += ["--policy", "policy/adr030_v1.json"]
    runner = CliRunner()

    honest = runner.invoke(determinism_cli.app, arguments)
    assert honest.exit_code == 2
    assert json.loads(honest.output)["status"] == "fail"

    write_tensor_npy(tmp_path / "sha256" / candidate_id.split(":", 1)[1] / "output_tensor.npy", baseline)
    tampered = runner.invoke(determinism_cli.app, arguments)

    assert tampered.exit_code == 1
    assert isinstance(tampered.exception, ValueError)
    assert "does not match its artifact id" in str(tampered.exception)
    assert '"status"' not in tampered.output


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
