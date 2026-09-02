"""Canonical execution-plan carrier contracts for depth workers/backends."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from transformation_portal.depth.backends import da3_worker, depth_pro_worker
from transformation_portal.depth.backends.da3 import DA3Backend, _load_verified_worker_depth
from transformation_portal.depth.backends.depth_pro import DepthProBackend
from transformation_portal.depth.backends.registry import DepthBackendRegistry
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
from transformation_portal.lux_depth_v3.execution_lifecycle import (
    backend_candidate_authority,
    consume_lux_worker_execution_plan,
    prepare_lux_execution,
    runtime_config_from_execution_plan,
)
from transformation_portal.lux_depth_v3.execution_plan_adapter import LuxExecutionPlanAuthorityError
from transformation_portal.lux_depth_v3.pipeline_coordinator import select_backend

pytestmark = pytest.mark.unit


def _write_bound_runtime_token(
    path: Path,
    *,
    runtime_identity_sha256: str,
    device: str,
    executed_backend: str,
) -> tuple[Path, str]:
    import transformation_portal.depth.backends.da3_runtime_identity as identity_module

    observed_path = path.with_name("runtime-token-input")
    observed_path.write_bytes(b"stable")
    observed = observed_path.stat()
    import_environment = identity_module._worker_import_environment_payload()
    payload = {
        "schema": "tp.da3.runtime-verification-token.v1",
        "worker_runtime_identity_sha256": "b" * 64,
        "worker_import_environment_sha256": identity_module._sha256_payload(import_environment),
        "worker_import_environment": import_environment,
        "prepared_runtime": {
            "schema": "tp.da3.prepared-runtime-binding.v1",
            "runtime_identity_sha256": runtime_identity_sha256,
            "requested_device": device,
            "actual_device": device,
            "executed_backend": executed_backend,
        },
        "source_revision_probe": None,
        "entries": [
            {
                "path": str(observed_path.resolve()),
                "kind": "file",
                "device": observed.st_dev,
                "inode": observed.st_ino,
                "size_bytes": observed.st_size,
                "mtime_ns": observed.st_mtime_ns,
                "ctime_ns": observed.st_ctime_ns,
            }
        ],
    }
    path.write_bytes(canonicalize_json(payload))
    return path, identity_module.runtime_verification_token_sha256(payload)


@pytest.fixture()
def prepared_da3(tmp_path):
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(b"fixture")
    return prepare_lux_execution(
        EnhanceConfig(depth_backend="da3"),
        tmp_path,
        [input_path],
    )


@pytest.fixture()
def prepared_fallback_chain(tmp_path):
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(b"fixture")
    return prepare_lux_execution(EnhanceConfig(), tmp_path, [input_path])


@pytest.fixture()
def prepared_depth_pro(tmp_path):
    input_path = tmp_path / "input.jpg"
    checkpoint = tmp_path / "depth_pro.pt"
    input_path.write_bytes(b"fixture")
    checkpoint.write_bytes(b"checkpoint-fixture")
    return prepare_lux_execution(
        EnhanceConfig(
            depth_backend="depth_pro",
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
            depth_pro_checkpoint_path=str(checkpoint),
        ),
        tmp_path,
        [input_path],
    )


@pytest.fixture()
def prepared_ensemble(tmp_path):
    input_path = tmp_path / "input.jpg"
    checkpoint = tmp_path / "depth_pro.pt"
    input_path.write_bytes(b"fixture")
    checkpoint.write_bytes(b"checkpoint-fixture")
    return prepare_lux_execution(
        EnhanceConfig(
            depth_backend="ensemble",
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
            accept_research_tools_license=True,
            depth_pro_checkpoint_path=str(checkpoint),
        ),
        tmp_path,
        [input_path],
    )


@pytest.mark.parametrize("worker", [da3_worker, depth_pro_worker])
def test_canonical_worker_mode_accepts_only_plan_candidate_selectors(worker) -> None:
    parser = worker._build_parser()
    args = parser.parse_args(
        [
            "--execution-plan-stdin",
            "--candidate-id",
            "ensemble",
            "--model-backend-id",
            "da3",
            "--check",
        ]
    )

    worker._validate_execution_mode(parser, args)
    assert args.execution_plan_stdin is True
    assert args.candidate_id == "ensemble"
    assert args.model_backend_id == "da3"


@pytest.mark.parametrize(
    ("worker", "mixed_flag", "mixed_value"),
    [
        (da3_worker, "--model-variant", "METRIC_LARGE"),
        (da3_worker, "--model-key", "da3_metric"),
        (da3_worker, "--model-revision", "a" * 40),
        (da3_worker, "--device", "cpu"),
        (depth_pro_worker, "--checkpoint", "checkpoint.pt"),
        (depth_pro_worker, "--device", "cpu"),
    ],
)
def test_canonical_worker_mode_rejects_legacy_selectors(worker, mixed_flag: str, mixed_value: str) -> None:
    parser = worker._build_parser()
    args = parser.parse_args(
        [
            "--execution-plan-stdin",
            "--candidate-id",
            "da3" if worker is da3_worker else "depth_pro",
            mixed_flag,
            mixed_value,
            "--check",
        ]
    )

    with pytest.raises(SystemExit):
        worker._validate_execution_mode(parser, args)


@pytest.mark.parametrize("mixed_flag", ["--use-coreml", "--non-commercial-ok"])
def test_da3_canonical_worker_mode_rejects_legacy_boolean_selectors(mixed_flag: str) -> None:
    parser = da3_worker._build_parser()
    args = parser.parse_args(
        [
            "--execution-plan-stdin",
            "--candidate-id",
            "da3",
            mixed_flag,
            "--check",
        ]
    )

    with pytest.raises(SystemExit):
        da3_worker._validate_execution_mode(parser, args)


@pytest.mark.parametrize("worker", [da3_worker, depth_pro_worker])
def test_candidate_selector_requires_canonical_plan_mode(worker) -> None:
    parser = worker._build_parser()
    legacy = ["--model-variant", "METRIC_LARGE"] if worker is da3_worker else ["--checkpoint", "checkpoint.pt"]
    args = parser.parse_args([*legacy, "--candidate-id", "da3", "--check"])

    with pytest.raises(SystemExit):
        worker._validate_execution_mode(parser, args)


@pytest.mark.parametrize(
    ("backend_cls", "backend_id"),
    [(DA3Backend, "da3"), (DepthProBackend, "depth_pro")],
)
def test_parent_rejects_worker_authority_echo_mismatch(backend_cls, backend_id: str) -> None:
    backend = backend_cls.__new__(backend_cls)
    backend._candidate_authority = SimpleNamespace(
        plan_fingerprint_sha256="a" * 64,
        candidate_id=backend_id,
        constituent_backend_id=None,
    )

    with pytest.raises(LuxExecutionPlanAuthorityError, match="execution-authority echo"):
        backend._verify_worker_authority_echo(
            {
                "execution_authority": {
                    "plan_fingerprint_sha256": "b" * 64,
                    "candidate_id": backend_id,
                    "model_backend_id": None,
                    "executed_backend_id": backend_id,
                }
            }
        )


@pytest.mark.parametrize(
    ("backend_cls", "backend_id"),
    [(DA3Backend, "da3"), (DepthProBackend, "depth_pro")],
)
def test_parent_accepts_exact_worker_authority_echo(backend_cls, backend_id: str) -> None:
    backend = backend_cls.__new__(backend_cls)
    backend._candidate_authority = SimpleNamespace(
        plan_fingerprint_sha256="a" * 64,
        candidate_id=backend_id,
        constituent_backend_id=None,
    )
    payload = {
        "execution_authority": {
            "plan_fingerprint_sha256": "a" * 64,
            "candidate_id": backend_id,
            "model_backend_id": None,
            "executed_backend_id": backend_id,
        }
    }

    backend._verify_worker_authority_echo(payload)


def test_da3_parent_rejects_runtime_identity_echo_mismatch() -> None:
    backend = DA3Backend.__new__(DA3Backend)
    backend._candidate_authority = SimpleNamespace(
        plan_fingerprint_sha256="a" * 64,
        candidate_id="da3",
        constituent_backend_id=None,
    )
    backend._prepared_cache_runtime_identity = SimpleNamespace(runtime_identity_sha256="b" * 64)

    with pytest.raises(LuxExecutionPlanAuthorityError, match="runtime-identity echo"):
        backend._verify_worker_authority_echo(
            {
                "execution_authority": {
                    "plan_fingerprint_sha256": "a" * 64,
                    "candidate_id": "da3",
                    "model_backend_id": None,
                    "executed_backend_id": "da3",
                },
                "runtime_identity_sha256": "c" * 64,
            }
        )


def test_da3_cache_boundary_verifier_requires_exact_prepared_identity_and_live_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = DA3Backend.__new__(DA3Backend)
    backend._prepared_cache_runtime_identity = SimpleNamespace(runtime_identity_sha256="a" * 64)
    backend._prepared_cache_runtime_verification_token = {"token": "retained"}
    backend._prepared_cache_runtime_verification_token_sha256 = "b" * 64
    backend._prepared_worker_runtime_identity_sha256 = "c" * 64
    backend._prepared_parent_runtime_identity = {"parent": "retained"}
    backend._device = "cpu"
    backend._cache_runtime_authority_disabled = False

    import transformation_portal.depth.backends.da3_runtime_identity as identity_module

    observed = []

    def verify(payload, **kwargs):
        observed.append((payload, kwargs))
        return True

    monkeypatch.setattr(identity_module, "verify_runtime_verification_token", verify)
    monkeypatch.setattr(
        identity_module, "verify_parent_output_runtime_identity", lambda evidence: evidence == {"parent": "retained"}
    )

    assert backend.verify_prepared_cache_runtime_identity(runtime_identity_sha256="d" * 64) is False
    assert observed == []
    assert backend.verify_prepared_cache_runtime_identity(runtime_identity_sha256="a" * 64) is True
    assert observed == [
        (
            {"token": "retained"},
            {
                "expected_token_sha256": "b" * 64,
                "expected_worker_runtime_identity_sha256": "c" * 64,
                "expected_prepared_runtime_identity_sha256": "a" * 64,
                "expected_requested_device": "cpu",
                "expected_actual_device": "cpu",
                "expected_executed_backend": "pytorch_cpu",
            },
        )
    ]

    monkeypatch.setattr(identity_module, "verify_runtime_verification_token", lambda *_args, **_kwargs: False)
    assert backend.verify_prepared_cache_runtime_identity(runtime_identity_sha256="a" * 64) is False
    assert backend._cache_runtime_authority_disabled is True
    assert backend._prepared_cache_runtime_identity is None


def test_da3_parent_runtime_drift_requires_restart_before_reauthorization(
    prepared_da3,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = backend_candidate_authority(prepared_da3.plan, "da3")
    backend = DepthBackendRegistry().get_backend(
        "da3",
        prepared_da3.runtime_config,
        candidate_authority=authority,
        canonical_plan_bytes=prepared_da3.canonical_plan_bytes,
    )
    backend._prepared_cache_runtime_identity = SimpleNamespace(runtime_identity_sha256="a" * 64)
    backend._prepared_cache_runtime_verification_token = {"token": "retained"}
    backend._prepared_cache_runtime_verification_token_sha256 = "b" * 64
    backend._prepared_worker_runtime_identity_sha256 = "c" * 64
    backend._prepared_parent_runtime_identity = {"parent": "retained"}

    import transformation_portal.depth.backends.da3_runtime_identity as identity_module

    monkeypatch.setattr(identity_module, "verify_runtime_verification_token", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(identity_module, "verify_parent_output_runtime_identity", lambda _evidence: False)

    assert backend.verify_prepared_cache_runtime_identity(runtime_identity_sha256="a" * 64) is False
    assert backend._cache_runtime_authority_disabled is True
    assert (
        backend.prepare_cache_runtime_identity(
            execution_plan=prepared_da3.plan,
            candidate_id=authority.candidate_id,
            canonical_plan_bytes=prepared_da3.canonical_plan_bytes,
        )
        is None
    )
    backend._python_executable = "/usr/bin/python-worker"

    def fake_run(command, **kwargs):
        assert "--execution-plan-stdin" in command
        assert kwargs["input"] == prepared_da3.canonical_plan_bytes.decode("utf-8")
        output_json = Path(command[command.index("--output-json") + 1])
        output_depth = Path(command[command.index("--output-depth") + 1])
        np.save(output_depth, np.ones((2, 2), dtype=np.float32), allow_pickle=False)
        output_json.write_bytes(
            canonicalize_json(
                {
                    "metadata": {"device": "cpu"},
                    "device": "cpu",
                    "dtype": "float32",
                    "input_size": [2, 2],
                    "execution_authority": {
                        "plan_fingerprint_sha256": authority.plan_fingerprint_sha256,
                        "candidate_id": authority.candidate_id,
                        "model_backend_id": authority.constituent_backend_id,
                        "executed_backend_id": "da3",
                    },
                }
            )
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    import transformation_portal.depth.backends.da3 as da3_module

    monkeypatch.setattr(da3_module.subprocess, "run", fake_run)
    result = backend._compute_subprocess(np.zeros((2, 2, 3), dtype=np.uint8))

    assert result.depth_map.shape == (2, 2)
    assert backend._cache_runtime_authority_disabled is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"candidate_id": "depth_pro"}, "different candidate"),
        ({"canonical_plan_bytes": b"{}"}, "different canonical plan bytes"),
    ],
)
def test_da3_cache_runtime_preparation_rejects_integration_authority_mismatch(
    prepared_da3,
    overrides: dict[str, Any],
    message: str,
) -> None:
    authority = backend_candidate_authority(prepared_da3.plan, "da3")
    backend = DepthBackendRegistry().get_backend(
        "da3",
        prepared_da3.runtime_config,
        candidate_authority=authority,
        canonical_plan_bytes=prepared_da3.canonical_plan_bytes,
    )
    backend._python_executable = "/usr/bin/python-worker"
    arguments = {
        "execution_plan": prepared_da3.plan,
        "candidate_id": "da3",
        "canonical_plan_bytes": prepared_da3.canonical_plan_bytes,
    }
    arguments.update(overrides)

    with pytest.raises(LuxExecutionPlanAuthorityError, match=message):
        backend.prepare_cache_runtime_identity(**arguments)


def test_da3_cache_runtime_preparation_rejects_different_execution_plan(
    prepared_da3,
    prepared_fallback_chain,
) -> None:
    authority = backend_candidate_authority(prepared_da3.plan, "da3")
    backend = DepthBackendRegistry().get_backend(
        "da3",
        prepared_da3.runtime_config,
        candidate_authority=authority,
        canonical_plan_bytes=prepared_da3.canonical_plan_bytes,
    )

    with pytest.raises(LuxExecutionPlanAuthorityError, match="different execution plan"):
        backend.prepare_cache_runtime_identity(
            execution_plan=prepared_fallback_chain.plan,
            candidate_id="da3",
            canonical_plan_bytes=prepared_da3.canonical_plan_bytes,
        )


def test_worker_consumer_rejects_noncanonical_plan_bytes(prepared_da3) -> None:
    with pytest.raises(ValueError, match="exact canonical serialization"):
        consume_lux_worker_execution_plan(b" " + prepared_da3.canonical_plan_bytes)


def test_worker_consumer_rejects_modified_plan_with_stale_fingerprint(prepared_da3) -> None:
    payload = json.loads(prepared_da3.canonical_plan_bytes)
    payload["quality_tier"] = "apex"

    with pytest.raises(ValueError, match="fingerprint|quality tier"):
        consume_lux_worker_execution_plan(canonicalize_json(payload))


def test_worker_consumer_rejects_structural_legacy_plan(prepared_da3) -> None:
    from transformation_portal.core.execution_plan import with_execution_plan_fingerprint

    payload = json.loads(prepared_da3.canonical_plan_bytes)
    payload["configuration_completeness"] = "structural_legacy"
    for node in payload["nodes"]:
        node["configuration"]["configuration_completeness"] = "structural_legacy"
    payload = with_execution_plan_fingerprint(payload)

    with pytest.raises(ValueError, match="parse-only|execution authority"):
        consume_lux_worker_execution_plan(canonicalize_json(payload))


def test_candidate_mismatch_fails_closed(prepared_da3) -> None:
    with pytest.raises(ValueError, match="absent or ambiguous"):
        backend_candidate_authority(prepared_da3.plan, "depth_pro")


def test_ensemble_constituent_mismatch_fails_closed(prepared_ensemble) -> None:
    with pytest.raises(ValueError, match="no unique enabled model"):
        backend_candidate_authority(
            prepared_ensemble.plan,
            "ensemble",
            model_backend_id="depthcrafter",
        )


def test_default_da3_plan_projects_metric_variant_without_second_resolution(
    prepared_da3,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.depth.backends.da3 as da3_module

    assert prepared_da3.plan.resolved_model is not None
    assert prepared_da3.plan.resolved_model.canonical_key == "da3_metric"
    assert prepared_da3.plan.resolved_model.legacy_model_variant_name is None
    authority = backend_candidate_authority(prepared_da3.plan, "da3")

    def forbidden_resolution(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("canonical backend attempted a second model resolution")

    monkeypatch.setattr(da3_module, "resolve_model_contract", forbidden_resolution)
    backend = DepthBackendRegistry().get_backend(
        "da3",
        prepared_da3.runtime_config,
        candidate_authority=authority,
        canonical_plan_bytes=prepared_da3.canonical_plan_bytes,
    )

    assert isinstance(backend, DA3Backend)
    assert backend._resolved_model_contract is authority.resolved_model_contract
    assert backend._resolved_model_contract.canonical_key == "da3_metric"
    assert backend._model_variant is ModelVariant.METRIC_LARGE


def test_da3_worker_consumes_default_metric_authority_without_legacy_provenance(
    prepared_da3,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        da3_worker.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(prepared_da3.canonical_plan_bytes)),
    )

    import transformation_portal.lux_depth_v3.execution_lifecycle as lifecycle_module

    def forbidden_resolution(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("canonical worker attempted model selection")

    monkeypatch.setattr(lifecycle_module, "resolve_model_contract", forbidden_resolution)
    authority = da3_worker._consume_canonical_worker_authority(candidate_id="da3", model_backend_id=None)

    assert authority.resolved_model_contract.canonical_key == "da3_metric"
    assert authority.resolved_model_contract.legacy_model_variant_name is None
    assert authority.device == prepared_da3.runtime_config.depth_device


def test_da3_worker_rejects_prepared_runtime_mismatch_before_inference(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_image = tmp_path / "input.png"
    from PIL import Image

    Image.new("RGB", (2, 2), color="white").save(input_image)
    predicted = False

    class FakeEngine:
        def predict(self, _image):
            nonlocal predicted
            predicted = True
            raise AssertionError("inference ran before runtime identity validation")

    monkeypatch.setattr(da3_worker, "_build_inference_engine", lambda **_kwargs: FakeEngine())
    import transformation_portal.depth.backends.da3_runtime_identity as identity_module

    token_path = tmp_path / "runtime-token.json"
    token_path.write_bytes(
        canonicalize_json(
            {
                "schema": "tp.da3.runtime-verification-token.v1",
                "worker_runtime_identity_sha256": "b" * 64,
                "prepared_runtime": {
                    "schema": "tp.da3.prepared-runtime-binding.v1",
                    "runtime_identity_sha256": "a" * 64,
                    "requested_device": "cpu",
                    "actual_device": "cpu",
                    "executed_backend": "pytorch_cpu",
                },
                "source_revision_probe": None,
                "entries": [],
            }
        )
    )
    monkeypatch.setattr(identity_module, "verify_runtime_verification_token", lambda *_args, **_kwargs: False)

    with pytest.raises(RuntimeError, match="verification token is stale or invalid"):
        da3_worker._run_inference(
            input_image=input_image,
            output_depth=tmp_path / "depth.npy",
            output_json=tmp_path / "result.json",
            model_variant_name="METRIC_LARGE",
            model_key="da3_metric",
            model_revision="4" * 40,
            device="cpu",
            use_coreml=False,
            non_commercial_ok=False,
            expected_runtime_identity_sha256="a" * 64,
            runtime_verification_token_path=token_path,
            runtime_verification_token_sha256="c" * 64,
        )
    assert predicted is False


def test_da3_worker_rejects_cross_token_runtime_identity_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    input_image = tmp_path / "input.png"
    Image.new("RGB", (2, 2), color="white").save(input_image)
    token_path, token_sha256 = _write_bound_runtime_token(
        tmp_path / "runtime-token.json",
        runtime_identity_sha256="a" * 64,
        device="cpu",
        executed_backend="pytorch_cpu",
    )
    predicted = False

    def forbidden_engine(**_kwargs):
        nonlocal predicted
        predicted = True
        raise AssertionError("engine constructed before cross-token replay rejection")

    monkeypatch.setattr(da3_worker, "_build_inference_engine", forbidden_engine)

    with pytest.raises(RuntimeError, match="verification token is stale or invalid"):
        da3_worker._run_inference(
            input_image=input_image,
            output_depth=tmp_path / "depth.npy",
            output_json=tmp_path / "result.json",
            model_variant_name="METRIC_LARGE",
            model_key="da3_metric",
            model_revision="4" * 40,
            device="cpu",
            use_coreml=False,
            non_commercial_ok=False,
            expected_runtime_identity_sha256="c" * 64,
            runtime_verification_token_path=token_path,
            runtime_verification_token_sha256=token_sha256,
        )
    assert predicted is False


def test_da3_worker_rejects_prepared_mps_runtime_with_fresh_cpu_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    input_image = tmp_path / "input.png"
    Image.new("RGB", (2, 2), color="white").save(input_image)
    token_path, token_sha256 = _write_bound_runtime_token(
        tmp_path / "runtime-token.json",
        runtime_identity_sha256="a" * 64,
        device="mps",
        executed_backend="pytorch_mps",
    )
    import transformation_portal.depth.backends.da3_runtime_identity as identity_module

    monkeypatch.setattr(identity_module, "_verify_source_revision_probe", lambda _payload: True)

    class FreshCpuEngine:
        backend = SimpleNamespace(value="pytorch_cpu")
        device = "cpu"

        def predict(self, _image):
            raise AssertionError("inference ran with a drifted fresh engine")

    monkeypatch.setattr(da3_worker, "_build_inference_engine", lambda **_kwargs: FreshCpuEngine())

    with pytest.raises(RuntimeError, match="differs from prepared cache authority"):
        da3_worker._run_inference(
            input_image=input_image,
            output_depth=tmp_path / "depth.npy",
            output_json=tmp_path / "result.json",
            model_variant_name="METRIC_LARGE",
            model_key="da3_metric",
            model_revision="4" * 40,
            device="mps",
            use_coreml=False,
            non_commercial_ok=False,
            expected_runtime_identity_sha256="a" * 64,
            runtime_verification_token_path=token_path,
            runtime_verification_token_sha256=token_sha256,
        )
    assert not (tmp_path / "depth.npy").exists()


@pytest.mark.parametrize("drift_phase", ("before", "after", "result"))
def test_da3_worker_rejects_fresh_engine_runtime_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_phase: str,
) -> None:
    from PIL import Image

    input_image = tmp_path / "input.png"
    Image.new("RGB", (2, 2), color="white").save(input_image)
    token_path = tmp_path / "runtime-token.json"
    token_path.write_bytes(
        canonicalize_json(
            {
                "schema": "tp.da3.runtime-verification-token.v1",
                "worker_runtime_identity_sha256": "b" * 64,
                "prepared_runtime": {
                    "schema": "tp.da3.prepared-runtime-binding.v1",
                    "runtime_identity_sha256": "a" * 64,
                    "requested_device": "cpu",
                    "actual_device": "cpu",
                    "executed_backend": "pytorch_cpu",
                },
                "source_revision_probe": None,
                "entries": [],
            }
        )
    )
    predicted = False

    class BackendValue:
        value = "pytorch_mps" if drift_phase == "before" else "pytorch_cpu"

    class FakeEngine:
        backend = BackendValue()
        device = "mps" if drift_phase == "before" else "cpu"

        def predict(self, _image):
            nonlocal predicted
            predicted = True
            if drift_phase == "after":
                self.backend = SimpleNamespace(value="pytorch_mps")
                self.device = "mps"
            metadata = {
                "backend": "pytorch_mps" if drift_phase == "result" else "pytorch_cpu",
                "device": "mps" if drift_phase == "result" else "cpu",
            }
            return SimpleNamespace(
                depth_map=np.ones((2, 2), dtype=np.float32),
                original_image=np.zeros((2, 2, 3), dtype=np.uint8),
                metadata=metadata,
            )

    monkeypatch.setattr(da3_worker, "_build_inference_engine", lambda **_kwargs: FakeEngine())
    import transformation_portal.depth.backends.da3_runtime_identity as identity_module

    monkeypatch.setattr(identity_module, "verify_runtime_verification_token", lambda *_args, **_kwargs: True)

    with pytest.raises(RuntimeError, match="differs from prepared cache authority"):
        da3_worker._run_inference(
            input_image=input_image,
            output_depth=tmp_path / "depth.npy",
            output_json=tmp_path / "result.json",
            model_variant_name="METRIC_LARGE",
            model_key="da3_metric",
            model_revision="4" * 40,
            device="cpu",
            use_coreml=False,
            non_commercial_ok=False,
            expected_runtime_identity_sha256="a" * 64,
            runtime_verification_token_path=token_path,
            runtime_verification_token_sha256="c" * 64,
        )
    assert predicted is (drift_phase != "before")
    assert not (tmp_path / "depth.npy").exists()


@pytest.mark.parametrize(
    ("prepared_fixture", "backend_id", "legacy_flags"),
    [
        ("prepared_da3", "da3", {"--model-variant", "--model-key", "--model-revision", "--device"}),
        ("prepared_depth_pro", "depth_pro", {"--checkpoint", "--device"}),
    ],
)
def test_canonical_availability_worker_receives_only_plan_authority(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    prepared_fixture: str,
    backend_id: str,
    legacy_flags: set[str],
) -> None:
    prepared = request.getfixturevalue(prepared_fixture)
    authority = backend_candidate_authority(prepared.plan, backend_id)
    backend = DepthBackendRegistry().get_backend(
        backend_id,
        prepared.runtime_config,
        candidate_authority=authority,
        canonical_plan_bytes=prepared.canonical_plan_bytes,
    )
    backend._python_executable = "/usr/bin/python-worker"
    captured: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    module = __import__(backend.__class__.__module__, fromlist=["subprocess"])
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    if backend_id == "da3":
        backend._ensure_subprocess_available()
    else:
        backend._ensure_subprocess_available(device=backend._device)

    command = captured["command"]
    assert "--execution-plan-stdin" in command
    assert command[command.index("--candidate-id") + 1] == backend_id
    assert not legacy_flags.intersection(command)
    assert captured["kwargs"]["input"].encode("utf-8") == prepared.canonical_plan_bytes


@pytest.mark.parametrize(
    ("prepared_fixture", "backend_id"),
    [("prepared_da3", "da3"), ("prepared_depth_pro", "depth_pro")],
)
def test_parent_verifies_worker_echo_before_loading_array(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    prepared_fixture: str,
    backend_id: str,
) -> None:
    prepared = request.getfixturevalue(prepared_fixture)
    authority = backend_candidate_authority(prepared.plan, backend_id)
    backend = DepthBackendRegistry().get_backend(
        backend_id,
        prepared.runtime_config,
        candidate_authority=authority,
        canonical_plan_bytes=prepared.canonical_plan_bytes,
    )
    backend._python_executable = "/usr/bin/python-worker"
    module = __import__(backend.__class__.__module__, fromlist=["subprocess"])
    loaded = False

    def fake_run(command, **kwargs):
        output_json = command[command.index("--output-json") + 1]
        output_depth = command[command.index("--output-depth") + 1]
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "execution_authority": {
                        "plan_fingerprint_sha256": "0" * 64,
                        "candidate_id": backend_id,
                        "model_backend_id": None,
                        "executed_backend_id": backend_id,
                    }
                },
                handle,
            )
        with open(output_depth, "wb") as handle:
            handle.write(b"not-an-array")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def forbidden_array_load(*args, **kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("array was loaded before authority verification")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.np, "load", forbidden_array_load)

    with pytest.raises(LuxExecutionPlanAuthorityError, match="execution-authority echo"):
        backend._compute_subprocess(np.zeros((2, 2, 3), dtype=np.uint8))
    assert loaded is False


def test_da3_parent_verifies_runtime_echo_before_loading_array(
    prepared_da3,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = backend_candidate_authority(prepared_da3.plan, "da3")
    backend = DepthBackendRegistry().get_backend(
        "da3",
        prepared_da3.runtime_config,
        candidate_authority=authority,
        canonical_plan_bytes=prepared_da3.canonical_plan_bytes,
    )
    backend._python_executable = "/usr/bin/python-worker"
    backend._prepared_cache_runtime_identity = SimpleNamespace(runtime_identity_sha256="a" * 64)
    backend._prepared_cache_runtime_verification_token = {
        "schema": "tp.da3.runtime-verification-token.v1",
        "worker_runtime_identity_sha256": "d" * 64,
        "source_revision_probe": None,
        "entries": [],
    }
    backend._prepared_cache_runtime_verification_token_sha256 = "e" * 64
    loaded = False

    def fake_run(command, **kwargs):
        del kwargs
        assert command[command.index("--expected-runtime-identity-sha256") + 1] == "a" * 64
        output_json = command[command.index("--output-json") + 1]
        output_depth = command[command.index("--output-depth") + 1]
        payload = {
            "execution_authority": {
                "plan_fingerprint_sha256": authority.plan_fingerprint_sha256,
                "candidate_id": "da3",
                "model_backend_id": None,
                "executed_backend_id": "da3",
            },
            "runtime_identity_sha256": "b" * 64,
        }
        with open(output_json, "wb") as handle:
            handle.write(canonicalize_json(payload))
        with open(output_depth, "wb") as handle:
            handle.write(b"not-an-array")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def forbidden_array_load(*args, **kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("array was loaded before runtime identity verification")

    import transformation_portal.depth.backends.da3 as da3_module

    monkeypatch.setattr(da3_module.subprocess, "run", fake_run)
    monkeypatch.setattr(da3_module.np, "load", forbidden_array_load)

    with pytest.raises(LuxExecutionPlanAuthorityError, match="runtime-identity echo"):
        backend._compute_subprocess(np.zeros((2, 2, 3), dtype=np.uint8))
    assert loaded is False


def test_da3_parent_records_verified_runtime_identity_after_inference(
    prepared_da3,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = backend_candidate_authority(prepared_da3.plan, "da3")
    backend = DepthBackendRegistry().get_backend(
        "da3",
        prepared_da3.runtime_config,
        candidate_authority=authority,
        canonical_plan_bytes=prepared_da3.canonical_plan_bytes,
    )
    backend._python_executable = "/usr/bin/python-worker"
    backend._prepared_cache_runtime_identity = SimpleNamespace(runtime_identity_sha256="a" * 64)
    backend._prepared_cache_runtime_verification_token = {
        "schema": "tp.da3.runtime-verification-token.v1",
        "worker_runtime_identity_sha256": "d" * 64,
        "source_revision_probe": None,
        "entries": [],
    }
    backend._prepared_cache_runtime_verification_token_sha256 = "e" * 64

    def fake_run(command, **kwargs):
        del kwargs
        output_json = command[command.index("--output-json") + 1]
        output_depth = command[command.index("--output-depth") + 1]
        payload = {
            "metadata": {"device": "cpu"},
            "device": "cpu",
            "dtype": "float32",
            "input_size": [2, 2],
            "execution_authority": {
                "plan_fingerprint_sha256": authority.plan_fingerprint_sha256,
                "candidate_id": "da3",
                "model_backend_id": None,
                "executed_backend_id": "da3",
            },
            "runtime_identity_sha256": "a" * 64,
        }
        np.save(output_depth, np.ones((2, 2), dtype=np.float32), allow_pickle=False)
        depth_bytes = Path(output_depth).read_bytes()
        payload["depth_artifact"] = {
            "sha256": hashlib.sha256(depth_bytes).hexdigest(),
            "size_bytes": len(depth_bytes),
            "shape": [2, 2],
            "dtype": "float32",
            "fortran_order": False,
        }
        with open(output_json, "wb") as handle:
            handle.write(canonicalize_json(payload))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    import transformation_portal.depth.backends.da3 as da3_module

    monkeypatch.setattr(da3_module.subprocess, "run", fake_run)

    result = backend._compute_subprocess(np.zeros((2, 2, 3), dtype=np.uint8))

    assert result.metadata["runtime_identity_sha256"] == "a" * 64


def test_da3_depth_artifact_is_loaded_from_digest_verified_open_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "depth.npy"
    np.save(path, np.ones((2, 3), dtype=np.float32), allow_pickle=False)
    raw = path.read_bytes()
    artifact = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "shape": [2, 3],
        "dtype": "float32",
        "fortran_order": False,
    }
    original_load = np.load
    observed_open_handle = False

    def checked_load(value, **kwargs):
        nonlocal observed_open_handle
        observed_open_handle = hasattr(value, "read") and hasattr(value, "fileno")
        return original_load(value, **kwargs)

    import transformation_portal.depth.backends.da3 as da3_module

    monkeypatch.setattr(da3_module.np, "load", checked_load)
    loaded = _load_verified_worker_depth(path, artifact)

    assert observed_open_handle is True
    assert loaded.shape == (2, 3)
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="digest mismatch"):
        _load_verified_worker_depth(path, artifact)


def test_da3_depth_artifact_rejects_forged_huge_header_before_np_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
            "fortran_order": False,
            "shape": (1_000_000_000, 1_000_000_000),
        },
    )
    raw = buffer.getvalue()
    path = tmp_path / "forged.npy"
    path.write_bytes(raw)
    artifact = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "shape": [1, 1],
        "dtype": "float32",
        "fortran_order": False,
    }

    import transformation_portal.depth.backends.da3 as da3_module

    monkeypatch.setattr(
        da3_module.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("np.load ran before bounded NPY header validation"),
    )
    with pytest.raises(ValueError, match="header disagrees"):
        _load_verified_worker_depth(path, artifact)


def test_ensemble_preserves_exact_constituent_order_and_contracts(prepared_ensemble) -> None:
    from transformation_portal.depth.backends.ensemble import DepthEnsembleBackend

    authority = backend_candidate_authority(prepared_ensemble.plan, "ensemble")
    backend = DepthBackendRegistry().get_backend(
        "ensemble",
        prepared_ensemble.runtime_config,
        candidate_authority=authority,
        canonical_plan_bytes=prepared_ensemble.canonical_plan_bytes,
    )

    assert isinstance(backend, DepthEnsembleBackend)
    assert [model.name for model in backend._models] == ["depth_pro", "da3"]
    assert [model.weight for model in backend._models] == [0.625, 0.375]
    assert [model.device for model in backend._models] == ["cpu", "cpu"]
    assert [model.model_contract for model in backend._models] == list(authority.candidate.model_contracts)
    da3_backend = backend._get_backend(backend._models[1])
    assert da3_backend._candidate_authority.candidate_id == "ensemble"
    assert da3_backend._candidate_authority.constituent_backend_id == "da3"
    assert da3_backend._resolved_model_contract.canonical_key == "da3_metric"


def test_ensemble_preserves_disabled_constituent_without_instantiating_it(prepared_ensemble) -> None:
    from transformation_portal.core.execution_plan import with_execution_plan_fingerprint

    payload = json.loads(prepared_ensemble.canonical_plan_bytes)
    disabled = dict(payload["backend_candidates"][0]["model_contracts"][1])
    disabled["model"] = dict(disabled["model"])
    disabled["backend_id"] = "depthcrafter"
    disabled["enabled"] = False
    disabled["weight"] = 0.1
    disabled["model"]["requested_selector"] = "backend:depthcrafter"
    disabled["model"]["canonical_key"] = "disabled_depthcrafter"
    payload["backend_candidates"][0]["model_contracts"].append(disabled)
    plan_bytes = canonicalize_json(with_execution_plan_fingerprint(payload))
    plan = consume_lux_worker_execution_plan(plan_bytes)
    authority = backend_candidate_authority(plan, "ensemble")
    runtime_config = runtime_config_from_execution_plan(plan)
    backend = DepthBackendRegistry().get_backend(
        "ensemble",
        runtime_config,
        candidate_authority=authority,
        canonical_plan_bytes=plan_bytes,
    )

    assert [model.name for model in backend._models] == ["depth_pro", "da3", "depthcrafter"]
    assert backend._models[-1].enabled is False
    assert backend._models[-1].weight == 0.1
    assert "depthcrafter" not in backend._backends


def test_canonical_ensemble_rejects_missing_planned_constituent(prepared_ensemble, monkeypatch) -> None:
    from transformation_portal.depth.backends.protocol import DepthResult

    authority = backend_candidate_authority(prepared_ensemble.plan, "ensemble")
    backend = DepthBackendRegistry().get_backend(
        "ensemble",
        prepared_ensemble.runtime_config,
        candidate_authority=authority,
        canonical_plan_bytes=prepared_ensemble.canonical_plan_bytes,
    )

    class SuccessfulBackend:
        def compute(self, image, device=None):
            return DepthResult(
                depth_map=np.ones((2, 2), dtype=np.float32),
                original_image=np.zeros((2, 2, 3), dtype=np.uint8),
                metadata={},
                depth_units="meters",
                backend_id="depth_pro",
                device="cpu",
                dtype="float32",
                input_size=(2, 2),
            )

    class FailingBackend:
        def compute(self, image, device=None):
            raise RuntimeError("planned DA3 failed")

    monkeypatch.setattr(
        backend,
        "_get_backend",
        lambda model: SuccessfulBackend() if model.name == "depth_pro" else FailingBackend(),
    )

    with pytest.raises(LuxExecutionPlanAuthorityError, match="exact planned membership"):
        backend._run_models(np.zeros((2, 2, 3), dtype=np.uint8), device=None)


def test_orchestrator_never_falls_back_after_execution_authority_failure(tmp_path, monkeypatch) -> None:
    from PIL import Image

    import transformation_portal.lux_depth_v3.orchestrator as orchestrator_module
    from transformation_portal.depth.backends.protocol import DepthResult, LicenseType
    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    image_path = tmp_path / "authority.png"
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(image_path)

    class FakeBackend:
        license_type = LicenseType.COMMERCIAL

        def __init__(self, name: str, *, authority_failure: bool) -> None:
            self.name = name
            self.authority_failure = authority_failure
            self.compute_calls = 0

        def ensure_available(self) -> None:
            return None

        def compute(self, image, device=None):
            self.compute_calls += 1
            if self.authority_failure:
                raise LuxExecutionPlanAuthorityError("worker echo mismatch")
            return DepthResult(
                depth_map=np.ones((8, 8), dtype=np.float32),
                original_image=np.zeros((8, 8, 3), dtype=np.uint8),
                metadata={},
                depth_units="relative",
                backend_id=self.name,
                device="cpu",
                dtype="float32",
                input_size=(8, 8),
            )

    da3_backend = FakeBackend("da3", authority_failure=True)
    da2_backend = FakeBackend("da2", authority_failure=False)

    class FakeRegistry:
        def get_backend(self, backend_id, config=None, **kwargs):
            return da3_backend if backend_id == "da3" else da2_backend

    monkeypatch.setattr(orchestrator_module, "DepthBackendRegistry", FakeRegistry)
    config = EnhanceConfig(
        depth_backend="da3",
        depth_operational_fallback_chain=("da3", "da2"),
        depth_fallback="skip",
        enable_v2=False,
        enable_materials_v3=False,
        enable_depth_cache=False,
    )
    orchestrator = orchestrator_module.EnhanceOrchestrator(config, tmp_path / "output")
    orchestrator.postprocessor = SimpleNamespace(process=lambda result: result)

    with pytest.raises(LuxExecutionPlanAuthorityError, match="worker echo mismatch"):
        orchestrator._compute_depth_stage(
            ImageInput(path=image_path),
            tmp_path / "authority",
            orchestrator.depth_dir / "authority.png",
            orchestrator.depth_dir / "authority.npy",
            orchestrator.manifests_dir / "authority.json",
            False,
        )

    assert da3_backend.compute_calls == 1
    assert da2_backend.compute_calls == 0
    assert orchestrator._active_depth_attempts[-1]["failure_kind"] == "authority"
    assert orchestrator._active_depth_attempts[-1]["error_code"] == "EXECUTION_AUTHORITY_REJECTED"


def test_production_select_backend_passes_exact_carried_authority(
    prepared_da3,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DA3Backend, "ensure_available", lambda self: None)

    selection = select_backend(
        "da3",
        prepared_da3.runtime_config,
        DepthBackendRegistry(),
    )

    assert selection.resolved_backend == "da3"
    assert selection.backend is not None
    assert selection.backend._candidate_authority == backend_candidate_authority(prepared_da3.plan, "da3")
    assert selection.backend._canonical_plan_bytes == prepared_da3.canonical_plan_bytes


def test_production_fallback_selects_next_exact_carried_candidate(
    prepared_fallback_chain,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformation_portal.depth.backends.da2 import DA2Backend

    def unavailable(self) -> None:
        raise ImportError("fixture primary unavailable")

    monkeypatch.setattr(DA3Backend, "ensure_available", unavailable)
    monkeypatch.setattr(DA2Backend, "ensure_available", lambda self: None)

    selection = select_backend(
        None,
        prepared_fallback_chain.runtime_config,
        DepthBackendRegistry(),
    )

    assert selection.resolved_backend == "da2"
    assert selection.backend is not None
    assert selection.backend._candidate_authority == backend_candidate_authority(
        prepared_fallback_chain.plan,
        "da2",
    )
    assert selection.backend._canonical_plan_bytes == prepared_fallback_chain.canonical_plan_bytes


def test_da2_loader_uses_exact_carried_model_revision(
    prepared_fallback_chain,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.depth.models.depth_anything_v2 as da2_model_module

    authority = backend_candidate_authority(prepared_fallback_chain.plan, "da2")
    assert authority.model_contract is not None
    captured: dict[str, Any] = {}

    class FakeDepthAnythingV2Model:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(da2_model_module, "DepthAnythingV2Model", FakeDepthAnythingV2Model)
    backend = DepthBackendRegistry().get_backend(
        "da2",
        runtime_config_from_execution_plan(
            prepared_fallback_chain.plan,
            candidate_authority=authority,
        ),
        candidate_authority=authority,
        canonical_plan_bytes=prepared_fallback_chain.canonical_plan_bytes,
    )

    backend._load_model()

    assert captured["model_revision"] == authority.model_contract.model.revision


def test_da2_canonical_mps_unavailable_fails_without_device_drift(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Import the model module before replacing the narrow runtime probe.
    from importlib import import_module

    from transformation_portal.depth.backends.da2 import DA2Backend

    import_module("transformation_portal.depth.models.depth_anything_v2")

    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(b"fixture")
    prepared = prepare_lux_execution(
        EnhanceConfig(
            depth_backend="da2",
            depth_device="mps",
            enable_v2=False,
        ),
        tmp_path,
        [input_path],
    )
    authority = backend_candidate_authority(prepared.plan, "da2")
    backend = DA2Backend(
        runtime_config_from_execution_plan(
            prepared.plan,
            candidate_authority=authority,
        ),
        candidate_authority=authority,
        canonical_plan_bytes=prepared.canonical_plan_bytes,
    )
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="Canonical DA2 candidate planned device='mps'.*MPS is unavailable"):
        backend._load_model()

    assert backend._device == "mps"
    assert backend._model is None


def test_da2_legacy_mps_unavailable_preserves_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.depth.models.depth_anything_v2 as da2_model_module
    from transformation_portal.depth.backends.da2 import DA2Backend

    class FakeDepthAnythingV2Model:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False),
        )
    )
    monkeypatch.setattr(da2_model_module, "DepthAnythingV2Model", FakeDepthAnythingV2Model)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    backend = DA2Backend(
        EnhanceConfig(
            depth_backend="da2",
            depth_device="mps",
            enable_v2=False,
        )
    )

    backend._load_model()

    assert backend._device == "cpu"
    assert backend._model is not None
