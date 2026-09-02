"""Canonical execution-plan carrier contracts for depth workers/backends."""

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from transformation_portal.depth.backends import da3_worker, depth_pro_worker
from transformation_portal.depth.backends.da3 import DA3Backend
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
