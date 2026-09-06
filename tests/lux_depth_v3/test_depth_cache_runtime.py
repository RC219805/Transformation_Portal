"""Contracts for the pre-lookup depth-cache runtime evidence hand-off."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from transformation_portal.core.execution_identity_v3 import BackendRuntimeIdentity, ExecutionIdentityV3
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.depth_cache_runtime import (
    DEPTH_CACHE_RUNTIME_EVIDENCE_SCHEMA,
    DepthCacheRuntimeEvidenceError,
    PreparedDepthCacheRuntimeEvidence,
)
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
from transformation_portal.lux_depth_v3.execution_plan_adapter import LuxExecutionPlanAuthorityError
from transformation_portal.stage_graph.registry import StageRegistryIdentifier

pytestmark = pytest.mark.unit


def _backend_identity(**overrides: object) -> BackendRuntimeIdentity:
    values: dict[str, object] = {
        "constituent_ordinal": 0,
        "backend_id": "da3",
        "model_canonical_key": "da3_metric",
        "model_lock_revision": "1" * 40,
        "planned_model_artifact_sha256": None,
        "planned_model_license_contract_sha256": "2" * 64,
        "materialized_weights_sha256": "3" * 64,
        "dependency_lock_sha256": "4" * 64,
        "interpreter_identity_sha256": "5" * 64,
        "platform_identity_sha256": "6" * 64,
        "accelerator_identity_sha256": "7" * 64,
        "source_identity_sha256": "8" * 64,
    }
    values.update(overrides)
    return BackendRuntimeIdentity(**values)  # type: ignore[arg-type]


def _aggregate_runtime_field(identities: tuple[BackendRuntimeIdentity, ...], field_name: str) -> str:
    if len(identities) == 1:
        return getattr(identities[0], field_name)
    projection = [
        {
            "constituent_ordinal": identity.constituent_ordinal,
            "backend_id": identity.backend_id,
            "model_canonical_key": identity.model_canonical_key,
            "model_lock_revision": identity.model_lock_revision,
            field_name: getattr(identity, field_name),
        }
        for identity in identities
    ]
    return hashlib.sha256(
        b"tp.execution.runtime-aggregate.v1\0" + field_name.encode("ascii") + b"\0" + canonicalize_json(projection)
    ).hexdigest()


def _evidence(
    identity: BackendRuntimeIdentity | tuple[BackendRuntimeIdentity, ...] | None = None,
) -> PreparedDepthCacheRuntimeEvidence:
    runtimes = identity if isinstance(identity, tuple) else (identity or _backend_identity(),)
    return PreparedDepthCacheRuntimeEvidence.create(
        backend_runtime_identities=runtimes,
        dependency_lock_sha256=_aggregate_runtime_field(runtimes, "dependency_lock_sha256"),
        interpreter_identity_sha256=_aggregate_runtime_field(runtimes, "interpreter_identity_sha256"),
        platform_identity_sha256=_aggregate_runtime_field(runtimes, "platform_identity_sha256"),
        accelerator_identity_sha256=_aggregate_runtime_field(runtimes, "accelerator_identity_sha256"),
        source_identity_sha256=_aggregate_runtime_field(runtimes, "source_identity_sha256"),
    )


def test_runtime_evidence_is_closed_canonical_and_round_trips() -> None:
    evidence = _evidence()

    assert evidence.schema == DEPTH_CACHE_RUNTIME_EVIDENCE_SCHEMA
    assert len(evidence.runtime_identity_sha256) == 64
    assert PreparedDepthCacheRuntimeEvidence.from_payload(evidence.to_payload()) == evidence
    assert PreparedDepthCacheRuntimeEvidence.from_payload(evidence.to_payload()).runtime_identity_sha256 == (
        evidence.runtime_identity_sha256
    )


def test_runtime_digest_changes_with_materialized_weight_bytes() -> None:
    first = _evidence()
    second = _evidence(_backend_identity(materialized_weights_sha256="9" * 64))

    assert first.runtime_identity_sha256 != second.runtime_identity_sha256


def test_runtime_evidence_rejects_aggregate_not_derived_from_constituent() -> None:
    runtime = _backend_identity()

    with pytest.raises(DepthCacheRuntimeEvidenceError, match="does not match"):
        PreparedDepthCacheRuntimeEvidence.create(
            backend_runtime_identities=(runtime,),
            dependency_lock_sha256="a" * 64,
            interpreter_identity_sha256=runtime.interpreter_identity_sha256,
            platform_identity_sha256=runtime.platform_identity_sha256,
            accelerator_identity_sha256=runtime.accelerator_identity_sha256,
            source_identity_sha256=runtime.source_identity_sha256,
        )


def test_runtime_evidence_rejects_unknown_fields_and_placeholders() -> None:
    payload = _evidence().to_payload()
    payload["unknown"] = True
    with pytest.raises(DepthCacheRuntimeEvidenceError, match="exact supported field set"):
        PreparedDepthCacheRuntimeEvidence.from_payload(payload)

    placeholder = copy.deepcopy(_evidence().to_payload())
    placeholder["source_identity_sha256"] = "0" * 64
    with pytest.raises(DepthCacheRuntimeEvidenceError, match="placeholder"):
        PreparedDepthCacheRuntimeEvidence.from_payload(placeholder)


def test_runtime_evidence_requires_bounded_unique_ascending_constituents() -> None:
    first = _backend_identity()
    second = _backend_identity(
        constituent_ordinal=1,
        backend_id="depth_pro",
        model_canonical_key="depth_pro",
        model_lock_revision="9" * 40,
        planned_model_license_contract_sha256="a" * 64,
        materialized_weights_sha256="b" * 64,
    )

    with pytest.raises(DepthCacheRuntimeEvidenceError, match="unique ascending"):
        _evidence((first, first))
    with pytest.raises(DepthCacheRuntimeEvidenceError, match="unique ascending"):
        _evidence((second, first))
    with pytest.raises(DepthCacheRuntimeEvidenceError, match="between one and eight"):
        _evidence(tuple(_backend_identity(constituent_ordinal=index) for index in range(9)))


def test_ensemble_runtime_evidence_round_trips_with_stable_domain_digest() -> None:
    runtimes = (
        _backend_identity(),
        _backend_identity(
            constituent_ordinal=1,
            backend_id="depth_pro",
            model_canonical_key="depth_pro",
            model_lock_revision="9" * 40,
            planned_model_license_contract_sha256="a" * 64,
            materialized_weights_sha256="b" * 64,
        ),
    )
    evidence = _evidence(runtimes)

    assert PreparedDepthCacheRuntimeEvidence.from_payload(evidence.to_payload()) == evidence
    assert evidence.runtime_identity_sha256 == "867533b6e140e7e19a8ad3a14afd1ae913b4a22ddf007c0c3d555fc33c6f5ce5"


def _prepared_da2_evidence(tmp_path):
    from PIL import Image

    root = tmp_path / "inputs"
    root.mkdir()
    image = root / "image.png"
    Image.new("RGB", (64, 64), color="white").save(image)
    prepared = prepare_lux_execution(
        EnhanceConfig(
            depth_backend="da2",
            depth_operational_fallback_chain=("da2",),
            allow_synthetic_fallback=False,
            enable_depth_cache=True,
            enable_v2=False,
        ),
        root,
        [image],
    )
    depth_node = next(node for node in prepared.plan.nodes if node.stage_registry_id is StageRegistryIdentifier.LUX_DEPTH)
    seed = ExecutionIdentityV3.from_plan(
        prepared.plan,
        stage_node_id=depth_node.node_id,
        candidate_id="da2",
        input_id=prepared.plan.inputs[0].input_id,
    )
    runtime = BackendRuntimeIdentity.from_seed(
        seed,
        materialized_weights_sha256="3" * 64,
        dependency_lock_sha256="4" * 64,
        interpreter_identity_sha256="5" * 64,
        platform_identity_sha256="6" * 64,
        accelerator_identity_sha256="7" * 64,
        source_identity_sha256="8" * 64,
    )
    return prepared, image.resolve(), _evidence(runtime)


def test_orchestrator_materializes_identity_before_cache_access(tmp_path) -> None:
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    prepared, image, evidence = _prepared_da2_evidence(tmp_path)

    class Backend:
        @staticmethod
        def prepare_cache_runtime_identity(**kwargs):
            assert kwargs == {
                "execution_plan": prepared.plan,
                "candidate_id": "da2",
                "canonical_plan_bytes": prepared.canonical_plan_bytes,
            }
            return evidence

        @staticmethod
        def verify_prepared_cache_runtime_identity(*, runtime_identity_sha256: str) -> bool:
            return runtime_identity_sha256 == evidence.runtime_identity_sha256

    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator._prepared_execution = prepared

    authority = orchestrator._prepare_depth_cache_authority(
        backend=Backend(),
        backend_id="da2",
        image_path=image,
        input_content_sha256="a" * 64,
    )

    assert authority is not None
    assert authority.identity.input_content_sha256 == "a" * 64
    assert authority.identity.backend_runtime_identities == evidence.backend_runtime_identities


def test_opened_prepared_input_must_still_match_the_plan_bound_path(tmp_path) -> None:
    from PIL import Image

    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    prepared, image, _evidence = _prepared_da2_evidence(tmp_path)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (64, 64), color="black").save(outside)
    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator._prepared_execution = prepared

    orchestrator._validate_opened_prepared_image_input(image, image.stat())
    with pytest.raises(LuxExecutionPlanAuthorityError, match="no longer the file bound"):
        # Models a target-path symlink swap whose outside handle remains open
        # after the attacker restores the original planned pathname.
        orchestrator._validate_opened_prepared_image_input(image, outside.stat())


def test_prepared_cache_hit_is_authorized_before_inference_and_uses_uint8_snapshot(tmp_path) -> None:
    from transformation_portal.core.execution_identity_v3 import MaterializedExecutionIdentityV3
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    prepared, image, evidence = _prepared_da2_evidence(tmp_path)

    class Backend:
        name = "da2"
        license_type = SimpleNamespace(value="commercial")

        @staticmethod
        def ensure_available() -> None:
            return None

        @staticmethod
        def prepare_cache_runtime_identity(**kwargs):
            assert kwargs == {
                "execution_plan": prepared.plan,
                "candidate_id": "da2",
                "canonical_plan_bytes": prepared.canonical_plan_bytes,
            }
            return evidence

        @staticmethod
        def verify_prepared_cache_runtime_identity(*, runtime_identity_sha256: str) -> bool:
            return runtime_identity_sha256 == evidence.runtime_identity_sha256

        @staticmethod
        def compute(_image):  # pragma: no cover - a cache hit must preclude inference
            raise AssertionError("inference ran after an authorized cache hit")

    backend = Backend()
    registry = Mock()
    registry.get_backend.return_value = backend
    cache = Mock()
    cache.get.return_value = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)

    observed: dict[str, object] = {}
    from transformation_portal.depth.backends.protocol import DepthResult as RealDepthResult

    def _capture_depth_result(*args, **kwargs):
        original_image = kwargs.get("original_image")
        if original_image is None and len(args) >= 2:
            original_image = args[1]
        observed["dtype"] = str(original_image.dtype)
        return RealDepthResult(*args, **kwargs)

    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry", return_value=registry):
        with patch("transformation_portal.depth.backends.protocol.DepthResult", side_effect=_capture_depth_result):
            orchestrator = EnhanceOrchestrator.from_prepared(prepared, tmp_path / "output")
            orchestrator.depth_cache = cache
            orchestrator.postprocessor = Mock(process=lambda result: result)
            with patch.object(
                orchestrator,
                "_enforce_apex_depth_validity_gate",
                return_value={
                    "passed": True,
                    "failure_codes": [],
                    "warnings": [],
                    "metrics": {},
                    "thresholds": {},
                },
            ):
                output = orchestrator.enhance_batch(
                    prepared.input_root,
                    input_files=[image],
                )[0]

    assert output["status"] == "ok"
    assert observed["dtype"] == "uint8"
    planned_candidate = next(candidate for candidate in prepared.plan.backend_candidates if candidate.backend_id == "da2")
    assert len(planned_candidate.model_contracts) == 1
    assert output["attempts"][0]["device"] == planned_candidate.model_contracts[0].device
    assert output["attempts"][0]["cached"] is True
    cache.get.assert_called_once()
    assert isinstance(cache.get.call_args.args[0], MaterializedExecutionIdentityV3)
    cache.store.assert_not_called()


def test_prepared_cache_miss_stores_only_after_runtime_echo_and_semantic_gate(tmp_path) -> None:
    from transformation_portal.core.execution_identity_v3 import MaterializedExecutionIdentityV3
    from transformation_portal.depth.backends.protocol import DepthResult
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    prepared, image, evidence = _prepared_da2_evidence(tmp_path)
    native_depth = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)

    class Backend:
        name = "da2"
        license_type = SimpleNamespace(value="commercial")

        @staticmethod
        def ensure_available() -> None:
            return None

        @staticmethod
        def prepare_cache_runtime_identity(**_kwargs):
            return evidence

        @staticmethod
        def verify_prepared_cache_runtime_identity(*, runtime_identity_sha256: str) -> bool:
            return runtime_identity_sha256 == evidence.runtime_identity_sha256

        @staticmethod
        def compute(_image):
            return DepthResult(
                depth_map=native_depth,
                original_image=np.zeros((64, 64, 3), dtype=np.uint8),
                metadata={"runtime_identity_sha256": evidence.runtime_identity_sha256},
                depth_units="relative",
                backend_id="da2",
                device="cpu",
                dtype="float32",
                input_size=(64, 64),
            )

    backend = Backend()
    registry = Mock()
    registry.get_backend.return_value = backend
    cache = Mock()
    cache.get.return_value = None

    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry", return_value=registry):
        orchestrator = EnhanceOrchestrator.from_prepared(prepared, tmp_path / "output")
        orchestrator.depth_cache = cache
        orchestrator.postprocessor = Mock(process=lambda result: result)
        with patch.object(
            orchestrator,
            "_enforce_apex_depth_validity_gate",
            return_value={
                "passed": True,
                "failure_codes": [],
                "warnings": [],
                "metrics": {},
                "thresholds": {},
            },
        ):
            output = orchestrator.enhance_batch(prepared.input_root, input_files=[image])[0]

    assert output["status"] == "ok"
    cache.get.assert_called_once()
    cache.store.assert_called_once()
    stored_identity, stored_depth = cache.store.call_args.args
    assert isinstance(stored_identity, MaterializedExecutionIdentityV3)
    np.testing.assert_array_equal(stored_depth, native_depth)


@pytest.mark.parametrize("cached", [False, True])
def test_live_runtime_drift_revokes_cache_authority_without_failing_execution(tmp_path, cached: bool) -> None:
    from transformation_portal.depth.backends.protocol import DepthResult
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    prepared, image, evidence = _prepared_da2_evidence(tmp_path)
    native_depth = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    verification_calls: list[str] = []

    class Backend:
        name = "da2"
        license_type = SimpleNamespace(value="commercial")

        @staticmethod
        def ensure_available() -> None:
            return None

        @staticmethod
        def prepare_cache_runtime_identity(**_kwargs):
            return evidence

        @staticmethod
        def verify_prepared_cache_runtime_identity(*, runtime_identity_sha256: str) -> bool:
            verification_calls.append(runtime_identity_sha256)
            # Preparation is live.  The symmetric post-get or pre-store check
            # observes drift and must revoke only the cache optimization.
            return len(verification_calls) == 1

        @staticmethod
        def compute(_image):
            return DepthResult(
                depth_map=native_depth,
                original_image=np.zeros((64, 64, 3), dtype=np.uint8),
                metadata={"runtime_identity_sha256": evidence.runtime_identity_sha256},
                depth_units="relative",
                backend_id="da2",
                device="cpu",
                dtype="float32",
                input_size=(64, 64),
            )

    backend = Backend()
    registry = Mock()
    registry.get_backend.return_value = backend
    cache = Mock()
    cache.get.return_value = native_depth.copy() if cached else None

    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry", return_value=registry):
        orchestrator = EnhanceOrchestrator.from_prepared(prepared, tmp_path / f"output-live-drift-{cached}")
        orchestrator.depth_cache = cache
        orchestrator.postprocessor = Mock(process=lambda result: result)
        with patch.object(
            orchestrator,
            "_enforce_apex_depth_validity_gate",
            return_value={
                "passed": True,
                "failure_codes": [],
                "warnings": [],
                "metrics": {},
                "thresholds": {},
            },
        ):
            output = orchestrator.enhance_batch(prepared.input_root, input_files=[image])[0]

    assert output["status"] == "ok"
    assert verification_calls == [evidence.runtime_identity_sha256] * 2
    cache.get.assert_called_once()
    cache.store.assert_not_called()
    if cached:
        # The tentative hit was discarded, so the operational backend ran.
        assert output["attempts"][0]["cached"] is False


@pytest.mark.parametrize("failing_operation", ["get", "store"])
def test_cache_api_failure_cannot_replace_successful_backend_execution(tmp_path, failing_operation: str) -> None:
    from transformation_portal.depth.backends.protocol import DepthResult
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    prepared, image, evidence = _prepared_da2_evidence(tmp_path)
    native_depth = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    compute_calls: list[bool] = []

    class Backend:
        name = "da2"
        license_type = SimpleNamespace(value="commercial")

        @staticmethod
        def ensure_available() -> None:
            return None

        @staticmethod
        def prepare_cache_runtime_identity(**_kwargs):
            return evidence

        @staticmethod
        def verify_prepared_cache_runtime_identity(*, runtime_identity_sha256: str) -> bool:
            return runtime_identity_sha256 == evidence.runtime_identity_sha256

        @staticmethod
        def compute(_image):
            compute_calls.append(True)
            return DepthResult(
                depth_map=native_depth,
                original_image=np.zeros((64, 64, 3), dtype=np.uint8),
                metadata={"runtime_identity_sha256": evidence.runtime_identity_sha256},
                depth_units="relative",
                backend_id="da2",
                device="cpu",
                dtype="float32",
                input_size=(64, 64),
            )

    backend = Backend()
    registry = Mock()
    registry.get_backend.return_value = backend
    cache = Mock()
    cache.get.return_value = None
    cache.get.side_effect = RuntimeError("cache get failed") if failing_operation == "get" else None
    cache.store.side_effect = RuntimeError("cache store failed") if failing_operation == "store" else None

    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry", return_value=registry):
        orchestrator = EnhanceOrchestrator.from_prepared(prepared, tmp_path / f"output-cache-{failing_operation}")
        orchestrator.depth_cache = cache
        orchestrator.postprocessor = Mock(process=lambda result: result)
        with patch.object(
            orchestrator,
            "_enforce_apex_depth_validity_gate",
            return_value={
                "passed": True,
                "failure_codes": [],
                "warnings": [],
                "metrics": {},
                "thresholds": {},
            },
        ):
            output = orchestrator.enhance_batch(prepared.input_root, input_files=[image])[0]

    assert output["status"] == "ok"
    assert output["backend"] == "da2"
    assert output["attempts"][0]["status"] == "success"
    assert compute_calls == [True]
    cache.get.assert_called_once()
    if failing_operation == "get":
        cache.store.assert_not_called()
    else:
        cache.store.assert_called_once()


@pytest.mark.parametrize("failure", ["runtime_echo", "semantic_gate"])
def test_prepared_cache_miss_refuses_store_before_authority_and_semantic_acceptance(tmp_path, failure: str) -> None:
    from transformation_portal.depth.backends.protocol import DepthResult
    from transformation_portal.lux_depth_v3.execution_evidence import ExecutionEvidenceError
    from transformation_portal.lux_depth_v3.orchestrator import ApexStrictGateError, EnhanceOrchestrator

    prepared, image, evidence = _prepared_da2_evidence(tmp_path)
    native_depth = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    runtime_echo = "b" * 64 if failure == "runtime_echo" else evidence.runtime_identity_sha256

    class Backend:
        name = "da2"
        license_type = SimpleNamespace(value="commercial")

        @staticmethod
        def ensure_available() -> None:
            return None

        @staticmethod
        def prepare_cache_runtime_identity(**_kwargs):
            return evidence

        @staticmethod
        def verify_prepared_cache_runtime_identity(*, runtime_identity_sha256: str) -> bool:
            return runtime_identity_sha256 == evidence.runtime_identity_sha256

        @staticmethod
        def compute(_image):
            return DepthResult(
                depth_map=native_depth,
                original_image=np.zeros((64, 64, 3), dtype=np.uint8),
                metadata={"runtime_identity_sha256": runtime_echo},
                depth_units="relative",
                backend_id="da2",
                device="cpu",
                dtype="float32",
                input_size=(64, 64),
            )

    backend = Backend()
    registry = Mock()
    registry.get_backend.return_value = backend
    cache = Mock()
    cache.get.return_value = None

    def _gate(*_args, **_kwargs):
        if failure == "semantic_gate":
            raise ApexStrictGateError("APEX_TEST_REJECTED", "test semantic rejection")
        return {
            "passed": True,
            "failure_codes": [],
            "warnings": [],
            "metrics": {},
            "thresholds": {},
        }

    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry", return_value=registry):
        orchestrator = EnhanceOrchestrator.from_prepared(prepared, tmp_path / f"output-{failure}")
        orchestrator.depth_cache = cache
        orchestrator.postprocessor = Mock(process=lambda result: result)
        with patch.object(orchestrator, "_enforce_apex_depth_validity_gate", side_effect=_gate):
            with pytest.raises(ExecutionEvidenceError, match="failed required artifact accounting"):
                orchestrator.enhance_batch(
                    prepared.input_root,
                    input_files=list(prepared.input_files),
                )

    cache.get.assert_called_once()
    cache.store.assert_not_called()


def test_orchestrator_bypasses_missing_capability_and_rejects_bad_echo(tmp_path) -> None:
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    prepared, image, evidence = _prepared_da2_evidence(tmp_path)
    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator._prepared_execution = prepared

    assert (
        orchestrator._prepare_depth_cache_authority(
            backend=object(),
            backend_id="da2",
            image_path=image,
            input_content_sha256="a" * 64,
        )
        is None
    )
    assert (
        orchestrator._prepare_depth_cache_authority(
            backend=SimpleNamespace(prepare_cache_runtime_identity=lambda **_kwargs: evidence),
            backend_id="da2",
            image_path=image,
            input_content_sha256="a" * 64,
        )
        is None
    )

    backend = SimpleNamespace(
        prepare_cache_runtime_identity=lambda **_kwargs: evidence,
        verify_prepared_cache_runtime_identity=lambda **kwargs: (
            kwargs == {"runtime_identity_sha256": evidence.runtime_identity_sha256}
        ),
    )
    authority = orchestrator._prepare_depth_cache_authority(
        backend=backend,
        backend_id="da2",
        image_path=image,
        input_content_sha256="a" * 64,
    )
    assert authority is not None
    with pytest.raises(LuxExecutionPlanAuthorityError, match="did not echo"):
        orchestrator._verify_depth_cache_runtime_echo(
            SimpleNamespace(metadata={"runtime_identity_sha256": "b" * 64}),
            authority,
            backend_id="da2",
        )


def test_prepared_cache_enabled_run_rejects_legacy_manifest_depth_resume(tmp_path) -> None:
    from transformation_portal.lux_depth_v3.input_manager import ImageInput
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    prepared, image, _evidence = _prepared_da2_evidence(tmp_path)
    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator._prepared_execution = prepared
    orchestrator.depth_cache = Mock()

    assert orchestrator._authorize_legacy_depth_resume(True) is False
    assert orchestrator._authorize_legacy_depth_resume(False) is False

    orchestrator.config = prepared.runtime_config
    orchestrator.depth_dir = tmp_path / "depth"
    orchestrator.manifests_dir = tmp_path / "manifests"
    orchestrator.should_skip_depth = Mock(return_value=True)
    preprocessed = orchestrator._preprocess_single(
        ImageInput(path=image),
        prepared.input_root,
    )
    assert preprocessed["should_skip"] is False

    orchestrator.depth_cache = None
    assert orchestrator._authorize_legacy_depth_resume(True) is True

    orchestrator._prepared_execution = None
    orchestrator.depth_cache = Mock()
    assert orchestrator._authorize_legacy_depth_resume(True) is True
