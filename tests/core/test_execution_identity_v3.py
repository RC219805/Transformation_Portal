"""Focused contract tests for the inert ExecutionIdentity v3 seed."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import FrozenInstanceError, replace

import pytest

from tests.core.test_execution_plan import _refingerprint, _valid_payload, _with_backend_shape
from transformation_portal.core.execution_identity_v3 import (
    EXECUTION_IDENTITY_V3_INCOMPLETE,
    EXECUTION_IDENTITY_V3_MATERIALIZED,
    EXECUTION_IDENTITY_V3_SCHEMA,
    BackendRuntimeIdentity,
    ExecutionIdentityV3,
    ExecutionIdentityV3SeedError,
    IncompleteExecutionIdentityV3Error,
    MaterializedExecutionIdentityV3,
    MaterializedExecutionIdentityV3Error,
)
from transformation_portal.core.execution_plan import CanonicalExecutionPlan
from transformation_portal.ingest.canonical_json import canonicalize_json

pytestmark = pytest.mark.unit

_INPUT_SHA256 = "a" * 64
_WEIGHTS_SHA256 = "b" * 64
_DEPENDENCY_SHA256 = "c" * 64
_INTERPRETER_SHA256 = "d" * 64
_PLATFORM_SHA256 = "e" * 64
_ACCELERATOR_SHA256 = "f" * 64
_SOURCE_SHA256 = "1" * 64


def _synthetic_plan() -> CanonicalExecutionPlan:
    return CanonicalExecutionPlan.from_payload(_valid_payload())


def _seed(plan: CanonicalExecutionPlan, **overrides) -> ExecutionIdentityV3:
    arguments = {
        "stage_node_id": "lux.depth",
        "candidate_id": plan.candidate_fallback_chain[0],
        "input_id": "input-000001",
    }
    arguments.update(overrides)
    return ExecutionIdentityV3.from_plan(plan, **arguments)


def _runtime_identity(seed: ExecutionIdentityV3, **overrides: str) -> BackendRuntimeIdentity:
    arguments = {
        "materialized_weights_sha256": seed.planned_model_artifact_sha256 or _WEIGHTS_SHA256,
        "dependency_lock_sha256": _DEPENDENCY_SHA256,
        "interpreter_identity_sha256": _INTERPRETER_SHA256,
        "platform_identity_sha256": _PLATFORM_SHA256,
        "accelerator_identity_sha256": _ACCELERATOR_SHA256,
        "source_identity_sha256": _SOURCE_SHA256,
    }
    arguments.update(overrides)
    return BackendRuntimeIdentity.from_seed(seed, **arguments)


def _runtime_aggregate(runtimes: tuple[BackendRuntimeIdentity, ...], field_name: str) -> str:
    if len(runtimes) == 1:
        return getattr(runtimes[0], field_name)
    projection = [
        {
            "constituent_ordinal": runtime.constituent_ordinal,
            "backend_id": runtime.backend_id,
            "model_canonical_key": runtime.model_canonical_key,
            "model_lock_revision": runtime.model_lock_revision,
            field_name: getattr(runtime, field_name),
        }
        for runtime in runtimes
    ]
    digest = hashlib.sha256()
    digest.update(b"tp.execution.runtime-aggregate.v1\0")
    digest.update(field_name.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonicalize_json(projection))
    return digest.hexdigest()


def _materialized(
    plan: CanonicalExecutionPlan,
    *,
    input_content_sha256: str = _INPUT_SHA256,
    runtime_overrides: dict[str, str] | None = None,
) -> MaterializedExecutionIdentityV3:
    candidate_id = plan.candidate_fallback_chain[0]
    candidate = plan.backend_candidates[0]
    enabled = [contract for contract in candidate.model_contracts if contract.enabled]
    runtimes = []
    for contract in enabled:
        seed = _seed(
            plan,
            model_backend_id=contract.backend_id if len(enabled) > 1 else None,
        )
        runtimes.append(_runtime_identity(seed, **(runtime_overrides or {})))
    runtime_tuple = tuple(runtimes)
    return MaterializedExecutionIdentityV3.from_plan(
        plan,
        stage_node_id="lux.depth",
        candidate_id=candidate_id,
        input_id="input-000001",
        executed_backend=candidate_id,
        input_content_sha256=input_content_sha256,
        backend_runtime_identities=runtime_tuple,
        dependency_lock_sha256=_runtime_aggregate(runtime_tuple, "dependency_lock_sha256"),
        interpreter_identity_sha256=_runtime_aggregate(runtime_tuple, "interpreter_identity_sha256"),
        platform_identity_sha256=_runtime_aggregate(runtime_tuple, "platform_identity_sha256"),
        accelerator_identity_sha256=_runtime_aggregate(runtime_tuple, "accelerator_identity_sha256"),
        source_identity_sha256=_runtime_aggregate(runtime_tuple, "source_identity_sha256"),
    )


def _reparse_materialized(
    plan: CanonicalExecutionPlan,
    payload: dict[str, object],
    *,
    input_content_sha256: str = _INPUT_SHA256,
) -> MaterializedExecutionIdentityV3:
    candidate_id = plan.candidate_fallback_chain[0]
    return MaterializedExecutionIdentityV3.from_payload(
        payload,
        expected_plan=plan,
        expected_stage_node_id="lux.depth",
        expected_candidate_id=candidate_id,
        expected_input_id="input-000001",
        expected_executed_backend=candidate_id,
        expected_input_content_sha256=input_content_sha256,
    )


def test_seed_is_immutable_deterministic_and_explicitly_incomplete() -> None:
    plan = _synthetic_plan()

    first = _seed(plan)
    second = _seed(plan)

    assert first == second
    assert first.schema == EXECUTION_IDENTITY_V3_SCHEMA
    assert first.completeness == EXECUTION_IDENTITY_V3_INCOMPLETE
    assert first.cacheable is False
    assert first.plan_schema == plan.schema
    assert first.plan_fingerprint_sha256 == plan.plan_fingerprint_sha256
    assert first.stage_node_id == "lux.depth"
    assert first.candidate_id == "synthetic"
    assert first.model_backend_id is None
    assert first.config_fingerprint_sha256 == plan.config_fingerprint_sha256
    assert first.input_relative_path == "a.jpg"
    assert first.to_canonical_bytes() == second.to_canonical_bytes()
    assert first.seed_sha256() == second.seed_sha256()
    assert len(first.stage_configuration_sha256) == 64
    assert len(first.planned_model_license_contract_sha256) == 64
    assert set(first.to_payload()["materialized"]) == set(first.missing_authority_fields)
    assert all(value is None for value in first.to_payload()["materialized"].values())

    with pytest.raises(FrozenInstanceError):
        first.candidate_id = "forged"  # type: ignore[misc]


def test_cache_key_fails_closed_without_importing_or_calling_a_cache() -> None:
    seed = _seed(_synthetic_plan())

    with pytest.raises(IncompleteExecutionIdentityV3Error, match="incomplete and non-cacheable"):
        seed.cache_key()


def test_core_root_exports_v3_without_replacing_legacy_identity() -> None:
    from transformation_portal import core

    assert core.ExecutionIdentityV3 is ExecutionIdentityV3
    assert core.ExecutionIdentity is not ExecutionIdentityV3
    assert core.EXECUTION_IDENTITY_V3_SCHEMA == EXECUTION_IDENTITY_V3_SCHEMA


def test_seed_rejects_structural_legacy_plan() -> None:
    payload = _valid_payload()
    payload["configuration_completeness"] = "structural_legacy"
    for node in payload["nodes"]:
        node["configuration"]["configuration_completeness"] = "structural_legacy"
    plan = CanonicalExecutionPlan.from_payload(_refingerprint(payload))

    with pytest.raises(ExecutionIdentityV3SeedError, match="execution_complete"):
        _seed(plan)


def test_seed_revalidates_fingerprint_before_reading_plan_authority() -> None:
    plan = _synthetic_plan()
    forged = replace(plan, quality_tier="premium")

    with pytest.raises(ExecutionIdentityV3SeedError, match="canonically validated"):
        _seed(forged)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({"stage_node_id": "lux.unknown"}, "stage_node_id"),
        ({"candidate_id": "unknown"}, "candidate_id"),
        ({"input_id": "input-999999"}, "input_id"),
        ({"model_backend_id": "da3"}, "no enabled model constituent"),
    ),
)
def test_seed_rejects_unknown_or_inapplicable_ids(overrides: dict[str, str], expected: str) -> None:
    with pytest.raises(ExecutionIdentityV3SeedError, match=expected):
        _seed(_synthetic_plan(), **overrides)


def test_single_model_candidate_derives_and_validates_exact_backend_id() -> None:
    plan = CanonicalExecutionPlan.from_payload(_with_backend_shape(_valid_payload(), ["da3"]))

    inferred = _seed(plan)
    explicit = _seed(plan, model_backend_id="da3")

    assert inferred.model_backend_id == "da3"
    assert inferred.constituent_ordinal == 0
    assert inferred.model_canonical_key == "da3_metric"
    assert inferred.model_lock_revision == "1" * 40
    assert inferred.planned_model_artifact_sha256 is None
    assert inferred == explicit
    with pytest.raises(ExecutionIdentityV3SeedError, match="does not carry"):
        _seed(plan, model_backend_id="depth_pro")


def test_multi_model_candidate_requires_and_binds_exact_constituent() -> None:
    plan = CanonicalExecutionPlan.from_payload(_with_backend_shape(_valid_payload(), ["ensemble"]))

    with pytest.raises(ExecutionIdentityV3SeedError, match="requires an exact model_backend_id"):
        _seed(plan)

    depth_pro = _seed(plan, model_backend_id="depth_pro")
    da3 = _seed(plan, model_backend_id="da3")

    assert depth_pro.model_backend_id == "depth_pro"
    assert depth_pro.constituent_ordinal == 0
    assert da3.model_backend_id == "da3"
    assert da3.constituent_ordinal == 1
    assert depth_pro.planned_model_license_contract_sha256 != da3.planned_model_license_contract_sha256
    assert depth_pro.to_canonical_bytes() != da3.to_canonical_bytes()
    with pytest.raises(ExecutionIdentityV3SeedError, match="no unique enabled model_backend_id"):
        _seed(plan, model_backend_id="depthcrafter")


def test_multi_model_candidate_rejects_disabled_constituent() -> None:
    payload = _with_backend_shape(_valid_payload(), ["ensemble"])
    contracts = payload["backend_candidates"][0]["model_contracts"]
    disabled = copy.deepcopy(contracts[0])
    disabled["backend_id"] = "depthcrafter"
    disabled["model"]["requested_selector"] = "backend:depthcrafter"
    disabled["model"]["canonical_key"] = "depthcrafter"
    disabled["model"]["repo_id"] = "depthcrafter/repository"
    disabled["enabled"] = False
    disabled["weight"] = 0.2
    contracts.insert(0, disabled)
    plan = CanonicalExecutionPlan.from_payload(_refingerprint(payload))

    with pytest.raises(ExecutionIdentityV3SeedError, match="no unique enabled model_backend_id"):
        _seed(plan, model_backend_id="depthcrafter")

    selected = _seed(plan, model_backend_id="depth_pro")
    assert selected.constituent_ordinal == 1


def test_seed_digest_changes_with_selected_stage_configuration() -> None:
    base_payload = _valid_payload()
    changed_payload = copy.deepcopy(base_payload)
    changed_payload["nodes"][1]["configuration"]["verify_writes"] = False
    changed_plan = CanonicalExecutionPlan.from_payload(_refingerprint(changed_payload))

    assert _seed(_synthetic_plan()).stage_configuration_sha256 != _seed(changed_plan).stage_configuration_sha256


def test_single_model_materialization_is_closed_deterministic_and_plan_bound() -> None:
    plan = CanonicalExecutionPlan.from_payload(_with_backend_shape(_valid_payload(), ["da3"]))

    first = _materialized(plan)
    second = _materialized(plan)
    payload = first.to_payload()

    assert first == second
    assert first.completeness == EXECUTION_IDENTITY_V3_MATERIALIZED
    assert first.cacheable is True
    assert first.config_fingerprint_sha256 == plan.config_fingerprint_sha256
    assert first.model_canonical_key == "da3_metric"
    assert first.model_lock_revision == "1" * 40
    assert first.materialized_weights_sha256 == _WEIGHTS_SHA256
    assert first.dependency_lock_sha256 == _DEPENDENCY_SHA256
    assert payload["execution_identity_sha256"] == first.execution_identity_sha256
    assert payload["model_constituents"][0]["model_canonical_key"] == "da3_metric"
    assert len(first.execution_identity_sha256) == 64
    assert first.to_canonical_bytes() == second.to_canonical_bytes()
    assert _reparse_materialized(plan, payload) == first

    with pytest.raises(FrozenInstanceError):
        first.candidate_id = "forged"  # type: ignore[misc]

    with pytest.raises(MaterializedExecutionIdentityV3Error, match="factory-only"):
        MaterializedExecutionIdentityV3()


def test_core_root_exports_materialized_identity_without_replacing_legacy_or_seed() -> None:
    from transformation_portal import core

    assert core.BackendRuntimeIdentity is BackendRuntimeIdentity
    assert core.MaterializedExecutionIdentityV3 is MaterializedExecutionIdentityV3
    assert core.ExecutionIdentityV3 is ExecutionIdentityV3
    assert core.ExecutionIdentity is not MaterializedExecutionIdentityV3


def test_cache_key_is_domain_separated_by_cache_schema_and_complete_identity() -> None:
    plan = CanonicalExecutionPlan.from_payload(_with_backend_shape(_valid_payload(), ["da3"]))
    identity = _materialized(plan)
    changed_input = _materialized(plan, input_content_sha256="2" * 64)

    assert identity.cache_key("tp.lux.depth-cache.v3") == identity.cache_key("tp.lux.depth-cache.v3")
    assert identity.cache_key("tp.lux.depth-cache.v3") != identity.cache_key("tp.lux.depth-cache.v4")
    assert identity.cache_key("tp.lux.depth-cache.v3") != changed_input.cache_key("tp.lux.depth-cache.v3")
    assert identity.execution_identity_sha256 == "c3b366477d0da830738b6acbe892c1d5dacd2e6da9759535a631c2a4308b327d"
    assert identity.cache_key("tp.lux.depth-cache.v3") == "bbea4bb492e5826cfc5be7ca439bdc662ea6c7029bf83e5983bbd3839bc400e4"
    with pytest.raises(MaterializedExecutionIdentityV3Error, match="cache_schema"):
        identity.cache_key("not canonical schema")


def test_runtime_digest_changes_invalidate_execution_identity() -> None:
    plan = CanonicalExecutionPlan.from_payload(_with_backend_shape(_valid_payload(), ["da3"]))
    baseline = _materialized(plan)
    changed_weights = _materialized(plan, runtime_overrides={"materialized_weights_sha256": "3" * 64})
    changed_dependency = _materialized(plan, runtime_overrides={"dependency_lock_sha256": "4" * 64})
    changed_source = _materialized(plan, runtime_overrides={"source_identity_sha256": "5" * 64})

    assert (
        len(
            {
                baseline.execution_identity_sha256,
                changed_weights.execution_identity_sha256,
                changed_dependency.execution_identity_sha256,
                changed_source.execution_identity_sha256,
            }
        )
        == 4
    )


def test_model_lock_revision_change_invalidates_execution_identity() -> None:
    base_payload = _with_backend_shape(_valid_payload(), ["da3"])
    changed_payload = copy.deepcopy(base_payload)
    changed_revision = "9" * 40
    changed_payload["backend_candidates"][0]["model_contracts"][0]["model"]["revision"] = changed_revision
    changed_payload["resolved_model"]["revision"] = changed_revision
    changed_payload["nodes"][1]["configuration"]["resolved_model_revision"] = changed_revision
    changed_payload = _refingerprint(changed_payload)

    baseline = _materialized(CanonicalExecutionPlan.from_payload(base_payload))
    changed = _materialized(CanonicalExecutionPlan.from_payload(changed_payload))

    assert changed.model_lock_revision == changed_revision
    assert changed.execution_identity_sha256 != baseline.execution_identity_sha256


def test_ensemble_requires_every_enabled_constituent_in_plan_order() -> None:
    plan = CanonicalExecutionPlan.from_payload(_with_backend_shape(_valid_payload(), ["ensemble"]))
    identity = _materialized(plan)

    assert identity.model_canonical_key is None
    assert identity.model_lock_revision is None
    assert [item.backend_id for item in identity.model_constituents] == ["depth_pro", "da3"]
    assert [item.constituent_ordinal for item in identity.model_constituents] == [0, 1]
    assert identity.materialized_weights_sha256 not in {
        item.materialized_weights_sha256 for item in identity.model_constituents
    }

    with pytest.raises(MaterializedExecutionIdentityV3Error, match="all and only"):
        MaterializedExecutionIdentityV3.from_plan(
            plan,
            stage_node_id="lux.depth",
            candidate_id="ensemble",
            input_id="input-000001",
            executed_backend="ensemble",
            input_content_sha256=_INPUT_SHA256,
            backend_runtime_identities=identity.model_constituents[:1],
            dependency_lock_sha256=identity.dependency_lock_sha256,
            interpreter_identity_sha256=identity.interpreter_identity_sha256,
            platform_identity_sha256=identity.platform_identity_sha256,
            accelerator_identity_sha256=identity.accelerator_identity_sha256,
            source_identity_sha256=identity.source_identity_sha256,
        )
    with pytest.raises(MaterializedExecutionIdentityV3Error, match="exact planned constituent"):
        MaterializedExecutionIdentityV3.from_plan(
            plan,
            stage_node_id="lux.depth",
            candidate_id="ensemble",
            input_id="input-000001",
            executed_backend="ensemble",
            input_content_sha256=_INPUT_SHA256,
            backend_runtime_identities=tuple(reversed(identity.model_constituents)),
            dependency_lock_sha256=identity.dependency_lock_sha256,
            interpreter_identity_sha256=identity.interpreter_identity_sha256,
            platform_identity_sha256=identity.platform_identity_sha256,
            accelerator_identity_sha256=identity.accelerator_identity_sha256,
            source_identity_sha256=identity.source_identity_sha256,
        )


def test_disabled_ensemble_constituent_is_excluded_without_renumbering() -> None:
    payload = _with_backend_shape(_valid_payload(), ["ensemble"])
    disabled = copy.deepcopy(payload["backend_candidates"][0]["model_contracts"][0])
    disabled["backend_id"] = "depthcrafter"
    disabled["model"]["requested_selector"] = "backend:depthcrafter"
    disabled["model"]["canonical_key"] = "depthcrafter"
    disabled["model"]["repo_id"] = "depthcrafter/repository"
    disabled["model"]["revision"] = "8" * 40
    disabled["artifact_path"] = None
    disabled["artifact_sha256"] = None
    disabled["enabled"] = False
    payload["backend_candidates"][0]["model_contracts"].insert(0, disabled)
    plan = CanonicalExecutionPlan.from_payload(_refingerprint(payload))

    identity = _materialized(plan)

    assert [item.backend_id for item in identity.model_constituents] == ["depth_pro", "da3"]
    assert [item.constituent_ordinal for item in identity.model_constituents] == [1, 2]


@pytest.mark.parametrize(
    ("field_name", "value", "expected"),
    (
        ("materialized_weights_sha256", "0" * 64, "non-placeholder"),
        ("dependency_lock_sha256", "sha256:" + "a" * 64, "lowercase SHA-256"),
        ("model_lock_revision", "main", "pinned lowercase"),
    ),
)
def test_backend_runtime_identity_rejects_placeholder_or_unpinned_evidence(
    field_name: str,
    value: str,
    expected: str,
) -> None:
    plan = CanonicalExecutionPlan.from_payload(_with_backend_shape(_valid_payload(), ["da3"]))
    seed = _seed(plan)
    runtime = _runtime_identity(seed)

    with pytest.raises(MaterializedExecutionIdentityV3Error, match=expected):
        replace(runtime, **{field_name: value})


def test_plan_bound_artifact_requires_exact_materialized_weight_digest() -> None:
    plan = CanonicalExecutionPlan.from_payload(_with_backend_shape(_valid_payload(), ["depth_pro"]))
    seed = _seed(plan)

    assert seed.model_lock_revision is None
    assert seed.planned_model_artifact_sha256 == "4" * 64
    with pytest.raises(MaterializedExecutionIdentityV3Error, match="plan-bound model artifact"):
        _runtime_identity(seed, materialized_weights_sha256="5" * 64)


def test_synthetic_seed_remains_non_cacheable_after_materialized_contract_exists() -> None:
    plan = _synthetic_plan()

    with pytest.raises(MaterializedExecutionIdentityV3Error, match="model-less candidate"):
        MaterializedExecutionIdentityV3.from_plan(
            plan,
            stage_node_id="lux.depth",
            candidate_id="synthetic",
            input_id="input-000001",
            executed_backend="synthetic",
            input_content_sha256=_INPUT_SHA256,
            backend_runtime_identities=(),
            dependency_lock_sha256=_DEPENDENCY_SHA256,
            interpreter_identity_sha256=_INTERPRETER_SHA256,
            platform_identity_sha256=_PLATFORM_SHA256,
            accelerator_identity_sha256=_ACCELERATOR_SHA256,
            source_identity_sha256=_SOURCE_SHA256,
        )


def test_materialized_payload_rejects_extra_missing_or_forged_fields() -> None:
    plan = CanonicalExecutionPlan.from_payload(_with_backend_shape(_valid_payload(), ["da3"]))
    identity = _materialized(plan)

    extra = identity.to_payload()
    extra["unexpected"] = True
    with pytest.raises(MaterializedExecutionIdentityV3Error, match="non-canonical field set"):
        _reparse_materialized(plan, extra)

    missing = identity.to_payload()
    missing.pop("dependency_lock_sha256")
    with pytest.raises(MaterializedExecutionIdentityV3Error, match="non-canonical field set"):
        _reparse_materialized(plan, missing)

    forged = identity.to_payload()
    forged["execution_identity_sha256"] = "6" * 64
    with pytest.raises(MaterializedExecutionIdentityV3Error, match="does not match"):
        _reparse_materialized(plan, forged)

    inconsistent = identity.to_payload()
    inconsistent["materialized_weights_sha256"] = "7" * 64
    with pytest.raises(MaterializedExecutionIdentityV3Error, match="inconsistent derived fields"):
        _reparse_materialized(plan, inconsistent)


def test_materialized_payload_cannot_choose_its_own_plan_or_input_authority() -> None:
    plan = CanonicalExecutionPlan.from_payload(_with_backend_shape(_valid_payload(), ["da3"]))
    payload = _materialized(plan).to_payload()
    changed_payload = _with_backend_shape(_valid_payload(), ["da3"])
    changed_payload["nodes"][1]["configuration"]["verify_writes"] = False
    changed_plan = CanonicalExecutionPlan.from_payload(_refingerprint(changed_payload))

    with pytest.raises(MaterializedExecutionIdentityV3Error, match="inconsistent derived fields"):
        _reparse_materialized(changed_plan, payload)
    with pytest.raises(MaterializedExecutionIdentityV3Error, match="inconsistent derived fields"):
        _reparse_materialized(plan, payload, input_content_sha256="2" * 64)
