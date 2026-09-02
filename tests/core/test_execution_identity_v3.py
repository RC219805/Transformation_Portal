"""Focused contract tests for the inert ExecutionIdentity v3 seed."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace

import pytest

from tests.core.test_execution_plan import _refingerprint, _valid_payload, _with_backend_shape
from transformation_portal.core.execution_identity_v3 import (
    EXECUTION_IDENTITY_V3_INCOMPLETE,
    EXECUTION_IDENTITY_V3_SCHEMA,
    ExecutionIdentityV3,
    ExecutionIdentityV3SeedError,
    IncompleteExecutionIdentityV3Error,
)
from transformation_portal.core.execution_plan import CanonicalExecutionPlan

pytestmark = pytest.mark.unit


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
