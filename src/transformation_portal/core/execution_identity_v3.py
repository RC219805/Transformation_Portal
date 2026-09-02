"""Complete, materialized ExecutionIdentity v3 contracts.

``ExecutionIdentityV3`` remains the inert plan-time seed introduced by
#2065. It cannot authorize cache access. ``MaterializedExecutionIdentityV3``
is the distinct cache-authorizing type: it can only be built by revalidating an
execution-complete plan and matching complete runtime evidence for every
enabled model constituent.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from ..ingest.canonical_json import TP_CANONICAL_JSON_PROFILE, canonicalize_json
from .execution_plan import (
    EXECUTION_COMPLETE,
    EXECUTION_PLAN_SCHEMA,
    CanonicalExecutionPlan,
    ExecutionPlanError,
)

EXECUTION_IDENTITY_V3_SCHEMA = "tp.execution.identity.v3"
EXECUTION_IDENTITY_V3_INCOMPLETE = "incomplete_seed"
EXECUTION_IDENTITY_V3_MATERIALIZED = "materialized"

_EXECUTION_IDENTITY_HASH_DOMAIN = b"tp.execution.identity.v3\x00"
_EXECUTION_CACHE_KEY_HASH_DOMAIN = b"tp.execution.cache-key.v1\x00"
_RUNTIME_AGGREGATE_HASH_DOMAIN = b"tp.execution.runtime-aggregate.v1\x00"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PINNED_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CACHE_SCHEMA_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_ZERO_SHA256 = "0" * 64

_MISSING_AUTHORITY_FIELDS = (
    "executed_backend",
    "input_content_sha256",
    "backend_runtime_identities",
    "materialized_weights_sha256",
    "dependency_lock_sha256",
    "interpreter_identity_sha256",
    "platform_identity_sha256",
    "accelerator_identity_sha256",
    "source_identity_sha256",
)

_BACKEND_RUNTIME_PAYLOAD_KEYS = frozenset(
    {
        "constituent_ordinal",
        "backend_id",
        "model_canonical_key",
        "model_lock_revision",
        "planned_model_artifact_sha256",
        "planned_model_license_contract_sha256",
        "materialized_weights_sha256",
        "dependency_lock_sha256",
        "interpreter_identity_sha256",
        "platform_identity_sha256",
        "accelerator_identity_sha256",
        "source_identity_sha256",
    }
)

_MATERIALIZED_BODY_KEYS = frozenset(
    {
        "schema",
        "canonicalization",
        "completeness",
        "cacheable",
        "plan_schema",
        "plan_fingerprint_sha256",
        "config_fingerprint_sha256",
        "stage_node_id",
        "stage_registry_id",
        "stage_configuration_sha256",
        "candidate_id",
        "candidate_ordinal",
        "executed_backend",
        "model_canonical_key",
        "model_lock_revision",
        "input_id",
        "input_relative_path",
        "input_content_sha256",
        "model_constituents",
        "materialized_weights_sha256",
        "dependency_lock_sha256",
        "interpreter_identity_sha256",
        "platform_identity_sha256",
        "accelerator_identity_sha256",
        "source_identity_sha256",
    }
)
_MATERIALIZED_PAYLOAD_KEYS = _MATERIALIZED_BODY_KEYS | {"execution_identity_sha256"}


class ExecutionIdentityV3SeedError(ValueError):
    """An identity-v3 seed cannot be selected safely from the plan."""


class IncompleteExecutionIdentityV3Error(RuntimeError):
    """An incomplete seed was asked to authorize cache access."""


class MaterializedExecutionIdentityV3Error(ValueError):
    """Runtime evidence cannot authorize a complete identity-v3 value."""


def _require_exact_keys(payload: Mapping[str, Any], expected: frozenset[str], *, context: str) -> None:
    if not isinstance(payload, Mapping):
        raise MaterializedExecutionIdentityV3Error(f"{context} must be an object")
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise MaterializedExecutionIdentityV3Error(
            f"{context} has a non-canonical field set (missing={missing}, extra={extra})"
        )


def _require_sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None or value == _ZERO_SHA256:
        raise MaterializedExecutionIdentityV3Error(f"{field_name} must be a non-placeholder lowercase SHA-256 digest")
    return value


def _require_optional_sha256(value: object, *, field_name: str) -> Optional[str]:
    if value is None:
        return None
    return _require_sha256(value, field_name=field_name)


def _require_text(value: object, *, field_name: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value or value != value.strip():
        raise MaterializedExecutionIdentityV3Error(f"{field_name} must be a bounded canonical string")
    return value


def _require_identifier(value: object, *, field_name: str) -> str:
    text = _require_text(value, field_name=field_name, maximum=128)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        raise MaterializedExecutionIdentityV3Error(f"{field_name} is not a canonical identifier")
    return text


def _require_ordinal(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 255:
        raise MaterializedExecutionIdentityV3Error(f"{field_name} must be an integer between 0 and 255")
    return value


def _require_model_revision(value: object, *, artifact_sha256: Optional[str]) -> Optional[str]:
    if value is None:
        if artifact_sha256 is None:
            raise MaterializedExecutionIdentityV3Error("model_lock_revision may be null only for a plan-bound model artifact")
        return None
    if not isinstance(value, str) or _PINNED_REVISION_RE.fullmatch(value) is None:
        raise MaterializedExecutionIdentityV3Error("model_lock_revision must be a pinned lowercase 40-character commit SHA")
    return value


def _validated_execution_complete_plan(plan: CanonicalExecutionPlan) -> CanonicalExecutionPlan:
    if not isinstance(plan, CanonicalExecutionPlan):
        raise TypeError("plan must be a CanonicalExecutionPlan")
    try:
        validated = CanonicalExecutionPlan.from_payload(plan.to_payload())
    except ExecutionPlanError as exc:
        raise ExecutionIdentityV3SeedError("ExecutionIdentity v3 requires a canonically validated execution plan") from exc
    if validated.configuration_completeness != EXECUTION_COMPLETE:
        raise ExecutionIdentityV3SeedError("ExecutionIdentity v3 requires an execution_complete plan")
    return validated


def _planned_model_contract_sha256(
    plan: CanonicalExecutionPlan,
    *,
    candidate_id: str,
    model_backend_id: Optional[str],
    selected_contract: Optional[Any],
) -> str:
    payload = {
        "candidate_id": candidate_id,
        "model_backend_id": model_backend_id,
        "model_contract": None if selected_contract is None else selected_contract.to_payload(),
        "resolved_model": None if plan.resolved_model is None else plan.resolved_model.to_payload(),
        "license_acknowledgements": plan.license_acknowledgements.to_payload(),
        "license_evaluation": plan.license_evaluation.to_payload(),
    }
    return hashlib.sha256(canonicalize_json(payload)).hexdigest()


@dataclass(frozen=True)
class ExecutionIdentityV3:
    """Immutable, deterministic, and explicitly non-cacheable plan seed."""

    plan_schema: str
    plan_fingerprint_sha256: str
    stage_node_id: str
    stage_registry_id: str
    stage_configuration_sha256: str
    candidate_id: str
    candidate_ordinal: int
    model_backend_id: Optional[str]
    constituent_ordinal: Optional[int]
    planned_model_license_contract_sha256: str
    input_id: str
    input_relative_path: str
    # Optional defaults preserve direct-construction compatibility. from_plan
    # always populates these, and materialization rejects their absence.
    config_fingerprint_sha256: Optional[str] = None
    model_canonical_key: Optional[str] = None
    model_lock_revision: Optional[str] = None
    planned_model_artifact_sha256: Optional[str] = None
    schema: str = field(default=EXECUTION_IDENTITY_V3_SCHEMA, init=False)
    completeness: str = field(default=EXECUTION_IDENTITY_V3_INCOMPLETE, init=False)
    cacheable: bool = field(default=False, init=False)
    executed_backend: None = field(default=None, init=False)
    input_content_sha256: None = field(default=None, init=False)
    materialized_weights_sha256: None = field(default=None, init=False)
    dependency_lock_sha256: None = field(default=None, init=False)
    interpreter_identity: None = field(default=None, init=False)
    platform_identity: None = field(default=None, init=False)
    accelerator_identity: None = field(default=None, init=False)
    source_identity: None = field(default=None, init=False)

    @classmethod
    def from_plan(
        cls,
        plan: CanonicalExecutionPlan,
        *,
        stage_node_id: str,
        candidate_id: str,
        input_id: str,
        model_backend_id: Optional[str] = None,
    ) -> "ExecutionIdentityV3":
        """Select one exact stage, candidate/constituent, and planned input."""

        plan = _validated_execution_complete_plan(plan)
        node = next((item for item in plan.nodes if item.node_id == stage_node_id), None)
        if node is None:
            raise ExecutionIdentityV3SeedError(f"Unknown stage_node_id: {stage_node_id!r}")
        candidate_ordinal = next(
            (index for index, item in enumerate(plan.backend_candidates) if item.backend_id == candidate_id),
            None,
        )
        if candidate_ordinal is None:
            raise ExecutionIdentityV3SeedError(f"Unknown candidate_id: {candidate_id!r}")
        candidate = plan.backend_candidates[candidate_ordinal]
        planned_input = next((item for item in plan.inputs if item.input_id == input_id), None)
        if planned_input is None:
            raise ExecutionIdentityV3SeedError(f"Unknown input_id: {input_id!r}")

        indexed_contracts = tuple(
            (index, contract) for index, contract in enumerate(candidate.model_contracts) if contract.enabled
        )
        constituent_ordinal: Optional[int]
        selected_contract: Optional[Any]
        if not indexed_contracts:
            if model_backend_id is not None:
                raise ExecutionIdentityV3SeedError(
                    f"Candidate {candidate_id!r} has no enabled model constituent {model_backend_id!r}"
                )
            selected_contract = None
            constituent_ordinal = None
        elif len(indexed_contracts) == 1:
            constituent_ordinal, selected_contract = indexed_contracts[0]
            if model_backend_id is not None and model_backend_id != selected_contract.backend_id:
                raise ExecutionIdentityV3SeedError(
                    f"Candidate {candidate_id!r} does not carry model_backend_id {model_backend_id!r}"
                )
            model_backend_id = selected_contract.backend_id
        else:
            if model_backend_id is None:
                raise ExecutionIdentityV3SeedError(
                    f"Multi-model candidate {candidate_id!r} requires an exact model_backend_id"
                )
            matches = [(index, contract) for index, contract in indexed_contracts if contract.backend_id == model_backend_id]
            if len(matches) != 1:
                raise ExecutionIdentityV3SeedError(
                    f"Candidate {candidate_id!r} has no unique enabled model_backend_id {model_backend_id!r}"
                )
            constituent_ordinal, selected_contract = matches[0]

        stage_configuration_sha256 = hashlib.sha256(canonicalize_json(dict(node.configuration))).hexdigest()
        planned_contract_sha256 = _planned_model_contract_sha256(
            plan,
            candidate_id=candidate.backend_id,
            model_backend_id=model_backend_id,
            selected_contract=selected_contract,
        )
        return cls(
            plan_schema=plan.schema,
            plan_fingerprint_sha256=plan.plan_fingerprint_sha256,
            stage_node_id=node.node_id,
            stage_registry_id=node.stage_registry_id.value,
            stage_configuration_sha256=stage_configuration_sha256,
            candidate_id=candidate.backend_id,
            candidate_ordinal=candidate_ordinal,
            model_backend_id=model_backend_id,
            constituent_ordinal=constituent_ordinal,
            planned_model_license_contract_sha256=planned_contract_sha256,
            input_id=planned_input.input_id,
            input_relative_path=planned_input.path,
            config_fingerprint_sha256=plan.config_fingerprint_sha256,
            model_canonical_key=None if selected_contract is None else selected_contract.model.canonical_key,
            model_lock_revision=None if selected_contract is None else selected_contract.model.revision,
            planned_model_artifact_sha256=None if selected_contract is None else selected_contract.artifact_sha256,
        )

    @property
    def missing_authority_fields(self) -> tuple[str, ...]:
        return _MISSING_AUTHORITY_FIELDS

    def to_payload(self) -> dict[str, Any]:
        """Return the closed, truthful seed payload."""

        return {
            "schema": self.schema,
            "completeness": self.completeness,
            "cacheable": self.cacheable,
            "plan_schema": self.plan_schema,
            "plan_fingerprint_sha256": self.plan_fingerprint_sha256,
            "config_fingerprint_sha256": self.config_fingerprint_sha256,
            "stage_node_id": self.stage_node_id,
            "stage_registry_id": self.stage_registry_id,
            "stage_configuration_sha256": self.stage_configuration_sha256,
            "candidate_id": self.candidate_id,
            "candidate_ordinal": self.candidate_ordinal,
            "model_backend_id": self.model_backend_id,
            "constituent_ordinal": self.constituent_ordinal,
            "model_canonical_key": self.model_canonical_key,
            "model_lock_revision": self.model_lock_revision,
            "planned_model_artifact_sha256": self.planned_model_artifact_sha256,
            "planned_model_license_contract_sha256": self.planned_model_license_contract_sha256,
            "input_id": self.input_id,
            "input_relative_path": self.input_relative_path,
            "materialized": {field_name: None for field_name in self.missing_authority_fields},
            "missing_authority_fields": list(self.missing_authority_fields),
        }

    def to_canonical_bytes(self) -> bytes:
        return canonicalize_json(self.to_payload())

    def seed_sha256(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()

    def cache_key(self) -> str:
        raise IncompleteExecutionIdentityV3Error(
            "ExecutionIdentity v3 seed is incomplete and non-cacheable; use MaterializedExecutionIdentityV3"
        )


@dataclass(frozen=True)
class BackendRuntimeIdentity:
    """Closed runtime identity for one exact enabled model constituent."""

    constituent_ordinal: int
    backend_id: str
    model_canonical_key: str
    model_lock_revision: Optional[str]
    planned_model_artifact_sha256: Optional[str]
    planned_model_license_contract_sha256: str
    materialized_weights_sha256: str
    dependency_lock_sha256: str
    interpreter_identity_sha256: str
    platform_identity_sha256: str
    accelerator_identity_sha256: str
    source_identity_sha256: str

    def __post_init__(self) -> None:
        _require_ordinal(self.constituent_ordinal, field_name="constituent_ordinal")
        _require_identifier(self.backend_id, field_name="backend_id")
        _require_identifier(self.model_canonical_key, field_name="model_canonical_key")
        artifact_sha256 = _require_optional_sha256(
            self.planned_model_artifact_sha256,
            field_name="planned_model_artifact_sha256",
        )
        _require_model_revision(self.model_lock_revision, artifact_sha256=artifact_sha256)
        _require_sha256(
            self.planned_model_license_contract_sha256,
            field_name="planned_model_license_contract_sha256",
        )
        materialized_weights_sha256 = _require_sha256(
            self.materialized_weights_sha256,
            field_name="materialized_weights_sha256",
        )
        if artifact_sha256 is not None and materialized_weights_sha256 != artifact_sha256:
            raise MaterializedExecutionIdentityV3Error(
                "materialized_weights_sha256 does not match the plan-bound model artifact"
            )
        for field_name in (
            "dependency_lock_sha256",
            "interpreter_identity_sha256",
            "platform_identity_sha256",
            "accelerator_identity_sha256",
            "source_identity_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name=field_name)

    @classmethod
    def from_seed(
        cls,
        seed: ExecutionIdentityV3,
        *,
        materialized_weights_sha256: str,
        dependency_lock_sha256: str,
        interpreter_identity_sha256: str,
        platform_identity_sha256: str,
        accelerator_identity_sha256: str,
        source_identity_sha256: str,
    ) -> "BackendRuntimeIdentity":
        if not isinstance(seed, ExecutionIdentityV3):
            raise TypeError("seed must be an ExecutionIdentityV3")
        if seed.constituent_ordinal is None or seed.model_backend_id is None or seed.model_canonical_key is None:
            raise MaterializedExecutionIdentityV3Error(
                "A model-less or legacy-incomplete seed cannot produce backend runtime identity"
            )
        return cls(
            constituent_ordinal=seed.constituent_ordinal,
            backend_id=seed.model_backend_id,
            model_canonical_key=seed.model_canonical_key,
            model_lock_revision=seed.model_lock_revision,
            planned_model_artifact_sha256=seed.planned_model_artifact_sha256,
            planned_model_license_contract_sha256=seed.planned_model_license_contract_sha256,
            materialized_weights_sha256=materialized_weights_sha256,
            dependency_lock_sha256=dependency_lock_sha256,
            interpreter_identity_sha256=interpreter_identity_sha256,
            platform_identity_sha256=platform_identity_sha256,
            accelerator_identity_sha256=accelerator_identity_sha256,
            source_identity_sha256=source_identity_sha256,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BackendRuntimeIdentity":
        _require_exact_keys(payload, _BACKEND_RUNTIME_PAYLOAD_KEYS, context="backend runtime identity")
        return cls(**{key: payload[key] for key in _BACKEND_RUNTIME_PAYLOAD_KEYS})

    def to_payload(self) -> dict[str, Any]:
        return {
            "constituent_ordinal": self.constituent_ordinal,
            "backend_id": self.backend_id,
            "model_canonical_key": self.model_canonical_key,
            "model_lock_revision": self.model_lock_revision,
            "planned_model_artifact_sha256": self.planned_model_artifact_sha256,
            "planned_model_license_contract_sha256": self.planned_model_license_contract_sha256,
            "materialized_weights_sha256": self.materialized_weights_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "interpreter_identity_sha256": self.interpreter_identity_sha256,
            "platform_identity_sha256": self.platform_identity_sha256,
            "accelerator_identity_sha256": self.accelerator_identity_sha256,
            "source_identity_sha256": self.source_identity_sha256,
        }


def _runtime_aggregate(runtime_identities: tuple[BackendRuntimeIdentity, ...], field_name: str) -> str:
    if len(runtime_identities) == 1:
        return getattr(runtime_identities[0], field_name)
    projection = [
        {
            "constituent_ordinal": identity.constituent_ordinal,
            "backend_id": identity.backend_id,
            "model_canonical_key": identity.model_canonical_key,
            "model_lock_revision": identity.model_lock_revision,
            field_name: getattr(identity, field_name),
        }
        for identity in runtime_identities
    ]
    digest = hashlib.sha256()
    digest.update(_RUNTIME_AGGREGATE_HASH_DOMAIN)
    digest.update(field_name.encode("ascii"))
    digest.update(b"\x00")
    digest.update(canonicalize_json(projection))
    return digest.hexdigest()


@dataclass(frozen=True, init=False)
class MaterializedExecutionIdentityV3:
    """Complete cache-authorizing identity derived from plan plus runtime."""

    plan_schema: str
    plan_fingerprint_sha256: str
    config_fingerprint_sha256: str
    stage_node_id: str
    stage_registry_id: str
    stage_configuration_sha256: str
    candidate_id: str
    candidate_ordinal: int
    executed_backend: str
    input_id: str
    input_relative_path: str
    input_content_sha256: str
    backend_runtime_identities: tuple[BackendRuntimeIdentity, ...]
    schema: str = field(default=EXECUTION_IDENTITY_V3_SCHEMA, init=False)
    canonicalization: str = field(default=TP_CANONICAL_JSON_PROFILE, init=False)
    completeness: str = field(default=EXECUTION_IDENTITY_V3_MATERIALIZED, init=False)
    cacheable: bool = field(default=True, init=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise MaterializedExecutionIdentityV3Error(
            "MaterializedExecutionIdentityV3 is factory-only; use from_plan or plan-bound from_payload"
        )

    @classmethod
    def _from_validated_plan_values(
        cls,
        *,
        plan_schema: str,
        plan_fingerprint_sha256: str,
        config_fingerprint_sha256: str,
        stage_node_id: str,
        stage_registry_id: str,
        stage_configuration_sha256: str,
        candidate_id: str,
        candidate_ordinal: int,
        executed_backend: str,
        input_id: str,
        input_relative_path: str,
        input_content_sha256: str,
        backend_runtime_identities: tuple[BackendRuntimeIdentity, ...],
    ) -> "MaterializedExecutionIdentityV3":
        """Construct only after the public factories have rebound plan authority."""

        result = object.__new__(cls)
        values: dict[str, object] = {
            "plan_schema": plan_schema,
            "plan_fingerprint_sha256": plan_fingerprint_sha256,
            "config_fingerprint_sha256": config_fingerprint_sha256,
            "stage_node_id": stage_node_id,
            "stage_registry_id": stage_registry_id,
            "stage_configuration_sha256": stage_configuration_sha256,
            "candidate_id": candidate_id,
            "candidate_ordinal": candidate_ordinal,
            "executed_backend": executed_backend,
            "input_id": input_id,
            "input_relative_path": input_relative_path,
            "input_content_sha256": input_content_sha256,
            "backend_runtime_identities": backend_runtime_identities,
            "schema": EXECUTION_IDENTITY_V3_SCHEMA,
            "canonicalization": TP_CANONICAL_JSON_PROFILE,
            "completeness": EXECUTION_IDENTITY_V3_MATERIALIZED,
            "cacheable": True,
        }
        for field_name, value in values.items():
            object.__setattr__(result, field_name, value)
        result.__post_init__()
        return result

    def __post_init__(self) -> None:
        if self.plan_schema != EXECUTION_PLAN_SCHEMA:
            raise MaterializedExecutionIdentityV3Error("plan_schema is not the authoritative execution-plan schema")
        for field_name in (
            "plan_fingerprint_sha256",
            "config_fingerprint_sha256",
            "stage_configuration_sha256",
            "input_content_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name=field_name)
        _require_text(self.stage_node_id, field_name="stage_node_id", maximum=128)
        _require_text(self.stage_registry_id, field_name="stage_registry_id", maximum=256)
        _require_identifier(self.candidate_id, field_name="candidate_id")
        _require_ordinal(self.candidate_ordinal, field_name="candidate_ordinal")
        _require_identifier(self.executed_backend, field_name="executed_backend")
        if self.executed_backend != self.candidate_id:
            raise MaterializedExecutionIdentityV3Error("executed_backend must equal the exact selected candidate_id")
        _require_text(self.input_id, field_name="input_id", maximum=128)
        _require_text(self.input_relative_path, field_name="input_relative_path")
        if not isinstance(self.backend_runtime_identities, tuple):
            raise MaterializedExecutionIdentityV3Error("backend_runtime_identities must be an immutable tuple")
        if not self.backend_runtime_identities or len(self.backend_runtime_identities) > 8:
            raise MaterializedExecutionIdentityV3Error(
                "backend_runtime_identities must contain between one and eight constituents"
            )
        if any(not isinstance(item, BackendRuntimeIdentity) for item in self.backend_runtime_identities):
            raise MaterializedExecutionIdentityV3Error("backend_runtime_identities contains an unsupported value")
        ordinals = tuple(item.constituent_ordinal for item in self.backend_runtime_identities)
        if ordinals != tuple(sorted(set(ordinals))):
            raise MaterializedExecutionIdentityV3Error(
                "backend runtime identities must use unique ascending constituent ordinals"
            )
        if self.candidate_id == "ensemble":
            if len(self.backend_runtime_identities) < 2:
                raise MaterializedExecutionIdentityV3Error("An ensemble identity requires every enabled constituent")
        elif len(self.backend_runtime_identities) != 1 or self.backend_runtime_identities[0].backend_id != self.candidate_id:
            raise MaterializedExecutionIdentityV3Error(
                "A non-ensemble identity requires the selected backend as its sole constituent"
            )

    @classmethod
    def from_plan(
        cls,
        plan: CanonicalExecutionPlan,
        *,
        stage_node_id: str,
        candidate_id: str,
        input_id: str,
        executed_backend: str,
        input_content_sha256: str,
        backend_runtime_identities: Sequence[BackendRuntimeIdentity],
        dependency_lock_sha256: str,
        interpreter_identity_sha256: str,
        platform_identity_sha256: str,
        accelerator_identity_sha256: str,
        source_identity_sha256: str,
    ) -> "MaterializedExecutionIdentityV3":
        """Revalidate the plan and bind all enabled constituent evidence."""

        try:
            plan = _validated_execution_complete_plan(plan)
        except ExecutionIdentityV3SeedError as exc:
            raise MaterializedExecutionIdentityV3Error(str(exc)) from exc
        node = next((item for item in plan.nodes if item.node_id == stage_node_id), None)
        if node is None:
            raise MaterializedExecutionIdentityV3Error(f"Unknown stage_node_id: {stage_node_id!r}")
        candidate_ordinal = next(
            (index for index, item in enumerate(plan.backend_candidates) if item.backend_id == candidate_id),
            None,
        )
        if candidate_ordinal is None:
            raise MaterializedExecutionIdentityV3Error(f"Unknown candidate_id: {candidate_id!r}")
        candidate = plan.backend_candidates[candidate_ordinal]
        planned_input = next((item for item in plan.inputs if item.input_id == input_id), None)
        if planned_input is None:
            raise MaterializedExecutionIdentityV3Error(f"Unknown input_id: {input_id!r}")
        if executed_backend != candidate.backend_id:
            raise MaterializedExecutionIdentityV3Error("executed_backend does not match the exact selected candidate")

        indexed_contracts = tuple(
            (index, contract) for index, contract in enumerate(candidate.model_contracts) if contract.enabled
        )
        if not indexed_contracts:
            raise MaterializedExecutionIdentityV3Error(
                "A model-less candidate has incomplete model and weight identity and is non-cacheable"
            )
        runtime_identities = tuple(backend_runtime_identities)
        if len(runtime_identities) != len(indexed_contracts):
            raise MaterializedExecutionIdentityV3Error("Runtime evidence must cover all and only enabled model constituents")

        for (expected_ordinal, contract), runtime_identity in zip(indexed_contracts, runtime_identities):
            if not isinstance(runtime_identity, BackendRuntimeIdentity):
                raise MaterializedExecutionIdentityV3Error("Runtime evidence contains an unsupported backend identity")
            seed = ExecutionIdentityV3.from_plan(
                plan,
                stage_node_id=stage_node_id,
                candidate_id=candidate_id,
                input_id=input_id,
                model_backend_id=contract.backend_id if len(indexed_contracts) > 1 else None,
            )
            expected = {
                "constituent_ordinal": expected_ordinal,
                "backend_id": contract.backend_id,
                "model_canonical_key": contract.model.canonical_key,
                "model_lock_revision": contract.model.revision,
                "planned_model_artifact_sha256": contract.artifact_sha256,
                "planned_model_license_contract_sha256": seed.planned_model_license_contract_sha256,
            }
            observed = {field_name: getattr(runtime_identity, field_name) for field_name in expected}
            if observed != expected:
                raise MaterializedExecutionIdentityV3Error(
                    "Backend runtime identity does not match its exact planned constituent"
                )

        result = cls._from_validated_plan_values(
            plan_schema=plan.schema,
            plan_fingerprint_sha256=plan.plan_fingerprint_sha256,
            config_fingerprint_sha256=plan.config_fingerprint_sha256,
            stage_node_id=node.node_id,
            stage_registry_id=node.stage_registry_id.value,
            stage_configuration_sha256=hashlib.sha256(canonicalize_json(dict(node.configuration))).hexdigest(),
            candidate_id=candidate.backend_id,
            candidate_ordinal=candidate_ordinal,
            executed_backend=executed_backend,
            input_id=planned_input.input_id,
            input_relative_path=planned_input.path,
            input_content_sha256=input_content_sha256,
            backend_runtime_identities=runtime_identities,
        )
        expected_runtime_projections = {
            "dependency_lock_sha256": result.dependency_lock_sha256,
            "interpreter_identity_sha256": result.interpreter_identity_sha256,
            "platform_identity_sha256": result.platform_identity_sha256,
            "accelerator_identity_sha256": result.accelerator_identity_sha256,
            "source_identity_sha256": result.source_identity_sha256,
        }
        supplied_runtime_projections = {
            "dependency_lock_sha256": dependency_lock_sha256,
            "interpreter_identity_sha256": interpreter_identity_sha256,
            "platform_identity_sha256": platform_identity_sha256,
            "accelerator_identity_sha256": accelerator_identity_sha256,
            "source_identity_sha256": source_identity_sha256,
        }
        for field_name, value in supplied_runtime_projections.items():
            _require_sha256(value, field_name=field_name)
        if supplied_runtime_projections != expected_runtime_projections:
            raise MaterializedExecutionIdentityV3Error(
                "Runtime evidence projections do not match the ordered backend identities"
            )
        return result

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_plan: CanonicalExecutionPlan,
        expected_stage_node_id: str,
        expected_candidate_id: str,
        expected_input_id: str,
        expected_executed_backend: str,
        expected_input_content_sha256: str,
    ) -> "MaterializedExecutionIdentityV3":
        """Rebind a closed payload to independently supplied plan authority."""

        _require_exact_keys(payload, _MATERIALIZED_PAYLOAD_KEYS, context="materialized execution identity")
        if payload["schema"] != EXECUTION_IDENTITY_V3_SCHEMA:
            raise MaterializedExecutionIdentityV3Error("Unsupported execution identity schema")
        if payload["canonicalization"] != TP_CANONICAL_JSON_PROFILE:
            raise MaterializedExecutionIdentityV3Error("Unsupported execution identity canonicalization")
        if payload["completeness"] != EXECUTION_IDENTITY_V3_MATERIALIZED or payload["cacheable"] is not True:
            raise MaterializedExecutionIdentityV3Error("Serialized execution identity is not complete and cacheable")
        runtime_payloads = payload["model_constituents"]
        if not isinstance(runtime_payloads, list):
            raise MaterializedExecutionIdentityV3Error("model_constituents must be an array")
        runtime_identities = tuple(BackendRuntimeIdentity.from_payload(item) for item in runtime_payloads)
        result = cls.from_plan(
            expected_plan,
            stage_node_id=expected_stage_node_id,
            candidate_id=expected_candidate_id,
            input_id=expected_input_id,
            executed_backend=expected_executed_backend,
            input_content_sha256=expected_input_content_sha256,
            backend_runtime_identities=runtime_identities,
            dependency_lock_sha256=payload["dependency_lock_sha256"],
            interpreter_identity_sha256=payload["interpreter_identity_sha256"],
            platform_identity_sha256=payload["platform_identity_sha256"],
            accelerator_identity_sha256=payload["accelerator_identity_sha256"],
            source_identity_sha256=payload["source_identity_sha256"],
        )
        expected_body = result._identity_payload()
        observed_body = {key: payload[key] for key in _MATERIALIZED_BODY_KEYS}
        if observed_body != expected_body:
            raise MaterializedExecutionIdentityV3Error("Serialized execution identity contains inconsistent derived fields")
        claimed_digest = _require_sha256(payload["execution_identity_sha256"], field_name="execution_identity_sha256")
        if claimed_digest != result.execution_identity_sha256:
            raise MaterializedExecutionIdentityV3Error(
                "execution_identity_sha256 does not match the canonical identity payload"
            )
        return result

    @property
    def model_canonical_key(self) -> Optional[str]:
        return self.backend_runtime_identities[0].model_canonical_key if len(self.backend_runtime_identities) == 1 else None

    @property
    def model_lock_revision(self) -> Optional[str]:
        return self.backend_runtime_identities[0].model_lock_revision if len(self.backend_runtime_identities) == 1 else None

    @property
    def model_constituents(self) -> tuple[BackendRuntimeIdentity, ...]:
        """Stable alias used by cache projections."""

        return self.backend_runtime_identities

    @property
    def materialized_weights_sha256(self) -> str:
        return _runtime_aggregate(self.backend_runtime_identities, "materialized_weights_sha256")

    @property
    def dependency_lock_sha256(self) -> str:
        return _runtime_aggregate(self.backend_runtime_identities, "dependency_lock_sha256")

    @property
    def interpreter_identity_sha256(self) -> str:
        return _runtime_aggregate(self.backend_runtime_identities, "interpreter_identity_sha256")

    @property
    def platform_identity_sha256(self) -> str:
        return _runtime_aggregate(self.backend_runtime_identities, "platform_identity_sha256")

    @property
    def accelerator_identity_sha256(self) -> str:
        return _runtime_aggregate(self.backend_runtime_identities, "accelerator_identity_sha256")

    @property
    def source_identity_sha256(self) -> str:
        return _runtime_aggregate(self.backend_runtime_identities, "source_identity_sha256")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "canonicalization": self.canonicalization,
            "completeness": self.completeness,
            "cacheable": self.cacheable,
            "plan_schema": self.plan_schema,
            "plan_fingerprint_sha256": self.plan_fingerprint_sha256,
            "config_fingerprint_sha256": self.config_fingerprint_sha256,
            "stage_node_id": self.stage_node_id,
            "stage_registry_id": self.stage_registry_id,
            "stage_configuration_sha256": self.stage_configuration_sha256,
            "candidate_id": self.candidate_id,
            "candidate_ordinal": self.candidate_ordinal,
            "executed_backend": self.executed_backend,
            "model_canonical_key": self.model_canonical_key,
            "model_lock_revision": self.model_lock_revision,
            "input_id": self.input_id,
            "input_relative_path": self.input_relative_path,
            "input_content_sha256": self.input_content_sha256,
            "model_constituents": [identity.to_payload() for identity in self.backend_runtime_identities],
            "materialized_weights_sha256": self.materialized_weights_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "interpreter_identity_sha256": self.interpreter_identity_sha256,
            "platform_identity_sha256": self.platform_identity_sha256,
            "accelerator_identity_sha256": self.accelerator_identity_sha256,
            "source_identity_sha256": self.source_identity_sha256,
        }

    @property
    def execution_identity_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(_EXECUTION_IDENTITY_HASH_DOMAIN)
        digest.update(canonicalize_json(self._identity_payload()))
        return digest.hexdigest()

    def to_payload(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["execution_identity_sha256"] = self.execution_identity_sha256
        return payload

    def to_canonical_bytes(self) -> bytes:
        return canonicalize_json(self.to_payload())

    def cache_key(self, cache_schema: str) -> str:
        if not isinstance(cache_schema, str) or _CACHE_SCHEMA_RE.fullmatch(cache_schema) is None:
            raise MaterializedExecutionIdentityV3Error("cache_schema is not a canonical bounded schema identifier")
        digest = hashlib.sha256()
        digest.update(_EXECUTION_CACHE_KEY_HASH_DOMAIN)
        digest.update(
            canonicalize_json(
                {
                    "cache_schema": cache_schema,
                    "execution_identity_sha256": self.execution_identity_sha256,
                }
            )
        )
        return digest.hexdigest()


__all__ = [
    "EXECUTION_IDENTITY_V3_INCOMPLETE",
    "EXECUTION_IDENTITY_V3_MATERIALIZED",
    "EXECUTION_IDENTITY_V3_SCHEMA",
    "BackendRuntimeIdentity",
    "ExecutionIdentityV3",
    "ExecutionIdentityV3SeedError",
    "IncompleteExecutionIdentityV3Error",
    "MaterializedExecutionIdentityV3",
    "MaterializedExecutionIdentityV3Error",
]
