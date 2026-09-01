"""Inert ExecutionIdentity v3 seed for canonical execution plans.

ADR-051 designates identity v3 as future cache authority.  This #2065 seed
contains only plan-time facts.  It deliberately cannot produce a cache key
until #2064 materializes and verifies every runtime identity input.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

from ..ingest.canonical_json import canonicalize_json
from .execution_plan import EXECUTION_COMPLETE, CanonicalExecutionPlan, ExecutionPlanError

EXECUTION_IDENTITY_V3_SCHEMA = "tp.execution.identity.v3"
EXECUTION_IDENTITY_V3_INCOMPLETE = "incomplete_seed"

_MISSING_AUTHORITY_FIELDS = (
    "executed_backend",
    "input_content_sha256",
    "materialized_weights_sha256",
    "dependency_lock_sha256",
    "interpreter_identity",
    "platform_identity",
    "accelerator_identity",
    "source_identity",
)


class ExecutionIdentityV3SeedError(ValueError):
    """An identity-v3 seed cannot be selected safely from the plan."""


class IncompleteExecutionIdentityV3Error(RuntimeError):
    """An incomplete seed was asked to authorize cache access."""


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

        if not isinstance(plan, CanonicalExecutionPlan):
            raise TypeError("plan must be a CanonicalExecutionPlan")
        try:
            # ``CanonicalExecutionPlan`` is a public frozen dataclass, so a
            # caller can construct or ``dataclasses.replace`` an instance
            # without passing through ``from_payload``. Re-parse its complete
            # payload before trusting the carried fingerprint or any selected
            # authority, and derive every seed field from that validated copy.
            plan = CanonicalExecutionPlan.from_payload(plan.to_payload())
        except ExecutionPlanError as exc:
            raise ExecutionIdentityV3SeedError("ExecutionIdentity v3 requires a canonically validated execution plan") from exc
        if plan.configuration_completeness != EXECUTION_COMPLETE:
            raise ExecutionIdentityV3SeedError("ExecutionIdentity v3 requires an execution_complete plan")

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
        model_license_payload = {
            "candidate_id": candidate.backend_id,
            "model_backend_id": model_backend_id,
            "model_contract": None if selected_contract is None else selected_contract.to_payload(),
            "resolved_model": None if plan.resolved_model is None else plan.resolved_model.to_payload(),
            "license_acknowledgements": plan.license_acknowledgements.to_payload(),
            "license_evaluation": plan.license_evaluation.to_payload(),
        }
        planned_contract_sha256 = hashlib.sha256(canonicalize_json(model_license_payload)).hexdigest()
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
            "stage_node_id": self.stage_node_id,
            "stage_registry_id": self.stage_registry_id,
            "stage_configuration_sha256": self.stage_configuration_sha256,
            "candidate_id": self.candidate_id,
            "candidate_ordinal": self.candidate_ordinal,
            "model_backend_id": self.model_backend_id,
            "constituent_ordinal": self.constituent_ordinal,
            "planned_model_license_contract_sha256": self.planned_model_license_contract_sha256,
            "input_id": self.input_id,
            "input_relative_path": self.input_relative_path,
            "materialized": {field_name: None for field_name in self.missing_authority_fields},
            "missing_authority_fields": list(self.missing_authority_fields),
        }

    def to_canonical_bytes(self) -> bytes:
        """Serialize deterministic seed evidence; these bytes are not a cache key."""

        return canonicalize_json(self.to_payload())

    def seed_sha256(self) -> str:
        """Hash deterministic seed evidence without granting cache authority."""

        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()

    def cache_key(self) -> str:
        """Fail closed until #2064 materializes all missing identity fields."""

        raise IncompleteExecutionIdentityV3Error("ExecutionIdentity v3 is incomplete and non-cacheable until #2064")


__all__ = [
    "EXECUTION_IDENTITY_V3_INCOMPLETE",
    "EXECUTION_IDENTITY_V3_SCHEMA",
    "ExecutionIdentityV3",
    "ExecutionIdentityV3SeedError",
    "IncompleteExecutionIdentityV3Error",
]
