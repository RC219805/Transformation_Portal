"""Materialized multi-input stage identity; independent of legacy identity v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from transformation_portal.core.execution_plan import ExecutionPlanError, decode_bounded_json_object
from transformation_portal.core.execution_plan_v2 import ExecutionPlanV2, digest_payload, require_digest
from transformation_portal.core.execution_plan_v3 import ExecutionPlanV3, PhotographyPlan, parse_photography_plan
from transformation_portal.ingest.canonical_json import canonicalize_json

IDENTITY_SCHEMA = "tp.execution.identity.v4"


@dataclass(frozen=True, init=False)
class MaterializedExecutionIdentityV4:
    """Factory-only identity binding every named input plus materialized runtime."""

    canonical_bytes: bytes
    execution_identity_sha256: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ExecutionPlanError("MaterializedExecutionIdentityV4 is factory-only; use from_plan")

    @classmethod
    def from_plan(
        cls,
        plan: PhotographyPlan,
        *,
        node_id: str,
        input_id: str,
        inputs: Mapping[str, str],
        source_identity_sha256: str,
        runtime_identity_sha256: str,
        model_identity_sha256: str | None = None,
    ) -> MaterializedExecutionIdentityV4:
        if type(plan) not in (ExecutionPlanV2, ExecutionPlanV3):
            raise ExecutionPlanError("Execution identity requires the exact core-owned plan carrier")
        # Reconstruct from immutable bytes rather than trusting a forged object
        # or any caller-supplied projection of the plan.
        plan = parse_photography_plan(plan.canonical_bytes)
        payload = plan.to_payload()
        selected_inputs = [item for item in payload["inputs"] if item["id"] == input_id]
        if len(selected_inputs) != 1:
            raise ExecutionPlanError("Execution identity input must belong to the prepared inventory")
        nodes = [node for node in payload["nodes"] if node["id"] == node_id]
        if len(nodes) != 1 or set(inputs) != set(nodes[0]["inputs"]):
            raise ExecutionPlanError("Identity must bind every declared stage input exactly once")
        node = nodes[0]
        if node_id == "depth" and model_identity_sha256 is None:
            raise ExecutionPlanError("Depth identity requires materialized model evidence")
        if node_id != "depth" and model_identity_sha256 is not None:
            raise ExecutionPlanError("Model-free stage identity cannot invent a model constituent")
        body = {
            "schema": IDENTITY_SCHEMA,
            "plan_schema": plan.schema,
            "plan_fingerprint_sha256": plan.plan_fingerprint_sha256,
            "stage_node_id": node_id,
            "stage_registry_id": node["stage"],
            "stage_configuration_sha256": digest_payload(node["configuration"]),
            "source_input": selected_inputs[0],
            "inputs": [
                {"name": name, "binding": node["inputs"][name], "sha256": require_digest(value)}
                for name, value in sorted(inputs.items())
            ],
            "source_identity_sha256": require_digest(source_identity_sha256),
            "runtime_identity_sha256": require_digest(runtime_identity_sha256),
            "model_identity_sha256": None if model_identity_sha256 is None else require_digest(model_identity_sha256),
        }
        digest = digest_payload(body)
        instance = object.__new__(cls)
        object.__setattr__(instance, "canonical_bytes", canonicalize_json(body))
        object.__setattr__(instance, "execution_identity_sha256", digest)
        return instance

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_plan: PhotographyPlan,
        node_id: str,
        input_id: str,
        inputs: Mapping[str, str],
        source_identity_sha256: str,
        runtime_identity_sha256: str,
        model_identity_sha256: str | None = None,
    ) -> MaterializedExecutionIdentityV4:
        """Rebind serialized identity to independently materialized authority."""
        expected = cls.from_plan(
            expected_plan,
            node_id=node_id,
            input_id=input_id,
            inputs=inputs,
            source_identity_sha256=source_identity_sha256,
            runtime_identity_sha256=runtime_identity_sha256,
            model_identity_sha256=model_identity_sha256,
        )
        try:
            observed = decode_bounded_json_object(canonicalize_json(payload))
        except (TypeError, ValueError) as exc:
            raise ExecutionPlanError("Invalid serialized execution identity") from exc
        if canonicalize_json(observed) != expected.canonical_bytes:
            raise ExecutionPlanError("Serialized identity differs from independently materialized authority")
        return expected

    def to_payload(self) -> dict[str, Any]:
        return decode_bounded_json_object(self.canonical_bytes)
