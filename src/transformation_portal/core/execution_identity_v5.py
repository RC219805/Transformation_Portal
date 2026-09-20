"""Plan-bound stage identity with separate, complete native-inference equivalence."""

from __future__ import annotations

from typing import Mapping

from transformation_portal.core.cas_dag_executor import AuthoritativeStageIdentity
from transformation_portal.core.execution_plan import ExecutionPlanError
from transformation_portal.core.execution_plan_v2 import digest_payload, require_digest
from transformation_portal.core.execution_plan_v4 import ExecutionPlanV4, parse_plan_v4
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.stage_graph.stage import Stage


def materialize_stage_identity(
    plan: ExecutionPlanV4,
    stage: Stage,
    input_id: str,
    inputs: Mapping[str, str],
    source_sha256: str,
    runtime_sha256: str,
    model_sha256: str,
) -> AuthoritativeStageIdentity:
    if type(plan) is not ExecutionPlanV4:
        raise ExecutionPlanError("V5 identity requires the exact core plan")
    payload = parse_plan_v4(plan.canonical_bytes).to_payload()
    selected = [item for item in payload["inputs"] if item["id"] == input_id]
    nodes = [item for item in payload["nodes"] if item["id"] == stage.name]
    if len(selected) != 1 or len(nodes) != 1 or nodes[0]["stage"] != stage.version or set(inputs) != set(nodes[0]["inputs"]):
        raise ExecutionPlanError("Identity must bind the exact prepared stage and every named input")
    native = stage.name == "depth"
    body = {
        "schema": "tp.execution.identity.v5",
        "stage_registry_id": stage.version,
        "source_sha256": selected[0]["sha256"],
        "stage_configuration": nodes[0]["configuration"],
        "inputs": {name: require_digest(value) for name, value in sorted(inputs.items())},
        "source_identity_sha256": require_digest(source_sha256),
        "runtime_identity_sha256": require_digest(runtime_sha256),
        "model_identity_sha256": require_digest(model_sha256) if native else None,
    }
    if native:
        body["inference_recipe"] = "tp.da3.explicit_precision_sky.v1"
        body["device"] = payload["device"]
        body["model"] = payload["model"]
        body["seed_policy"] = "proxy_content_v1"
    else:
        body["plan_fingerprint_sha256"] = plan.plan_fingerprint_sha256
        body["input_id"] = input_id
    return AuthoritativeStageIdentity(stage.name, stage.version, digest_payload(body), canonicalize_json(body))
