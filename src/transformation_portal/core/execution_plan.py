"""Canonical, non-executing ``tp.execution.plan.v1`` contract.

The types and validators in this module describe execution intent.  They do
not construct ``StageGraph`` objects, load models, or activate
``CASDAGExecutor``.  ADR-051 requires a separate vertical slice before any
plan may become execution authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from importlib import resources
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from ..ingest.canonical_json import TP_CANONICAL_JSON_PROFILE, canonicalize_json
from ..stage_graph.registry import (
    StageRegistryIdentifier,
    UnknownStageRegistryIdentifier,
    get_output_definition,
    get_stage_definition,
)

EXECUTION_PLAN_SCHEMA = "tp.execution.plan.v1"
EXECUTION_PLAN_SCHEMA_RESOURCE = "plan.v1.schema.json"
STRUCTURAL_LEGACY = "structural_legacy"
EXECUTION_COMPLETE = "execution_complete"

# Pre-parse limits apply before json.loads.  Collection limits are duplicated
# in the shipped schema and checked semantically where graph context matters.
MAX_PLAN_BODY_BYTES = 1_048_576
MAX_PLAN_JSON_DEPTH = 24
MAX_PLAN_STRING_LENGTH = 8_192
MAX_PLAN_INTEGER_DIGITS = 20
MAX_PLAN_NODES = 32
MAX_PLAN_EDGES = 128
MAX_PLAN_FANOUT = 16
MAX_PLAN_INPUTS = 4_096
MAX_PLAN_REQUESTED_OUTPUTS = 128
MAX_DECODED_PIXELS_PER_INPUT = 268_435_456
MAX_TOTAL_DECODED_PIXELS = 17_179_869_184
MAX_INPUT_DECOMPRESSION_RATIO = 1_000


class ExecutionPlanError(ValueError):
    """Base error for an invalid or unsupported execution plan."""


class UnsupportedExecutionPlanSchema(ExecutionPlanError):
    """Raised when a payload does not name the one supported plan schema."""


class ExecutionPlanLimitError(ExecutionPlanError):
    """Raised when a payload exceeds a pre-parse or semantic safety limit."""


class DuplicateExecutionPlanKey(ExecutionPlanError):
    """Raised when JSON contains a duplicate object member."""


@dataclass(frozen=True)
class ResourceRange:
    """Minimum and maximum resource values declared by a plan node."""

    minimum: int
    maximum: int

    def to_payload(self) -> dict[str, int]:
        return {"minimum": self.minimum, "maximum": self.maximum}


@dataclass(frozen=True)
class PlanResources:
    """Closed resource declaration for a stage node."""

    cpu_cores: ResourceRange
    gpu_count: ResourceRange
    memory_mib: ResourceRange
    disk_mib: ResourceRange
    wall_time_seconds: ResourceRange

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PlanResources":
        def resource_range(name: str) -> ResourceRange:
            value = payload[name]
            return ResourceRange(minimum=value["minimum"], maximum=value["maximum"])

        return cls(
            cpu_cores=resource_range("cpu_cores"),
            gpu_count=resource_range("gpu_count"),
            memory_mib=resource_range("memory_mib"),
            disk_mib=resource_range("disk_mib"),
            wall_time_seconds=resource_range("wall_time_seconds"),
        )

    def to_payload(self) -> dict[str, dict[str, int]]:
        return {
            "cpu_cores": self.cpu_cores.to_payload(),
            "gpu_count": self.gpu_count.to_payload(),
            "memory_mib": self.memory_mib.to_payload(),
            "disk_mib": self.disk_mib.to_payload(),
            "wall_time_seconds": self.wall_time_seconds.to_payload(),
        }


@dataclass(frozen=True)
class PlanInput:
    """One frozen input selection entry.

    V1 compatibility carries a contained relative path but no content digest.
    ExecutionIdentity v3 will decide cacheability; this contract does not
    imply that a path-only input is cache-authorizing.
    """

    input_id: str
    path: str

    def to_payload(self) -> dict[str, str]:
        return {"id": self.input_id, "path": self.path}


@dataclass(frozen=True)
class InputSafetyLimits:
    """Image-allocation limits an eventual executor must enforce."""

    max_decoded_pixels_per_input: int
    max_total_decoded_pixels: int
    max_decompression_ratio: int

    def to_payload(self) -> dict[str, int]:
        return {
            "max_decoded_pixels_per_input": self.max_decoded_pixels_per_input,
            "max_total_decoded_pixels": self.max_total_decoded_pixels,
            "max_decompression_ratio": self.max_decompression_ratio,
        }


@dataclass(frozen=True)
class ResolvedModelIntent:
    """Carried model intent; schema validity alone grants no authority."""

    requested_selector: str
    resolution_reason: str
    canonical_key: str
    repo_id: Optional[str]
    revision: Optional[str]
    license_id: Optional[str]
    usage_class: Optional[str]
    requires_non_commercial_ok: bool
    accelerator_kind: Optional[str]
    legacy_model_variant_name: Optional[str]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ResolvedModelIntent":
        return cls(
            requested_selector=payload["requested_selector"],
            resolution_reason=payload["resolution_reason"],
            canonical_key=payload["canonical_key"],
            repo_id=payload["repo_id"],
            revision=payload["revision"],
            license_id=payload["license_id"],
            usage_class=payload["usage_class"],
            requires_non_commercial_ok=payload["requires_non_commercial_ok"],
            accelerator_kind=payload["accelerator_kind"],
            legacy_model_variant_name=payload["legacy_model_variant_name"],
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "requested_selector": self.requested_selector,
            "resolution_reason": self.resolution_reason,
            "canonical_key": self.canonical_key,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "license_id": self.license_id,
            "usage_class": self.usage_class,
            "requires_non_commercial_ok": self.requires_non_commercial_ok,
            "accelerator_kind": self.accelerator_kind,
            "legacy_model_variant_name": self.legacy_model_variant_name,
        }


@dataclass(frozen=True)
class BackendModelIntent:
    """One carried model identity for a backend candidate.

    ``model`` preserves resolver provenance for domain authority checks;
    ``artifact_sha256`` covers checkpoint-backed candidates whose immutable
    identity is a byte digest rather than a repository revision. Weight and
    device are explicit so ensemble constituents never need to be inferred.
    """

    role: str
    backend_id: str
    model: ResolvedModelIntent
    artifact_path: Optional[str]
    artifact_sha256: Optional[str]
    enabled: bool
    weight: Optional[float]
    device: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BackendModelIntent":
        return cls(
            role=payload["role"],
            backend_id=payload["backend_id"],
            model=ResolvedModelIntent.from_payload(payload["model"]),
            artifact_path=payload["artifact_path"],
            artifact_sha256=payload["artifact_sha256"],
            enabled=payload["enabled"],
            weight=payload["weight"],
            device=payload["device"],
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "backend_id": self.backend_id,
            "model": self.model.to_payload(),
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "enabled": self.enabled,
            "weight": self.weight,
            "device": self.device,
        }


@dataclass(frozen=True)
class BackendCandidateIntent:
    """One ordered fallback candidate and all model identities it would use."""

    backend_id: str
    model_contracts: tuple[BackendModelIntent, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BackendCandidateIntent":
        return cls(
            backend_id=payload["backend_id"],
            model_contracts=tuple(BackendModelIntent.from_payload(item) for item in payload["model_contracts"]),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "model_contracts": [contract.to_payload() for contract in self.model_contracts],
        }


@dataclass(frozen=True)
class LicenseAcknowledgements:
    """Individually attributable acknowledgement values."""

    non_commercial_ok: bool
    apple_depth_pro_research: bool
    research_tools: bool

    def to_payload(self) -> dict[str, bool]:
        return {
            "non_commercial_ok": self.non_commercial_ok,
            "apple_depth_pro_research": self.apple_depth_pro_research,
            "research_tools": self.research_tools,
        }


@dataclass(frozen=True)
class LicenseEvaluation:
    """Plan-time license evaluation result."""

    enforced: bool
    status: str

    def to_payload(self) -> dict[str, Any]:
        return {"enforced": self.enforced, "status": self.status}


@dataclass(frozen=True)
class OutputDeclaration:
    """Typed logical output declaration for one node."""

    output_id: str
    artifact_kind: str
    scope: str
    cardinality: str
    required: bool
    disposition: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OutputDeclaration":
        return cls(
            output_id=payload["id"],
            artifact_kind=payload["artifact_kind"],
            scope=payload["scope"],
            cardinality=payload["cardinality"],
            required=payload["required"],
            disposition=payload["disposition"],
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.output_id,
            "artifact_kind": self.artifact_kind,
            "scope": self.scope,
            "cardinality": self.cardinality,
            "required": self.required,
            "disposition": self.disposition,
        }


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class StageNode:
    """One typed semantic plan node."""

    node_id: str
    stage_registry_id: StageRegistryIdentifier
    configuration: Mapping[str, Any]
    resources: PlanResources
    outputs: tuple[OutputDeclaration, ...]
    optional: bool
    failure_policy: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StageNode":
        return cls(
            node_id=payload["id"],
            stage_registry_id=StageRegistryIdentifier(payload["stage_registry_id"]),
            configuration=_freeze_json(payload["configuration"]),
            resources=PlanResources.from_payload(payload["resources"]),
            outputs=tuple(OutputDeclaration.from_payload(item) for item in payload["outputs"]),
            optional=payload["optional"],
            failure_policy=payload["failure_policy"],
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "stage_registry_id": self.stage_registry_id.value,
            "configuration": _thaw_json(self.configuration),
            "resources": self.resources.to_payload(),
            "outputs": [output.to_payload() for output in self.outputs],
            "optional": self.optional,
            "failure_policy": self.failure_policy,
        }


@dataclass(frozen=True)
class DependencyEdge:
    """Explicit dependency edge between two node IDs."""

    source: str
    target: str

    def to_payload(self) -> dict[str, str]:
        return {"from": self.source, "to": self.target}


@dataclass(frozen=True)
class ExecutionPlan:
    """Immutable typed representation of a validated canonical plan."""

    plan_fingerprint_sha256: str
    configuration_completeness: str
    planned_backend: str
    candidate_fallback_chain: tuple[str, ...]
    backend_candidates: tuple[BackendCandidateIntent, ...]
    resolved_model: Optional[ResolvedModelIntent]
    license_acknowledgements: LicenseAcknowledgements
    license_evaluation: LicenseEvaluation
    quality_tier: str
    preset_requested: Optional[str]
    preset_resolved: Optional[str]
    input_root: str
    inputs: tuple[PlanInput, ...]
    input_limits: InputSafetyLimits
    config_fingerprint_sha256: str
    nodes: tuple[StageNode, ...]
    edges: tuple[DependencyEdge, ...]
    requested_outputs: tuple[str, ...]
    warnings: tuple[str, ...]
    schema: str = EXECUTION_PLAN_SCHEMA
    canonicalization: str = TP_CANONICAL_JSON_PROFILE

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ExecutionPlan":
        """Validate and freeze an untrusted mapping."""

        validate_execution_plan_payload(payload)
        model_payload = payload["resolved_model"]
        input_selection = payload["input_selection"]
        input_limits = payload["input_limits"]
        acknowledgements = payload["license_acknowledgements"]
        evaluation = payload["license_evaluation"]
        return cls(
            schema=payload["schema"],
            canonicalization=payload["canonicalization"],
            plan_fingerprint_sha256=payload["plan_fingerprint_sha256"],
            configuration_completeness=payload["configuration_completeness"],
            planned_backend=payload["planned_backend"],
            candidate_fallback_chain=tuple(payload["candidate_fallback_chain"]),
            backend_candidates=tuple(BackendCandidateIntent.from_payload(item) for item in payload["backend_candidates"]),
            resolved_model=None if model_payload is None else ResolvedModelIntent.from_payload(model_payload),
            license_acknowledgements=LicenseAcknowledgements(
                non_commercial_ok=acknowledgements["non_commercial_ok"],
                apple_depth_pro_research=acknowledgements["apple_depth_pro_research"],
                research_tools=acknowledgements["research_tools"],
            ),
            license_evaluation=LicenseEvaluation(
                enforced=evaluation["enforced"],
                status=evaluation["status"],
            ),
            quality_tier=payload["quality_tier"],
            preset_requested=payload["preset_requested"],
            preset_resolved=payload["preset_resolved"],
            input_root=input_selection["root"],
            inputs=tuple(PlanInput(input_id=item["id"], path=item["path"]) for item in input_selection["files"]),
            input_limits=InputSafetyLimits(
                max_decoded_pixels_per_input=input_limits["max_decoded_pixels_per_input"],
                max_total_decoded_pixels=input_limits["max_total_decoded_pixels"],
                max_decompression_ratio=input_limits["max_decompression_ratio"],
            ),
            config_fingerprint_sha256=payload["config_fingerprint_sha256"],
            nodes=tuple(StageNode.from_payload(item) for item in payload["nodes"]),
            edges=tuple(DependencyEdge(source=item["from"], target=item["to"]) for item in payload["edges"]),
            requested_outputs=tuple(payload["requested_outputs"]),
            warnings=tuple(payload["warnings"]),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "canonicalization": self.canonicalization,
            "plan_fingerprint_sha256": self.plan_fingerprint_sha256,
            "configuration_completeness": self.configuration_completeness,
            "planned_backend": self.planned_backend,
            "candidate_fallback_chain": list(self.candidate_fallback_chain),
            "backend_candidates": [candidate.to_payload() for candidate in self.backend_candidates],
            "resolved_model": None if self.resolved_model is None else self.resolved_model.to_payload(),
            "license_acknowledgements": self.license_acknowledgements.to_payload(),
            "license_evaluation": self.license_evaluation.to_payload(),
            "quality_tier": self.quality_tier,
            "preset_requested": self.preset_requested,
            "preset_resolved": self.preset_resolved,
            "input_selection": {
                "root": self.input_root,
                "files": [item.to_payload() for item in self.inputs],
            },
            "input_limits": self.input_limits.to_payload(),
            "config_fingerprint_sha256": self.config_fingerprint_sha256,
            "nodes": [node.to_payload() for node in self.nodes],
            "edges": [edge.to_payload() for edge in self.edges],
            "requested_outputs": list(self.requested_outputs),
            "warnings": list(self.warnings),
        }

    def to_canonical_json(self) -> str:
        """Return deterministic canonical JSON without filesystem effects."""

        return canonicalize_json(self.to_payload()).decode("utf-8")

    def ordered_node_ids(self) -> tuple[str, ...]:
        """Return deterministic topological order, using payload order as tie-break."""

        return _topological_order(
            [node.node_id for node in self.nodes],
            [(edge.source, edge.target) for edge in self.edges],
        )


# Explicit public alias avoids confusion with the retained Lux flat
# ``pipeline_coordinator.ExecutionPlan`` compatibility type.
CanonicalExecutionPlan = ExecutionPlan


def load_execution_plan_schema() -> dict[str, Any]:
    """Load the schema from package resources, including installed wheels."""

    schema_text = (
        resources.files("transformation_portal.schemas.execution")
        .joinpath(EXECUTION_PLAN_SCHEMA_RESOURCE)
        .read_text(encoding="utf-8")
    )
    payload = json.loads(schema_text)
    if not isinstance(payload, dict):  # pragma: no cover - governed resource
        raise ExecutionPlanError("Packaged execution plan schema is not a JSON object")
    return payload


def compute_execution_plan_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash the canonical plan body, excluding its self-describing digest."""

    body = dict(payload)
    body.pop("plan_fingerprint_sha256", None)
    try:
        canonical_body = canonicalize_json(body)
    except UnicodeEncodeError as exc:
        raise ExecutionPlanError("Execution plan strings must not contain Unicode surrogate code points") from exc
    return hashlib.sha256(canonical_body).hexdigest()


def with_execution_plan_fingerprint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow top-level copy carrying its canonical body digest."""

    result = dict(payload)
    result["plan_fingerprint_sha256"] = compute_execution_plan_fingerprint(result)
    return result


def _format_schema_path(parts: Sequence[Any]) -> str:
    if not parts:
        return "$"
    return "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in parts)


def _validate_relative_input_path(path_value: str) -> None:
    if "\x00" in path_value or "\\" in path_value or path_value.startswith("./") or "/./" in path_value or "//" in path_value:
        raise ExecutionPlanError(f"Input path is not a portable contained POSIX path: {path_value!r}")
    path = PurePosixPath(path_value)
    if (
        not path_value
        or not path.parts
        or path.is_absolute()
        or path.as_posix() != path_value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ExecutionPlanError(f"Input path is not a contained relative path: {path_value!r}")
    if len(path.parts) > 64:
        raise ExecutionPlanLimitError("Input path exceeds the 64-component limit")


def _topological_order(node_ids: Sequence[str], edges: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    node_order = {node_id: index for index, node_id in enumerate(node_ids)}
    incoming = {node_id: 0 for node_id in node_ids}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for source, target in edges:
        incoming[target] += 1
        outgoing[source].append(target)
    ready = [node_id for node_id in node_ids if incoming[node_id] == 0]
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for target in outgoing[current]:
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
        ready.sort(key=node_order.__getitem__)
    if len(ordered) != len(node_ids):
        raise ExecutionPlanError("Execution plan dependency graph contains a cycle")
    return tuple(ordered)


def _validate_backend_candidates(payload: Mapping[str, Any]) -> None:
    """Validate carrier completeness without granting domain authority."""

    completeness = payload["configuration_completeness"]
    candidate_chain = payload["candidate_fallback_chain"]
    candidates = payload["backend_candidates"]
    candidate_ids = [candidate["backend_id"] for candidate in candidates]
    if candidate_ids != candidate_chain:
        raise ExecutionPlanError("backend_candidates must exactly match candidate_fallback_chain order")

    da3_models: list[Mapping[str, Any]] = []
    for candidate in candidates:
        candidate_id = candidate["backend_id"]
        contracts = candidate["model_contracts"]
        contract_backend_ids = [contract["backend_id"] for contract in contracts]
        if len(contract_backend_ids) != len(set(contract_backend_ids)):
            raise ExecutionPlanError(f"Backend candidate {candidate_id!r} contains duplicate model backend identities")

        if candidate_id == "ensemble":
            if any(contract["role"] != "ensemble_constituent" for contract in contracts):
                raise ExecutionPlanError("Ensemble model contracts must use role='ensemble_constituent'")
            if any(backend_id in {"ensemble", "synthetic"} for backend_id in contract_backend_ids):
                raise ExecutionPlanError("Ensemble constituents must name concrete non-synthetic backends")
            if completeness == EXECUTION_COMPLETE:
                enabled_contracts = [contract for contract in contracts if contract["enabled"]]
                if len(enabled_contracts) < 2:
                    raise ExecutionPlanError("An execution-complete ensemble requires at least two enabled constituents")
                weights = [contract["weight"] for contract in enabled_contracts]
                if any(weight is None or weight <= 0 for weight in weights):
                    raise ExecutionPlanError("Enabled ensemble constituents require positive weights")
        else:
            if any(contract["role"] != "primary" for contract in contracts):
                raise ExecutionPlanError(f"Backend candidate {candidate_id!r} model contracts must use role='primary'")
            if any(backend_id != candidate_id for backend_id in contract_backend_ids):
                raise ExecutionPlanError(f"Backend candidate {candidate_id!r} carries a model for another backend")
            if any(contract["weight"] is not None for contract in contracts):
                raise ExecutionPlanError("Non-ensemble model contracts must not carry weights")
            if completeness == EXECUTION_COMPLETE:
                expected_contract_count = 0 if candidate_id == "synthetic" else 1
                if len(contracts) != expected_contract_count:
                    raise ExecutionPlanError(
                        f"Execution-complete backend {candidate_id!r} requires {expected_contract_count} model contract(s)"
                    )
                if any(not contract["enabled"] for contract in contracts):
                    raise ExecutionPlanError("A non-ensemble candidate model contract must be enabled")

        for contract in contracts:
            model = contract["model"]
            if contract["backend_id"] == "da3":
                da3_models.append(model)
            if completeness != EXECUTION_COMPLETE:
                continue
            missing_identity = [
                field_name for field_name in ("repo_id", "license_id", "usage_class") if model[field_name] is None
            ]
            if missing_identity:
                raise ExecutionPlanError(
                    f"Execution-complete model {model['canonical_key']!r} lacks identity fields {missing_identity}"
                )
            if model["revision"] is None and contract["artifact_sha256"] is None:
                raise ExecutionPlanError(
                    f"Execution-complete model {model['canonical_key']!r} requires a revision or artifact SHA-256"
                )
            if (contract["artifact_path"] is None) != (contract["artifact_sha256"] is None):
                raise ExecutionPlanError(
                    f"Execution-complete model {model['canonical_key']!r} must pair artifact_path and artifact_sha256"
                )
            if model["requires_non_commercial_ok"] and not payload["license_acknowledgements"]["non_commercial_ok"]:
                raise ExecutionPlanError(f"Model {model['canonical_key']!r} requires the non-commercial acknowledgement")

    resolved_model = payload["resolved_model"]
    if completeness == EXECUTION_COMPLETE:
        if da3_models and resolved_model is None:
            raise ExecutionPlanError("An execution-complete plan using DA3 requires resolved_model")
        if not da3_models and resolved_model is not None:
            raise ExecutionPlanError("An execution-complete plan without a DA3 candidate must not carry resolved_model")
    if resolved_model is not None and da3_models and any(model != resolved_model for model in da3_models):
        raise ExecutionPlanError("DA3 candidate model contracts must match the top-level resolved_model")


def _validate_plan_semantics(payload: Mapping[str, Any]) -> None:
    candidate_chain = payload["candidate_fallback_chain"]
    if candidate_chain[0] != payload["planned_backend"]:
        raise ExecutionPlanError("candidate_fallback_chain must start with planned_backend")
    _validate_backend_candidates(payload)

    if "\x00" in payload["input_selection"]["root"]:
        raise ExecutionPlanError("input_selection.root contains a NUL character")
    input_ids: set[str] = set()
    input_paths: set[str] = set()
    for item in payload["input_selection"]["files"]:
        if item["id"] in input_ids:
            raise ExecutionPlanError(f"Duplicate input id: {item['id']!r}")
        if item["path"] in input_paths:
            raise ExecutionPlanError(f"Duplicate input path: {item['path']!r}")
        input_ids.add(item["id"])
        input_paths.add(item["path"])
        _validate_relative_input_path(item["path"])

    input_limits = payload["input_limits"]
    for field_name, value in input_limits.items():
        if type(value) is not int:
            raise ExecutionPlanError(f"input_limits.{field_name} must be an exact integer")
    if input_limits["max_total_decoded_pixels"] < input_limits["max_decoded_pixels_per_input"]:
        raise ExecutionPlanError("input_limits.max_total_decoded_pixels must be at least the per-input limit")

    nodes = payload["nodes"]
    node_ids = [node["id"] for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ExecutionPlanError("Execution plan node ids must be unique")
    stage_registry_ids = [node["stage_registry_id"] for node in nodes]
    if stage_registry_ids.count(StageRegistryIdentifier.LUX_DEPTH.value) != 1:
        raise ExecutionPlanError("A Lux execution plan must contain exactly one depth node")
    if stage_registry_ids.count(StageRegistryIdentifier.LUX_OUTPUT.value) != 1:
        raise ExecutionPlanError("A Lux execution plan must contain exactly one output node")

    requested_declarations: set[str] = set()
    output_ids: set[str] = set()
    for node in nodes:
        try:
            definition = get_stage_definition(node["stage_registry_id"])
        except UnknownStageRegistryIdentifier as exc:
            raise ExecutionPlanError(str(exc)) from exc
        configuration_schema = node["configuration"]["schema"]
        if configuration_schema != definition.configuration_schema:
            raise ExecutionPlanError(
                f"Node {node['id']!r} configuration schema {configuration_schema!r} does not match "
                f"registry schema {definition.configuration_schema!r}"
            )
        if node["configuration"]["configuration_completeness"] != payload["configuration_completeness"]:
            raise ExecutionPlanError(f"Node {node['id']!r} configuration completeness does not match the plan")
        if definition.identifier is StageRegistryIdentifier.LUX_DEPTH:
            expected_model_key = None
            expected_model_revision = None
            if payload["resolved_model"] is not None:
                expected_model_key = payload["resolved_model"]["canonical_key"]
                expected_model_revision = payload["resolved_model"]["revision"]
            expected_depth_values = {
                "planned_backend": payload["planned_backend"],
                "candidate_fallback_chain": payload["candidate_fallback_chain"],
                "resolved_model_key": expected_model_key,
                "resolved_model_revision": expected_model_revision,
            }
            for field_name, expected_value in expected_depth_values.items():
                if node["configuration"][field_name] != expected_value:
                    raise ExecutionPlanError(
                        f"Node {node['id']!r} depth configuration field {field_name!r} "
                        "does not match the authoritative plan value"
                    )
            if payload["configuration_completeness"] == EXECUTION_COMPLETE:
                ensemble_expected = "ensemble" in payload["candidate_fallback_chain"]
                if (node["configuration"]["ensemble"] is not None) != ensemble_expected:
                    raise ExecutionPlanError("Depth ensemble configuration must match the candidate chain")
                if node["configuration"]["apex_gate"]["quality_tier"] != payload["quality_tier"]:
                    raise ExecutionPlanError("Depth APEX gate quality tier does not match the plan")
                if node["configuration"]["apex_gate"]["depth_fallback"] != node["configuration"]["fallback_mode"]:
                    raise ExecutionPlanError("Depth APEX gate fallback does not match the depth fallback mode")
                depth_pro_paths = [
                    contract["artifact_path"]
                    for candidate in payload["backend_candidates"]
                    for contract in candidate["model_contracts"]
                    if contract["backend_id"] == "depth_pro" and contract["enabled"]
                ]
                if depth_pro_paths and any(
                    path != node["configuration"]["depth_pro_checkpoint_path"] for path in depth_pro_paths
                ):
                    raise ExecutionPlanError("Depth Pro candidate artifact paths do not match the depth configuration")
        elif definition.identifier is StageRegistryIdentifier.LUX_OUTPUT:
            output_bit_depth = node["configuration"]["output_bit_depth"]
            if type(output_bit_depth) is not int or output_bit_depth not in {8, 16}:
                raise ExecutionPlanError(f"Node {node['id']!r} output_bit_depth must be the exact integer 8 or 16")
            if node["configuration"]["requested_outputs"] != payload["requested_outputs"]:
                raise ExecutionPlanError(f"Node {node['id']!r} output configuration does not match requested_outputs")
            if payload["configuration_completeness"] == EXECUTION_COMPLETE:
                run_card_requested = "run_card" in payload["requested_outputs"]
                if node["configuration"]["run_card_enabled"] != run_card_requested:
                    raise ExecutionPlanError("Output run-card configuration does not match requested_outputs")
        for resource_name, resource_range in node["resources"].items():
            for bound_name, value in resource_range.items():
                if type(value) is not int:
                    raise ExecutionPlanError(
                        f"Node {node['id']!r} resource {resource_name!r} {bound_name!r} " "must be an exact integer"
                    )
            if resource_range["minimum"] > resource_range["maximum"]:
                raise ExecutionPlanError(f"Node {node['id']!r} resource {resource_name!r} has minimum above maximum")
            registry_range = getattr(definition.resources, resource_name)
            if resource_range["minimum"] < registry_range.minimum or resource_range["maximum"] > registry_range.maximum:
                raise ExecutionPlanError(f"Node {node['id']!r} resource {resource_name!r} exceeds its registry profile")
        allowed_outputs = set(definition.allowed_output_kinds)
        node_output_kinds: set[str] = set()
        for output in node["outputs"]:
            if output["id"] in output_ids:
                raise ExecutionPlanError(f"Duplicate output id: {output['id']!r}")
            output_ids.add(output["id"])
            if output["artifact_kind"] not in allowed_outputs:
                raise ExecutionPlanError(
                    f"Node {node['id']!r} declares output {output['artifact_kind']!r} "
                    f"outside registry identifier {definition.identifier.value!r}"
                )
            output_definition = get_output_definition(output["artifact_kind"])
            if output["scope"] != output_definition.scope.value:
                raise ExecutionPlanError(
                    f"Output {output['artifact_kind']!r} scope {output['scope']!r} does not match "
                    f"registry scope {output_definition.scope.value!r}"
                )
            if output["cardinality"] != output_definition.cardinality.value:
                raise ExecutionPlanError(
                    f"Output {output['artifact_kind']!r} cardinality {output['cardinality']!r} does not match "
                    f"registry cardinality {output_definition.cardinality.value!r}"
                )
            if output["artifact_kind"] in node_output_kinds:
                raise ExecutionPlanError(f"Node {node['id']!r} declares duplicate artifact kind {output['artifact_kind']!r}")
            node_output_kinds.add(output["artifact_kind"])
            if output["artifact_kind"] == "run_card" and output["required"]:
                raise ExecutionPlanError("The current Lux run card is a non-required requested output")
            if output["disposition"] == "requested":
                if output["artifact_kind"] in requested_declarations:
                    raise ExecutionPlanError(
                        f"Requested artifact kind {output['artifact_kind']!r} has multiple producer nodes"
                    )
                requested_declarations.add(output["artifact_kind"])
        if node["optional"]:
            if node["failure_policy"] != "omit_outputs":
                raise ExecutionPlanError("An optional node must use failure_policy='omit_outputs'")
            if any(output["required"] for output in node["outputs"]):
                raise ExecutionPlanError("An optional node cannot declare required outputs")
        elif node["failure_policy"] != "abort_plan":
            raise ExecutionPlanError("A required node must use failure_policy='abort_plan'")
        if definition.identifier is StageRegistryIdentifier.LUX_PBR and not node["optional"]:
            raise ExecutionPlanError("The current Lux PBR stage must be optional")
        if definition.identifier is StageRegistryIdentifier.LUX_MATERIALS_V3:
            materials_optional = str(payload["quality_tier"]).lower() != "apex"
            if node["optional"] != materials_optional:
                raise ExecutionPlanError("Materials V3 optionality does not match the current Lux quality-tier policy")

    requested_outputs = set(payload["requested_outputs"])
    if requested_outputs != requested_declarations:
        missing = sorted(requested_outputs - requested_declarations)
        unexpected = sorted(requested_declarations - requested_outputs)
        raise ExecutionPlanError(
            f"Requested output declarations do not match plan intent (missing={missing}, unexpected={unexpected})"
        )

    node_id_set = set(node_ids)
    edge_pairs: list[tuple[str, str]] = []
    fanout = {node_id: 0 for node_id in node_ids}
    for edge in payload["edges"]:
        source = edge["from"]
        target = edge["to"]
        if source not in node_id_set or target not in node_id_set:
            raise ExecutionPlanError(f"Dependency edge references an unknown node: {source!r} -> {target!r}")
        if source == target:
            raise ExecutionPlanError(f"Dependency edge cannot be a self-edge: {source!r}")
        pair = (source, target)
        if pair in edge_pairs:
            raise ExecutionPlanError(f"Duplicate dependency edge: {source!r} -> {target!r}")
        edge_pairs.append(pair)
        fanout[source] += 1
        if fanout[source] > MAX_PLAN_FANOUT:
            raise ExecutionPlanLimitError(f"Node {source!r} exceeds maximum fanout {MAX_PLAN_FANOUT}")
    _topological_order(node_ids, edge_pairs)

    if payload["configuration_completeness"] == EXECUTION_COMPLETE:
        required_stage_ids = (
            StageRegistryIdentifier.LUX_PREPROCESS.value,
            StageRegistryIdentifier.LUX_DEPTH.value,
            StageRegistryIdentifier.LUX_OUTPUT.value,
        )
        lifecycle_nodes: dict[str, str] = {}
        for stage_registry_id in required_stage_ids:
            matching_node_ids = [node["id"] for node in nodes if node["stage_registry_id"] == stage_registry_id]
            if len(matching_node_ids) != 1:
                raise ExecutionPlanError(f"An execution-complete Lux plan must contain exactly one {stage_registry_id!r} node")
            lifecycle_nodes[stage_registry_id] = matching_node_ids[0]

        outgoing: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
        incoming: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
        for source, target in edge_pairs:
            outgoing[source].add(target)
            incoming[target].add(source)

        preprocess_id = lifecycle_nodes[StageRegistryIdentifier.LUX_PREPROCESS.value]
        depth_id = lifecycle_nodes[StageRegistryIdentifier.LUX_DEPTH.value]
        output_id = lifecycle_nodes[StageRegistryIdentifier.LUX_OUTPUT.value]
        sources = {node_id for node_id, predecessors in incoming.items() if not predecessors}
        sinks = {node_id for node_id, successors in outgoing.items() if not successors}
        if sources != {preprocess_id}:
            raise ExecutionPlanError("An execution-complete Lux plan must have preprocess as its only source node")
        if sinks != {output_id}:
            raise ExecutionPlanError("An execution-complete Lux plan must have output as its only sink node")

        reachable_from_preprocess: set[str] = set()
        pending = [preprocess_id]
        while pending:
            current = pending.pop()
            if current in reachable_from_preprocess:
                continue
            reachable_from_preprocess.add(current)
            pending.extend(outgoing[current])
        if reachable_from_preprocess != node_id_set:
            raise ExecutionPlanError("Every execution-complete Lux node must be reachable from preprocess")

        reaching_output: set[str] = set()
        pending = [output_id]
        while pending:
            current = pending.pop()
            if current in reaching_output:
                continue
            reaching_output.add(current)
            pending.extend(incoming[current])
        if reaching_output != node_id_set:
            raise ExecutionPlanError("Every execution-complete Lux node must have a dependency path to output")

        # Connectivity alone is insufficient execution authority: a single
        # source/sink DAG can still schedule a depth consumer before the depth
        # producer (for example preprocess -> pbr -> depth -> output). Require
        # every current depth-dependent Lux stage to be downstream of the one
        # authoritative depth node. Paths may be indirect so valid fan-out and
        # future intermediary stages remain representable.
        reachable_from_depth: set[str] = set()
        pending = [depth_id]
        while pending:
            current = pending.pop()
            if current in reachable_from_depth:
                continue
            reachable_from_depth.add(current)
            pending.extend(outgoing[current])
        depth_dependent_stage_ids = {
            StageRegistryIdentifier.LUX_MATERIALS_V3.value,
            StageRegistryIdentifier.LUX_PBR.value,
            StageRegistryIdentifier.LUX_V2.value,
            StageRegistryIdentifier.LUX_RECONSTRUCTION.value,
        }
        invalid_depth_consumers = sorted(
            node["id"]
            for node in nodes
            if node["stage_registry_id"] in depth_dependent_stage_ids and node["id"] not in reachable_from_depth
        )
        if invalid_depth_consumers:
            raise ExecutionPlanError(
                "Every execution-complete Lux depth consumer must have a "
                f"dependency path from depth (invalid={invalid_depth_consumers})"
            )

    expected_fingerprint = compute_execution_plan_fingerprint(payload)
    if payload["plan_fingerprint_sha256"] != expected_fingerprint:
        raise ExecutionPlanError("plan_fingerprint_sha256 does not match the canonical plan body")


def validate_execution_plan_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed on schema, registry, topology, bounds, or digest drift."""

    if not isinstance(payload, Mapping):
        raise ExecutionPlanError("Execution plan payload must be a JSON object")
    schema_value = payload.get("schema")
    if schema_value != EXECUTION_PLAN_SCHEMA:
        raise UnsupportedExecutionPlanSchema(
            f"Unsupported execution plan schema {schema_value!r}; expected {EXECUTION_PLAN_SCHEMA!r}"
        )

    import jsonschema

    validator = jsonschema.Draft202012Validator(load_execution_plan_schema())
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: _format_schema_path(list(error.absolute_path)),
    )
    if errors:
        first = errors[0]
        raise ExecutionPlanError(
            f"Execution plan schema validation failed at {_format_schema_path(list(first.absolute_path))}: " f"{first.message}"
        ) from first
    _validate_plan_semantics(payload)


def _preflight_json_text(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    string_length = 0
    for character in text:
        if in_string:
            if escaped:
                escaped = False
                string_length += 1
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            else:
                string_length += 1
            if string_length > MAX_PLAN_STRING_LENGTH:
                raise ExecutionPlanLimitError(f"JSON string exceeds maximum length {MAX_PLAN_STRING_LENGTH}")
            continue
        if character == '"':
            in_string = True
            string_length = 0
        elif character in "[{":
            depth += 1
            if depth > MAX_PLAN_JSON_DEPTH:
                raise ExecutionPlanLimitError(f"JSON nesting exceeds maximum depth {MAX_PLAN_JSON_DEPTH}")
        elif character in "]}":
            depth -= 1


def decode_bounded_json_object(data: str | bytes) -> dict[str, Any]:
    """Decode one bounded JSON object with duplicate-member rejection."""

    if isinstance(data, bytes):
        if len(data) > MAX_PLAN_BODY_BYTES:
            raise ExecutionPlanLimitError(f"Plan body exceeds maximum size {MAX_PLAN_BODY_BYTES} bytes")
        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ExecutionPlanError("Plan body must be valid UTF-8") from exc
    elif isinstance(data, str):
        try:
            encoded_size = len(data.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ExecutionPlanError("Plan JSON must not contain Unicode surrogate code points") from exc
        if encoded_size > MAX_PLAN_BODY_BYTES:
            raise ExecutionPlanLimitError(f"Plan body exceeds maximum size {MAX_PLAN_BODY_BYTES} bytes")
        text = data
    else:
        raise TypeError("Plan JSON must be str or bytes")

    _preflight_json_text(text)

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateExecutionPlanKey(f"Duplicate JSON object member: {key!r}")
            result[key] = value
        return result

    def reject_non_finite(value: str) -> Any:
        raise ExecutionPlanError(f"Non-finite JSON number is forbidden: {value}")

    def parse_bounded_int(value: str) -> int:
        if len(value.removeprefix("-")) > MAX_PLAN_INTEGER_DIGITS:
            raise ExecutionPlanLimitError(f"JSON integer exceeds maximum length {MAX_PLAN_INTEGER_DIGITS} digits")
        return int(value)

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ExecutionPlanError(f"Non-finite JSON number is forbidden: {value}")
        return parsed

    try:
        payload = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
            parse_float=parse_finite_float,
            parse_int=parse_bounded_int,
        )
    except ExecutionPlanError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ExecutionPlanError(f"Invalid execution plan JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExecutionPlanError("Execution plan JSON must contain one object")
    return payload


def parse_execution_plan_json(data: str | bytes) -> ExecutionPlan:
    """Bounded-decode, structurally validate, and freeze a canonical plan.

    This generic entrypoint does not grant backend/model authority. A domain
    consumer must additionally revalidate an ``execution_complete`` plan.
    """

    return ExecutionPlan.from_payload(decode_bounded_json_object(data))


__all__ = [
    "BackendCandidateIntent",
    "BackendModelIntent",
    "CanonicalExecutionPlan",
    "DependencyEdge",
    "DuplicateExecutionPlanKey",
    "EXECUTION_PLAN_SCHEMA",
    "EXECUTION_COMPLETE",
    "ExecutionPlan",
    "ExecutionPlanError",
    "ExecutionPlanLimitError",
    "InputSafetyLimits",
    "LicenseAcknowledgements",
    "LicenseEvaluation",
    "MAX_PLAN_BODY_BYTES",
    "MAX_PLAN_EDGES",
    "MAX_PLAN_FANOUT",
    "MAX_PLAN_INPUTS",
    "MAX_PLAN_INTEGER_DIGITS",
    "MAX_PLAN_JSON_DEPTH",
    "MAX_PLAN_NODES",
    "MAX_PLAN_REQUESTED_OUTPUTS",
    "MAX_PLAN_STRING_LENGTH",
    "MAX_DECODED_PIXELS_PER_INPUT",
    "MAX_INPUT_DECOMPRESSION_RATIO",
    "MAX_TOTAL_DECODED_PIXELS",
    "OutputDeclaration",
    "PlanInput",
    "PlanResources",
    "ResolvedModelIntent",
    "ResourceRange",
    "StageNode",
    "STRUCTURAL_LEGACY",
    "UnsupportedExecutionPlanSchema",
    "compute_execution_plan_fingerprint",
    "decode_bounded_json_object",
    "load_execution_plan_schema",
    "parse_execution_plan_json",
    "validate_execution_plan_payload",
    "with_execution_plan_fingerprint",
]
