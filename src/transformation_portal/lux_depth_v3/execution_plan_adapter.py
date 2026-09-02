"""Read-only Lux v1 compatibility adapter for canonical execution plans.

The adapter consumes the landed ``tp.lux.resolved_invocation.v1`` contract,
revalidates its carried model/backend authority, and returns an immutable
``tp.execution.plan.v1`` semantic plan.  It never constructs a StageGraph,
loads a model, writes an artifact, or changes the live Lux executor.
"""

from __future__ import annotations

import math
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping, Optional

from ..core.execution_plan import (
    EXECUTION_COMPLETE,
    EXECUTION_PLAN_SCHEMA,
    MAX_DECODED_PIXELS_PER_INPUT,
    MAX_INPUT_DECOMPRESSION_RATIO,
    MAX_PLAN_INPUTS,
    MAX_PLAN_NODES,
    MAX_PLAN_REQUESTED_OUTPUTS,
    MAX_TOTAL_DECODED_PIXELS,
    STRUCTURAL_LEGACY,
    CanonicalExecutionPlan,
    ExecutionPlanError,
    decode_bounded_json_object,
    parse_execution_plan_json,
    with_execution_plan_fingerprint,
)
from ..depth.backends.protocol import LicenseRestrictionError
from ..depth.backends.registry import DepthBackendRegistry, UnknownDepthBackendError
from ..ingest.canonical_json import TP_CANONICAL_JSON_PROFILE
from ..stage_graph.registry import (
    StageRegistryIdentifier,
    get_output_definition,
    get_stage_definition,
)
from .config import DA3Config, Preset, _parse_legacy_output_bool
from .model_registry import (
    DEFAULT_MODEL_KEY,
    DEFAULT_MODEL_SELECTOR,
    LEGACY_MODEL_VARIANT_ALIASES,
    AcceleratorKind,
    get_model_spec,
    resolve_registry_key,
)
from .model_resolution import (
    ModelLicenseError,
    ResolvedModel,
    UntrustedModelContractError,
    validate_authoritative_model_contract,
)
from .resolved_invocation import (
    RESOLVED_INVOCATION_SCHEMA,
    ResolvedInvocation,
    validate_resolved_invocation_payload,
)


class ResolvedInvocationCompatibilityError(ExecutionPlanError):
    """A legacy v1 payload cannot be safely promoted to the core contract."""


class LuxExecutionPlanAuthorityError(ExecutionPlanError):
    """A canonical Lux plan fails carried model/backend/license authority checks."""


_CANONICAL_LEGACY_STAGE_ORDER = (
    "preprocess",
    "depth",
    "materials_v3",
    "pbr",
    "v2",
    "reconstruction",
    "output",
)

LEGACY_STAGE_REGISTRY_IDS = MappingProxyType(
    {
        "preprocess": StageRegistryIdentifier.LUX_PREPROCESS,
        "depth": StageRegistryIdentifier.LUX_DEPTH,
        "materials_v3": StageRegistryIdentifier.LUX_MATERIALS_V3,
        "pbr": StageRegistryIdentifier.LUX_PBR,
        "v2": StageRegistryIdentifier.LUX_V2,
        "reconstruction": StageRegistryIdentifier.LUX_RECONSTRUCTION,
        "output": StageRegistryIdentifier.LUX_OUTPUT,
    }
)

_LEGACY_NODE_IDS = MappingProxyType({stage: f"lux.{stage}" for stage in _CANONICAL_LEGACY_STAGE_ORDER})

_REQUESTED_ARTIFACT_OWNER = MappingProxyType(
    {
        "depth_u16_png": "depth",
        "depth_metadata_json": "depth",
        "depth_float_npy": "depth",
        "materials_v3_masks": "materials_v3",
        "pbr_maps": "pbr",
        "v2_enhanced_image": "v2",
        "reconstruction_bundle": "reconstruction",
        "combined_manifest_json": "output",
        "batch_manifest_json": "output",
        "run_card": "output",
    }
)

_INTERNAL_OUTPUTS = MappingProxyType(
    {
        "preprocess": ("preprocessed_image",),
        "depth": ("depth_map",),
    }
)

_MAX_LEGACY_BACKENDS = 8
_MAX_LEGACY_WARNINGS = 64
_LEGACY_OUTPUT_DEPTH_ALIASES = ("emit_master16", "emit_upscaled16")
_LEGACY_16_BIT_ARTIFACT = "bit_depth_16_intermediates"


def _normalize_legacy_output_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy and reconcile pre-#2068 output intent with the current schema.

    Older ``tp.lux.resolved_invocation.v1`` payloads had no
    ``output_bit_depth`` field.  Their only serialized 16-bit signal was the
    now-retired ``bit_depth_16_intermediates`` requested artifact, although
    additive producers could also carry the deprecated config aliases.  The
    v1 compatibility promise requires consuming those shapes without letting
    the aliases or fictional artifact cross the canonical boundary.
    """

    normalized = dict(payload)
    requested_artifacts = normalized.get("requested_artifacts")
    artifact_marker = False
    if isinstance(requested_artifacts, list):
        artifact_marker = _LEGACY_16_BIT_ARTIFACT in requested_artifacts
        normalized["requested_artifacts"] = [
            artifact for artifact in requested_artifacts if artifact != _LEGACY_16_BIT_ARTIFACT
        ]

    truthy_aliases: list[str] = []
    for alias in _LEGACY_OUTPUT_DEPTH_ALIASES:
        value = normalized.pop(alias, None)
        # EnhanceConfig and the portal both treat null aliases as omitted.
        if value is None:
            continue
        try:
            if _parse_legacy_output_bool(value, alias):
                truthy_aliases.append(alias)
        except ValueError as exc:
            raise ResolvedInvocationCompatibilityError(f"Legacy output-depth alias {alias!r} is invalid: {exc}") from exc

    truthy_markers = list(truthy_aliases)
    if artifact_marker:
        truthy_markers.append(_LEGACY_16_BIT_ARTIFACT)
    has_truthy_marker = bool(truthy_markers)
    if "output_bit_depth" not in normalized:
        normalized["output_bit_depth"] = 16 if has_truthy_marker else 8
    else:
        output_bit_depth = normalized["output_bit_depth"]
        if type(output_bit_depth) is not int or output_bit_depth not in {8, 16}:
            raise ResolvedInvocationCompatibilityError("output_bit_depth must be the exact integer 8 or 16")
        if output_bit_depth == 8 and has_truthy_marker:
            markers = ", ".join(truthy_markers)
            raise ResolvedInvocationCompatibilityError(
                f"output_bit_depth=8 conflicts with truthy legacy 16-bit marker(s): {markers}"
            )

    return normalized


def _validate_legacy_stage_order(stages: list[str]) -> None:
    if len(stages) > MAX_PLAN_NODES:
        raise ResolvedInvocationCompatibilityError(f"Legacy plan exceeds the {MAX_PLAN_NODES}-node safety limit")
    if len(stages) != len(set(stages)):
        raise ResolvedInvocationCompatibilityError("Legacy plan contains duplicate stages")
    unknown = [stage for stage in stages if stage not in LEGACY_STAGE_REGISTRY_IDS]
    if unknown:
        raise ResolvedInvocationCompatibilityError(f"Legacy plan names unknown stage registry identifiers: {unknown}")
    if not stages or stages[0] != "preprocess" or "depth" not in stages or stages[-1] != "output":
        raise ResolvedInvocationCompatibilityError(
            "Legacy Lux plan must start with preprocess, contain depth, and end with output"
        )
    expected = [stage for stage in _CANONICAL_LEGACY_STAGE_ORDER if stage in stages]
    if stages != expected:
        raise ResolvedInvocationCompatibilityError(
            f"Legacy stage order {stages!r} does not match canonical Lux order {expected!r}"
        )


def _revalidate_model_payload(
    model_payload: Optional[Mapping[str, Any]],
    *,
    candidate_chain: tuple[str, ...],
    non_commercial_ok: bool,
) -> Optional[dict[str, Any]]:
    da3_permitted = "da3" in candidate_chain
    if model_payload is None:
        if da3_permitted:
            raise ResolvedInvocationCompatibilityError("A legacy plan permitting DA3 is missing resolved_model")
        return None
    if not da3_permitted:
        raise ResolvedInvocationCompatibilityError(
            "A legacy plan without DA3 in its candidate chain must not carry a DA3 resolved_model"
        )

    canonical_key = model_payload["canonical_key"]
    try:
        spec = get_model_spec(canonical_key)
    except KeyError as exc:
        raise UntrustedModelContractError(f"Legacy plan names unknown model registry key {canonical_key!r}") from exc

    expected_registry_fields = {
        "repo_id": spec.repo_id,
        "license_id": spec.license_id,
        "usage_class": spec.usage_class.value,
        "requires_non_commercial_ok": spec.requires_non_commercial_ok,
    }
    for field_name, expected_value in expected_registry_fields.items():
        if model_payload[field_name] != expected_value:
            raise UntrustedModelContractError(
                f"Legacy plan model field {field_name!r} disagrees with registry key "
                f"{canonical_key!r}; refusing compatibility promotion"
            )

    requested_selector = model_payload["requested_selector"]
    legacy_variant_name = model_payload.get("legacy_model_variant_name")
    selector_key: Optional[str]
    if requested_selector == DEFAULT_MODEL_SELECTOR:
        if legacy_variant_name is not None:
            raise UntrustedModelContractError("Default model provenance must not carry a legacy model variant")
        selector_key = DEFAULT_MODEL_KEY
    elif requested_selector.startswith("preset:"):
        if legacy_variant_name is not None:
            raise UntrustedModelContractError("Typed preset model provenance must not carry a legacy model variant")
        preset_value = requested_selector.removeprefix("preset:")
        try:
            preset_variant = DA3Config.from_preset(Preset(preset_value)).model_variant
            selector_key = LEGACY_MODEL_VARIANT_ALIASES[preset_variant.name]
        except (KeyError, ValueError) as exc:
            raise UntrustedModelContractError(
                f"Legacy plan carries unknown typed preset selector {requested_selector!r}"
            ) from exc
    elif legacy_variant_name is not None:
        selector_key = LEGACY_MODEL_VARIANT_ALIASES.get(legacy_variant_name)
        if selector_key is None or requested_selector != legacy_variant_name:
            raise UntrustedModelContractError(
                "Legacy model variant provenance is unknown or disagrees with requested_selector"
            )
    else:
        selector_key = resolve_registry_key(requested_selector)

    if selector_key != canonical_key:
        raise UntrustedModelContractError(
            f"Legacy plan requested selector {requested_selector!r} resolves to {selector_key!r}, "
            f"not carried canonical key {canonical_key!r}"
        )

    accelerator_value = model_payload["accelerator_kind"]
    try:
        accelerator = AcceleratorKind(accelerator_value)
    except (TypeError, ValueError) as exc:
        raise UntrustedModelContractError(f"Legacy plan carries unknown accelerator kind {accelerator_value!r}") from exc
    if accelerator is AcceleratorKind.COREML and not spec.supports_coreml:
        raise UntrustedModelContractError(
            f"Legacy plan requests CoreML for registry key {canonical_key!r}, which does not support it"
        )

    contract = ResolvedModel(
        requested_selector=model_payload["requested_selector"],
        resolution_reason=model_payload.get("resolution_reason", ""),
        canonical_key=canonical_key,
        spec=spec,
        revision=model_payload["revision"],
        fallback_chain=(),
        accelerator_kind=accelerator,
        legacy_model_variant_name=model_payload.get("legacy_model_variant_name"),
    )
    validated = validate_authoritative_model_contract(
        contract,
        non_commercial_ok=non_commercial_ok,
    )
    return {
        "requested_selector": validated.requested_selector,
        "resolution_reason": validated.resolution_reason,
        "canonical_key": validated.canonical_key,
        "repo_id": validated.spec.repo_id,
        "revision": validated.revision,
        "license_id": validated.spec.license_id,
        "usage_class": validated.spec.usage_class.value,
        "requires_non_commercial_ok": validated.spec.requires_non_commercial_ok,
        "accelerator_kind": validated.accelerator_kind.value,
        "legacy_model_variant_name": validated.legacy_model_variant_name,
    }


def _revalidate_backend_chain(payload: Mapping[str, Any]) -> tuple[str, ...]:
    acknowledgements = payload["license_acknowledgements"]
    config_view: Any = SimpleNamespace(
        non_commercial_ok=acknowledgements["non_commercial_ok"],
        accept_apple_depth_pro_research_license=acknowledgements["apple_depth_pro_research"],
        accept_research_tools_license=acknowledgements["research_tools"],
    )
    registry = DepthBackendRegistry()
    chain = tuple(
        registry.validate_backend_request(backend_id, config_view) for backend_id in payload["candidate_fallback_chain"]
    )
    if not chain or chain[0] != payload["planned_backend"]:
        raise ResolvedInvocationCompatibilityError("Legacy candidate_fallback_chain must start with planned_backend")
    if len(chain) != len(set(chain)):
        raise ResolvedInvocationCompatibilityError("Legacy candidate fallback chain contains duplicates")
    return chain


def _node_configuration(
    stage: str,
    *,
    definition_schema: str,
    planned_backend: str,
    candidate_chain: tuple[str, ...],
    resolved_model: Optional[Mapping[str, Any]],
    requested_outputs: list[str],
    output_bit_depth: int,
) -> dict[str, Any]:
    configuration: dict[str, Any] = {
        "schema": definition_schema,
        "configuration_completeness": STRUCTURAL_LEGACY,
    }
    if stage == "depth":
        configuration.update(
            {
                "planned_backend": planned_backend,
                "candidate_fallback_chain": list(candidate_chain),
                "resolved_model_key": None if resolved_model is None else resolved_model["canonical_key"],
                "resolved_model_revision": None if resolved_model is None else resolved_model["revision"],
            }
        )
    elif stage == "output":
        configuration.update(
            {
                "requested_outputs": list(requested_outputs),
                "output_bit_depth": output_bit_depth,
            }
        )
    return configuration


def _legacy_stage_is_optional(stage: str, *, quality_tier: str) -> bool:
    """Project the live Lux failure policy without inventing missing config."""

    if stage == "pbr":
        return True
    if stage == "materials_v3":
        return quality_tier.lower() != "apex"
    return False


def _legacy_requested_output_is_required(stage: str, artifact_kind: str, *, quality_tier: str) -> bool:
    """Return whether the current executor fails the run when this output fails."""

    if artifact_kind == "run_card":
        return False
    return not _legacy_stage_is_optional(stage, quality_tier=quality_tier)


def _output_declaration(
    node_id: str,
    artifact_kind: str,
    *,
    disposition: str,
    required: bool = True,
) -> dict[str, Any]:
    definition = get_output_definition(artifact_kind)
    return {
        "id": f"{node_id}.output.{artifact_kind}",
        "artifact_kind": artifact_kind,
        "scope": definition.scope.value,
        "cardinality": definition.cardinality.value,
        "required": required,
        "disposition": disposition,
    }


def adapt_resolved_invocation_payload(payload: Mapping[str, Any]) -> CanonicalExecutionPlan:
    """Promote a schema-valid Lux v1 payload without mutating or executing it.

    The legacy schema's documented unknown-field compatibility is preserved
    here.  Unknown fields are ignored only while producing a new closed-world
    payload; they can never survive into ``tp.execution.plan.v1``.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("ResolvedInvocation compatibility payload must be a mapping")
    if payload.get("schema") != RESOLVED_INVOCATION_SCHEMA:
        raise ResolvedInvocationCompatibilityError(
            f"Unsupported compatibility schema {payload.get('schema')!r}; " f"expected {RESOLVED_INVOCATION_SCHEMA!r}"
        )
    prevalidation_limits = (
        ("candidate_fallback_chain", _MAX_LEGACY_BACKENDS),
        ("stages", MAX_PLAN_NODES),
        ("input_files", MAX_PLAN_INPUTS),
        ("requested_artifacts", MAX_PLAN_REQUESTED_OUTPUTS),
        ("warnings", _MAX_LEGACY_WARNINGS),
    )
    for field_name, maximum in prevalidation_limits:
        value = payload.get(field_name)
        if isinstance(value, (list, tuple)) and len(value) > maximum:
            raise ResolvedInvocationCompatibilityError(
                f"Legacy plan field {field_name!r} exceeds its {maximum}-item safety limit"
            )
    normalized_payload = _normalize_legacy_output_contract(payload)
    validate_resolved_invocation_payload(normalized_payload)
    payload = normalized_payload

    stages = list(payload["stages"])
    _validate_legacy_stage_order(stages)
    input_files = list(payload["input_files"])
    requested_outputs = list(payload["requested_artifacts"])
    if len(requested_outputs) != len(set(requested_outputs)):
        raise ResolvedInvocationCompatibilityError("Legacy plan contains duplicate requested artifacts")

    unknown_artifacts = [artifact for artifact in requested_outputs if artifact not in _REQUESTED_ARTIFACT_OWNER]
    if unknown_artifacts:
        raise ResolvedInvocationCompatibilityError(f"Legacy plan names unknown requested artifacts: {unknown_artifacts}")
    absent_owners = sorted(
        {
            _REQUESTED_ARTIFACT_OWNER[artifact]
            for artifact in requested_outputs
            if _REQUESTED_ARTIFACT_OWNER[artifact] not in stages
        }
    )
    if absent_owners:
        raise ResolvedInvocationCompatibilityError(f"Requested artifacts require absent stages: {absent_owners}")

    license_evaluation = payload["license_evaluation"]
    # The provisional v1 schema permits additive fields, so compare only the
    # fields that carry v1 authority.
    if license_evaluation.get("enforced") is not True or license_evaluation.get("status") != "allowed":
        raise ResolvedInvocationCompatibilityError("Compatibility promotion requires an enforced, allowed license evaluation")
    acknowledgements = payload["license_acknowledgements"]
    if "reconstruction" in stages and (
        acknowledgements["non_commercial_ok"] is not True or acknowledgements["research_tools"] is not True
    ):
        raise ResolvedInvocationCompatibilityError(
            "Legacy reconstruction intent requires non-commercial and research-tools acknowledgements"
        )
    candidate_chain = _revalidate_backend_chain(payload)
    if "ensemble" in candidate_chain:
        raise ResolvedInvocationCompatibilityError(
            "Legacy ensemble intent lacks constituent model identities required for canonical promotion"
        )
    resolved_model = _revalidate_model_payload(
        payload["resolved_model"],
        candidate_chain=candidate_chain,
        non_commercial_ok=acknowledgements["non_commercial_ok"],
    )

    requested_by_stage: dict[str, list[str]] = {stage: [] for stage in stages}
    for artifact in requested_outputs:
        requested_by_stage[_REQUESTED_ARTIFACT_OWNER[artifact]].append(artifact)

    nodes: list[dict[str, Any]] = []
    for stage in stages:
        registry_id = LEGACY_STAGE_REGISTRY_IDS[stage]
        definition = get_stage_definition(registry_id)
        node_id = _LEGACY_NODE_IDS[stage]
        node_optional = _legacy_stage_is_optional(stage, quality_tier=payload["quality_tier"])
        outputs = [
            _output_declaration(node_id, artifact, disposition="intermediate") for artifact in _INTERNAL_OUTPUTS.get(stage, ())
        ]
        outputs.extend(
            _output_declaration(
                node_id,
                artifact,
                disposition="requested",
                required=_legacy_requested_output_is_required(
                    stage,
                    artifact,
                    quality_tier=payload["quality_tier"],
                ),
            )
            for artifact in requested_by_stage[stage]
        )
        nodes.append(
            {
                "id": node_id,
                "stage_registry_id": registry_id.value,
                "configuration": _node_configuration(
                    stage,
                    definition_schema=definition.configuration_schema,
                    planned_backend=payload["planned_backend"],
                    candidate_chain=candidate_chain,
                    resolved_model=resolved_model,
                    requested_outputs=requested_outputs,
                    output_bit_depth=payload["output_bit_depth"],
                ),
                "resources": definition.resources.to_payload(),
                "outputs": outputs,
                "optional": node_optional,
                "failure_policy": "omit_outputs" if node_optional else "abort_plan",
            }
        )

    edges = [{"from": _LEGACY_NODE_IDS[source], "to": _LEGACY_NODE_IDS[target]} for source, target in zip(stages, stages[1:])]
    canonical_payload = with_execution_plan_fingerprint(
        {
            "schema": EXECUTION_PLAN_SCHEMA,
            "canonicalization": TP_CANONICAL_JSON_PROFILE,
            "configuration_completeness": STRUCTURAL_LEGACY,
            "planned_backend": payload["planned_backend"],
            "candidate_fallback_chain": list(candidate_chain),
            "backend_candidates": [
                {
                    "backend_id": backend_id,
                    "model_contracts": (
                        [
                            {
                                "role": "primary",
                                "backend_id": "da3",
                                "model": dict(resolved_model),
                                "artifact_path": None,
                                "artifact_sha256": None,
                                "enabled": True,
                                "weight": None,
                                "device": "legacy_unspecified",
                            }
                        ]
                        if backend_id == "da3" and resolved_model is not None
                        else []
                    ),
                }
                for backend_id in candidate_chain
            ],
            "resolved_model": resolved_model,
            "license_acknowledgements": {
                "non_commercial_ok": acknowledgements["non_commercial_ok"],
                "apple_depth_pro_research": acknowledgements["apple_depth_pro_research"],
                "research_tools": acknowledgements["research_tools"],
            },
            "license_evaluation": {
                "enforced": license_evaluation["enforced"],
                "status": license_evaluation["status"],
            },
            "quality_tier": payload["quality_tier"],
            "preset_requested": payload.get("preset_requested"),
            "preset_resolved": payload.get("preset_resolved"),
            "input_selection": {
                "root": payload["input_dir"],
                "files": [{"id": f"input-{index:06d}", "path": path} for index, path in enumerate(input_files, start=1)],
            },
            "input_limits": {
                "max_decoded_pixels_per_input": MAX_DECODED_PIXELS_PER_INPUT,
                "max_total_decoded_pixels": MAX_TOTAL_DECODED_PIXELS,
                "max_decompression_ratio": MAX_INPUT_DECOMPRESSION_RATIO,
            },
            "config_fingerprint_sha256": payload["config_fingerprint_sha256"],
            "nodes": nodes,
            "edges": edges,
            "requested_outputs": requested_outputs,
            "warnings": list(payload["warnings"]),
        }
    )
    return CanonicalExecutionPlan.from_payload(canonical_payload)


def adapt_resolved_invocation(invocation: ResolvedInvocation | Mapping[str, Any]) -> CanonicalExecutionPlan:
    """Adapt a trusted in-process object or an already-decoded v1 mapping."""

    if isinstance(invocation, ResolvedInvocation):
        return adapt_resolved_invocation_payload(invocation.to_payload())
    return adapt_resolved_invocation_payload(invocation)


def adapt_resolved_invocation_json(data: str | bytes) -> CanonicalExecutionPlan:
    """Bounded-decode and adapt serialized Lux v1 compatibility JSON."""

    return adapt_resolved_invocation_payload(decode_bounded_json_object(data))


def _authority_config_view(acknowledgements: Mapping[str, Any]) -> Any:
    return SimpleNamespace(
        non_commercial_ok=acknowledgements["non_commercial_ok"],
        accept_apple_depth_pro_research_license=acknowledgements["apple_depth_pro_research"],
        accept_research_tools_license=acknowledgements["research_tools"],
    )


def _require_exact_model_fields(
    model: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    backend_id: str,
) -> None:
    for field_name, expected_value in expected.items():
        if model[field_name] != expected_value:
            raise LuxExecutionPlanAuthorityError(
                f"Canonical {backend_id} model field {field_name!r} disagrees with current authority"
            )


def _revalidate_candidate_model_contract(
    contract: Mapping[str, Any],
    *,
    acknowledgements: Mapping[str, Any],
) -> None:
    """Re-anchor one complete carried identity without choosing a model."""

    backend_id = contract["backend_id"]
    model = contract["model"]
    if backend_id == "da3":
        revalidated = _revalidate_model_payload(
            model,
            candidate_chain=("da3",),
            non_commercial_ok=acknowledgements["non_commercial_ok"],
        )
        if revalidated != model or contract["artifact_sha256"] is not None:
            raise UntrustedModelContractError("Canonical DA3 candidate contract changed during authority revalidation")
        return

    if backend_id == "depth_pro":
        from ..depth.backends.depth_pro import DepthProBackend
        from .config_resolver import DEPTH_PRO_MODEL_ID

        _require_exact_model_fields(
            model,
            {
                "requested_selector": "backend:depth_pro",
                "canonical_key": "depth_pro",
                "repo_id": DEPTH_PRO_MODEL_ID,
                "revision": None,
                "license_id": "apple_amlr",
                "usage_class": "non_commercial_only",
                "requires_non_commercial_ok": True,
                "accelerator_kind": "none",
                "legacy_model_variant_name": None,
            },
            backend_id=backend_id,
        )
        if contract["artifact_sha256"] != DepthProBackend.EXPECTED_SHA256:
            raise LuxExecutionPlanAuthorityError("Canonical depth_pro artifact SHA-256 disagrees with the pinned checkpoint")
        return

    if backend_id == "da2":
        from ..core.security.model_lock import manifest_revision_for_repo
        from ..depth.models.depth_anything_v2 import ModelVariant as DA2ModelVariant

        repo_id = DA2ModelVariant.SMALL.value
        revision = manifest_revision_for_repo(repo_id)
        if revision is None:
            raise LuxExecutionPlanAuthorityError("The current DA2 model lock has no pinned Small-model revision")
        _require_exact_model_fields(
            model,
            {
                "requested_selector": "backend:da2",
                "canonical_key": "da2_small",
                "repo_id": repo_id,
                "revision": revision,
                "license_id": "apache-2.0",
                "usage_class": "commercial_ok",
                "requires_non_commercial_ok": False,
                "accelerator_kind": "none",
                "legacy_model_variant_name": None,
            },
            backend_id=backend_id,
        )
        if contract["artifact_sha256"] is not None:
            raise LuxExecutionPlanAuthorityError("Canonical DA2 Hugging Face identity must not carry a local artifact digest")
        return

    if backend_id == "depthcrafter":
        raise LuxExecutionPlanAuthorityError(
            "DepthCrafter has no pinned executable model identity in the current Lux backend; authority is unavailable"
        )

    raise LuxExecutionPlanAuthorityError(f"Backend {backend_id!r} must not carry a model contract")


def _runtime_node_configuration(
    payload: Mapping[str, Any],
    stage_registry_id: StageRegistryIdentifier,
) -> Optional[Mapping[str, Any]]:
    for node in payload["nodes"]:
        if node["stage_registry_id"] == stage_registry_id.value:
            return node["configuration"]
    return None


def _optional_nonempty_string(configuration: Mapping[str, Any], field_name: str) -> Optional[str]:
    value = configuration.get(field_name)
    if value is None:
        return None
    if type(value) is not str or not value.strip():
        raise LuxExecutionPlanAuthorityError(f"Canonical field {field_name!r} must be null or a non-empty string")
    return value


def _revalidate_runtime_policy_nodes(payload: Mapping[str, Any]) -> None:
    """Validate execution-complete policy fields consumed after preparation."""

    output = _runtime_node_configuration(payload, StageRegistryIdentifier.LUX_OUTPUT)
    if output is None or not isinstance(output.get("captioning"), Mapping):
        raise LuxExecutionPlanAuthorityError("Canonical output authority is missing its captioning configuration")
    captioning = output["captioning"]
    required_captioning_fields = {
        "enabled",
        "backend",
        "selector",
        "model_id",
        "model_revision",
        "proxy_format",
        "max_side_px",
        "python_executable",
        "mlx_vlm_dir",
        "timeout_seconds",
    }
    missing_captioning_fields = required_captioning_fields.difference(captioning)
    if missing_captioning_fields:
        raise LuxExecutionPlanAuthorityError(
            "Canonical captioning authority is missing fields: " + ", ".join(sorted(missing_captioning_fields))
        )
    if type(captioning["enabled"]) is not bool:
        raise LuxExecutionPlanAuthorityError("Canonical captioning enabled must be an exact boolean")
    for field_name in (
        "model_path",
        "review_model_path",
        "python_executable",
        "mlx_vlm_dir",
    ):
        _optional_nonempty_string(captioning, field_name)
    if captioning["enabled"]:
        if captioning["backend"] != "fastvlm":
            raise LuxExecutionPlanAuthorityError("Enabled canonical captioning requires backend 'fastvlm'")
        if type(captioning["selector"]) is not str or not captioning["selector"].strip():
            raise LuxExecutionPlanAuthorityError("Canonical captioning selector must be a non-empty string")
        if type(captioning["max_side_px"]) is not int or captioning["max_side_px"] <= 0:
            raise LuxExecutionPlanAuthorityError("Canonical captioning max_side_px must be a positive integer")
        if type(captioning["timeout_seconds"]) is not int or captioning["timeout_seconds"] <= 0:
            raise LuxExecutionPlanAuthorityError("Canonical FastVLM timeout_seconds must be a positive integer")
        max_tokens = captioning.get("max_tokens", 120)
        if type(max_tokens) is not int or max_tokens <= 0:
            raise LuxExecutionPlanAuthorityError("Canonical FastVLM max_tokens must be a positive integer")
        temperature = captioning.get("temperature", 0.0)
        if type(temperature) is not float or not math.isfinite(temperature) or temperature < 0:
            raise LuxExecutionPlanAuthorityError("Canonical FastVLM temperature must be a finite non-negative float")

    reconstruction = _runtime_node_configuration(payload, StageRegistryIdentifier.LUX_RECONSTRUCTION)
    if reconstruction is None:
        return
    required_reconstruction_fields = {
        "grouping_mode",
        "cameras_sidecar_path",
        "cameras_sidecar_sha256",
        "iterations",
        "tier",
        "emit_scene_debug_bundle",
        "risk_threshold",
    }
    missing_reconstruction_fields = required_reconstruction_fields.difference(reconstruction)
    if missing_reconstruction_fields:
        raise LuxExecutionPlanAuthorityError(
            "Canonical reconstruction authority is missing fields: " + ", ".join(sorted(missing_reconstruction_fields))
        )
    if reconstruction["grouping_mode"] not in {"single", "parent_dir"}:
        raise LuxExecutionPlanAuthorityError("Canonical reconstruction grouping_mode is unsupported")
    cameras_path = _optional_nonempty_string(reconstruction, "cameras_sidecar_path")
    cameras_sha256 = _optional_nonempty_string(reconstruction, "cameras_sidecar_sha256")
    if cameras_path is None:
        if cameras_sha256 is not None:
            raise LuxExecutionPlanAuthorityError("Canonical reconstruction sidecar digest has no sidecar path")
    elif (
        cameras_sha256 is None
        or len(cameras_sha256) != 64
        or any(character not in "0123456789abcdef" for character in cameras_sha256)
    ):
        raise LuxExecutionPlanAuthorityError("Canonical reconstruction sidecar SHA-256 is invalid")
    if type(reconstruction["iterations"]) is not int or reconstruction["iterations"] <= 0:
        raise LuxExecutionPlanAuthorityError("Canonical reconstruction iterations must be a positive integer")
    if type(reconstruction["tier"]) is not str or not reconstruction["tier"].strip():
        raise LuxExecutionPlanAuthorityError("Canonical reconstruction tier must be a non-empty string")
    if type(reconstruction["emit_scene_debug_bundle"]) is not bool:
        raise LuxExecutionPlanAuthorityError("Canonical reconstruction debug flag must be an exact boolean")
    risk_threshold = reconstruction["risk_threshold"]
    if type(risk_threshold) is not float or not math.isfinite(risk_threshold) or not 0 <= risk_threshold <= 1:
        raise LuxExecutionPlanAuthorityError("Canonical reconstruction risk threshold must be between 0 and 1")


def revalidate_lux_execution_plan_authority(
    plan: CanonicalExecutionPlan | Mapping[str, Any],
) -> CanonicalExecutionPlan:
    """Revalidate carried Lux authority without resolving or executing a model.

    The core parser establishes only structural validity.  This domain boundary
    additionally checks the current backend registry and license gates plus the
    exact carried DA3 registry/lock contract.  It never fills a missing model
    identity and therefore preserves fail-closed legacy carrier behavior.
    """

    payload = plan.to_payload() if isinstance(plan, CanonicalExecutionPlan) else dict(plan)
    validated_plan = CanonicalExecutionPlan.from_payload(payload)
    validated_payload = validated_plan.to_payload()

    if validated_plan.configuration_completeness != EXECUTION_COMPLETE:
        raise LuxExecutionPlanAuthorityError(
            "A structural_legacy plan is parse-only and cannot become Lux execution authority"
        )

    _revalidate_runtime_policy_nodes(validated_payload)

    try:
        candidate_chain = _revalidate_backend_chain(validated_payload)
    except (LicenseRestrictionError, UnknownDepthBackendError) as exc:
        raise LuxExecutionPlanAuthorityError(f"Canonical backend/license authority revalidation failed: {exc}") from exc
    if candidate_chain != validated_plan.candidate_fallback_chain:
        raise LuxExecutionPlanAuthorityError("Canonical candidate fallback chain changed during revalidation")

    acknowledgements = validated_payload["license_acknowledgements"]
    registry = DepthBackendRegistry()
    config_view = _authority_config_view(acknowledgements)
    for candidate in validated_payload["backend_candidates"]:
        for contract in candidate["model_contracts"]:
            if not contract["enabled"]:
                continue
            try:
                registry.validate_backend_request(contract["backend_id"], config_view)
            except (LicenseRestrictionError, UnknownDepthBackendError) as exc:
                raise LuxExecutionPlanAuthorityError(
                    f"Canonical constituent backend/license authority revalidation failed: {exc}"
                ) from exc
            try:
                _revalidate_candidate_model_contract(contract, acknowledgements=acknowledgements)
            except (ModelLicenseError, UntrustedModelContractError) as exc:
                raise LuxExecutionPlanAuthorityError(f"Canonical model authority revalidation failed: {exc}") from exc

    has_reconstruction = any(
        node.stage_registry_id is StageRegistryIdentifier.LUX_RECONSTRUCTION for node in validated_plan.nodes
    )
    if has_reconstruction and (
        acknowledgements["non_commercial_ok"] is not True or acknowledgements["research_tools"] is not True
    ):
        raise LuxExecutionPlanAuthorityError(
            "Canonical reconstruction intent requires non-commercial and research-tools acknowledgements"
        )

    return validated_plan


def revalidate_lux_execution_plan_json(data: str | bytes) -> CanonicalExecutionPlan:
    """Bounded-parse canonical JSON, then revalidate carried Lux authority."""

    return revalidate_lux_execution_plan_authority(parse_execution_plan_json(data))


__all__ = [
    "LEGACY_STAGE_REGISTRY_IDS",
    "LuxExecutionPlanAuthorityError",
    "ResolvedInvocationCompatibilityError",
    "adapt_resolved_invocation",
    "adapt_resolved_invocation_json",
    "adapt_resolved_invocation_payload",
    "revalidate_lux_execution_plan_authority",
    "revalidate_lux_execution_plan_json",
]
