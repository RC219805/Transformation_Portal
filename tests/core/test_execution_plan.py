"""Contract tests for the non-activating canonical execution plan."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from transformation_portal.core.execution_plan import (
    EXECUTION_COMPLETE,
    EXECUTION_PLAN_SCHEMA,
    MAX_DECODED_PIXELS_PER_INPUT,
    MAX_INPUT_DECOMPRESSION_RATIO,
    MAX_PLAN_BODY_BYTES,
    MAX_PLAN_INTEGER_DIGITS,
    MAX_PLAN_JSON_DEPTH,
    MAX_PLAN_STRING_LENGTH,
    MAX_TOTAL_DECODED_PIXELS,
    CanonicalExecutionPlan,
    DuplicateExecutionPlanKey,
    ExecutionPlanError,
    ExecutionPlanLimitError,
    UnsupportedExecutionPlanSchema,
    load_execution_plan_schema,
    parse_execution_plan_json,
    validate_execution_plan_payload,
    with_execution_plan_fingerprint,
)
from transformation_portal.stage_graph.registry import (
    StageRegistryIdentifier,
    get_stage_definition,
)

pytestmark = pytest.mark.unit


def _node(
    node_id: str,
    registry_id: StageRegistryIdentifier,
    *,
    configuration: dict,
    outputs: list[dict],
    optional: bool = False,
) -> dict:
    definition = get_stage_definition(registry_id)
    return {
        "id": node_id,
        "stage_registry_id": registry_id.value,
        "configuration": configuration,
        "resources": definition.resources.to_payload(),
        "outputs": outputs,
        "optional": optional,
        "failure_policy": "omit_outputs" if optional else "abort_plan",
    }


def _valid_payload() -> dict:
    preprocess = _node(
        "lux.preprocess",
        StageRegistryIdentifier.LUX_PREPROCESS,
        configuration={
            "schema": "tp.stage.config.lux.preprocess.v1",
            "configuration_completeness": EXECUTION_COMPLETE,
            "verify_images": True,
            "raw_ingest_mode": "auto",
            "raw_wb_mode": "camera",
            "raw_demosaic": "AHD",
            "raw_python_executable": None,
            "raw_preview_escape_enabled": False,
            "output_key_hash_algorithm": "sha1",
            "parallel_enabled": True,
            "max_workers": 4,
            "max_gpu_workers": 1,
        },
        outputs=[
            {
                "id": "lux.preprocess.output.preprocessed_image",
                "artifact_kind": "preprocessed_image",
                "scope": "per_input",
                "cardinality": "one",
                "required": True,
                "disposition": "intermediate",
            }
        ],
    )
    depth = _node(
        "lux.depth",
        StageRegistryIdentifier.LUX_DEPTH,
        configuration={
            "schema": "tp.stage.config.lux.depth.v1",
            "configuration_completeness": EXECUTION_COMPLETE,
            "planned_backend": "synthetic",
            "candidate_fallback_chain": ["synthetic"],
            "resolved_model_key": None,
            "resolved_model_revision": None,
            "device": "cpu",
            "quantization": "none",
            "verify_writes": True,
            "save_float_depth": False,
            "force": False,
            "fallback_mode": "fail",
            "allow_semantic_fallback": False,
            "allow_synthetic_fallback": True,
            "da3_python_executable": None,
            "da3_subprocess_timeout_seconds": 900,
            "depth_pro_checkpoint_path": None,
            "depth_pro_python_executable": None,
            "hash_mode": "if_manifest_exists",
            "manifest_cache_enabled": True,
            "depth_cache_enabled": False,
            "depth_cache_max_size_gb": 10.0,
            "postprocessing": {
                "apply_metric_scaling": True,
                "scale_factor": 1.0,
                "apply_median_filter": False,
                "median_kernel_size": 3,
                "apply_bilateral_filter": False,
                "bilateral_sigma_color": 0.0,
                "bilateral_sigma_space": 0.0,
                "preserve_edges": True,
                "edge_threshold": 0.1,
                "fusion_mode": "weighted",
                "refinement": None,
            },
            "apex_gate": {
                "quality_tier": "standard",
                "min_finite_pct": 0.999,
                "min_upper_iqr": 0.0001,
                "max_high_saturation_fraction": 0.02,
                "max_low_saturation_fraction": 0.02,
                "scaled_saturation_margin": 0.0025,
                "low_saturation_warning_band": 0.0075,
                "saturation_high_value": 0.999,
                "saturation_low_value": 0.001,
                "min_gradient_energy": 0.0005,
                "threshold_epsilon": 0.000001,
                "hist_bins": 64,
                "depth_fallback": "fail",
            },
            "ensemble": None,
        },
        outputs=[
            {
                "id": "lux.depth.output.depth_map",
                "artifact_kind": "depth_map",
                "scope": "per_input",
                "cardinality": "one",
                "required": True,
                "disposition": "intermediate",
            }
        ],
    )
    output = _node(
        "lux.output",
        StageRegistryIdentifier.LUX_OUTPUT,
        configuration={
            "schema": "tp.stage.config.lux.output.v1",
            "configuration_completeness": EXECUTION_COMPLETE,
            "requested_outputs": ["batch_manifest_json"],
            "output_bit_depth": 8,
            "hash_mode": "if_manifest_exists",
            "run_card_enabled": False,
            "run_card_version": "v1",
            "run_card_include_proofs": False,
            "keep_intermediates": False,
            "captioning": {
                "enabled": False,
                "backend": "fastvlm",
                "selector": "default",
                "model_id": None,
                "model_revision": None,
                "proxy_format": "png",
                "max_side_px": 1600,
                "python_executable": None,
                "mlx_vlm_dir": None,
                "timeout_seconds": 180,
            },
        },
        outputs=[
            {
                "id": "lux.output.output.batch_manifest_json",
                "artifact_kind": "batch_manifest_json",
                "scope": "per_run",
                "cardinality": "one",
                "required": True,
                "disposition": "requested",
            }
        ],
    )
    return with_execution_plan_fingerprint(
        {
            "schema": EXECUTION_PLAN_SCHEMA,
            "canonicalization": "tp.canonical.json.v1",
            "configuration_completeness": EXECUTION_COMPLETE,
            "planned_backend": "synthetic",
            "candidate_fallback_chain": ["synthetic"],
            "backend_candidates": [{"backend_id": "synthetic", "model_contracts": []}],
            "resolved_model": None,
            "license_acknowledgements": {
                "non_commercial_ok": False,
                "apple_depth_pro_research": False,
                "research_tools": False,
            },
            "license_evaluation": {"enforced": True, "status": "allowed"},
            "quality_tier": "standard",
            "preset_requested": None,
            "preset_resolved": "quality_tier:standard",
            "input_selection": {
                "root": "/contained/inputs",
                "files": [{"id": "input-000001", "path": "a.jpg"}],
            },
            "input_limits": {
                "max_decoded_pixels_per_input": 268_435_456,
                "max_total_decoded_pixels": 17_179_869_184,
                "max_decompression_ratio": 1_000,
            },
            "config_fingerprint_sha256": "a" * 64,
            "nodes": [preprocess, depth, output],
            "edges": [
                {"from": "lux.preprocess", "to": "lux.depth"},
                {"from": "lux.depth", "to": "lux.output"},
            ],
            "requested_outputs": ["batch_manifest_json"],
            "warnings": [],
        }
    )


def _refingerprint(payload: dict) -> dict:
    payload.pop("plan_fingerprint_sha256", None)
    return with_execution_plan_fingerprint(payload)


def _model_contract(backend_id: str, *, role: str = "primary", weight: float | None = None) -> dict:
    model_fields = {
        "da3": ("backend:da3", "da3_metric", "depth-anything/DA3METRIC-LARGE", "1" * 40, "apache-2.0"),
        "depth_pro": ("backend:depth_pro", "depth_pro", "apple/ml-depth-pro", None, "apple_amlr"),
        "da2": (
            "backend:da2",
            "da2_small",
            "depth-anything/Depth-Anything-V2-Small-hf",
            "2" * 40,
            "apache-2.0",
        ),
        "depthcrafter": (
            "backend:depthcrafter",
            "depthcrafter",
            "Tencent/DepthCrafter",
            "3" * 40,
            "apache-2.0",
        ),
    }
    selector, canonical_key, repo_id, revision, license_id = model_fields[backend_id]
    research_only = backend_id == "depth_pro"
    return {
        "role": role,
        "backend_id": backend_id,
        "model": {
            "requested_selector": selector,
            "resolution_reason": "native execution-plan fixture",
            "canonical_key": canonical_key,
            "repo_id": repo_id,
            "revision": revision,
            "license_id": license_id,
            "usage_class": "non_commercial_only" if research_only else "commercial_ok",
            "requires_non_commercial_ok": research_only,
            "accelerator_kind": "none",
            "legacy_model_variant_name": None,
        },
        "artifact_path": (
            "checkpoints/depth_pro.pt"
            if backend_id == "depth_pro"
            else "checkpoints/depthcrafter_v1.pt" if backend_id == "depthcrafter" else None
        ),
        "artifact_sha256": ("4" * 64 if backend_id == "depth_pro" else "5" * 64 if backend_id == "depthcrafter" else None),
        "enabled": True,
        "weight": weight,
        "device": "cpu",
    }


def _with_backend_shape(payload: dict, chain: list[str]) -> dict:
    candidates: list[dict] = []
    da3_model: dict | None = None
    for backend_id in chain:
        if backend_id == "synthetic":
            contracts: list[dict] = []
        elif backend_id == "ensemble":
            contracts = [
                _model_contract("depth_pro", role="ensemble_constituent", weight=0.5),
                _model_contract("da3", role="ensemble_constituent", weight=0.3),
            ]
            da3_model = contracts[1]["model"]
        else:
            contracts = [_model_contract(backend_id)]
            if backend_id == "da3":
                da3_model = contracts[0]["model"]
        candidates.append({"backend_id": backend_id, "model_contracts": contracts})

    payload["planned_backend"] = chain[0]
    payload["candidate_fallback_chain"] = chain
    payload["backend_candidates"] = candidates
    payload["resolved_model"] = copy.deepcopy(da3_model)
    payload["license_acknowledgements"] = {
        "non_commercial_ok": True,
        "apple_depth_pro_research": True,
        "research_tools": True,
    }
    depth_config = payload["nodes"][1]["configuration"]
    depth_config["planned_backend"] = chain[0]
    depth_config["candidate_fallback_chain"] = chain
    depth_config["resolved_model_key"] = None if da3_model is None else da3_model["canonical_key"]
    depth_config["resolved_model_revision"] = None if da3_model is None else da3_model["revision"]
    if "depth_pro" in chain or "ensemble" in chain:
        depth_config["depth_pro_checkpoint_path"] = "checkpoints/depth_pro.pt"
    depth_config["ensemble"] = (
        {
            "fusion_method": "variance_weighted",
            "max_variance_threshold": 0.15,
            "temporal_post_filter": {"mode": "ema", "alpha": 0.3},
        }
        if "ensemble" in chain
        else None
    )
    return _refingerprint(payload)


def _authoritative_model_contract(
    backend_id: str,
    *,
    role: str = "primary",
    weight: float | None = None,
) -> dict:
    contract = _model_contract(backend_id, role=role, weight=weight)
    if backend_id == "da3":
        from transformation_portal.core.security.model_lock import manifest_revision_for_repo
        from transformation_portal.lux_depth_v3.model_registry import get_model_spec

        spec = get_model_spec("da3_metric")
        contract["model"].update(
            {
                "requested_selector": "da3-metric",
                "canonical_key": spec.key,
                "repo_id": spec.repo_id,
                "revision": manifest_revision_for_repo(spec.repo_id),
                "license_id": spec.license_id,
                "usage_class": spec.usage_class.value,
                "requires_non_commercial_ok": spec.requires_non_commercial_ok,
            }
        )
    elif backend_id == "depth_pro":
        from transformation_portal.depth.backends.depth_pro import DepthProBackend

        contract["model"].update(
            {
                "requested_selector": "backend:depth_pro",
                "canonical_key": "depth_pro",
                "repo_id": "apple/ml-depth-pro",
                "revision": None,
                "license_id": "apple_amlr",
                "usage_class": "non_commercial_only",
                "requires_non_commercial_ok": True,
            }
        )
        contract["artifact_sha256"] = DepthProBackend.EXPECTED_SHA256
    elif backend_id == "da2":
        from transformation_portal.core.security.model_lock import manifest_revision_for_repo
        from transformation_portal.depth.models.depth_anything_v2 import ModelVariant as DA2ModelVariant

        repo_id = DA2ModelVariant.SMALL.value
        contract["model"].update(
            {
                "requested_selector": "backend:da2",
                "canonical_key": "da2_small",
                "repo_id": repo_id,
                "revision": manifest_revision_for_repo(repo_id),
                "license_id": "apache-2.0",
                "usage_class": "commercial_ok",
                "requires_non_commercial_ok": False,
            }
        )
    return contract


def _authoritative_backend_shape(chain: list[str]) -> dict:
    payload = _with_backend_shape(_valid_payload(), chain)
    da3_model = None
    for candidate in payload["backend_candidates"]:
        backend_id = candidate["backend_id"]
        if backend_id == "ensemble":
            candidate["model_contracts"] = [
                _authoritative_model_contract("depth_pro", role="ensemble_constituent", weight=0.5),
                _authoritative_model_contract("da3", role="ensemble_constituent", weight=0.3),
            ]
        elif backend_id not in {"synthetic", "depthcrafter"}:
            candidate["model_contracts"] = [_authoritative_model_contract(backend_id)]
        for contract in candidate["model_contracts"]:
            if contract["backend_id"] == "da3":
                da3_model = contract["model"]
    payload["resolved_model"] = copy.deepcopy(da3_model)
    depth_config = payload["nodes"][1]["configuration"]
    depth_config["resolved_model_key"] = None if da3_model is None else da3_model["canonical_key"]
    depth_config["resolved_model_revision"] = None if da3_model is None else da3_model["revision"]
    return _refingerprint(payload)


def test_packaged_schema_is_valid_and_closed() -> None:
    schema = load_execution_plan_schema()
    jsonschema.Draft202012Validator.check_schema(schema)

    assert schema["$id"] == EXECUTION_PLAN_SCHEMA
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["stageNode"]["additionalProperties"] is False
    assert schema["$defs"]["depthConfiguration"]["additionalProperties"] is False
    output_configuration = schema["$defs"]["outputConfiguration"]
    assert "output_bit_depth" in output_configuration["properties"]
    assert "output_bit_depth" in output_configuration["required"]
    assert {"bit_depth", "emit_master16", "emit_upscaled16"}.isdisjoint(output_configuration["properties"])
    assert "bit_depth_16_intermediates" not in schema["$defs"]["allArtifactKind"]["enum"]
    captioning_configuration = schema["$defs"]["captioningConfiguration"]
    frozen_fastvlm_fields = {"model_path", "review_model_path", "max_tokens", "temperature"}
    assert frozen_fastvlm_fields.issubset(captioning_configuration["properties"])
    assert frozen_fastvlm_fields.isdisjoint(captioning_configuration["required"])
    input_limit_properties = schema["$defs"]["inputLimits"]["properties"]
    assert input_limit_properties["max_decoded_pixels_per_input"]["maximum"] == MAX_DECODED_PIXELS_PER_INPUT
    assert input_limit_properties["max_total_decoded_pixels"]["maximum"] == MAX_TOTAL_DECODED_PIXELS
    assert input_limit_properties["max_decompression_ratio"]["maximum"] == MAX_INPUT_DECOMPRESSION_RATIO


def test_payload_round_trip_is_canonical_and_immutable() -> None:
    payload = _valid_payload()
    plan = CanonicalExecutionPlan.from_payload(payload)

    assert plan.to_payload() == payload
    assert plan.ordered_node_ids() == ("lux.preprocess", "lux.depth", "lux.output")
    assert parse_execution_plan_json(plan.to_canonical_json()).to_canonical_json() == plan.to_canonical_json()
    with pytest.raises(TypeError):
        plan.nodes[0].configuration["schema"] = "forged"  # type: ignore[index]


def test_output_bit_depth_requires_an_exact_integer() -> None:
    payload = _valid_payload()
    payload["nodes"][-1]["configuration"]["output_bit_depth"] = 16.0

    with pytest.raises(ExecutionPlanError, match="exact integer 8 or 16"):
        CanonicalExecutionPlan.from_payload(_refingerprint(payload))


def test_execution_complete_lifecycle_requires_preprocess() -> None:
    payload = _valid_payload()
    payload["nodes"] = payload["nodes"][1:]
    payload["edges"] = [{"from": "lux.depth", "to": "lux.output"}]

    with pytest.raises(ExecutionPlanError, match="exactly one 'tp.stage.lux.preprocess.v1'"):
        CanonicalExecutionPlan.from_payload(_refingerprint(payload))


def test_execution_complete_lifecycle_rejects_disconnected_required_nodes() -> None:
    payload = _valid_payload()
    payload["edges"] = []

    with pytest.raises(ExecutionPlanError, match="preprocess as its only source"):
        CanonicalExecutionPlan.from_payload(_refingerprint(payload))


def test_execution_complete_depth_consumers_require_dependency_path_from_depth() -> None:
    payload = _valid_payload()
    consumer = _node(
        "lux.pbr",
        StageRegistryIdentifier.LUX_PBR,
        configuration={
            "schema": "tp.stage.config.lux.pbr.v1",
            "configuration_completeness": EXECUTION_COMPLETE,
            "normal_strength": 1.0,
            "normal_blur_radius": 1,
            "roughness_strength": 1.0,
            "roughness_blur_radius": 1,
            "ao_strength": 1.0,
            "ao_blur_radius": 1,
            "ao_bias": 0.0,
        },
        outputs=[],
        optional=True,
    )
    payload["nodes"].insert(1, consumer)
    payload["edges"] = [
        {"from": "lux.preprocess", "to": "lux.pbr"},
        {"from": "lux.pbr", "to": "lux.depth"},
        {"from": "lux.depth", "to": "lux.output"},
    ]

    with pytest.raises(ExecutionPlanError, match="dependency path from depth"):
        CanonicalExecutionPlan.from_payload(_refingerprint(payload))

    payload["edges"] = [
        {"from": "lux.preprocess", "to": "lux.depth"},
        {"from": "lux.depth", "to": "lux.pbr"},
        {"from": "lux.pbr", "to": "lux.output"},
    ]
    plan = CanonicalExecutionPlan.from_payload(_refingerprint(payload))
    assert plan.ordered_node_ids() == (
        "lux.preprocess",
        "lux.depth",
        "lux.pbr",
        "lux.output",
    )


@pytest.mark.parametrize(
    "candidate_chain",
    (
        ["depth_pro"],
        ["da3"],
        ["da2"],
        ["depthcrafter"],
        ["ensemble"],
        ["depth_pro", "da3", "da2"],
    ),
)
def test_native_execution_complete_carrier_supports_current_backend_shapes(candidate_chain: list[str]) -> None:
    payload = _with_backend_shape(_valid_payload(), candidate_chain)

    plan = CanonicalExecutionPlan.from_payload(payload)

    assert plan.configuration_completeness == EXECUTION_COMPLETE
    assert [candidate.backend_id for candidate in plan.backend_candidates] == candidate_chain


@pytest.mark.parametrize(
    "candidate_chain",
    (["synthetic"], ["depth_pro"], ["da3"], ["da2"], ["ensemble"], ["depth_pro", "da3", "da2"]),
)
def test_lux_authority_accepts_current_pinned_execution_complete_shapes(candidate_chain: list[str]) -> None:
    from transformation_portal.lux_depth_v3.execution_plan_adapter import revalidate_lux_execution_plan_authority

    plan = revalidate_lux_execution_plan_authority(_authoritative_backend_shape(candidate_chain))

    assert plan.candidate_fallback_chain == tuple(candidate_chain)


@pytest.mark.parametrize(
    ("candidate_chain", "mutation", "expected"),
    (
        (
            ["da3"],
            lambda payload: payload["backend_candidates"][0]["model_contracts"][0]["model"].update(
                {"repo_id": "attacker/forged-da3"}
            ),
            "disagrees with registry",
        ),
        (
            ["da2"],
            lambda payload: payload["backend_candidates"][0]["model_contracts"][0]["model"].update({"revision": "f" * 40}),
            "revision.*authority",
        ),
        (
            ["depth_pro"],
            lambda payload: payload["backend_candidates"][0]["model_contracts"][0]["model"].update(
                {"license_id": "forged-license"}
            ),
            "license_id.*authority",
        ),
        (
            ["depth_pro"],
            lambda payload: payload["license_acknowledgements"].update({"apple_depth_pro_research": False}),
            "Apple.*license|accept_apple",
        ),
    ),
)
def test_generic_parser_is_not_lux_authority(candidate_chain, mutation, expected: str) -> None:
    from transformation_portal.lux_depth_v3.execution_plan_adapter import revalidate_lux_execution_plan_authority

    payload = _authoritative_backend_shape(candidate_chain)
    mutation(payload)
    if candidate_chain == ["da3"]:
        payload["resolved_model"] = copy.deepcopy(payload["backend_candidates"][0]["model_contracts"][0]["model"])
    payload = _refingerprint(payload)

    # Structural parsing verifies the closed carrier and digest, not live Lux
    # model/backend authority.
    parse_execution_plan_json(json.dumps(payload))
    with pytest.raises(ExecutionPlanError, match=expected):
        revalidate_lux_execution_plan_authority(payload)


def test_canonical_json_and_fingerprint_are_repeatable() -> None:
    first = CanonicalExecutionPlan.from_payload(_valid_payload())
    second = CanonicalExecutionPlan.from_payload(_valid_payload())

    assert first.plan_fingerprint_sha256 == second.plan_fingerprint_sha256
    assert first.to_canonical_json().encode("utf-8") == second.to_canonical_json().encode("utf-8")


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        (lambda payload: payload.update({"unknown": True}), "Additional properties"),
        (
            lambda payload: payload["nodes"][0]["configuration"].update({"module": "evil.module"}),
            "not valid under any of the given schemas",
        ),
        (
            lambda payload: payload["nodes"][0].update({"stage_registry_id": "tp.stage.attacker.import.v1"}),
            "is not one of",
        ),
    ),
)
def test_closed_schema_rejects_unknown_fields_and_registry_ids(mutation, expected_error: str) -> None:
    payload = _valid_payload()
    mutation(payload)
    payload = _refingerprint(payload)

    with pytest.raises(ExecutionPlanError, match=expected_error):
        validate_execution_plan_payload(payload)


def test_unknown_schema_version_fails_before_generic_validation() -> None:
    payload = _valid_payload()
    payload["schema"] = "tp.execution.plan.v2"

    with pytest.raises(UnsupportedExecutionPlanSchema, match="Unsupported"):
        validate_execution_plan_payload(payload)


def test_fingerprint_drift_fails_closed() -> None:
    payload = _valid_payload()
    payload["warnings"].append("drifted after fingerprinting")

    with pytest.raises(ExecutionPlanError, match="fingerprint"):
        validate_execution_plan_payload(payload)


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "../escape.jpg",
        "/absolute.jpg",
        "dir\\escape.jpg",
        "./a.jpg",
        ".",
        "a/.",
        "a/",
    ),
)
def test_input_paths_must_be_contained_relative_posix_paths(unsafe_path: str) -> None:
    payload = _valid_payload()
    payload["input_selection"]["files"][0]["path"] = unsafe_path
    payload = _refingerprint(payload)

    with pytest.raises(ExecutionPlanError, match="contained|portable"):
        validate_execution_plan_payload(payload)


def test_cycles_and_resource_inversions_fail_semantic_validation() -> None:
    cyclic = _valid_payload()
    cyclic["edges"].append({"from": "lux.output", "to": "lux.preprocess"})
    cyclic = _refingerprint(cyclic)
    with pytest.raises(ExecutionPlanError, match="cycle"):
        validate_execution_plan_payload(cyclic)

    inverted = _valid_payload()
    inverted["nodes"][0]["resources"]["memory_mib"] = {"minimum": 1024, "maximum": 512}
    inverted = _refingerprint(inverted)
    with pytest.raises(ExecutionPlanError, match="minimum above maximum"):
        validate_execution_plan_payload(inverted)

    invalid_input_limits = _valid_payload()
    invalid_input_limits["input_limits"]["max_total_decoded_pixels"] = 1
    invalid_input_limits = _refingerprint(invalid_input_limits)
    with pytest.raises(ExecutionPlanError, match="at least the per-input"):
        validate_execution_plan_payload(invalid_input_limits)

    floating_input_limit = _valid_payload()
    floating_input_limit["input_limits"]["max_decompression_ratio"] = 1_000.0
    floating_input_limit = _refingerprint(floating_input_limit)
    with pytest.raises(ExecutionPlanError, match="exact integer"):
        validate_execution_plan_payload(floating_input_limit)

    floating_resource = _valid_payload()
    floating_resource["nodes"][0]["resources"]["cpu_cores"]["minimum"] = 1.0
    floating_resource = _refingerprint(floating_resource)
    with pytest.raises(ExecutionPlanError, match="exact integer"):
        validate_execution_plan_payload(floating_resource)


def test_registry_profiles_and_requested_output_projection_fail_closed() -> None:
    mismatched_output_config = _valid_payload()
    mismatched_output_config["nodes"][-1]["configuration"]["requested_outputs"] = ["run_card"]
    mismatched_output_config = _refingerprint(mismatched_output_config)
    with pytest.raises(ExecutionPlanError, match="output configuration"):
        validate_execution_plan_payload(mismatched_output_config)

    excessive_resource = _valid_payload()
    excessive_resource["nodes"][0]["resources"]["gpu_count"] = {"minimum": 0, "maximum": 1}
    excessive_resource = _refingerprint(excessive_resource)
    with pytest.raises(ExecutionPlanError, match="registry profile"):
        validate_execution_plan_payload(excessive_resource)

    duplicate_artifact = _valid_payload()
    duplicate_output = copy.deepcopy(duplicate_artifact["nodes"][-1]["outputs"][0])
    duplicate_output["id"] = "lux.output.output.batch_manifest_json.copy"
    duplicate_artifact["nodes"][-1]["outputs"].append(duplicate_output)
    duplicate_artifact = _refingerprint(duplicate_artifact)
    with pytest.raises(ExecutionPlanError, match="duplicate artifact kind"):
        validate_execution_plan_payload(duplicate_artifact)

    optional_request = _valid_payload()
    optional_request["nodes"][-1]["outputs"][0]["required"] = False
    optional_request = _refingerprint(optional_request)
    validate_execution_plan_payload(optional_request)
    assert CanonicalExecutionPlan.from_payload(optional_request).nodes[-1].outputs[0].required is False

    optional_required_output = _valid_payload()
    optional_required_output["nodes"][-1]["optional"] = True
    optional_required_output["nodes"][-1]["failure_policy"] = "omit_outputs"
    optional_required_output = _refingerprint(optional_required_output)
    with pytest.raises(ExecutionPlanError, match="optional node cannot declare required"):
        validate_execution_plan_payload(optional_required_output)

    wrong_output_shape = _valid_payload()
    wrong_output_shape["nodes"][-1]["outputs"][0].update({"scope": "per_input", "cardinality": "many"})
    wrong_output_shape = _refingerprint(wrong_output_shape)
    with pytest.raises(ExecutionPlanError, match="scope .* does not match"):
        validate_execution_plan_payload(wrong_output_shape)


def test_required_lux_stage_nodes_and_requested_producers_are_unique() -> None:
    missing_depth = _valid_payload()
    missing_depth["nodes"] = [missing_depth["nodes"][0], missing_depth["nodes"][-1]]
    missing_depth["edges"] = [{"from": "lux.preprocess", "to": "lux.output"}]
    missing_depth = _refingerprint(missing_depth)
    with pytest.raises(ExecutionPlanError, match="exactly one depth"):
        validate_execution_plan_payload(missing_depth)

    missing_output = _valid_payload()
    missing_output["nodes"] = missing_output["nodes"][:-1]
    missing_output["edges"] = [{"from": "lux.preprocess", "to": "lux.depth"}]
    missing_output = _refingerprint(missing_output)
    with pytest.raises(ExecutionPlanError, match="exactly one output"):
        validate_execution_plan_payload(missing_output)

    duplicate_requested_producer = _valid_payload()
    duplicate_requested_producer["configuration_completeness"] = "structural_legacy"
    for node in duplicate_requested_producer["nodes"]:
        node["configuration"]["configuration_completeness"] = "structural_legacy"
    requested_pbr_output = {
        "id": "lux.pbr.one.output.pbr_maps",
        "artifact_kind": "pbr_maps",
        "scope": "per_input",
        "cardinality": "many",
        "required": False,
        "disposition": "requested",
    }
    pbr_one = _node(
        "lux.pbr.one",
        StageRegistryIdentifier.LUX_PBR,
        configuration={
            "schema": "tp.stage.config.lux.pbr.v1",
            "configuration_completeness": "structural_legacy",
        },
        outputs=[requested_pbr_output],
        optional=True,
    )
    pbr_two_output = copy.deepcopy(requested_pbr_output)
    pbr_two_output["id"] = "lux.pbr.two.output.pbr_maps"
    pbr_two = _node(
        "lux.pbr.two",
        StageRegistryIdentifier.LUX_PBR,
        configuration={
            "schema": "tp.stage.config.lux.pbr.v1",
            "configuration_completeness": "structural_legacy",
        },
        outputs=[pbr_two_output],
        optional=True,
    )
    duplicate_requested_producer["nodes"][-1]["configuration"]["requested_outputs"] = ["pbr_maps"]
    duplicate_requested_producer["nodes"][-1]["outputs"] = []
    duplicate_requested_producer["nodes"][2:2] = [pbr_one, pbr_two]
    duplicate_requested_producer["requested_outputs"] = ["pbr_maps"]
    duplicate_requested_producer = _refingerprint(duplicate_requested_producer)
    with pytest.raises(ExecutionPlanError, match="multiple producer nodes"):
        validate_execution_plan_payload(duplicate_requested_producer)


def test_topological_order_uses_node_order_to_break_ready_ties() -> None:
    payload = _valid_payload()
    payload["configuration_completeness"] = "structural_legacy"
    for node in payload["nodes"]:
        node["configuration"]["configuration_completeness"] = "structural_legacy"
    pbr = _node(
        "lux.pbr",
        StageRegistryIdentifier.LUX_PBR,
        configuration={
            "schema": "tp.stage.config.lux.pbr.v1",
            "configuration_completeness": "structural_legacy",
        },
        outputs=[],
        optional=True,
    )
    materials = _node(
        "lux.materials_v3",
        StageRegistryIdentifier.LUX_MATERIALS_V3,
        configuration={
            "schema": "tp.stage.config.lux.materials_v3.v1",
            "configuration_completeness": "structural_legacy",
        },
        outputs=[],
        optional=True,
    )
    payload["nodes"][2:2] = [pbr, materials]
    payload["edges"] = [
        {"from": "lux.preprocess", "to": "lux.depth"},
        # Deliberately list the later payload node first.
        {"from": "lux.depth", "to": "lux.materials_v3"},
        {"from": "lux.depth", "to": "lux.pbr"},
        {"from": "lux.pbr", "to": "lux.output"},
        {"from": "lux.materials_v3", "to": "lux.output"},
    ]
    plan = CanonicalExecutionPlan.from_payload(_refingerprint(payload))

    assert plan.ordered_node_ids() == (
        "lux.preprocess",
        "lux.depth",
        "lux.pbr",
        "lux.materials_v3",
        "lux.output",
    )


def test_execution_complete_candidates_require_carried_model_authority() -> None:
    missing_fallback_model = _valid_payload()
    missing_fallback_model["candidate_fallback_chain"] = ["synthetic", "da3"]
    missing_fallback_model["backend_candidates"] = [
        {"backend_id": "synthetic", "model_contracts": []},
        {"backend_id": "da3", "model_contracts": []},
    ]
    missing_fallback_model["nodes"][1]["configuration"]["candidate_fallback_chain"] = ["synthetic", "da3"]
    missing_fallback_model = _refingerprint(missing_fallback_model)

    with pytest.raises(ExecutionPlanError, match="requires 1 model contract"):
        validate_execution_plan_payload(missing_fallback_model)

    incomplete_ensemble = _valid_payload()
    incomplete_ensemble["planned_backend"] = "ensemble"
    incomplete_ensemble["candidate_fallback_chain"] = ["ensemble"]
    incomplete_ensemble["backend_candidates"] = [{"backend_id": "ensemble", "model_contracts": []}]
    incomplete_ensemble["nodes"][1]["configuration"].update(
        {
            "planned_backend": "ensemble",
            "candidate_fallback_chain": ["ensemble"],
        }
    )
    incomplete_ensemble = _refingerprint(incomplete_ensemble)
    with pytest.raises(ExecutionPlanError, match="at least two enabled constituents"):
        validate_execution_plan_payload(incomplete_ensemble)


@pytest.mark.security
def test_bounded_decoder_rejects_duplicate_keys_non_finite_values_and_limits() -> None:
    canonical = CanonicalExecutionPlan.from_payload(_valid_payload()).to_canonical_json()
    duplicate = canonical.replace(
        '"schema":"tp.execution.plan.v1"',
        '"schema":"tp.execution.plan.v1","schema":"tp.execution.plan.v1"',
        1,
    )
    with pytest.raises(DuplicateExecutionPlanKey, match="Duplicate"):
        parse_execution_plan_json(duplicate)
    with pytest.raises(ExecutionPlanError, match="Non-finite"):
        parse_execution_plan_json('{"schema":NaN}')
    with pytest.raises(ExecutionPlanError, match="Non-finite"):
        parse_execution_plan_json('{"schema":1e999}')
    with pytest.raises(ExecutionPlanLimitError, match="integer exceeds"):
        parse_execution_plan_json('{"schema":' + "9" * (MAX_PLAN_INTEGER_DIGITS + 1) + "}")
    with pytest.raises(ExecutionPlanLimitError, match="maximum size"):
        parse_execution_plan_json(b"{" + b" " * MAX_PLAN_BODY_BYTES + b"}")
    with pytest.raises(ExecutionPlanLimitError, match="maximum depth"):
        parse_execution_plan_json("[" * (MAX_PLAN_JSON_DEPTH + 1))
    with pytest.raises(ExecutionPlanLimitError, match="maximum length"):
        parse_execution_plan_json(json.dumps({"x": "a" * (MAX_PLAN_STRING_LENGTH + 1)}))


def test_lone_surrogates_are_normalized_to_execution_plan_errors() -> None:
    with pytest.raises(ExecutionPlanError, match="surrogate"):
        parse_execution_plan_json('{"schema":"\ud800"}')
    with pytest.raises(ExecutionPlanError, match="surrogate"):
        with_execution_plan_fingerprint({"schema": EXECUTION_PLAN_SCHEMA, "warning": "\udfff"})


def test_core_root_uses_explicit_alias_without_replacing_legacy_name() -> None:
    import transformation_portal.core as core
    from transformation_portal.core.execution_plan import ExecutionPlan

    assert core.CanonicalExecutionPlan is ExecutionPlan
    assert "ExecutionPlan" not in core.__all__


def test_schema_validation_does_not_construct_stage_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformation_portal.stage_graph.graph import StageGraph

    monkeypatch.setattr(StageGraph, "__init__", lambda self, *args, **kwargs: pytest.fail("StageGraph constructed"))
    payload = copy.deepcopy(_valid_payload())

    validate_execution_plan_payload(payload)


def test_core_plan_import_does_not_load_graph_or_optional_ml_dependencies() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    script = """
import sys

blocked = (
    "numpy",
    "torch",
    "transformation_portal.stage_graph.graph",
    "transformation_portal.stage_graph.stages",
)
attempted = []

class RecordBlockedImports:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            attempted.append(fullname)
        return None

sys.meta_path.insert(0, RecordBlockedImports())
from transformation_portal.core.execution_plan import load_execution_plan_schema

assert load_execution_plan_schema()["$id"] == "tp.execution.plan.v1"
assert not attempted, attempted
assert all(name not in sys.modules for name in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
