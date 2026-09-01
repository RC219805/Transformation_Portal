"""Lux v1 -> core execution-plan compatibility contract tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from transformation_portal.core.execution_plan import EXECUTION_PLAN_SCHEMA, ExecutionPlanError
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.execution_plan_adapter import (
    LEGACY_STAGE_REGISTRY_IDS,
    LuxExecutionPlanAuthorityError,
    ResolvedInvocationCompatibilityError,
    adapt_resolved_invocation,
    adapt_resolved_invocation_json,
    adapt_resolved_invocation_payload,
    revalidate_lux_execution_plan_authority,
)
from transformation_portal.lux_depth_v3.model_resolution import (
    ModelLicenseError,
    UntrustedModelContractError,
)
from transformation_portal.lux_depth_v3.resolved_invocation import build_resolved_invocation

pytestmark = pytest.mark.unit


def _build(config: EnhanceConfig, tmp_path: Path):
    input_root = tmp_path / "inputs"
    input_root.mkdir(exist_ok=True)
    inputs = [input_root / "b.jpg", input_root / "a.jpg"]
    return build_resolved_invocation(config, input_root, inputs)


def _full_config() -> EnhanceConfig:
    return EnhanceConfig(
        model_key="da3-metric",
        non_commercial_ok=True,
        accept_research_tools_license=True,
        enable_materials_v3=True,
        generate_pbr=True,
        enable_v2=True,
        v2_preset="signature",
        enable_reconstruction=True,
        save_float_depth=True,
        emit_run_card=True,
        emit_master16=True,
    )


def test_adapter_preserves_model_license_inputs_stage_order_and_artifact_intent(tmp_path: Path) -> None:
    invocation = _build(_full_config(), tmp_path)
    plan = adapt_resolved_invocation(invocation)

    assert plan.schema == EXECUTION_PLAN_SCHEMA
    assert plan.planned_backend == invocation.planned_backend
    assert plan.candidate_fallback_chain == invocation.candidate_fallback_chain
    assert plan.resolved_model is not None and invocation.resolved_model is not None
    assert plan.resolved_model.canonical_key == invocation.resolved_model.canonical_key
    assert plan.resolved_model.revision == invocation.resolved_model.revision
    assert plan.license_acknowledgements.to_payload() == invocation.license_acknowledgements.to_payload()
    assert [item.path for item in plan.inputs] == list(invocation.input_files)
    assert plan.input_limits.max_decoded_pixels_per_input > 0
    assert plan.input_limits.max_decompression_ratio == 1_000
    assert plan.config_fingerprint_sha256 == invocation.config_fingerprint_sha256
    assert plan.requested_outputs == invocation.requested_artifacts
    assert plan.ordered_node_ids() == tuple(f"lux.{stage}" for stage in invocation.stages)
    assert [node.stage_registry_id for node in plan.nodes] == [LEGACY_STAGE_REGISTRY_IDS[stage] for stage in invocation.stages]


@pytest.mark.parametrize("output_bit_depth", (8, 16))
def test_current_output_depth_is_the_only_canonical_output_configuration(
    tmp_path: Path,
    output_bit_depth: int,
) -> None:
    invocation = _build(
        EnhanceConfig(
            model_key="da3-metric",
            output_bit_depth=output_bit_depth,
            enable_materials_v3=True,
        ),
        tmp_path,
    )

    plan = adapt_resolved_invocation(invocation)
    output_node = next(node for node in plan.nodes if node.stage_registry_id == LEGACY_STAGE_REGISTRY_IDS["output"])
    configuration = dict(output_node.configuration)

    assert configuration["output_bit_depth"] == output_bit_depth
    assert {"bit_depth", "emit_master16", "emit_upscaled16"}.isdisjoint(configuration)
    assert "bit_depth_16_intermediates" not in plan.requested_outputs
    assert all(output.artifact_kind != "bit_depth_16_intermediates" for node in plan.nodes for output in node.outputs)


@pytest.mark.parametrize(
    ("legacy_alias", "legacy_value", "expected_depth"),
    (
        ("emit_master16", None, 8),
        ("emit_master16", False, 8),
        ("emit_upscaled16", "off", 8),
        ("emit_master16", True, 16),
        ("emit_upscaled16", "on", 16),
    ),
)
def test_pre_2068_alias_shapes_normalize_before_current_schema_validation(
    tmp_path: Path,
    legacy_alias: str,
    legacy_value: object,
    expected_depth: int,
) -> None:
    payload = _build(EnhanceConfig(model_key="da3-metric"), tmp_path).to_payload()
    payload.pop("output_bit_depth")
    payload[legacy_alias] = legacy_value

    plan = adapt_resolved_invocation_payload(payload)
    output_node = next(node for node in plan.nodes if node.stage_registry_id == LEGACY_STAGE_REGISTRY_IDS["output"])

    assert output_node.configuration["output_bit_depth"] == expected_depth
    assert legacy_alias not in output_node.configuration


def test_pre_2068_payload_without_output_marker_preserves_historical_8_bit_default(tmp_path: Path) -> None:
    payload = _build(EnhanceConfig(model_key="da3-metric"), tmp_path).to_payload()
    payload.pop("output_bit_depth")

    plan = adapt_resolved_invocation_payload(payload)
    output_node = next(node for node in plan.nodes if node.stage_registry_id == LEGACY_STAGE_REGISTRY_IDS["output"])

    assert output_node.configuration["output_bit_depth"] == 8


def test_pre_2068_fictional_artifact_selects_16_bit_then_is_consumed_without_mutation(tmp_path: Path) -> None:
    payload = _build(EnhanceConfig(model_key="da3-metric"), tmp_path).to_payload()
    payload.pop("output_bit_depth")
    payload["requested_artifacts"].append("bit_depth_16_intermediates")
    original = copy.deepcopy(payload)

    plan = adapt_resolved_invocation_payload(payload)
    output_node = next(node for node in plan.nodes if node.stage_registry_id == LEGACY_STAGE_REGISTRY_IDS["output"])

    assert payload == original
    assert output_node.configuration["output_bit_depth"] == 16
    assert "bit_depth_16_intermediates" not in plan.requested_outputs
    assert "bit_depth_16_intermediates" not in plan.to_canonical_json()


@pytest.mark.parametrize(
    "legacy_marker",
    (
        {"emit_master16": True},
        {"emit_upscaled16": 1},
        {"requested_artifacts": "bit_depth_16_intermediates"},
    ),
)
def test_explicit_8_bit_conflicts_with_truthy_legacy_marker(
    tmp_path: Path,
    legacy_marker: dict[str, object],
) -> None:
    payload = _build(EnhanceConfig(model_key="da3-metric", output_bit_depth=8), tmp_path).to_payload()
    artifact = legacy_marker.get("requested_artifacts")
    if artifact is not None:
        payload["requested_artifacts"].append(artifact)
    else:
        payload.update(legacy_marker)

    with pytest.raises(ResolvedInvocationCompatibilityError, match="conflicts with truthy legacy"):
        adapt_resolved_invocation_payload(payload)


def test_explicit_16_bit_consumes_all_legacy_markers(tmp_path: Path) -> None:
    payload = _build(EnhanceConfig(model_key="da3-metric", output_bit_depth=16), tmp_path).to_payload()
    payload.update({"emit_master16": True, "emit_upscaled16": None})
    payload["requested_artifacts"].append("bit_depth_16_intermediates")

    plan = adapt_resolved_invocation_payload(payload)
    output_node = next(node for node in plan.nodes if node.stage_registry_id == LEGACY_STAGE_REGISTRY_IDS["output"])

    assert output_node.configuration["output_bit_depth"] == 16
    assert "bit_depth_16_intermediates" not in plan.requested_outputs
    assert {"emit_master16", "emit_upscaled16"}.isdisjoint(output_node.configuration)


def test_output_depth_rejects_json_number_that_is_not_an_exact_integer(tmp_path: Path) -> None:
    payload = _build(EnhanceConfig(model_key="da3-metric", output_bit_depth=16), tmp_path).to_payload()
    payload["output_bit_depth"] = 16.0

    with pytest.raises(ResolvedInvocationCompatibilityError, match="exact integer 8 or 16"):
        adapt_resolved_invocation_payload(payload)


def test_unknown_legacy_extension_is_not_deep_copied_or_observed(tmp_path: Path) -> None:
    class NoDeepCopy:
        def __deepcopy__(self, _memo):
            raise AssertionError("unknown legacy extensions must remain unobserved")

    payload = _build(EnhanceConfig(model_key="da3-metric"), tmp_path).to_payload()
    payload["future_annotation"] = NoDeepCopy()

    plan = adapt_resolved_invocation_payload(payload)

    assert plan.schema == "tp.execution.plan.v1"


def test_legacy_projection_is_structural_only_and_preserves_live_failure_policies(tmp_path: Path) -> None:
    invocation = _build(_full_config(), tmp_path)
    plan = adapt_resolved_invocation(invocation)
    nodes = {node.stage_registry_id: node for node in plan.nodes}

    assert plan.configuration_completeness == "structural_legacy"
    assert all(node.configuration["configuration_completeness"] == "structural_legacy" for node in plan.nodes)
    assert nodes[LEGACY_STAGE_REGISTRY_IDS["pbr"]].optional is True
    assert nodes[LEGACY_STAGE_REGISTRY_IDS["materials_v3"]].optional is True
    assert nodes[LEGACY_STAGE_REGISTRY_IDS["reconstruction"]].optional is False
    required_by_kind = {
        output.artifact_kind: output.required
        for node in plan.nodes
        for output in node.outputs
        if output.disposition == "requested"
    }
    assert required_by_kind["pbr_maps"] is False
    assert required_by_kind["materials_v3_masks"] is False
    assert required_by_kind["run_card"] is False
    assert required_by_kind["reconstruction_bundle"] is True

    with pytest.raises(LuxExecutionPlanAuthorityError, match="parse-only"):
        revalidate_lux_execution_plan_authority(plan)


def test_apex_materials_failure_policy_is_blocking(tmp_path: Path) -> None:
    config = _full_config()
    config.quality_tier = "apex"
    plan = adapt_resolved_invocation(_build(config, tmp_path))
    materials = next(node for node in plan.nodes if node.stage_registry_id == LEGACY_STAGE_REGISTRY_IDS["materials_v3"])

    assert materials.optional is False
    assert materials.failure_policy == "abort_plan"
    assert all(output.required for output in materials.outputs)


def test_object_payload_and_json_adapters_are_byte_identical(tmp_path: Path) -> None:
    invocation = _build(EnhanceConfig(model_key="da3-metric"), tmp_path)

    object_plan = adapt_resolved_invocation(invocation)
    payload_plan = adapt_resolved_invocation_payload(invocation.to_payload())
    json_plan = adapt_resolved_invocation_json(invocation.to_canonical_json())

    assert object_plan.to_canonical_json() == payload_plan.to_canonical_json() == json_plan.to_canonical_json()
    assert object_plan.plan_fingerprint_sha256 == payload_plan.plan_fingerprint_sha256


def test_legacy_unknown_fields_are_consumed_but_never_promoted(tmp_path: Path) -> None:
    payload = _build(EnhanceConfig(model_key="da3-metric"), tmp_path).to_payload()
    payload["future_legacy_annotation"] = {"ignored": True}
    payload["license_acknowledgements"]["future_acknowledgement"] = True
    payload["license_evaluation"]["future_evaluation_note"] = "ignored"
    payload["resolved_model"]["future_model_field"] = "ignored"

    plan = adapt_resolved_invocation_payload(payload)

    assert "future_legacy_annotation" not in plan.to_payload()
    assert "future_acknowledgement" not in plan.to_payload()["license_acknowledgements"]
    assert "future_evaluation_note" not in plan.to_payload()["license_evaluation"]
    assert "future_model_field" not in plan.to_payload()["resolved_model"]


def test_unknown_legacy_version_stage_and_artifact_fail_closed(tmp_path: Path) -> None:
    base = _build(EnhanceConfig(model_key="da3-metric"), tmp_path).to_payload()

    unknown_version = copy.deepcopy(base)
    unknown_version["schema"] = "tp.lux.resolved_invocation.v2"
    with pytest.raises(ResolvedInvocationCompatibilityError, match="Unsupported"):
        adapt_resolved_invocation_payload(unknown_version)

    unknown_stage = copy.deepcopy(base)
    unknown_stage["stages"].insert(-1, "python.module:ArbitraryStage")
    with pytest.raises(ResolvedInvocationCompatibilityError, match="unknown stage"):
        adapt_resolved_invocation_payload(unknown_stage)

    unknown_artifact = copy.deepcopy(base)
    unknown_artifact["requested_artifacts"].append("arbitrary_executable")
    with pytest.raises(ResolvedInvocationCompatibilityError, match="unknown requested"):
        adapt_resolved_invocation_payload(unknown_artifact)

    oversized = copy.deepcopy(base)
    oversized["stages"] = ["preprocess"] * 33
    with pytest.raises(ResolvedInvocationCompatibilityError, match="safety limit"):
        adapt_resolved_invocation_payload(oversized)


def test_forged_model_registry_and_lock_contracts_fail_closed(tmp_path: Path) -> None:
    payload = _build(EnhanceConfig(model_key="da3-metric"), tmp_path).to_payload()

    forged_registry = copy.deepcopy(payload)
    forged_registry["resolved_model"]["repo_id"] = "attacker/forged-model"
    with pytest.raises(UntrustedModelContractError, match="disagrees with registry"):
        adapt_resolved_invocation_payload(forged_registry)

    forged_revision = copy.deepcopy(payload)
    forged_revision["resolved_model"]["revision"] = "f" * 40
    with pytest.raises(UntrustedModelContractError, match="disagrees with the model lock"):
        adapt_resolved_invocation_payload(forged_revision)

    forged_selector = copy.deepcopy(payload)
    forged_selector["resolved_model"]["requested_selector"] = "da3"
    with pytest.raises(UntrustedModelContractError, match="requested selector"):
        adapt_resolved_invocation_payload(forged_selector)

    forged_variant = copy.deepcopy(payload)
    forged_variant["resolved_model"]["legacy_model_variant_name"] = "METRIC_LARGE"
    with pytest.raises(UntrustedModelContractError, match="variant provenance"):
        adapt_resolved_invocation_payload(forged_variant)


def test_research_model_acknowledgement_is_revalidated(tmp_path: Path) -> None:
    invocation = _build(
        EnhanceConfig(model_key="da3-research", non_commercial_ok=True),
        tmp_path,
    )
    payload = invocation.to_payload()
    payload["license_acknowledgements"]["non_commercial_ok"] = False

    with pytest.raises(ModelLicenseError, match="non-commercial"):
        adapt_resolved_invocation_payload(payload)


def test_legacy_da3_fallback_without_carried_model_fails_closed(tmp_path: Path) -> None:
    invocation = _build(
        EnhanceConfig(
            depth_backend="depth_pro",
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
        ),
        tmp_path,
    )
    payload = invocation.to_payload()
    payload["candidate_fallback_chain"].append("da3")

    with pytest.raises(ResolvedInvocationCompatibilityError, match="permitting DA3"):
        adapt_resolved_invocation_payload(payload)


def test_legacy_reconstruction_requires_runtime_license_acknowledgements(tmp_path: Path) -> None:
    payload = _build(
        EnhanceConfig(model_key="da3-metric", enable_reconstruction=True),
        tmp_path,
    ).to_payload()

    with pytest.raises(ResolvedInvocationCompatibilityError, match="reconstruction intent"):
        adapt_resolved_invocation_payload(payload)


def test_legacy_ensemble_without_constituent_model_identity_fails_closed(tmp_path: Path) -> None:
    payload = _build(
        EnhanceConfig(
            depth_backend="ensemble",
            non_commercial_ok=True,
            accept_research_tools_license=True,
        ),
        tmp_path,
    ).to_payload()

    with pytest.raises(ResolvedInvocationCompatibilityError, match="constituent model identities"):
        adapt_resolved_invocation_payload(payload)


def test_legacy_absolute_or_parent_input_never_crosses_canonical_boundary(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    outside = tmp_path / "outside" / "a.jpg"
    invocation = build_resolved_invocation(
        EnhanceConfig(model_key="da3-metric"),
        input_root,
        [outside],
    )

    with pytest.raises(ExecutionPlanError, match="contained relative"):
        adapt_resolved_invocation(invocation)


def test_adapter_does_not_construct_or_execute_stage_graph_or_write_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformation_portal.stage_graph.graph import StageGraph

    output_root = tmp_path / "must-not-exist"
    invocation = _build(EnhanceConfig(model_key="da3-metric"), tmp_path)
    monkeypatch.setattr(StageGraph, "__init__", lambda self, *args, **kwargs: pytest.fail("StageGraph constructed"))

    plan = adapt_resolved_invocation(invocation)

    assert plan.nodes
    assert not output_root.exists()


def test_legacy_flat_plan_name_and_historical_order_are_preserved() -> None:
    from transformation_portal.lux_depth_v3.pipeline_coordinator import (
        ExecutionPlan,
        LegacyExecutionPlan,
        PipelineCoordinator,
    )

    assert LegacyExecutionPlan is ExecutionPlan
    config = EnhanceConfig(
        model_key="da3-metric",
        enable_materials_v3=True,
        generate_pbr=True,
    )
    flat_plan = PipelineCoordinator(config).plan()
    # Do not silently change the public flat projection in A1. The canonical
    # adapter takes its authoritative order from ResolvedInvocation instead.
    assert flat_plan.stages.index("pbr") < flat_plan.stages.index("materials_v3")
