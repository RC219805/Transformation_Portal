"""Unit tests for the config/presets response/request models (Phase 1.2 PR D).

These tests verify the typed models in
``transformation_portal.api.v1.config`` accept and reject the wire shapes
that ``app.py``'s config/presets handlers produce. They complement the
existing wire-contract tests in
``tests/test_app_orchestrator_contract_http.py`` — those exercise the
routes end-to-end; these exercise the models in isolation so a regression
is caught before it reaches the contract tests.

Test fixtures use real shapes drawn from ``PRESET_CATALOG``,
``_lux_config_metadata``, and ``_build_lux_config_preview``. If those
helpers' wire shapes drift, these tests should fail loudly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from transformation_portal.api.v1 import (
    ConfigMetadataData,
    ConfigMetadataEnvelope,
    ConfigPreviewData,
    ConfigPreviewEnvelope,
    ConfigPreviewRequest,
    PipelinePresetGroup,
    PresetEntry,
    PresetsAllPipelinesData,
    PresetsEnvelope,
    PresetsSinglePipelineData,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# PresetEntry — single preset within /v1/presets data
# ---------------------------------------------------------------------------


class TestPresetEntry:
    """Mirrors entries in app.py:PRESET_CATALOG."""

    def test_real_premium_preset_from_catalog(self) -> None:
        # Real premium preset from app.py:PRESET_CATALOG.
        preset = PresetEntry(
            name="premium",
            label="premium (Stable)",
            stability="stable",
            description="Balanced production quality preset",
            is_research=False,
            recommended_args={
                "quality_tier": "premium",
                "depth_backend": "da3",
                "model_key": "da3-metric",
                "enable_segmentation": True,
            },
            advanced_sections=[],
        )
        dumped = preset.model_dump(mode="json")
        assert dumped["name"] == "premium"
        assert dumped["recommended_args"]["depth_backend"] == "da3"
        assert dumped["advanced_sections"] == []

    def test_research_preset_with_advanced_sections(self) -> None:
        preset = PresetEntry(
            name="depth-anything-v3.1-research-m4",
            label="v3.1-m4 (Experimental)",
            stability="experimental",
            description="Research-only preset",
            is_research=True,
            recommended_args={"quality_tier": "apex", "model_key": "da3-research"},
            advanced_sections=["governance"],
        )
        assert preset.is_research is True
        assert preset.advanced_sections == ["governance"]

    def test_extra_keys_pass_through(self) -> None:
        # extra="allow" — recommended_args and the entry itself can grow new
        # keys without a model bump.
        preset = PresetEntry(
            name="x",
            label="x",
            stability="stable",
            description="d",
            is_research=False,
            recommended_args={},
            advanced_sections=[],
            future_field="not_yet_modelled",
        )
        assert preset.model_dump(mode="json")["future_field"] == "not_yet_modelled"

    def test_recommended_args_and_advanced_sections_required(self) -> None:
        with pytest.raises(ValidationError):
            PresetEntry(
                name="x",
                label="x",
                stability="stable",
                description="d",
                is_research=False,
            )


# ---------------------------------------------------------------------------
# /v1/presets payload shapes
# ---------------------------------------------------------------------------


class TestPresetsSinglePipelineData:
    """Shape returned when /v1/presets is called WITH ?pipeline=foo."""

    def test_real_lux_depth_v3_shape(self) -> None:
        # Sourced from contract test
        # tests/test_app_orchestrator_contract_http.py::test_presets_contract_for_lux_depth_pipeline.
        data = PresetsSinglePipelineData(
            pipeline="lux-depth-v3",
            presets=[
                PresetEntry(
                    name="premium",
                    label="premium (Stable)",
                    stability="stable",
                    description="d",
                    is_research=False,
                    recommended_args={"quality_tier": "premium", "model_key": "da3-metric"},
                    advanced_sections=[],
                ),
            ],
        )
        dumped = data.model_dump(mode="json")
        assert dumped["pipeline"] == "lux-depth-v3"
        assert dumped["presets"][0]["recommended_args"]["model_key"] == "da3-metric"

    def test_extra_top_level_keys_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PresetsSinglePipelineData(pipeline="x", presets=[], extra="nope")  # type: ignore[call-arg]


class TestPresetsAllPipelinesData:
    """Shape returned when /v1/presets is called WITHOUT ?pipeline=."""

    def test_empty_pipelines_list(self) -> None:
        data = PresetsAllPipelinesData(pipelines=[])
        assert data.model_dump(mode="json") == {"pipelines": []}

    def test_pipelines_list_with_groups(self) -> None:
        data = PresetsAllPipelinesData(
            pipelines=[
                PipelinePresetGroup(
                    pipeline="lux-depth-v3",
                    presets=[
                        PresetEntry(
                            name="premium",
                            label="premium (Stable)",
                            stability="stable",
                            description="d",
                            is_research=False,
                            recommended_args={},
                            advanced_sections=[],
                        ),
                    ],
                ),
            ],
        )
        dumped = data.model_dump(mode="json")
        assert dumped["pipelines"][0]["pipeline"] == "lux-depth-v3"
        assert dumped["pipelines"][0]["presets"][0]["name"] == "premium"


# ---------------------------------------------------------------------------
# /v1/config-metadata payload
# ---------------------------------------------------------------------------


class TestConfigMetadataData:
    """Mirrors the contract-stable subset of _lux_config_metadata output
    as asserted in
    tests/test_app_orchestrator_contract_http.py::test_config_metadata_contract_for_lux_depth_pipeline.
    """

    def test_minimal_pipeline_only(self) -> None:
        data = ConfigMetadataData(pipeline="lux-depth-v3")
        dumped = data.model_dump(mode="json")
        assert dumped["pipeline"] == "lux-depth-v3"
        # Stable defaults
        assert dumped["advanced_sections"] == []
        assert dumped["estimate_bands"] == {}
        assert dumped["backend_catalog"] == {}
        # Optional fields default to None on this minimal construction
        assert dumped["fields"] is None
        assert dumped["model_catalog"] is None
        assert dumped["debug_bundle_policy"] is None

    def test_full_realistic_shape(self) -> None:
        # Top-level shape extracted from
        # test_config_metadata_contract_for_lux_depth_pipeline assertions.
        data = ConfigMetadataData(
            pipeline="lux-depth-v3",
            advanced_sections=["advanced", "governance", "reconstruction"],
            estimate_bands={
                "runtime": ["low", "medium", "high"],
                "gpu_pressure": ["low", "medium", "high"],
                "research_risk": ["none", "research_only", "experimental"],
            },
            backend_catalog={
                "da3": {
                    "policy_posture": {"code": "governed_default"},
                    "default_model_key": "da3-metric",
                },
                "depth_pro": {
                    "required_acknowledgments": [{"field": "non_commercial_ok"}],
                },
            },
            fields={
                "model_key": {"options": [{"value": "da3-metric"}]},
                "reconstruction_tier": {"default": "apex_research"},
            },
            debug_bundle_policy={"acknowledgement_required": True},
        )
        dumped = data.model_dump(mode="json")
        assert dumped["backend_catalog"]["da3"]["default_model_key"] == "da3-metric"
        assert dumped["fields"]["reconstruction_tier"]["default"] == "apex_research"
        assert dumped["debug_bundle_policy"]["acknowledgement_required"] is True

    def test_extra_top_level_keys_pass_through(self) -> None:
        # extra="allow" — _lux_config_metadata may grow new top-level fields.
        data = ConfigMetadataData(pipeline="lux-depth-v3", future_section={"future": "value"})
        assert data.model_dump(mode="json")["future_section"] == {"future": "value"}


# ---------------------------------------------------------------------------
# /v1/config-preview payload
# ---------------------------------------------------------------------------


class TestConfigPreviewData:
    """Mirrors the contract-stable subset of _build_config_preview output
    as asserted in
    test_lux_config_preview_returns_execution_args_and_repair_warning_for_repo_local_shorthand.
    """

    def test_minimal_stable_wire_shape(self) -> None:
        data = ConfigPreviewData(
            pipeline="lux-depth-v3",
            normalized_args={},
            execution_args={},
            argv_preview="",
            field_errors=[],
            field_warnings=[],
            inactive_fields=[],
            readiness={},
            estimate_summary={},
            debug_bundle_summary={},
            next_best_action=None,
        )
        dumped = data.model_dump(mode="json")
        assert dumped["pipeline"] == "lux-depth-v3"
        assert dumped["argv_preview"] == ""
        assert dumped["next_best_action"] is None

    def test_realistic_lux_preview_shape(self) -> None:
        data = ConfigPreviewData(
            pipeline="lux-depth-v3",
            normalized_args={
                "input_dir": "./tests/fixtures/archive_small/archive_root",
                "output_dir": "./tests/fixtures/portal_contract_output/x",
            },
            execution_args={
                "input_dir": "./tests/fixtures/archive_small/archive_root",
                "output_dir": "./tests/fixtures/portal_contract_output/x",
            },
            argv_preview="python -m transformation_portal.cli --input-dir ./tests/fixtures/archive_small/archive_root",
            field_errors=[],
            field_warnings=[{"code": "repo_local_path_repaired", "field": "input_dir"}],
            inactive_fields=[],
            readiness={"ready": True},
            estimate_summary={"runtime": "medium"},
            debug_bundle_summary={"enabled": False},
            next_best_action={
                "action": "review_warning",
                "field": "input_dir",
                "label": "Resolve input dir",
                "detail": "Review the repaired path.",
                "tone": "warning",
            },
        )
        dumped = data.model_dump(mode="json")
        assert dumped["field_warnings"][0]["code"] == "repo_local_path_repaired"
        assert dumped["execution_args"]["input_dir"].startswith("./tests/")
        assert dumped["readiness"]["ready"] is True
        assert dumped["estimate_summary"]["runtime"] == "medium"
        assert dumped["next_best_action"]["tone"] == "warning"

    def test_pipeline_specific_extras_pass_through(self) -> None:
        # Different pipelines emit pipeline-specific keys; extra="allow"
        # accommodates them without a model bump.
        data = ConfigPreviewData(
            pipeline="archive-gate-a",
            normalized_args={},
            execution_args={},
            argv_preview="",
            field_errors=[],
            field_warnings=[],
            inactive_fields=[],
            readiness={},
            estimate_summary={},
            debug_bundle_summary={},
            next_best_action=None,
            archive_index_summary={"rows_total": 100},
        )
        assert data.model_dump(mode="json")["archive_index_summary"]["rows_total"] == 100

    def test_stable_wire_keys_required(self) -> None:
        with pytest.raises(ValidationError):
            ConfigPreviewData(
                pipeline="lux-depth-v3",
                normalized_args={},
                execution_args={},
                field_errors=[],
                field_warnings=[],
                inactive_fields=[],
                readiness={},
                estimate_summary={},
                debug_bundle_summary={},
                next_best_action=None,
            )


# ---------------------------------------------------------------------------
# ConfigPreviewRequest — defined for type discipline; not yet wired
# ---------------------------------------------------------------------------


class TestConfigPreviewRequest:
    def test_minimal_required_pipeline(self) -> None:
        req = ConfigPreviewRequest(pipeline="lux-depth-v3")
        assert req.pipeline == "lux-depth-v3"
        assert req.args == {}

    def test_pipeline_required(self) -> None:
        with pytest.raises(ValidationError):
            ConfigPreviewRequest()  # type: ignore[call-arg]

    def test_args_passthrough(self) -> None:
        req = ConfigPreviewRequest(pipeline="lux-depth-v3", args={"input_dir": "x"})
        assert req.args == {"input_dir": "x"}

    def test_extra_top_level_keys_passthrough(self) -> None:
        req = ConfigPreviewRequest(pipeline="x", overrides={"foo": "bar"})
        assert req.model_dump(mode="json")["overrides"] == {"foo": "bar"}


# ---------------------------------------------------------------------------
# Envelope aliases — full round-trip wraps for each schema
# ---------------------------------------------------------------------------


class TestConfigEnvelopes:
    def test_presets_envelope_accepts_single_pipeline_shape(self) -> None:
        env = PresetsEnvelope(
            schema="tp.orchestrator.presets.v1",
            success=True,
            data={"pipeline": "lux-depth-v3", "presets": []},
        )
        dumped = env.model_dump(mode="json")
        assert list(dumped.keys()) == ["schema", "success", "data", "error"]
        assert dumped["data"]["pipeline"] == "lux-depth-v3"
        assert dumped["error"] is None

    def test_presets_envelope_accepts_all_pipelines_shape(self) -> None:
        env = PresetsEnvelope(
            schema="tp.orchestrator.presets.v1",
            success=True,
            data={"pipelines": []},
        )
        dumped = env.model_dump(mode="json")
        assert dumped["data"] == {"pipelines": []}

    def test_presets_envelope_rejects_neither_shape(self) -> None:
        # The data Union is PresetsSinglePipelineData | PresetsAllPipelinesData.
        # A dict that matches neither (e.g. missing both required fields)
        # must fail validation.
        with pytest.raises(ValidationError):
            PresetsEnvelope(
                schema="tp.orchestrator.presets.v1",
                success=True,
                data={"unrelated_key": "value"},
            )

    def test_config_metadata_envelope_round_trip(self) -> None:
        env = ConfigMetadataEnvelope(
            schema="tp.orchestrator.config_metadata.v1",
            success=True,
            data=ConfigMetadataData(pipeline="lux-depth-v3"),
        )
        dumped = env.model_dump(mode="json")
        assert dumped["schema"] == "tp.orchestrator.config_metadata.v1"
        assert dumped["data"]["pipeline"] == "lux-depth-v3"

    def test_config_preview_envelope_round_trip(self) -> None:
        env = ConfigPreviewEnvelope(
            schema="tp.orchestrator.config_preview.v1",
            success=True,
            data=ConfigPreviewData(
                pipeline="lux-depth-v3",
                normalized_args={},
                execution_args={},
                argv_preview="",
                field_errors=[],
                field_warnings=[{"code": "warn"}],
                inactive_fields=[],
                readiness={},
                estimate_summary={},
                debug_bundle_summary={},
                next_best_action=None,
            ),
        )
        dumped = env.model_dump(mode="json")
        assert dumped["data"]["field_warnings"][0]["code"] == "warn"

    def test_envelope_aliases_carry_typed_data_payloads(self) -> None:
        # Behavioral check (not identity-based, per Copilot review on PR #1566):
        # each alias's `data` field is bound to the right payload type, so
        # invalid payload shapes are rejected at model construction time.
        with pytest.raises(ValidationError):
            ConfigMetadataEnvelope(
                schema="tp.orchestrator.config_metadata.v1",
                success=True,
                data={"missing": "pipeline field"},  # ConfigMetadataData requires pipeline
            )

    def test_unrecognised_schema_string_rejected_at_literal_level(self) -> None:
        # SchemaName Literal still rejects unknown strings even though the
        # alias doesn't pin to a single value.
        with pytest.raises(ValidationError):
            PresetsEnvelope(
                schema="tp.orchestrator.not_a_real_schema.v1",
                success=True,
                data={"pipelines": []},
            )
