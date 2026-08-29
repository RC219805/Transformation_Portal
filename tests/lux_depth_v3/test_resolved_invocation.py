"""Tests for the single-resolution ResolvedInvocation contract (P0-1, #2065).

Covers the acceptance criteria that are testable without ML dependencies:
- the builder performs THE license-enforcing resolution (fail-closed default),
- consumers (ConfigResolver, DA3Backend) receive the authoritative contract,
- the legacy model_variant round-trip cannot reach backend construction,
- plan serialization is deterministic and excludes runtime-only fields,
- building performs no filesystem writes.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant, Preset
from transformation_portal.lux_depth_v3.model_resolution import (
    DefaultModelSelectionChangedWarning,
    DeprecatedModelSelectorWarning,
    ModelLicenseError,
)
from transformation_portal.lux_depth_v3.resolved_invocation import (
    RESOLVED_INVOCATION_SCHEMA,
    authoritative_model_contract,
    build_resolved_invocation,
)

pytestmark = [pytest.mark.unit]


def _commercial_config() -> EnhanceConfig:
    return EnhanceConfig(model_key="da3-metric")


def _build(config: EnhanceConfig, tmp_path: Path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(exist_ok=True)
    files = [input_dir / "b.jpg", input_dir / "a.jpg"]
    return build_resolved_invocation(config, input_dir=input_dir, input_files=files)


class TestSingleResolution:
    def test_default_selector_resolves_commercial_safe_model(self, tmp_path: Path) -> None:
        # Repair 1.2 (#2066, option A): the bare default resolves the
        # Apache-2.0 metric model without any license acknowledgement, and
        # records the distinct "default" selector — never the "da3" alias,
        # whose (deprecated) meaning is still the research model.
        invocation = _build(EnhanceConfig(), tmp_path)
        assert invocation.resolved_model is not None
        assert invocation.resolved_model.canonical_key == "da3_metric"
        assert invocation.resolved_model.requested_selector == "default"
        assert invocation.resolved_model.resolution_reason == "no model selector supplied; defaulted to 'da3_metric'"

    def test_research_selector_fails_closed_on_license(self, tmp_path: Path) -> None:
        # The builder is THE enforcing resolution and must raise for the
        # research model without non_commercial_ok, matching the error a
        # real run surfaces at validation time.
        with pytest.raises(ModelLicenseError):
            _build(EnhanceConfig(model_key="da3-research"), tmp_path)

    def test_commercial_key_resolves_metric(self, tmp_path: Path) -> None:
        invocation = _build(_commercial_config(), tmp_path)
        assert invocation.resolved_model is not None
        assert invocation.resolved_model.canonical_key == "da3_metric"
        assert invocation.license_enforced is True
        assert invocation.planned_backend == "da3"
        assert invocation.candidate_fallback_chain[0] == "da3"

    def test_authoritative_helper_reads_config_carrier(self, tmp_path: Path) -> None:
        config = _commercial_config()
        assert authoritative_model_contract(config) is None
        invocation = _build(config, tmp_path)
        config.resolved_invocation = invocation
        assert authoritative_model_contract(config) is invocation.resolved_model


class TestLegacyRoundTripIsDead:
    """The compat model_variant mapping must not override the invocation.

    Without the authoritative carrier, a da3_metric selection whose config
    later carries model_key=None + model_variant=METRIC_LARGE re-resolves to
    da3_research (METRIC_LARGE legacy-maps to the research model). With the
    carrier attached, consumers must keep da3_metric.
    """

    def _drifted_config(self, tmp_path: Path) -> EnhanceConfig:
        config = _commercial_config()
        invocation = _build(config, tmp_path)
        config.resolved_invocation = invocation
        # Simulate the compat drift the re-resolution seams are vulnerable to.
        config.model_key = None
        config.model_variant = ModelVariant.METRIC_LARGE
        return config

    def test_config_resolver_consumes_authoritative_contract(self, tmp_path: Path) -> None:
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = self._drifted_config(tmp_path)
        resolved = ConfigResolver().resolve(config)
        assert resolved.resolved_model_contract is not None
        assert resolved.resolved_model_contract.canonical_key == "da3_metric"
        assert resolved.resolved_model_contract is config.resolved_invocation.resolved_model

    def test_da3_backend_consumes_authoritative_contract(self, tmp_path: Path) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend

        config = self._drifted_config(tmp_path)
        backend = DA3Backend(config)
        assert backend._resolved_model_contract is config.resolved_invocation.resolved_model
        assert backend._resolved_model_contract.canonical_key == "da3_metric"

    def test_without_carrier_drift_reproduces(self, tmp_path: Path) -> None:
        # Regression guard for the guard itself: absent the carrier, the
        # drifted config resolves back to the research model, proving the
        # authoritative path is what prevents it (not coincidence).
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = self._drifted_config(tmp_path)
        config.resolved_invocation = None
        resolved = ConfigResolver().resolve(config)
        assert resolved.resolved_model_contract is not None
        assert resolved.resolved_model_contract.canonical_key == "da3_research"


class TestPlanSerialization:
    def test_plan_json_is_byte_identical_across_builds(self, tmp_path: Path) -> None:
        first = _build(_commercial_config(), tmp_path).to_canonical_json()
        second = _build(_commercial_config(), tmp_path).to_canonical_json()
        assert first == second

    def test_payload_shape_and_runtime_field_exclusion(self, tmp_path: Path) -> None:
        payload = _build(_commercial_config(), tmp_path).to_payload()
        assert payload["schema"] == RESOLVED_INVOCATION_SCHEMA
        assert "planned_backend" in payload
        assert "candidate_fallback_chain" in payload
        # executed_backend is runtime state recorded in manifests only.
        assert "executed_backend" not in json.dumps(payload)
        assert payload["resolved_model"]["canonical_key"] == "da3_metric"
        assert payload["resolved_model"]["resolution_reason"] == (
            "explicit model selector 'da3-metric' resolved to 'da3_metric'"
        )
        assert payload["resolved_model"]["requires_non_commercial_ok"] is False
        assert payload["license_evaluation"] == {"enforced": True, "status": "allowed"}
        assert payload["input_files"] == ["a.jpg", "b.jpg"]

    def test_dead_flag_warnings_surface_in_plan(self, tmp_path: Path) -> None:
        config = EnhanceConfig(model_key="da3-metric", emit_marketing=True)
        payload = _build(config, tmp_path).to_payload()
        assert any("emit-marketing" in warning for warning in payload["warnings"])
        assert "marketing" not in " ".join(payload["requested_artifacts"])

    def test_default_selection_migration_notice_is_recorded_without_duplicate_emission(
        self,
        tmp_path: Path,
    ) -> None:
        config = EnhanceConfig(non_commercial_ok=True)
        with pytest.warns(DefaultModelSelectionChangedWarning, match="da3-research") as caught:
            invocation = _build(config, tmp_path)

        assert len(caught) == 1
        matching_notices = [warning for warning in invocation.warnings if "No model selector was given" in warning]
        assert len(matching_notices) == 1
        assert "da3-research" in matching_notices[0]

    def test_deprecated_alias_notice_is_recorded_without_duplicate_emission(
        self,
        tmp_path: Path,
    ) -> None:
        config = EnhanceConfig(model_key="da3", non_commercial_ok=True)
        with pytest.warns(DeprecatedModelSelectorWarning, match="model_key='da3' is deprecated") as caught:
            invocation = _build(config, tmp_path)

        assert len(caught) == 1
        matching_notices = [warning for warning in invocation.warnings if "model_key='da3' is deprecated" in warning]
        assert len(matching_notices) == 1
        assert "da3-research" in matching_notices[0]


class TestNonDa3AndOptionalStages:
    def test_depth_pro_plan_enforces_acknowledgements(self, tmp_path: Path) -> None:
        config = EnhanceConfig(
            depth_backend="depth_pro",
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
        )
        invocation = _build(config, tmp_path)
        assert invocation.planned_backend == "depth_pro"
        assert invocation.resolved_model is None
        # The registry license gate ran at build time, so the plan reports
        # enforcement — not deferral — matching what the run enforces.
        assert invocation.license_enforced is True
        payload = invocation.to_payload()
        assert payload["resolved_model"] is None
        assert payload["license_evaluation"] == {"enforced": True, "status": "allowed"}

    def test_depth_pro_plan_without_acknowledgements_fails_like_run(self, tmp_path: Path) -> None:
        from transformation_portal.depth.backends.protocol import LicenseRestrictionError

        with pytest.raises(LicenseRestrictionError):
            _build(EnhanceConfig(depth_backend="depth_pro"), tmp_path)

    def test_ensemble_plan_without_acknowledgement_fails_like_run(self, tmp_path: Path) -> None:
        from transformation_portal.depth.backends.protocol import LicenseRestrictionError, LicenseType
        from transformation_portal.depth.backends.registry import DepthBackendRegistry

        registry = DepthBackendRegistry()
        ensemble_cls = registry.get_backend_class("ensemble")
        if ensemble_cls is None or ensemble_cls.license_type != LicenseType.RESEARCH_ONLY:
            pytest.skip("ensemble backend unavailable or not research-only in this environment")
        with pytest.raises(LicenseRestrictionError):
            _build(EnhanceConfig(depth_backend="ensemble"), tmp_path)

    def test_unknown_backend_plan_fails_like_run_registry(self, tmp_path: Path) -> None:
        from transformation_portal.depth.backends.registry import UnknownDepthBackendError

        with pytest.raises(UnknownDepthBackendError):
            _build(EnhanceConfig(depth_backend="bogus"), tmp_path)

    def test_carrier_without_da3_contract_falls_back_to_local_resolution(self, tmp_path: Path) -> None:
        # A non-DA3 invocation carries resolved_model=None; consumers must
        # fall back to their existing resolution rather than treating None
        # as an authoritative contract.
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(
            depth_backend="depth_pro",
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
        )
        config.resolved_invocation = _build(config, tmp_path)
        resolved = ConfigResolver().resolve(config)
        assert resolved.resolved_model_contract is not None
        backend = DA3Backend(config)
        assert backend._resolved_model_contract is not None

    def test_all_optional_stages_and_artifacts_planned(self, tmp_path: Path) -> None:
        config = EnhanceConfig(
            model_key="da3-metric",
            enable_materials_v3=True,
            generate_pbr=True,
            enable_v2=True,
            v2_preset="signature",
            enable_reconstruction=True,
            save_float_depth=True,
            emit_master16=True,
            emit_upscaled16=True,
        )
        invocation = _build(config, tmp_path)
        assert invocation.stages == (
            "preprocess",
            "depth",
            "materials_v3",
            "pbr",
            "v2",
            "reconstruction",
            "output",
        )
        artifacts = set(invocation.requested_artifacts)
        assert {
            "depth_float_npy",
            "materials_v3_masks",
            "pbr_maps",
            "v2_enhanced_image",
            "reconstruction_bundle",
            "bit_depth_16_intermediates",
        } <= artifacts
        assert any("bit-depth switch" in warning for warning in invocation.warnings)

    def test_input_file_outside_input_dir_kept_as_posix_path(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        outside = tmp_path / "elsewhere" / "x.jpg"
        invocation = build_resolved_invocation(
            _commercial_config(),
            input_dir=input_dir,
            input_files=[outside],
        )
        assert invocation.input_files == (outside.as_posix(),)


class TestFingerprintIdentity:
    def test_fingerprint_distinguishes_resolved_models(self, tmp_path: Path) -> None:
        """Two plans that execute different models must never share a
        fingerprint, even though ConfigFingerprint's variant label is
        METRIC_LARGE for both (Codex P1 on PR #2070)."""
        metric = _build(EnhanceConfig(model_key="da3-metric"), tmp_path)
        research = _build(
            EnhanceConfig(model_key="da3-research", non_commercial_ok=True),
            tmp_path,
        )
        assert metric.resolved_model.canonical_key != research.resolved_model.canonical_key
        assert metric.config_fingerprint_sha256 != research.config_fingerprint_sha256

    def test_plan_and_runtime_share_one_fingerprint_algorithm(self, tmp_path: Path) -> None:
        """No split identities: the plan fingerprint IS the runtime
        ConfigFingerprint for the same resolved contract — the value the
        depth cache, manifests, and Stage-A reuse compare."""
        from transformation_portal.lux_depth_v3.config_resolver import compute_config_fingerprint

        config = _commercial_config()
        invocation = _build(config, tmp_path)
        config.resolved_invocation = invocation
        runtime_fingerprint = compute_config_fingerprint(config)
        assert invocation.config_fingerprint_sha256 == runtime_fingerprint.to_sha256()

    def test_runtime_fingerprint_distinguishes_models_with_carrier(self, tmp_path: Path) -> None:
        """The runtime identity (cache/manifest/Stage-A) itself now
        distinguishes da3_metric from da3_research when the invocation is
        carried — not only the plan serialization."""
        from transformation_portal.lux_depth_v3.config_resolver import compute_config_fingerprint

        metric_config = EnhanceConfig(model_key="da3-metric")
        metric_config.resolved_invocation = _build(metric_config, tmp_path)
        research_config = EnhanceConfig(model_key="da3-research", non_commercial_ok=True)
        research_config.resolved_invocation = _build(research_config, tmp_path)
        assert compute_config_fingerprint(metric_config).to_sha256() != compute_config_fingerprint(research_config).to_sha256()

    def test_direct_runtime_cache_and_manifest_identities_distinguish_models(self) -> None:
        """The direct Python path has no invocation carrier, so the resolved
        contract returned by ConfigResolver must still key every runtime cache
        and manifest identity. Metric and research both use the legacy
        METRIC_LARGE compatibility variant and previously collided here."""
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        metric = ConfigResolver().resolve(EnhanceConfig(model_key="da3-metric"))
        research = ConfigResolver().resolve(
            EnhanceConfig(model_key="da3-research", non_commercial_ok=True),
        )

        assert metric.fingerprint.model_variant.startswith("da3_metric:")
        assert research.fingerprint.model_variant.startswith("da3_research:")
        assert metric.fingerprint.depth_only().to_sha256() != research.fingerprint.depth_only().to_sha256()

        metric_orchestrator = object.__new__(EnhanceOrchestrator)
        metric_orchestrator.config = metric.enhance_config
        metric_orchestrator._resolved_model_contract = metric.resolved_model_contract
        research_orchestrator = object.__new__(EnhanceOrchestrator)
        research_orchestrator.config = research.enhance_config
        research_orchestrator._resolved_model_contract = research.resolved_model_contract

        assert metric_orchestrator.compute_config_fingerprint().to_sha256() != (
            research_orchestrator.compute_config_fingerprint().to_sha256()
        )
        assert metric_orchestrator._build_depth_cache_fingerprint("da3") != (
            research_orchestrator._build_depth_cache_fingerprint("da3")
        )

    def test_research_manifest_cannot_skip_metric_depth_stage(self, tmp_path: Path) -> None:
        """A research manifest must not satisfy a later metric Stage-A
        lookup, even though both models retain the METRIC_LARGE compatibility
        variant."""
        from unittest.mock import patch

        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.manifest import CombinedManifest, DepthMetadata
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        image_path = tmp_path / "input.jpg"
        image_path.write_bytes(b"input")
        depth_path = tmp_path / "depth.png"
        depth_path.write_bytes(b"depth")
        manifest_path = tmp_path / "manifest.json"

        with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
            research = EnhanceOrchestrator(
                EnhanceConfig(model_key="da3-research", non_commercial_ok=True, enable_v2=False),
                tmp_path / "research",
                verify_outputs=False,
            )
            metric = EnhanceOrchestrator(
                EnhanceConfig(model_key="da3-metric", enable_v2=False),
                tmp_path / "metric",
                verify_outputs=False,
            )

        CombinedManifest(
            config_fingerprint=research.compute_config_fingerprint(),
            depth=DepthMetadata(
                model="da3_research",
                depth_path=str(depth_path),
                runtime_seconds=0.1,
                scaling={},
            ),
            backend_selection=research._capture_backend_metadata(),
        ).save(manifest_path)

        assert (
            metric.should_skip_depth(
                depth_path,
                manifest_path,
                ImageInput(image_path),
            )
            is False
        )


class TestCarrierTrustBoundary:
    """A forged or drifted carrier must be rejected at every consumption
    boundary — the carrier is a transport, not an authority (review P1 on
    PR #2070: an injected research contract must not bypass licensing, and
    a contract with an unapproved repo_id must not reach inference)."""

    def _forged_research_config(self, tmp_path: Path) -> EnhanceConfig:
        # Build a legitimate research contract WITH acknowledgement, then
        # carry it on a config WITHOUT the acknowledgement.
        acknowledged = EnhanceConfig(model_key="da3-research", non_commercial_ok=True)
        invocation = _build(acknowledged, tmp_path)
        config = EnhanceConfig(model_key="da3-research", non_commercial_ok=False)
        config.resolved_invocation = invocation
        return config

    def _forged_repo_config(self, tmp_path: Path) -> EnhanceConfig:
        import dataclasses

        config = _commercial_config()
        invocation = _build(config, tmp_path)
        forged_model = dataclasses.replace(
            invocation.resolved_model,
            spec=dataclasses.replace(invocation.resolved_model.spec, repo_id="unapproved/example-model"),
            revision=None,
        )
        config.resolved_invocation = dataclasses.replace(invocation, resolved_model=forged_model)
        return config

    def test_config_resolver_rejects_unacknowledged_research_carrier(self, tmp_path: Path) -> None:
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        with pytest.raises(ModelLicenseError):
            ConfigResolver().resolve(self._forged_research_config(tmp_path))

    def test_da3_backend_rejects_unacknowledged_research_carrier(self, tmp_path: Path) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend

        with pytest.raises(ModelLicenseError):
            DA3Backend(self._forged_research_config(tmp_path))

    def test_config_resolver_rejects_forged_repo_id(self, tmp_path: Path) -> None:
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver
        from transformation_portal.lux_depth_v3.model_resolution import UntrustedModelContractError

        with pytest.raises(UntrustedModelContractError):
            ConfigResolver().resolve(self._forged_repo_config(tmp_path))

    def test_da3_backend_rejects_forged_repo_id(self, tmp_path: Path) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.model_resolution import UntrustedModelContractError

        with pytest.raises(UntrustedModelContractError):
            DA3Backend(self._forged_repo_config(tmp_path))

    def test_engine_rejects_forged_carrier(self, tmp_path: Path) -> None:
        import dataclasses

        from transformation_portal.lux_depth_v3.config import DA3Config
        from transformation_portal.lux_depth_v3.inference import DA3InferenceEngine
        from transformation_portal.lux_depth_v3.model_resolution import UntrustedModelContractError

        invocation = _build(_commercial_config(), tmp_path)
        forged = dataclasses.replace(
            invocation.resolved_model,
            spec=dataclasses.replace(invocation.resolved_model.spec, repo_id="unapproved/example-model"),
        )
        engine = DA3InferenceEngine.__new__(DA3InferenceEngine)
        engine.config = DA3Config(resolved_model_contract=forged)
        engine._resolved_model_contract = None
        with pytest.raises(UntrustedModelContractError):
            engine._resolve_model_contract()

    def test_forged_revision_rejected_in_non_strict_mode(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The revision leg must resolve the model lock INDEPENDENTLY and
        compare — passing the carried revision into the resolver echoes it
        back in non-strict mode (requested wins), which previously accepted
        any forged revision (review P1, second round)."""
        import dataclasses

        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver
        from transformation_portal.lux_depth_v3.model_resolution import UntrustedModelContractError

        monkeypatch.delenv("TP_STRICT_MODEL_LOCK", raising=False)
        config = _commercial_config()
        invocation = _build(config, tmp_path)
        forged_model = dataclasses.replace(invocation.resolved_model, revision="b" * 40)
        config.resolved_invocation = dataclasses.replace(invocation, resolved_model=forged_model)
        with pytest.raises(UntrustedModelContractError):
            ConfigResolver().resolve(config)

    def test_dropped_revision_rejected_when_lock_pins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import dataclasses

        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver
        from transformation_portal.lux_depth_v3.model_resolution import UntrustedModelContractError

        monkeypatch.delenv("TP_STRICT_MODEL_LOCK", raising=False)
        config = _commercial_config()
        invocation = _build(config, tmp_path)
        if invocation.resolved_model.revision is None:
            pytest.skip("model lock does not pin this repo in this environment")
        stripped_model = dataclasses.replace(invocation.resolved_model, revision=None)
        config.resolved_invocation = dataclasses.replace(invocation, resolved_model=stripped_model)
        with pytest.raises(UntrustedModelContractError):
            ConfigResolver().resolve(config)

    def test_lock_consistent_carrier_accepted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        monkeypatch.delenv("TP_STRICT_MODEL_LOCK", raising=False)
        config = _commercial_config()
        config.resolved_invocation = _build(config, tmp_path)
        resolved = ConfigResolver().resolve(config)
        assert resolved.resolved_model_contract is config.resolved_invocation.resolved_model

    def test_non_resolvedmodel_carrier_rejected(self, tmp_path: Path) -> None:
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver
        from transformation_portal.lux_depth_v3.model_resolution import UntrustedModelContractError

        config = _commercial_config()

        class _Fake:
            resolved_model = object()

        config.resolved_invocation = _Fake()
        with pytest.raises(UntrustedModelContractError):
            ConfigResolver().resolve(config)


class TestBackendSelectionParity:
    def test_apple_silicon_depth_pro_opt_in_matches_runtime(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """On Apple Silicon with the Depth Pro opt-in, the runtime's
        requested-backend resolution selects depth_pro — the plan must
        select the same backend, not default to da3."""
        import transformation_portal.lux_depth_v3.pipeline_coordinator as pc

        class _FakeApple:
            is_apple_silicon = True

        monkeypatch.setattr(pc, "CURRENT_PLATFORM", _FakeApple())
        config = EnhanceConfig(
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
        )
        if not pc._apple_silicon_depth_pro_opt_in(config):
            pytest.skip("depth_pro opt-in requires additional config in this build")
        invocation = _build(config, tmp_path)
        assert invocation.planned_backend == "depth_pro"

    def test_explicit_backend_chain_is_strict(self, tmp_path: Path) -> None:
        """An explicit backend request is strict at runtime, so the plan
        must not advertise fallback edges startup will never attempt."""
        explicit = _build(EnhanceConfig(depth_backend="da3", model_key="da3-metric"), tmp_path)
        assert explicit.candidate_fallback_chain == ("da3",)
        defaulted = _build(EnhanceConfig(model_key="da3-metric"), tmp_path)
        assert defaulted.candidate_fallback_chain[0] == "da3"


class TestSchemaValidation:
    def test_emitted_payload_validates_against_schema(self, tmp_path: Path) -> None:
        from transformation_portal.lux_depth_v3.resolved_invocation import validate_resolved_invocation_payload

        payload = _build(_commercial_config(), tmp_path).to_payload()
        validate_resolved_invocation_payload(payload)

    def test_schema_rejects_missing_required_field(self, tmp_path: Path) -> None:
        import jsonschema

        from transformation_portal.lux_depth_v3.resolved_invocation import validate_resolved_invocation_payload

        payload = _build(_commercial_config(), tmp_path).to_payload()
        del payload["config_fingerprint_sha256"]
        with pytest.raises(jsonschema.ValidationError):
            validate_resolved_invocation_payload(payload)

    def test_payload_is_marked_provisional(self, tmp_path: Path) -> None:
        payload = _build(_commercial_config(), tmp_path).to_payload()
        assert payload["stability"] == "provisional"

    def test_schema_loads_from_package_resources(self) -> None:
        """The schema ships as package data and loads via
        importlib.resources — the mechanism an installed wheel uses (review
        P2, second round: the repo-root path never reached wheels)."""
        from transformation_portal.lux_depth_v3.resolved_invocation import load_resolved_invocation_schema

        schema = load_resolved_invocation_schema()
        assert schema["$id"] == "tp.lux.resolved_invocation.v1"


class TestSingleResolutionCallCount:
    """The advertised contract is exactly one resolution per invocation
    (Codex P2 on PR #2070): the builder resolves once; ConfigResolver, the
    DA3 backend, and the inference engine consume the carried contract with
    zero further resolve_model_contract calls."""

    def _install_spy(self, monkeypatch: pytest.MonkeyPatch):
        import transformation_portal.depth.backends.da3 as da3_module
        import transformation_portal.lux_depth_v3.config_resolver as resolver_module
        import transformation_portal.lux_depth_v3.inference as inference_module
        import transformation_portal.lux_depth_v3.resolved_invocation as invocation_module
        from transformation_portal.lux_depth_v3.model_resolution import resolve_model_contract as real_resolve

        calls: list = []

        def spy(request):
            calls.append(request)
            return real_resolve(request)

        for module in (invocation_module, resolver_module, da3_module, inference_module):
            monkeypatch.setattr(module, "resolve_model_contract", spy)
        return calls

    def test_build_is_the_single_resolution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._install_spy(monkeypatch)
        invocation = _build(_commercial_config(), tmp_path)
        assert len(calls) == 1
        assert invocation.resolved_model is not None

    def test_consumers_add_zero_resolutions(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config import DA3Config
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver
        from transformation_portal.lux_depth_v3.inference import DA3InferenceEngine

        config = _commercial_config()
        invocation = _build(config, tmp_path)
        config.resolved_invocation = invocation

        calls = self._install_spy(monkeypatch)
        ConfigResolver().resolve(config)
        backend = DA3Backend(config)
        assert backend._resolved_model_contract is invocation.resolved_model

        engine = DA3InferenceEngine.__new__(DA3InferenceEngine)
        engine.config = DA3Config(resolved_model_contract=invocation.resolved_model)
        engine._resolved_model_contract = None
        resolved = engine._resolve_model_contract()
        assert resolved is invocation.resolved_model
        assert calls == []


class TestRevisionPinning:
    def test_worker_command_carries_planned_revision(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The subprocess boundary must carry the planned revision — the
        parent previously serialized only the canonical key, letting the
        worker re-resolve a drifted lock (review P1 on PR #2070). Captures
        the REAL availability-check argv, not a reconstruction. Uses the
        genuine lock-resolved revision — a forged one is now rejected at the
        carrier boundary."""
        import types

        import transformation_portal.depth.backends.da3 as da3_module

        config = _commercial_config()
        invocation = _build(config, tmp_path)
        planned_revision = invocation.resolved_model.revision
        if planned_revision is None:
            pytest.skip("model lock does not pin this repo in this environment")
        config.resolved_invocation = invocation
        backend = da3_module.DA3Backend(config)
        backend._python_executable = "/usr/bin/env-python-stub"

        captured: dict = {}

        def _fake_run(command, **kwargs):
            captured["command"] = list(command)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(da3_module.subprocess, "run", _fake_run)
        backend._ensure_subprocess_available()
        command = captured["command"]
        assert "--model-revision" in command
        assert command[command.index("--model-revision") + 1] == planned_revision
        assert command[command.index("--model-key") + 1] == "da3_metric"

    def test_model_request_threads_requested_revision(self) -> None:
        from transformation_portal.lux_depth_v3.model_resolution import ModelRequest, resolve_model_contract

        pinned = resolve_model_contract(ModelRequest(model_key="da3-metric", requested_revision="b" * 40))
        assert pinned.revision == "b" * 40

    def test_engine_config_revision_reaches_resolution(self) -> None:
        from transformation_portal.lux_depth_v3.config import DA3Config
        from transformation_portal.lux_depth_v3.inference import DA3InferenceEngine

        engine = DA3InferenceEngine.__new__(DA3InferenceEngine)
        engine.config = DA3Config(model_key="da3-metric", model_revision="c" * 40)
        engine._resolved_model_contract = None
        resolved = engine._resolve_model_contract()
        assert resolved.revision == "c" * 40


class TestNoWrites:
    def test_build_performs_no_filesystem_writes(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        (input_dir / "a.jpg").write_bytes(b"stub")
        before = sorted(p.as_posix() for p in tmp_path.rglob("*"))
        build_resolved_invocation(
            _commercial_config(),
            input_dir=input_dir,
            input_files=[input_dir / "a.jpg"],
        )
        after = sorted(p.as_posix() for p in tmp_path.rglob("*"))
        assert before == after


class TestPythonApiDefaultParity:
    """Repair 1.2 (#2066) — Codex P1 on PR #2081: the public Python path
    (EnhanceConfig -> ConfigResolver -> DA3Backend, no CLI carrier) must
    execute the same commercial-safe default the metadata advertises.
    The compat model_variant mutation (METRIC_LARGE) and the backend's
    fabricated variant fallback previously re-resolved the bare default
    to da3_research."""

    def test_resolver_then_backend_keeps_metric_default(self) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig()
        resolved = ConfigResolver().resolve(config)
        backend = DA3Backend(config)
        assert resolved.resolved_model_contract.canonical_key == "da3_metric"
        assert backend._resolved_model_contract.canonical_key == "da3_metric"

    def test_no_split_identity_with_acknowledgement(self) -> None:
        # Previously: metadata said da3_metric while the backend loaded
        # da3_research — provenance lying about the executed model.
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(non_commercial_ok=True)
        resolved = ConfigResolver().resolve(config)
        backend = DA3Backend(config)
        assert resolved.resolved_model_contract.canonical_key == "da3_metric"
        assert backend._resolved_model_contract.canonical_key == "da3_metric"

    def test_unresolved_config_backend_uses_default_not_fabricated_variant(self) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend

        backend = DA3Backend(EnhanceConfig())
        assert backend._resolved_model_contract.canonical_key == "da3_metric"

    def test_preset_variant_plane_preserved(self) -> None:
        # Presets keep their legacy variant semantics through the
        # deprecation cycle: EXTERIOR carries METRIC_BASE -> da3_base.
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config import Preset
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(preset=Preset.ARCHITECTURAL_EXTERIOR)
        ConfigResolver().resolve(config)
        backend = DA3Backend(config)
        assert backend._resolved_model_contract.canonical_key == "da3_base"

    @pytest.mark.parametrize(
        ("preset", "expected_model_key"),
        (
            (Preset.DEFAULT, "da3_research"),
            (Preset.ARCHITECTURAL_INTERIOR, "da3_research"),
            (Preset.ARCHITECTURAL_EXTERIOR, "da3_base"),
            (Preset.LUXURY_ESTATE, "da3_research"),
        ),
    )
    def test_typed_preset_identity_matches_plan_resolver_backend_and_run_card(
        self,
        tmp_path: Path,
        preset: Preset,
        expected_model_key: str,
    ) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import (
            ConfigResolver,
            build_orchestrator_run_card_config_fingerprint,
        )

        config = EnhanceConfig(
            preset=preset,
            non_commercial_ok=True,
        )
        invocation = _build(config, tmp_path)
        config.resolved_invocation = invocation
        resolved = ConfigResolver().resolve(config)
        backend = DA3Backend(config)
        run_card_fingerprint = build_orchestrator_run_card_config_fingerprint(
            config,
            resolved.model_variant,
            None,
            resolved_model_contract=resolved.resolved_model_contract,
        )

        assert invocation.resolved_model is not None
        assert invocation.resolved_model.canonical_key == expected_model_key
        assert invocation.resolved_model.requested_selector == f"preset:{preset.value}"
        assert invocation.resolved_model.resolution_reason == (
            f"typed preset {preset.value!r} selected {expected_model_key!r}"
        )
        assert resolved.resolved_model_contract is invocation.resolved_model
        assert backend._resolved_model_contract is invocation.resolved_model
        assert resolved.fingerprint is not None
        assert resolved.fingerprint.model_variant.startswith(f"{expected_model_key}:")
        assert run_card_fingerprint["model_variant"] == resolved.fingerprint.model_variant

    @pytest.mark.parametrize(
        ("preset", "expected_model_key", "requires_acknowledgement"),
        (
            (Preset.DEFAULT, "da3_research", True),
            (Preset.ARCHITECTURAL_INTERIOR, "da3_research", True),
            (Preset.ARCHITECTURAL_EXTERIOR, "da3_base", False),
            (Preset.LUXURY_ESTATE, "da3_research", True),
        ),
    )
    def test_typed_preset_plan_license_boundary(
        self,
        tmp_path: Path,
        preset: Preset,
        expected_model_key: str,
        requires_acknowledgement: bool,
    ) -> None:
        config = EnhanceConfig(preset=preset)

        if requires_acknowledgement:
            with pytest.raises(ModelLicenseError):
                _build(config, tmp_path)
            return

        invocation = _build(config, tmp_path)
        assert invocation.resolved_model is not None
        assert invocation.resolved_model.canonical_key == expected_model_key

    @pytest.mark.parametrize(
        ("preset", "expected_model_key", "requires_acknowledgement"),
        (
            (Preset.DEFAULT, "da3_research", True),
            (Preset.ARCHITECTURAL_INTERIOR, "da3_research", True),
            (Preset.ARCHITECTURAL_EXTERIOR, "da3_base", False),
            (Preset.LUXURY_ESTATE, "da3_research", True),
        ),
    )
    @pytest.mark.parametrize("non_commercial_ok", (False, True))
    def test_typed_preset_direct_python_boundary(
        self,
        preset: Preset,
        expected_model_key: str,
        requires_acknowledgement: bool,
        non_commercial_ok: bool,
    ) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(
            preset=preset,
            non_commercial_ok=non_commercial_ok,
        )
        resolved = ConfigResolver().resolve(config)
        assert resolved.resolved_model_contract is not None
        assert resolved.resolved_model_contract.canonical_key == expected_model_key
        assert resolved.resolved_model_contract.requested_selector == f"preset:{preset.value}"
        assert config.model_key is None

        if requires_acknowledgement and not non_commercial_ok:
            with pytest.raises(ModelLicenseError):
                DA3Backend(config)
            return

        backend = DA3Backend(config)
        assert backend._resolved_model_contract.canonical_key == expected_model_key

    def test_unacknowledged_research_preset_resolver_is_idempotent(self) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(preset=Preset.ARCHITECTURAL_INTERIOR)
        resolver = ConfigResolver()

        first = resolver.resolve(config)
        second = resolver.resolve(config)

        assert first.resolved_model_contract is second.resolved_model_contract
        assert second.resolved_model_contract.canonical_key == "da3_research"
        assert second.resolved_model_contract.requested_selector == "preset:architectural_interior"
        with pytest.raises(ModelLicenseError):
            DA3Backend(config)

    @pytest.mark.parametrize(
        ("selection", "expected_model_key"),
        (
            ({"model_key": "da3-metric"}, "da3_metric"),
            ({"raw_model_id": "depth-anything/DA3METRIC-LARGE"}, "da3_metric"),
            ({"model_variant": ModelVariant.METRIC_SMALL}, "da3_small"),
        ),
    )
    def test_explicit_model_selection_overrides_typed_preset(
        self,
        selection: dict[str, object],
        expected_model_key: str,
    ) -> None:
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(
            preset=Preset.ARCHITECTURAL_INTERIOR,
            non_commercial_ok=True,
            **selection,
        )
        if "model_variant" in selection:
            with pytest.warns(DeprecatedModelSelectorWarning):
                resolved = ConfigResolver().resolve(config)
        else:
            resolved = ConfigResolver().resolve(config)

        assert resolved.resolved_model_contract is not None
        assert resolved.resolved_model_contract.canonical_key == expected_model_key

    def test_repeated_resolution_replaces_resolver_owned_preset_variant(self) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(
            preset=Preset.ARCHITECTURAL_EXTERIOR,
            non_commercial_ok=True,
        )
        resolver = ConfigResolver()
        first = resolver.resolve(config)
        config.preset = Preset.ARCHITECTURAL_INTERIOR
        second = resolver.resolve(config)
        backend = DA3Backend(config)

        assert first.resolved_model_contract.canonical_key == "da3_base"
        assert second.resolved_model_contract.canonical_key == "da3_research"
        assert second.resolved_model_contract.requested_selector == "preset:architectural_interior"
        assert backend._resolved_model_contract is second.resolved_model_contract
        assert config.model_key is None

    def test_raw_model_id_overrides_prior_preset_projection(self) -> None:
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(
            preset=Preset.ARCHITECTURAL_EXTERIOR,
            non_commercial_ok=True,
        )
        resolver = ConfigResolver()
        resolver.resolve(config)
        config.raw_model_id = "depth-anything/DA3METRIC-LARGE"

        resolved = resolver.resolve(config)

        assert resolved.resolved_model_contract.canonical_key == "da3_metric"
        assert resolved.resolved_model_contract.requested_selector == config.raw_model_id
        assert config.model_key is None

    def test_legacy_variant_overrides_prior_preset_projection(self) -> None:
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(
            preset=Preset.ARCHITECTURAL_EXTERIOR,
            non_commercial_ok=True,
        )
        resolver = ConfigResolver()
        resolver.resolve(config)
        config.model_variant = ModelVariant.METRIC_SMALL

        with pytest.warns(DeprecatedModelSelectorWarning):
            resolved = resolver.resolve(config)

        assert resolved.resolved_model_contract.canonical_key == "da3_small"

    def test_typed_preset_replaces_prior_selector_free_default_projection(self) -> None:
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(non_commercial_ok=True)
        resolver = ConfigResolver()
        first = resolver.resolve(config)
        config.preset = Preset.ARCHITECTURAL_EXTERIOR

        second = resolver.resolve(config)

        assert first.resolved_model_contract.canonical_key == "da3_metric"
        assert second.resolved_model_contract.canonical_key == "da3_base"
        assert second.resolved_model_contract.requested_selector == "preset:architectural_exterior"
        assert config.model_key is None

    def test_explicit_legacy_variant_still_means_research_and_gates(self) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(model_variant=ModelVariant.METRIC_LARGE)
        with pytest.warns(DeprecatedModelSelectorWarning):
            ConfigResolver().resolve(config)
        with pytest.raises(ModelLicenseError):
            DA3Backend(config)

    def test_repeated_direct_resolution_preserves_default_provenance(self) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig()
        resolver = ConfigResolver()
        first = resolver.resolve(config)
        second = resolver.resolve(config)
        backend = DA3Backend(config)

        assert config.model_key == "da3_metric"
        assert first.resolved_model_contract.requested_selector == "default"
        assert second.resolved_model_contract.requested_selector == "default"
        assert second.resolved_model_contract.resolution_reason == first.resolved_model_contract.resolution_reason
        assert backend._resolved_model_contract.requested_selector == "default"

    def test_changed_selector_invalidates_direct_default_provenance(self) -> None:
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig()
        resolver = ConfigResolver()
        resolver.resolve(config)
        config.model_key = "da3-research"
        config.non_commercial_ok = True

        changed = resolver.resolve(config)

        assert changed.resolved_model_contract.canonical_key == "da3_research"
        assert changed.resolved_model_contract.requested_selector == "da3-research"
        assert changed.resolved_model_contract.resolution_reason.startswith("explicit model selector")

    def test_direct_orchestrator_pins_metric_default(self, tmp_path: Path) -> None:
        # The focused direct-orchestrator regression: constructing the
        # public orchestrator on a bare default pins the commercial-safe
        # identity onto the config for every downstream consumer.
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(allow_synthetic_fallback=True)
        EnhanceOrchestrator(config, tmp_path)
        assert config.model_key == "da3_metric"

    def test_direct_default_change_warning_is_emitted_once(self) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(non_commercial_ok=True)
        with pytest.warns(DefaultModelSelectionChangedWarning, match="da3-research") as caught:
            ConfigResolver().resolve(config)
            ConfigResolver().resolve(config)
            DA3Backend(config)
        assert len(caught) == 1

    def test_default_change_warning_emits_when_acknowledgement_changes(self) -> None:
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig()
        resolver = ConfigResolver()
        resolver.resolve(config)
        config.non_commercial_ok = True

        with pytest.warns(DefaultModelSelectionChangedWarning, match="da3-research") as caught:
            resolved = resolver.resolve(config)

        assert len(caught) == 1
        assert resolved.resolved_model_contract.canonical_key == "da3_metric"
        assert resolved.resolved_model_contract.requested_selector == "default"

    def test_acknowledgement_change_preserves_alias_and_warning_cardinality(
        self,
        tmp_path: Path,
    ) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(model_key="da3", non_commercial_ok=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = ConfigResolver().resolve(config)
            config.non_commercial_ok = False
            with pytest.raises(ModelLicenseError):
                DA3Backend(config)
            with pytest.raises(ModelLicenseError):
                _build(config, tmp_path)

        alias_warnings = [warning for warning in caught if issubclass(warning.category, DeprecatedModelSelectorWarning)]
        assert len(alias_warnings) == 1
        assert resolved.resolved_model_contract.canonical_key == "da3_research"
        assert resolved.resolved_model_contract.requested_selector == "da3"

    def test_acknowledgement_change_preserves_exterior_preset_provenance(
        self,
        tmp_path: Path,
    ) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(
            preset=Preset.ARCHITECTURAL_EXTERIOR,
            non_commercial_ok=True,
        )
        resolved = ConfigResolver().resolve(config)
        config.non_commercial_ok = False

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            backend = DA3Backend(config)
            invocation = _build(config, tmp_path)

        assert not [warning for warning in caught if issubclass(warning.category, DeprecatedModelSelectorWarning)]
        for contract in (
            resolved.resolved_model_contract,
            backend._resolved_model_contract,
            invocation.resolved_model,
        ):
            assert contract.canonical_key == "da3_base"
            assert contract.requested_selector == "preset:architectural_exterior"
            assert contract.legacy_model_variant_name is None

    def test_default_acknowledgement_transition_preserves_provenance_and_warns_once(
        self,
        tmp_path: Path,
    ) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig()
        resolved = ConfigResolver().resolve(config)
        config.non_commercial_ok = True

        with pytest.warns(DefaultModelSelectionChangedWarning, match="da3-research") as caught:
            backend = DA3Backend(config)
            invocation = _build(config, tmp_path)

        assert len(caught) == 1
        for contract in (
            resolved.resolved_model_contract,
            backend._resolved_model_contract,
            invocation.resolved_model,
        ):
            assert contract.canonical_key == "da3_metric"
            assert contract.requested_selector == "default"

    def test_revoked_acknowledgement_preserves_research_preset_and_fails_closed(
        self,
        tmp_path: Path,
    ) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(
            preset=Preset.ARCHITECTURAL_INTERIOR,
            non_commercial_ok=True,
        )
        resolved = ConfigResolver().resolve(config)
        config.non_commercial_ok = False

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ModelLicenseError):
                DA3Backend(config)
            with pytest.raises(ModelLicenseError):
                _build(config, tmp_path)

        assert not [warning for warning in caught if issubclass(warning.category, DeprecatedModelSelectorWarning)]
        assert resolved.resolved_model_contract.canonical_key == "da3_research"
        assert resolved.resolved_model_contract.requested_selector == "preset:architectural_interior"

    @pytest.mark.parametrize(
        ("selection", "expected_selector", "expected_alias_warnings"),
        (
            (
                {"preset": Preset.ARCHITECTURAL_INTERIOR},
                "preset:architectural_interior",
                0,
            ),
            ({"model_key": "da3"}, "da3", 1),
        ),
    )
    def test_granted_research_acknowledgement_preserves_provenance_across_consumers(
        self,
        tmp_path: Path,
        selection: dict[str, object],
        expected_selector: str,
        expected_alias_warnings: int,
    ) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(non_commercial_ok=False, **selection)
        resolver = ConfigResolver()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = resolver.resolve(config)
            config.non_commercial_ok = True
            second = resolver.resolve(config)
            backend = DA3Backend(config)
            invocation = _build(config, tmp_path)

        alias_warnings = [warning for warning in caught if issubclass(warning.category, DeprecatedModelSelectorWarning)]
        assert len(alias_warnings) == expected_alias_warnings
        for contract in (
            first.resolved_model_contract,
            second.resolved_model_contract,
            backend._resolved_model_contract,
            invocation.resolved_model,
        ):
            assert contract.canonical_key == "da3_research"
            assert contract.requested_selector == expected_selector

    def test_direct_deprecated_alias_warning_is_emitted_once(self) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(model_key="da3", non_commercial_ok=True)
        with pytest.warns(DeprecatedModelSelectorWarning, match="model_key='da3' is deprecated") as caught:
            ConfigResolver().resolve(config)
            ConfigResolver().resolve(config)
            DA3Backend(config)
        assert len(caught) == 1

    def test_cli_carrier_does_not_duplicate_default_change_warning(self, tmp_path: Path) -> None:
        from transformation_portal.depth.backends.da3 import DA3Backend
        from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver

        config = EnhanceConfig(non_commercial_ok=True)
        with pytest.warns(DefaultModelSelectionChangedWarning, match="da3-research") as caught:
            config.resolved_invocation = _build(config, tmp_path)
            ConfigResolver().resolve(config)
            DA3Backend(config)
        assert len(caught) == 1
