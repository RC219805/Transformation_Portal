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
from pathlib import Path

import pytest

from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
from transformation_portal.lux_depth_v3.model_resolution import ModelLicenseError
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
    def test_default_selector_fails_closed_on_license(self, tmp_path: Path) -> None:
        # The bare default resolves to the research-licensed model; the
        # builder is THE enforcing resolution and must raise, matching the
        # error a real run surfaces at validation time.
        with pytest.raises(ModelLicenseError):
            _build(EnhanceConfig(), tmp_path)

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
        assert payload["resolved_model"]["requires_non_commercial_ok"] is False
        assert payload["license_evaluation"] == {"enforced": True, "status": "allowed"}
        assert payload["input_files"] == ["a.jpg", "b.jpg"]

    def test_dead_flag_warnings_surface_in_plan(self, tmp_path: Path) -> None:
        config = EnhanceConfig(model_key="da3-metric", emit_marketing=True)
        payload = _build(config, tmp_path).to_payload()
        assert any("emit-marketing" in warning for warning in payload["warnings"])
        assert "marketing" not in " ".join(payload["requested_artifacts"])


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
        the REAL availability-check argv, not a reconstruction."""
        import dataclasses
        import types

        import transformation_portal.depth.backends.da3 as da3_module

        config = _commercial_config()
        invocation = _build(config, tmp_path)
        pinned_model = dataclasses.replace(invocation.resolved_model, revision="a" * 40)
        config.resolved_invocation = dataclasses.replace(invocation, resolved_model=pinned_model)
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
        assert command[command.index("--model-revision") + 1] == "a" * 40
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
