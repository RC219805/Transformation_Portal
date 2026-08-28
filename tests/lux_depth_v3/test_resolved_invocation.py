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
