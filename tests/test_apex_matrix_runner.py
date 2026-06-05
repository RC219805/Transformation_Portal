"""Focused regression tests for the APEX matrix runner."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


class _StopAfterConfig(Exception):
    """Raised by the fake orchestrator once it captures runner config."""


def test_real_da3_run_uses_governed_metric_model_key(monkeypatch, tmp_path):
    """The scheduled real DA3 lane must not fall back to the research selector."""
    from scripts import apex_matrix_runner
    from transformation_portal.metrics.contracts import RunSpec

    captured = {}
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    monkeypatch.setattr(
        apex_matrix_runner,
        "check_ml_dependencies",
        lambda backend_id: (True, []),
    )

    discovery_module = types.ModuleType("transformation_portal.lux_depth_v3.input_discovery")

    class DiscoveryConfig:
        def __init__(self, *, strict_mode: bool) -> None:
            self.strict_mode = strict_mode

    discovery_module.DiscoveryConfig = DiscoveryConfig
    discovery_module.discover_images = lambda input_dir, _config, _extensions: [Path(input_dir) / "sample.jpg"]

    raw_loader_module = types.ModuleType("transformation_portal.lux_depth_v3.raw_loader")
    raw_loader_module.RAW_EXTENSIONS = frozenset()

    orchestrator_module = types.ModuleType("transformation_portal.lux_depth_v3.orchestrator")

    class CapturingOrchestrator:
        def __init__(self, *, config, output_root, verify_outputs) -> None:
            captured["config"] = config
            captured["output_root"] = output_root
            captured["verify_outputs"] = verify_outputs
            raise _StopAfterConfig

    orchestrator_module.EnhanceOrchestrator = CapturingOrchestrator

    monkeypatch.setitem(sys.modules, discovery_module.__name__, discovery_module)
    monkeypatch.setitem(sys.modules, raw_loader_module.__name__, raw_loader_module)
    monkeypatch.setitem(sys.modules, orchestrator_module.__name__, orchestrator_module)

    run_spec = RunSpec(
        run_id="scheduled-run",
        commit_sha="abc123",
        workflow_version="v1",
        zones=["local"],
        device="cpu",
        backend_id="da3",
        timestamp="2026-06-04T00:00:00+00:00",
    )

    with pytest.raises(_StopAfterConfig):
        apex_matrix_runner.run_apex_for_config(
            run_spec=run_spec,
            zone="local",
            output_dir=tmp_path / "results",
            dry_run=False,
            synthetic=False,
            input_dir=input_dir,
            sample_size=1,
        )

    config = captured["config"]
    assert config.depth_backend == "da3"
    assert config.model_key == "da3-metric"
    assert config.non_commercial_ok is False
    assert captured["verify_outputs"] is False
