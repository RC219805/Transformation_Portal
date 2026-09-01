"""Focused regression tests for the APEX matrix runner."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import scripts.apex_matrix_runner as apex_matrix_runner  # pylint: disable=consider-using-from-import

pytestmark = pytest.mark.unit


class _StopAfterConfig(Exception):
    """Raised by the fake orchestrator once it captures runner config."""


@pytest.mark.parametrize("backend_id", ["da3", "depth-anything-v3", "depth_anything_v3"])
def test_real_da3_run_uses_governed_metric_model_key(monkeypatch, tmp_path, backend_id):
    """The scheduled real DA3 lane must not fall back to the research selector."""
    from transformation_portal.metrics.contracts import RunSpec

    captured = {}
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "sample.jpg").touch()

    # The public wrapper re-exports functions defined in the canonical module,
    # so patch the delegated function's globals rather than the wrapper module.
    monkeypatch.setitem(
        apex_matrix_runner.run_apex_for_config.__globals__,
        "check_ml_dependencies",
        lambda backend_id: (True, []),
    )

    discovery_module_name = "transformation_portal.lux_depth_v3.input_discovery"
    discovery_module = types.ModuleType(discovery_module_name)

    class DiscoveryConfig:
        def __init__(self, *, strict_mode: bool) -> None:
            self.strict_mode = strict_mode

    discovery_module.DiscoveryConfig = DiscoveryConfig
    discovery_module.discover_images = lambda input_dir, _config, _extensions: [Path(input_dir) / "sample.jpg"]

    raw_loader_module_name = "transformation_portal.lux_depth_v3.raw_loader"
    raw_loader_module = types.ModuleType(raw_loader_module_name)
    raw_loader_module.RAW_EXTENSIONS = frozenset()

    orchestrator_module_name = "transformation_portal.lux_depth_v3.orchestrator"
    orchestrator_module = types.ModuleType(orchestrator_module_name)

    class CapturingOrchestrator:
        @classmethod
        def from_prepared(cls, prepared, *, output_root, verify_outputs) -> None:
            captured["prepared"] = prepared
            captured["config"] = prepared.runtime_config
            captured["output_root"] = output_root
            captured["verify_outputs"] = verify_outputs
            raise _StopAfterConfig

    orchestrator_module.EnhanceOrchestrator = CapturingOrchestrator

    monkeypatch.setitem(sys.modules, discovery_module_name, discovery_module)
    monkeypatch.setitem(sys.modules, raw_loader_module_name, raw_loader_module)
    monkeypatch.setitem(sys.modules, orchestrator_module_name, orchestrator_module)

    run_spec = RunSpec(
        run_id="scheduled-run",
        commit_sha="abc123",
        workflow_version="v1",
        zones=["local"],
        device="cpu",
        backend_id=backend_id,
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
    assert config.model_variant is None
    assert config.non_commercial_ok is False
    assert captured["prepared"].input_files == (input_dir / "sample.jpg",)
    assert captured["verify_outputs"] is False


def test_real_run_uses_prepared_canonical_paths_after_alias_retarget(monkeypatch, tmp_path):
    """Direct dimension/hash reads must not reopen a mutable discovery alias."""

    import hashlib

    from PIL import Image

    from transformation_portal.metrics.contracts import RunSpec

    captured = {}
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    planned_image = input_dir / "planned.jpg"
    outside_image = tmp_path / "outside.jpg"
    Image.new("RGB", (3, 2), color=(10, 20, 30)).save(planned_image)
    Image.new("RGB", (11, 7), color=(200, 210, 220)).save(outside_image)
    alias = input_dir / "alias.jpg"
    alias.symlink_to(planned_image)

    monkeypatch.setitem(
        apex_matrix_runner.run_apex_for_config.__globals__,
        "check_ml_dependencies",
        lambda backend_id: (True, []),
    )
    monkeypatch.setitem(
        apex_matrix_runner.run_apex_for_config.__globals__,
        "_get_pipeline_version",
        lambda: "test",
    )

    discovery_module_name = "transformation_portal.lux_depth_v3.input_discovery"
    discovery_module = types.ModuleType(discovery_module_name)

    class DiscoveryConfig:
        def __init__(self, *, strict_mode: bool) -> None:
            self.strict_mode = strict_mode

    discovery_module.DiscoveryConfig = DiscoveryConfig
    discovery_module.discover_images = lambda input_dir, _config, _extensions: [alias]

    raw_loader_module_name = "transformation_portal.lux_depth_v3.raw_loader"
    raw_loader_module = types.ModuleType(raw_loader_module_name)
    raw_loader_module.RAW_EXTENSIONS = frozenset()

    orchestrator_module_name = "transformation_portal.lux_depth_v3.orchestrator"
    orchestrator_module = types.ModuleType(orchestrator_module_name)

    class CapturingOrchestrator:
        @classmethod
        def from_prepared(cls, prepared, *, output_root, verify_outputs):
            captured["prepared"] = prepared
            alias.unlink()
            alias.symlink_to(outside_image)
            return cls()

        def enhance_image(self, image_input, *, input_root):
            captured["image_path"] = image_input.path
            return {"status": "ok", "enforced_shape": (2, 3)}

    orchestrator_module.EnhanceOrchestrator = CapturingOrchestrator

    monkeypatch.setitem(sys.modules, discovery_module_name, discovery_module)
    monkeypatch.setitem(sys.modules, raw_loader_module_name, raw_loader_module)
    monkeypatch.setitem(sys.modules, orchestrator_module_name, orchestrator_module)

    run_spec = RunSpec(
        run_id="canonical-path-run",
        commit_sha="abc123",
        workflow_version="v1",
        zones=["local"],
        device="cpu",
        backend_id="synthetic",
        timestamp="2026-09-01T00:00:00+00:00",
    )

    observation = apex_matrix_runner.run_apex_for_config(
        run_spec=run_spec,
        zone="local",
        output_dir=tmp_path / "results",
        dry_run=False,
        synthetic=False,
        input_dir=input_dir,
        sample_size=1,
    )

    canonical_path = planned_image.resolve()
    capsule = observation.capsules[0]
    assert alias.resolve() == outside_image.resolve()
    assert captured["prepared"].input_files == (canonical_path,)
    assert captured["image_path"] == canonical_path
    assert capsule.image_path == str(canonical_path)
    assert capsule.original_shape == (2, 3)
    assert capsule.input_hash == hashlib.sha256(planned_image.read_bytes()).hexdigest()[:16]
