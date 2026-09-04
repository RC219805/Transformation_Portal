"""Focused regression tests for the APEX matrix runner."""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

import scripts.apex_matrix_runner as apex_matrix_runner  # pylint: disable=consider-using-from-import

canonical_apex_matrix_runner = apex_matrix_runner._impl  # pylint: disable=protected-access

pytestmark = pytest.mark.unit


class _StopAfterConfig(Exception):
    """Raised by the fake orchestrator once it captures runner config."""


def _valid_batch_result(*, runtime_s: float = 0.25) -> dict[str, object]:
    return {
        "status": "ok",
        "original_shape": (2, 3),
        "enforced_shape": (2, 3),
        "runtime_s": runtime_s,
        "input_sha256": "a" * 64,
    }


@pytest.mark.parametrize(
    ("invalid_result", "message"),
    [
        pytest.param(
            {key: value for key, value in _valid_batch_result().items() if key != "runtime_s"},
            "valid runtime_s",
            id="missing-runtime",
        ),
    ],
)
def test_batch_timing_evidence_rejects_incomplete_result_rows(tmp_path, invalid_result, message):
    images = (tmp_path / "first.jpg", tmp_path / "second.jpg")
    with pytest.raises(RuntimeError, match=message):
        canonical_apex_matrix_runner._validate_apex_batch_results(
            images,
            (_valid_batch_result(), invalid_result),
            batch_total_seconds=1.0,
        )


def test_batch_timing_evidence_preserves_partial_success(tmp_path):
    images = (tmp_path / "first.jpg", tmp_path / "second.jpg")

    rows, inner_total, shared_overhead = canonical_apex_matrix_runner._validate_apex_batch_results(
        images,
        (_valid_batch_result(), {**_valid_batch_result(), "status": "error"}),
        batch_total_seconds=1.0,
    )

    assert [row.image_path for row in rows] == [images[0]]
    assert inner_total == pytest.approx(0.25)
    assert shared_overhead == pytest.approx(0.75)


def test_batch_timing_evidence_rejects_inner_total_overrun(tmp_path):
    images = (tmp_path / "first.jpg", tmp_path / "second.jpg")
    with pytest.raises(RuntimeError, match="inner pipeline timings exceed"):
        canonical_apex_matrix_runner._validate_apex_batch_results(
            images,
            (_valid_batch_result(runtime_s=0.75), _valid_batch_result(runtime_s=0.75)),
            batch_total_seconds=1.0,
        )


def test_batch_timing_evidence_reconciles_monotonic_rounding_noise(tmp_path):
    images = (tmp_path / "first.jpg", tmp_path / "second.jpg")
    _rows, inner_total, shared_overhead = canonical_apex_matrix_runner._validate_apex_batch_results(
        images,
        (
            _valid_batch_result(runtime_s=0.5000002),
            _valid_batch_result(runtime_s=0.5000002),
        ),
        batch_total_seconds=1.0,
    )

    assert inner_total == pytest.approx(1.0000004)
    assert shared_overhead == 0.0


@pytest.mark.parametrize("backend_id", ["da3", "depth-anything-v3", "depth_anything_v3"])
def test_real_da3_run_uses_governed_metric_model_key(monkeypatch, tmp_path, backend_id):
    """The scheduled real DA3 lane must not fall back to the research selector."""
    from transformation_portal.metrics.contracts import RunSpec

    captured = {}
    monkeypatch.chdir(tmp_path)
    input_dir = Path("inputs")
    input_dir.mkdir()
    image_path = input_dir / "sample.jpg"
    image_path.touch()

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
    assert config.enable_parallel_processing is False
    assert captured["prepared"].input_root == input_dir.resolve()
    assert captured["prepared"].input_files == (image_path.resolve(),)
    assert captured["verify_outputs"] is False


def test_real_run_uses_prepared_canonical_paths_after_alias_retarget(monkeypatch, tmp_path):
    """Prepared results retain canonical paths and authoritative batch timing."""

    import hashlib

    from PIL import Image

    from transformation_portal.metrics.contracts import RunSpec

    captured = {}
    lifecycle_events = []

    @contextmanager
    def fake_timing_context(phase_name, timings, *, device):
        assert phase_name == "total"
        assert device == "cpu"
        lifecycle_events.append("timer:start")
        yield
        lifecycle_events.append("timer:stop")
        timings[phase_name] = 12.5

    monkeypatch.setitem(
        apex_matrix_runner.run_apex_for_config.__globals__,
        "timing_context",
        fake_timing_context,
    )
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    planned_image = input_dir / "planned.jpg"
    second_image = input_dir / "second.jpg"
    outside_image = tmp_path / "outside.jpg"
    Image.new("RGB", (3, 2), color=(10, 20, 30)).save(planned_image)
    Image.new("RGB", (5, 4), color=(40, 50, 60)).save(second_image)
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
    discovery_module.discover_images = lambda input_dir, _config, _extensions: [alias, second_image]

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

        def enhance_batch(self, input_root, *, input_files):
            captured["batch_input_root"] = input_root
            captured["batch_input_files"] = tuple(input_files)
            lifecycle_events.extend(
                [
                    "snapshot",
                    "dimension_probe",
                    "pipeline",
                    "carrier",
                    "run_card",
                    "execution_evidence",
                ]
            )
            return [
                {
                    "status": "ok",
                    "original_shape": (2, 3),
                    "enforced_shape": (2, 3),
                    "runtime_s": 0.25,
                    "input_sha256": hashlib.sha256(planned_image.read_bytes()).hexdigest(),
                },
                {
                    "status": "ok",
                    "original_shape": (4, 5),
                    "enforced_shape": (4, 5),
                    "runtime_s": 1.25,
                    "input_sha256": hashlib.sha256(second_image.read_bytes()).hexdigest(),
                },
            ]

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
        sample_size=2,
    )

    canonical_path = planned_image.resolve()
    second_canonical_path = second_image.resolve()
    first_capsule, second_capsule = observation.capsules
    assert alias.resolve() == outside_image.resolve()
    assert captured["prepared"].input_files == (canonical_path, second_canonical_path)
    assert captured["batch_input_root"] == input_dir.resolve()
    assert captured["batch_input_files"] == (canonical_path, second_canonical_path)
    assert first_capsule.image_path == str(canonical_path)
    assert first_capsule.original_shape == (2, 3)
    assert first_capsule.input_hash == hashlib.sha256(planned_image.read_bytes()).hexdigest()[:16]
    assert second_capsule.image_path == str(second_canonical_path)
    assert second_capsule.original_shape == (4, 5)
    assert second_capsule.input_hash == hashlib.sha256(second_image.read_bytes()).hexdigest()[:16]
    assert lifecycle_events == [
        "timer:start",
        "snapshot",
        "dimension_probe",
        "pipeline",
        "carrier",
        "run_card",
        "execution_evidence",
        "timer:stop",
    ]
    assert first_capsule.timings == {
        "total": 12.5,
        "pipeline": 0.25,
        "batch_shared_overhead": 11.0,
    }
    assert second_capsule.timings == {
        "total": 12.5,
        "pipeline": 1.25,
        "batch_shared_overhead": 11.0,
    }
    assert observation.phase_timings == {
        "authoritative_batch_total": 12.5,
        "inner_pipeline_total": 1.5,
        "shared_batch_overhead": 11.0,
    }


def test_real_run_timeout_cannot_be_swallowed_by_per_input_exception_handler(monkeypatch, tmp_path):
    """The batch deadline must escape inner ``except Exception`` handlers."""

    import signal

    from PIL import Image

    from transformation_portal.lux_depth_v3 import input_discovery as discovery_module
    from transformation_portal.lux_depth_v3 import orchestrator as orchestrator_module
    from transformation_portal.lux_depth_v3 import raw_loader as raw_loader_module
    from transformation_portal.metrics.contracts import RunSpec

    if not hasattr(signal, "SIGALRM"):
        pytest.skip("SIGALRM is unavailable on this platform")

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    image_path = input_dir / "sample.jpg"
    Image.new("RGB", (3, 2), color=(10, 20, 30)).save(image_path)
    installed_handler = {}

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
    monkeypatch.setattr(discovery_module, "discover_images", lambda *_args, **_kwargs: [image_path])
    monkeypatch.setattr(raw_loader_module, "RAW_EXTENSIONS", frozenset())
    monkeypatch.setattr(signal, "signal", lambda _signal_number, handler: installed_handler.setdefault("value", handler))
    monkeypatch.setattr(signal, "alarm", lambda _seconds: 0)

    class CatchingOrchestrator:
        @classmethod
        def from_prepared(cls, prepared, *, output_root, verify_outputs):
            return cls()

        def enhance_batch(self, input_root, *, input_files):
            try:
                installed_handler["value"](signal.SIGALRM, None)
            except Exception:
                return [_valid_batch_result()]
            raise AssertionError("timeout handler unexpectedly returned")

    monkeypatch.setattr(orchestrator_module, "EnhanceOrchestrator", CatchingOrchestrator)
    run_spec = RunSpec(
        run_id="timeout-run",
        commit_sha="abc123",
        workflow_version="v1",
        zones=["local"],
        device="cpu",
        backend_id="synthetic",
        timestamp="2026-09-05T00:00:00+00:00",
    )

    with pytest.raises(RuntimeError, match="authoritative batch timed out after 300 seconds"):
        apex_matrix_runner.run_apex_for_config(
            run_spec=run_spec,
            zone="local",
            output_dir=tmp_path / "results",
            dry_run=False,
            synthetic=False,
            input_dir=input_dir,
            sample_size=1,
        )


def test_real_run_uses_authoritative_raw_result_metadata(monkeypatch, tmp_path):
    """RAW capsule metadata must come from the prepared batch result."""
    import hashlib

    from transformation_portal.lux_depth_v3 import input_discovery as discovery_module
    from transformation_portal.lux_depth_v3 import orchestrator as orchestrator_module
    from transformation_portal.lux_depth_v3 import raw_loader as raw_loader_module
    from transformation_portal.metrics.contracts import RunSpec

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    raw_path = input_dir / "sample.dng"
    raw_bytes = b"raw-dimension-probe-fixture"
    raw_path.write_bytes(raw_bytes)

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
    monkeypatch.setattr(discovery_module, "discover_images", lambda *_args, **_kwargs: [raw_path])
    monkeypatch.setattr(raw_loader_module, "RAW_EXTENSIONS", frozenset({".dng"}))

    class CapturingOrchestrator:
        @classmethod
        def from_prepared(cls, prepared, *, output_root, verify_outputs):
            return cls()

        def enhance_batch(self, input_root, *, input_files):
            return [
                {
                    "status": "ok",
                    "original_shape": (7, 11),
                    "enforced_shape": (7, 12),
                    # This metadata-only stub returns immediately, so its
                    # claimed inner runtime must remain within measured wall time.
                    "runtime_s": 0.0,
                    "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                }
            ]

    monkeypatch.setattr(orchestrator_module, "EnhanceOrchestrator", CapturingOrchestrator)
    run_spec = RunSpec(
        run_id="raw-probe-run",
        commit_sha="abc123",
        workflow_version="v1",
        zones=["local"],
        device="cpu",
        backend_id="synthetic",
        timestamp="2026-09-04T00:00:00+00:00",
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

    assert observation.capsules[0].original_shape == (7, 11)
    assert observation.capsules[0].enforced_shape == (7, 12)
    assert observation.capsules[0].input_hash == hashlib.sha256(raw_bytes).hexdigest()[:16]
