"""Tests for universal hardening wrapper.

This module tests the UniversalHardenedWrapper from hardening/universal.py
which provides a generic hardening layer for any pipeline.

Covers:
- Pipeline protocol compliance
- ProcessingReport data class
- Input validation behavior
- Profiling and stamping
- Error handling
- Config hashing
- Runtime metadata gathering
- Function wrapping
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.security]


# =============================================================================
# Test ProcessingReport dataclass
# =============================================================================


@pytest.mark.security
class TestProcessingReport:
    """Tests for ProcessingReport dataclass."""

    def test_create_report(self) -> None:
        """ProcessingReport can be instantiated with valid data."""
        from transformation_portal.hardening.universal import ProcessingReport

        report = ProcessingReport(
            run_id="test-run-123",
            input_path="/path/to/input.jpg",
            config_hash="abc123def456",
            duration_ms=150.5,
            success=True,
            error=None,
            meta={"key": "value"},
        )

        assert report.run_id == "test-run-123"
        assert report.input_path == "/path/to/input.jpg"
        assert report.config_hash == "abc123def456"
        assert report.duration_ms == 150.5
        assert report.success is True
        assert report.error is None
        assert report.meta == {"key": "value"}

    def test_report_to_dict(self) -> None:
        """ProcessingReport.to_dict() returns proper dictionary."""
        from transformation_portal.hardening.universal import ProcessingReport

        report = ProcessingReport(
            run_id="test-run-123",
            input_path="/path/to/input.jpg",
            config_hash="abc123def456",
            duration_ms=100.0,
            success=True,
            error=None,
            meta={},
        )

        result = report.to_dict()

        assert isinstance(result, dict)
        assert result["run_id"] == "test-run-123"
        assert result["input_path"] == "/path/to/input.jpg"
        assert result["config_hash"] == "abc123def456"
        assert result["duration_ms"] == 100.0
        assert result["success"] is True
        assert result["error"] is None
        assert result["meta"] == {}

    def test_report_to_dict_matches_asdict(self) -> None:
        """ProcessingReport.to_dict() matches dataclasses.asdict()."""
        from transformation_portal.hardening.universal import ProcessingReport

        report = ProcessingReport(
            run_id="test-run-123",
            input_path="/path/to/input.jpg",
            config_hash="abc123",
            duration_ms=100.0,
            success=True,
            error=None,
            meta={"nested": {"key": "value"}},
        )

        assert report.to_dict() == asdict(report)

    def test_report_save(self, tmp_path: Path) -> None:
        """ProcessingReport.save() writes valid JSON."""
        from transformation_portal.hardening.universal import ProcessingReport

        report = ProcessingReport(
            run_id="test-run-save",
            input_path="/input.jpg",
            config_hash="hash123",
            duration_ms=50.0,
            success=True,
            error=None,
            meta={"saved": True},
        )

        output_path = tmp_path / "report.json"
        report.save(output_path)

        assert output_path.exists()

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["run_id"] == "test-run-save"
        assert loaded["meta"]["saved"] is True

    def test_report_with_error(self) -> None:
        """ProcessingReport correctly stores error information."""
        from transformation_portal.hardening.universal import ProcessingReport

        report = ProcessingReport(
            run_id="failed-run",
            input_path="/input.jpg",
            config_hash="hash",
            duration_ms=10.0,
            success=False,
            error="Something went wrong",
            meta={},
        )

        assert report.success is False
        assert report.error == "Something went wrong"

    def test_report_with_none_duration(self) -> None:
        """ProcessingReport handles None duration (no profiling)."""
        from transformation_portal.hardening.universal import ProcessingReport

        report = ProcessingReport(
            run_id="run",
            input_path="/input.jpg",
            config_hash="hash",
            duration_ms=None,
            success=True,
            error=None,
            meta={},
        )

        assert report.duration_ms is None
        result = report.to_dict()
        assert result["duration_ms"] is None


# =============================================================================
# Test UniversalHardenedWrapper
# =============================================================================


class MockPipeline:
    """Mock pipeline implementing the Pipeline protocol."""

    def __init__(self, return_value: Any = None, raise_exception: Exception | None = None):
        self.return_value = return_value or {"processed": True}
        self.raise_exception = raise_exception
        self.process_calls: list[tuple] = []

    def process(self, input_path: Path, **kwargs) -> Any:
        self.process_calls.append((input_path, kwargs))
        if self.raise_exception:
            raise self.raise_exception
        return self.return_value


@pytest.mark.security
class TestUniversalHardenedWrapper:
    """Tests for UniversalHardenedWrapper class."""

    def test_wrapper_initialization(self) -> None:
        """Wrapper initializes with valid pipeline."""
        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        mock_pipeline = MockPipeline()
        wrapper = UniversalHardenedWrapper(
            pipeline=mock_pipeline,
            policy=None,
            enable_profiling=True,
            enable_stamping=True,
            enable_input_validation=False,
        )

        assert wrapper.pipeline is mock_pipeline
        assert wrapper.enable_profiling is True
        assert wrapper.enable_stamping is True
        assert wrapper.enable_input_validation is False

    def test_wrapper_process_success(self, tmp_path: Path) -> None:
        """Wrapper processes input successfully."""
        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        mock_pipeline = MockPipeline(return_value={"output": "data"})
        wrapper = UniversalHardenedWrapper(
            pipeline=mock_pipeline,
            policy=None,
            enable_profiling=True,
            enable_stamping=True,
            enable_input_validation=False,
        )

        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"fake image data")

        result = wrapper.process(input_file, option="value")

        assert result["success"] is True
        assert result["result"] == {"output": "data"}
        assert "report" in result

    def test_wrapper_process_with_profiling(self, tmp_path: Path) -> None:
        """Wrapper records duration when profiling enabled."""
        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        mock_pipeline = MockPipeline()
        wrapper = UniversalHardenedWrapper(
            pipeline=mock_pipeline,
            policy=None,
            enable_profiling=True,
            enable_stamping=True,
            enable_input_validation=False,
        )

        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"data")

        result = wrapper.process(input_file)

        assert result["report"].duration_ms is not None
        assert result["report"].duration_ms >= 0

    def test_wrapper_process_without_profiling(self, tmp_path: Path) -> None:
        """Wrapper has None duration when profiling disabled."""
        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        mock_pipeline = MockPipeline()
        wrapper = UniversalHardenedWrapper(
            pipeline=mock_pipeline,
            policy=None,
            enable_profiling=False,
            enable_stamping=True,
            enable_input_validation=False,
        )

        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"data")

        result = wrapper.process(input_file)

        assert result["report"].duration_ms is None

    def test_wrapper_process_without_stamping(self, tmp_path: Path) -> None:
        """Wrapper returns no report when stamping disabled."""
        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        mock_pipeline = MockPipeline()
        wrapper = UniversalHardenedWrapper(
            pipeline=mock_pipeline,
            policy=None,
            enable_profiling=True,
            enable_stamping=False,
            enable_input_validation=False,
        )

        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"data")

        result = wrapper.process(input_file)

        assert "report" not in result
        assert result["success"] is True

    def test_wrapper_handles_pipeline_exception(self, tmp_path: Path) -> None:
        """Wrapper catches pipeline exceptions and reports error."""
        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        mock_pipeline = MockPipeline(raise_exception=ValueError("Pipeline failed"))
        wrapper = UniversalHardenedWrapper(
            pipeline=mock_pipeline,
            policy=None,
            enable_profiling=True,
            enable_stamping=True,
            enable_input_validation=False,
        )

        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"data")

        result = wrapper.process(input_file)

        assert result["success"] is False
        assert result["result"] is None
        assert result["report"].success is False
        assert "Pipeline failed" in result["report"].error

    def test_wrapper_generates_unique_run_ids(self, tmp_path: Path) -> None:
        """Wrapper generates unique run IDs for each process call."""
        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        mock_pipeline = MockPipeline()
        wrapper = UniversalHardenedWrapper(
            pipeline=mock_pipeline,
            policy=None,
            enable_profiling=False,
            enable_stamping=True,
            enable_input_validation=False,
        )

        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"data")

        result1 = wrapper.process(input_file)
        result2 = wrapper.process(input_file)

        assert result1["report"].run_id != result2["report"].run_id
        # Verify they are valid UUIDs
        uuid.UUID(result1["report"].run_id)
        uuid.UUID(result2["report"].run_id)


# =============================================================================
# Test config hashing
# =============================================================================


@pytest.mark.security
class TestConfigHashing:
    """Tests for deterministic config hashing."""

    def test_config_hash_deterministic(self, tmp_path: Path) -> None:
        """Config hash is deterministic for same input."""
        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        mock_pipeline = MockPipeline()
        wrapper = UniversalHardenedWrapper(
            pipeline=mock_pipeline,
            policy=None,
            enable_profiling=False,
            enable_stamping=True,
            enable_input_validation=False,
        )

        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"data")

        result1 = wrapper.process(input_file, option="value", count=5)
        result2 = wrapper.process(input_file, option="value", count=5)

        assert result1["report"].config_hash == result2["report"].config_hash

    def test_config_hash_different_for_different_config(self, tmp_path: Path) -> None:
        """Config hash differs for different configurations."""
        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        mock_pipeline = MockPipeline()
        wrapper = UniversalHardenedWrapper(
            pipeline=mock_pipeline,
            policy=None,
            enable_profiling=False,
            enable_stamping=True,
            enable_input_validation=False,
        )

        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"data")

        result1 = wrapper.process(input_file, option="value1")
        result2 = wrapper.process(input_file, option="value2")

        assert result1["report"].config_hash != result2["report"].config_hash

    def test_config_hash_order_independent(self, tmp_path: Path) -> None:
        """Config hash is order-independent (sorted keys)."""
        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        mock_pipeline = MockPipeline()
        wrapper = UniversalHardenedWrapper(
            pipeline=mock_pipeline,
            policy=None,
            enable_profiling=False,
            enable_stamping=True,
            enable_input_validation=False,
        )

        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"data")

        result1 = wrapper.process(input_file, a=1, b=2, c=3)
        result2 = wrapper.process(input_file, c=3, a=1, b=2)

        assert result1["report"].config_hash == result2["report"].config_hash


# =============================================================================
# Test metadata gathering
# =============================================================================


@pytest.mark.security
class TestMetadataGathering:
    """Tests for runtime metadata gathering."""

    def test_basic_runtime_info(self) -> None:
        """Basic runtime info includes expected fields."""
        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        mock_pipeline = MockPipeline()
        wrapper = UniversalHardenedWrapper(
            pipeline=mock_pipeline,
            policy=None,
            enable_profiling=False,
            enable_stamping=True,
            enable_input_validation=False,
        )

        info = wrapper._basic_runtime_info()

        assert "python_version" in info
        assert "platform" in info
        assert "machine" in info

    def test_gather_metadata_includes_wrapper_version(self, tmp_path: Path) -> None:
        """Gathered metadata includes wrapper version."""
        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        mock_pipeline = MockPipeline()
        wrapper = UniversalHardenedWrapper(
            pipeline=mock_pipeline,
            policy=None,
            enable_profiling=False,
            enable_stamping=True,
            enable_input_validation=False,
        )

        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"data")

        result = wrapper.process(input_file)

        assert result["report"].meta.get("wrapper_version") == "2.0.0"


# =============================================================================
# Test wrap_function utility
# =============================================================================


@pytest.mark.security
class TestWrapFunction:
    """Tests for wrap_function utility."""

    def test_wrap_function_basic(self, tmp_path: Path) -> None:
        """wrap_function wraps a simple function."""
        from transformation_portal.hardening.universal import wrap_function

        def simple_processor(input_path: Path, **kwargs) -> Dict[str, Any]:
            return {"processed": str(input_path)}

        wrapper = wrap_function(
            simple_processor,
            policy=None,
            enable_profiling=True,
            enable_stamping=True,
            enable_input_validation=False,
        )

        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"data")

        result = wrapper.process(input_file)

        assert result["success"] is True
        assert "processed" in result["result"]

    def test_wrap_function_with_kwargs(self, tmp_path: Path) -> None:
        """wrap_function passes kwargs to wrapped function."""
        from transformation_portal.hardening.universal import wrap_function

        def processor_with_options(input_path: Path, multiplier: int = 1, **kwargs) -> Dict[str, Any]:
            return {"value": 10 * multiplier}

        wrapper = wrap_function(
            processor_with_options,
            policy=None,
            enable_profiling=False,
            enable_stamping=True,
            enable_input_validation=False,
        )

        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"data")

        result = wrapper.process(input_file, multiplier=5)

        assert result["result"]["value"] == 50

    def test_wrap_function_exception_handling(self, tmp_path: Path) -> None:
        """wrap_function handles exceptions from wrapped function."""
        from transformation_portal.hardening.universal import wrap_function

        def failing_processor(input_path: Path, **kwargs) -> Dict[str, Any]:
            raise RuntimeError("Function failed")

        wrapper = wrap_function(
            failing_processor,
            policy=None,
            enable_profiling=True,
            enable_stamping=True,
            enable_input_validation=False,
        )

        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"data")

        result = wrapper.process(input_file)

        assert result["success"] is False
        assert "Function failed" in result["report"].error


# =============================================================================
# Test Pipeline protocol
# =============================================================================


@pytest.mark.security
class TestPipelineProtocol:
    """Tests for Pipeline protocol compliance."""

    def test_pipeline_protocol_check(self) -> None:
        """Mock pipeline satisfies Pipeline protocol."""
        from transformation_portal.hardening.universal import Pipeline

        mock = MockPipeline()
        assert isinstance(mock, Pipeline)

    def test_non_pipeline_does_not_satisfy(self) -> None:
        """Objects without process method don't satisfy protocol."""
        from transformation_portal.hardening.universal import Pipeline

        class NotAPipeline:
            def run(self):
                pass

        assert not isinstance(NotAPipeline(), Pipeline)

    def test_partial_pipeline_does_not_satisfy(self) -> None:
        """Objects with wrong process signature still satisfy protocol runtime check."""
        from transformation_portal.hardening.universal import Pipeline

        # Note: Runtime protocol checks only verify method exists, not signature
        class PartialPipeline:
            def process(self):  # Missing input_path arg
                pass

        # Runtime check only verifies method presence
        partial = PartialPipeline()
        assert isinstance(partial, Pipeline)


# =============================================================================
# Input-validation paths (policy auto-load + _validate_input branches)
# =============================================================================


class _OkPipeline:
    """Minimal pipeline that echoes a success marker."""

    def process(self, input_path: Path, **kwargs: Any) -> Dict[str, Any]:
        return {"ok": True, "path": str(input_path)}


def _install_fake_lux_depth_v2(monkeypatch: pytest.MonkeyPatch, *, safe_io=None, policy_cls=None) -> None:
    """Inject fake ``lux_depth_v2.hardening`` submodules into ``sys.modules``.

    The production import targets are optional (``lux_depth_v2`` no longer
    ships), so the validation branches are only reachable when those modules
    are present. We synthesize just enough surface for the import statements.
    """
    import sys
    import types

    root = types.ModuleType("lux_depth_v2")
    hardening = types.ModuleType("lux_depth_v2.hardening")
    root.hardening = hardening
    monkeypatch.setitem(sys.modules, "lux_depth_v2", root)
    monkeypatch.setitem(sys.modules, "lux_depth_v2.hardening", hardening)

    if safe_io is not None:
        mod = types.ModuleType("lux_depth_v2.hardening.safe_io")
        mod.validate_input_path = safe_io
        hardening.safe_io = mod
        monkeypatch.setitem(sys.modules, "lux_depth_v2.hardening.safe_io", mod)

    if policy_cls is not None:
        mod = types.ModuleType("lux_depth_v2.hardening.policy")
        mod.HardeningPolicy = policy_cls
        hardening.policy = mod
        monkeypatch.setitem(sys.modules, "lux_depth_v2.hardening.policy", mod)


class TestInputValidationPaths:
    """Cover the policy auto-load and ``_validate_input`` branches."""

    def test_init_disables_validation_when_policy_unavailable(self) -> None:
        """policy=None + validation enabled, but no lux_depth_v2 → fail closed to disabled."""
        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        wrapper = UniversalHardenedWrapper(
            pipeline=_OkPipeline(),
            policy=None,
            enable_input_validation=True,
        )
        # The auto-load import fails (lux_depth_v2 absent) → validation disabled.
        assert wrapper.policy is None
        assert wrapper.enable_input_validation is False

    def test_init_autoloads_policy_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel_policy = object()

        class FakePolicy:
            @staticmethod
            def load():
                return sentinel_policy

        _install_fake_lux_depth_v2(monkeypatch, policy_cls=FakePolicy)

        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        wrapper = UniversalHardenedWrapper(
            pipeline=_OkPipeline(),
            policy=None,
            enable_input_validation=True,
        )
        assert wrapper.policy is sentinel_policy
        assert wrapper.enable_input_validation is True

    def test_process_returns_error_response_when_validation_raises(self) -> None:
        """A validation failure short-circuits into a structured error response."""
        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        # Truthy policy keeps validation enabled; _validate_input then tries to
        # import lux_depth_v2 (absent) and the failure becomes an error response.
        wrapper = UniversalHardenedWrapper(
            pipeline=_OkPipeline(),
            policy=MagicMock(),
            enable_input_validation=True,
        )
        out = wrapper.process(Path("/whatever.jpg"))
        assert out["success"] is False
        assert out["result"] is None
        assert out["report"].success is False
        assert out["report"].error

    def test_validate_input_returns_path_when_policy_is_none(self) -> None:
        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        wrapper = UniversalHardenedWrapper(
            pipeline=_OkPipeline(),
            policy=None,
            enable_input_validation=False,
        )
        wrapper.policy = None
        p = Path("/unchanged.jpg")
        assert wrapper._validate_input(p) is p

    def test_validate_input_uses_injected_validator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        validated = Path("/validated/output.jpg")
        _install_fake_lux_depth_v2(monkeypatch, safe_io=lambda path, policy: validated)

        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        wrapper = UniversalHardenedWrapper(
            pipeline=_OkPipeline(),
            policy=MagicMock(),
            enable_input_validation=False,
        )
        assert wrapper._validate_input(Path("/in.jpg")) == validated

    def test_validate_input_wraps_validator_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(path, policy):
            raise RuntimeError("denied")

        _install_fake_lux_depth_v2(monkeypatch, safe_io=_boom)

        from transformation_portal.hardening.universal import UniversalHardenedWrapper

        wrapper = UniversalHardenedWrapper(
            pipeline=_OkPipeline(),
            policy=MagicMock(),
            enable_input_validation=False,
        )
        with pytest.raises(ValueError, match="Input validation failed"):
            wrapper._validate_input(Path("/in.jpg"))
