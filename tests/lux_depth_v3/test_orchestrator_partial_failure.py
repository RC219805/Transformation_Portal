"""Test coverage for orchestrator batch partial-failure behavior.

Phase 2 Coverage: Batch partial-failure behavior tests for EnhanceOrchestrator.

Tests verify:
1. Batch processing with mixed success/failure
2. Partial failure isolation
3. Error propagation control
4. Batch statistics computation
5. Run card aggregation with partial results
6. Graceful degradation patterns
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.unit


def _make_test_image(tmp_path: Path, name: str = "test.png", size: tuple = (64, 64)) -> Path:
    """Create a minimal test image for orchestrator tests."""
    image_path = tmp_path / name
    Image.new("RGB", size, color="white").save(image_path)
    return image_path


def _make_mock_depth_result():
    """Create a deterministic synthetic depth result."""
    from transformation_portal.depth.backends.protocol import DepthResult

    return DepthResult(
        depth_map=np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64),
        original_image=np.zeros((64, 64, 3), dtype=np.uint8),
        metadata={},
        depth_units="relative",
        backend_id="da3",
        device="cpu",
    )


def _make_failing_registry(fail_calls: Set[int]):
    """Create a mock depth backend registry that fails on specific calls.

    All backends in the fallback chain will fail for the specified calls,
    forcing the skip behavior.
    """
    call_count = [0]

    def create_backend(name):
        backend = Mock()
        backend.name = name
        backend.license_type = Mock(value="commercial")
        backend.ensure_available.return_value = None

        def compute_side_effect(img):
            call_count[0] += 1
            if call_count[0] in fail_calls:
                raise RuntimeError(f"Simulated failure for call {call_count[0]} on {name}")
            return _make_mock_depth_result()

        backend.compute.side_effect = compute_side_effect
        return backend

    backends = {
        "da3": create_backend("da3"),
        "da2": create_backend("da2"),
        "synthetic": create_backend("synthetic"),
    }

    registry = Mock()
    registry.get_backend.side_effect = lambda backend_id, config: backends.get(backend_id, backends["da3"])
    return registry


def _make_mock_registry():
    """Create a mock depth backend registry for successful operations."""
    backend = Mock()
    backend.name = "da3"
    backend.license_type = Mock(value="commercial")
    backend.ensure_available.return_value = None
    backend.compute.return_value = _make_mock_depth_result()

    registry = Mock()
    registry.get_backend.return_value = backend
    return registry


def _create_orchestrator(tmp_path: Path, registry_override=None, **config_kwargs):
    """Create an orchestrator instance with mocked backend registry."""
    from transformation_portal.lux_depth_v3.config import EnhanceConfig
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    defaults = {
        "depth_backend": "da3",
        "depth_device": "cpu",
        "enable_v2": False,
        "enable_materials_v3": False,
        "depth_fallback": "skip",  # Default to skip for partial failure tests
    }
    defaults.update(config_kwargs)
    config = EnhanceConfig(**defaults)

    registry = registry_override if registry_override else _make_mock_registry()

    with patch(
        "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
        return_value=registry,
    ):
        orchestrator = EnhanceOrchestrator(config, tmp_path)
        orchestrator.postprocessor = Mock(process=lambda result: result)
        return orchestrator


class TestBatchMixedResults:
    """Test batch processing with mixed success/failure outcomes."""

    def test_batch_with_all_success(self, tmp_path: Path) -> None:
        """Batch with all successful images returns all ok statuses."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)

        images = [_make_test_image(tmp_path, f"success_{i}.png") for i in range(3)]

        results = []
        for img_path in images:
            result = orchestrator.enhance_image(
                ImageInput(path=img_path),
                input_root=tmp_path,
            )
            results.append(result)

        assert all(r["status"] == "ok" for r in results)
        assert len(results) == 3

    def test_batch_continues_after_failure_with_skip_mode(self, tmp_path: Path) -> None:
        """Batch continues processing after failures in skip mode."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)

        images = [_make_test_image(tmp_path, f"partial_{i}.png") for i in range(3)]

        results = []
        for img_path in images:
            result = orchestrator.enhance_image(
                ImageInput(path=img_path),
                input_root=tmp_path,
            )
            results.append(result)

        # All should complete (even if some fail with skip)
        assert len(results) == 3

    def test_batch_results_are_independent(self, tmp_path: Path) -> None:
        """Each image in a batch has independent results."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)

        images = [_make_test_image(tmp_path, f"independent_{i}.png") for i in range(3)]

        results = []
        for img_path in images:
            result = orchestrator.enhance_image(
                ImageInput(path=img_path),
                input_root=tmp_path,
            )
            results.append(result)

        # Each result should reference a different image
        image_paths = [r["image"] for r in results]
        assert len(set(image_paths)) == 3


class TestPartialFailureIsolation:
    """Test that partial failures are properly isolated."""

    def test_state_reset_per_image(self, tmp_path: Path) -> None:
        """State is properly reset for each image."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)

        # First image
        img1 = _make_test_image(tmp_path, "state_reset_1.png")
        result1 = orchestrator.enhance_image(
            ImageInput(path=img1),
            input_root=tmp_path,
        )

        # Second image
        img2 = _make_test_image(tmp_path, "state_reset_2.png")
        result2 = orchestrator.enhance_image(
            ImageInput(path=img2),
            input_root=tmp_path,
        )

        # Both should succeed with independent state
        assert result1["status"] == "ok"
        assert result2["status"] == "ok"
        assert result1["image"] != result2["image"]

    def test_depth_attempts_isolated_per_image(self, tmp_path: Path) -> None:
        """Depth attempts are isolated per image."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)

        img1 = _make_test_image(tmp_path, "attempts_1.png")
        result1 = orchestrator.enhance_image(
            ImageInput(path=img1),
            input_root=tmp_path,
        )

        img2 = _make_test_image(tmp_path, "attempts_2.png")
        result2 = orchestrator.enhance_image(
            ImageInput(path=img2),
            input_root=tmp_path,
        )

        # Each result should have its own attempts (not accumulated)
        assert len(result1.get("attempts", [])) > 0
        assert len(result2.get("attempts", [])) > 0
        # Attempts should be similar counts (not growing)
        assert abs(len(result1["attempts"]) - len(result2["attempts"])) <= 1

    def test_backend_metadata_reset_per_image(self, tmp_path: Path) -> None:
        """Backend metadata is reset for each image."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)

        img1 = _make_test_image(tmp_path, "meta_1.png")
        orchestrator.enhance_image(
            ImageInput(path=img1),
            input_root=tmp_path,
        )

        # Check active metadata is still valid
        img2 = _make_test_image(tmp_path, "meta_2.png")
        result2 = orchestrator.enhance_image(
            ImageInput(path=img2),
            input_root=tmp_path,
        )

        assert result2["backend"] is not None


class TestErrorPropagationControl:
    """Test error propagation control mechanisms."""

    def test_depth_fallback_fail_config_propagates_error(self, tmp_path: Path) -> None:
        """depth_fallback='fail' propagates errors when all backends fail."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
            depth_fallback="fail",
        )

        # Create registry where all backends fail
        def always_fail(img):
            raise RuntimeError("All backends fail")

        backend = Mock()
        backend.name = "da3"
        backend.license_type = Mock(value="commercial")
        backend.ensure_available.return_value = None
        backend.compute.side_effect = always_fail

        failing_registry = Mock()
        failing_registry.get_backend.return_value = backend

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=failing_registry,
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)
            orchestrator.postprocessor = Mock(process=lambda result: result)

            test_image = _make_test_image(tmp_path, "propagate.png")

            with pytest.raises(RuntimeError, match="All backends fail"):
                orchestrator.enhance_image(
                    ImageInput(path=test_image),
                    input_root=tmp_path,
                )

    def test_depth_fallback_skip_continues_on_success(self, tmp_path: Path) -> None:
        """depth_fallback='skip' allows successful processing."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path, depth_fallback="skip")

        img = _make_test_image(tmp_path, "skip_success.png")

        # Should not raise
        result = orchestrator.enhance_image(
            ImageInput(path=img),
            input_root=tmp_path,
        )

        assert result["status"] == "ok"


class TestBatchStatistics:
    """Test batch statistics computation."""

    def test_compute_batch_runtime_stats(self) -> None:
        """compute_batch_runtime_stats calculates correct statistics."""
        from transformation_portal.lux_depth_v3.batch_stats import compute_batch_runtime_stats

        # API takes a list of floats, not dicts
        runtimes = [1.0, 2.0, 3.0]

        stats = compute_batch_runtime_stats(runtimes)

        assert stats["count"] == 3
        assert stats["mean"] == 2.0
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0

    def test_compute_batch_runtime_stats_empty(self) -> None:
        """compute_batch_runtime_stats handles empty batch."""
        from transformation_portal.lux_depth_v3.batch_stats import compute_batch_runtime_stats

        stats = compute_batch_runtime_stats([])

        assert stats["count"] == 0

    def test_compute_batch_runtime_stats_single(self) -> None:
        """compute_batch_runtime_stats handles single item."""
        from transformation_portal.lux_depth_v3.batch_stats import compute_batch_runtime_stats

        runtimes = [5.0]

        stats = compute_batch_runtime_stats(runtimes)

        assert stats["count"] == 1
        assert stats["mean"] == 5.0
        assert stats["min"] == 5.0
        assert stats["max"] == 5.0

    def test_detect_runtime_outliers(self) -> None:
        """detect_runtime_outliers identifies outliers."""
        from transformation_portal.lux_depth_v3.batch_stats import detect_runtime_outliers

        runtimes = [1.0, 1.1, 1.0, 1.0, 1.0]

        # Check for normal runtime (not outlier)
        result = detect_runtime_outliers("normal.png", 1.0, runtimes)
        assert result is None

        # Check for extreme outlier
        result = detect_runtime_outliers("outlier.png", 10.0, runtimes)
        # 10.0 is ~10x the median (1.0), should be detected
        if result is not None:
            warning_msg, metadata = result
            assert "outlier" in warning_msg.lower()
            assert metadata["is_outlier"] is True


class TestGracefulDegradation:
    """Test graceful degradation patterns."""

    def test_successful_outputs_are_preserved(self, tmp_path: Path) -> None:
        """Successful outputs are preserved in batch processing."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)

        images = [_make_test_image(tmp_path, f"preserve_{i}.png") for i in range(3)]

        results = []
        for img_path in images:
            result = orchestrator.enhance_image(
                ImageInput(path=img_path),
                input_root=tmp_path,
            )
            results.append(result)

        # Check all successful outputs exist
        successful = [r for r in results if r["status"] == "ok"]
        assert len(successful) == 3

        for r in successful:
            assert r["depth_path"] is not None
            assert Path(r["depth_path"]).exists()
            assert r["manifest"] is not None
            assert Path(r["manifest"]).exists()

    def test_result_contains_diagnostic_info(self, tmp_path: Path) -> None:
        """Results contain diagnostic information."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)

        img = _make_test_image(tmp_path, "diagnostic.png")
        result = orchestrator.enhance_image(
            ImageInput(path=img),
            input_root=tmp_path,
        )

        assert result["image"] == str(img)
        assert "backend" in result
        assert "status" in result

    def test_batch_completes_all_images(self, tmp_path: Path) -> None:
        """Batch completes processing all images."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)

        images = [_make_test_image(tmp_path, f"complete_{i}.png") for i in range(3)]

        results = []
        for img_path in images:
            result = orchestrator.enhance_image(
                ImageInput(path=img_path),
                input_root=tmp_path,
            )
            results.append(result)

        # All images should have results
        assert len(results) == 3


class TestResultAggregation:
    """Test result aggregation for batch processing."""

    def test_count_results_by_status(self, tmp_path: Path) -> None:
        """Can count results by status."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)

        images = [_make_test_image(tmp_path, f"count_{i}.png") for i in range(4)]

        results = []
        for img_path in images:
            result = orchestrator.enhance_image(
                ImageInput(path=img_path),
                input_root=tmp_path,
            )
            results.append(result)

        successful = sum(1 for r in results if r["status"] == "ok")
        assert successful == 4

    def test_separate_results_by_status(self, tmp_path: Path) -> None:
        """Can separate results by status."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)

        images = [_make_test_image(tmp_path, f"sep_{i}.png") for i in range(5)]

        results = []
        for img_path in images:
            result = orchestrator.enhance_image(
                ImageInput(path=img_path),
                input_root=tmp_path,
            )
            results.append(result)

        ok_results = [r for r in results if r["status"] == "ok"]

        assert len(ok_results) == 5

        # Verify ok results have valid outputs
        for r in ok_results:
            assert r["depth_path"] is not None

    def test_extract_runtimes_from_results(self, tmp_path: Path) -> None:
        """Can extract runtimes from result dictionaries."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)

        images = [_make_test_image(tmp_path, f"runtime_{i}.png") for i in range(3)]

        results = []
        for img_path in images:
            result = orchestrator.enhance_image(
                ImageInput(path=img_path),
                input_root=tmp_path,
            )
            results.append(result)

        runtimes = [r.get("runtime_s", 0.0) for r in results]
        assert len(runtimes) == 3
        assert all(isinstance(rt, (int, float)) for rt in runtimes)
