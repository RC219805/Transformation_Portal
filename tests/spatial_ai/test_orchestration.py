"""Tests for spatial_ai orchestration module (Phase 5 coverage).

Tests for:
- ProgressTracker and progress events
- ResourceManager and device selection
- ErrorHandler and retry logic
- Pipeline error handling

All tests use mocks - no ML model downloads or GPU requirements.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from transformation_portal.spatial_ai.orchestration.error_handler import (
    ErrorContext,
    ErrorHandler,
    ErrorRecoveryStrategy,
    PipelineError,
)
from transformation_portal.spatial_ai.orchestration.progress_tracker import (
    ProgressEvent,
    ProgressEventType,
    ProgressTracker,
    StageMetrics,
)
from transformation_portal.spatial_ai.orchestration.resource_manager import (
    ResourceLimits,
    ResourceManager,
)

pytestmark = [pytest.mark.unit, pytest.mark.ml]


class TestProgressEventType:
    """Test ProgressEventType enum."""

    def test_event_types_exist(self):
        """Test all event types are defined."""
        assert ProgressEventType.PIPELINE_START.value == "pipeline_start"
        assert ProgressEventType.PIPELINE_COMPLETE.value == "pipeline_complete"
        assert ProgressEventType.STAGE_START.value == "stage_start"
        assert ProgressEventType.STAGE_PROGRESS.value == "stage_progress"
        assert ProgressEventType.STAGE_COMPLETE.value == "stage_complete"
        assert ProgressEventType.STAGE_ERROR.value == "stage_error"


class TestProgressEvent:
    """Test ProgressEvent dataclass."""

    def test_progress_event_creation(self):
        """Test creating a progress event."""
        event = ProgressEvent(
            event_type=ProgressEventType.STAGE_START,
            stage="ingest",
            message="Starting ingest",
            progress_percent=25.0,
            stage_progress_percent=0.0,
            elapsed_time=1.5,
        )

        assert event.event_type == ProgressEventType.STAGE_START
        assert event.stage == "ingest"
        assert event.message == "Starting ingest"
        assert event.progress_percent == 25.0
        assert event.stage_progress_percent == 0.0
        assert event.elapsed_time == 1.5

    def test_progress_event_with_metadata(self):
        """Test progress event with metadata."""
        event = ProgressEvent(
            event_type=ProgressEventType.STAGE_COMPLETE,
            stage="segment",
            message="Segmentation complete",
            progress_percent=50.0,
            metadata={"duration": 5.2, "masks_found": 12},
        )

        assert event.metadata["duration"] == 5.2
        assert event.metadata["masks_found"] == 12


class TestStageMetrics:
    """Test StageMetrics dataclass."""

    def test_stage_metrics_creation(self):
        """Test creating stage metrics."""
        metrics = StageMetrics(
            name="ingest",
            display_name="Linear Ingest",
            start_time=1000.0,
        )

        assert metrics.name == "ingest"
        assert metrics.display_name == "Linear Ingest"
        assert metrics.start_time == 1000.0
        assert metrics.end_time is None
        assert metrics.success is None

    def test_stage_metrics_duration(self):
        """Test duration calculation."""
        metrics = StageMetrics(
            name="ingest",
            display_name="Linear Ingest",
            start_time=1000.0,
            end_time=1005.5,
        )

        assert metrics.duration == 5.5

    def test_stage_metrics_duration_none_when_incomplete(self):
        """Test duration is None when stage not complete."""
        metrics = StageMetrics(
            name="ingest",
            display_name="Linear Ingest",
            start_time=1000.0,
        )

        assert metrics.duration is None

    def test_stage_metrics_is_complete(self):
        """Test is_complete property."""
        incomplete = StageMetrics(
            name="ingest",
            display_name="Linear Ingest",
            start_time=1000.0,
        )
        assert not incomplete.is_complete

        complete = StageMetrics(
            name="ingest",
            display_name="Linear Ingest",
            start_time=1000.0,
            end_time=1005.0,
        )
        assert complete.is_complete


class TestProgressTracker:
    """Test ProgressTracker class."""

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = ProgressTracker(total_stages=4)

        assert tracker.total_stages == 4
        assert tracker.enable_time_estimation is True
        assert tracker.get_progress_percent() == 0.0

    def test_tracker_with_historical_times(self):
        """Test tracker with historical time estimates."""
        historical = {"ingest": 2.0, "segment": 5.0, "materials": 3.0}
        tracker = ProgressTracker(
            total_stages=3,
            historical_times=historical,
        )

        assert tracker.historical_times == historical

    def test_start_pipeline(self):
        """Test starting pipeline."""
        tracker = ProgressTracker(total_stages=3)
        tracker.start_pipeline()

        assert tracker._pipeline_start_time is not None
        assert tracker.get_progress_percent() == 0.0

    def test_start_and_complete_stage(self):
        """Test starting and completing a stage."""
        tracker = ProgressTracker(total_stages=3)
        tracker.start_pipeline()

        tracker.start_stage("ingest", "Linear Ingest")
        assert tracker._current_stage == "ingest"
        assert "ingest" in tracker._stage_metrics

        tracker.complete_stage("ingest", success=True)
        assert tracker._current_stage is None
        assert tracker._completed_stages == 1
        assert tracker.get_progress_percent() == pytest.approx(33.33, rel=0.1)

    def test_stage_failure(self):
        """Test stage failure tracking."""
        tracker = ProgressTracker(total_stages=3)
        tracker.start_pipeline()

        tracker.start_stage("ingest", "Linear Ingest")
        tracker.complete_stage("ingest", success=False, error_message="Failed to load image")

        metrics = tracker._stage_metrics["ingest"]
        assert metrics.success is False
        assert metrics.error_message == "Failed to load image"
        # Failed stage doesn't count toward progress
        assert tracker._completed_stages == 0

    def test_update_stage_progress(self):
        """Test updating stage-level progress."""
        tracker = ProgressTracker(total_stages=2)
        tracker.start_pipeline()
        tracker.start_stage("ingest", "Linear Ingest")

        tracker.update_stage("ingest", 50.0)
        # Base progress is 0%, stage contributes 50% of its 50% share = 25%
        assert tracker.get_progress_percent() == 0.0  # Still at base until complete

    def test_complete_pipeline(self):
        """Test completing entire pipeline."""
        tracker = ProgressTracker(total_stages=2)
        tracker.start_pipeline()

        tracker.start_stage("stage1", "Stage 1")
        tracker.complete_stage("stage1", success=True)

        tracker.start_stage("stage2", "Stage 2")
        tracker.complete_stage("stage2", success=True)

        tracker.complete_pipeline(success=True)

        assert tracker.get_progress_percent() == 100.0
        assert tracker._pipeline_end_time is not None

    def test_get_summary(self):
        """Test getting pipeline summary."""
        tracker = ProgressTracker(total_stages=2)
        tracker.start_pipeline()

        tracker.start_stage("ingest", "Linear Ingest")
        tracker.complete_stage("ingest", success=True)

        summary = tracker.get_summary()

        assert summary["total_stages"] == 2
        assert summary["completed_stages"] == 1
        assert summary["progress_percent"] == 50.0
        assert "ingest" in summary["stages"]

    def test_update_nonexistent_stage_warning(self):
        """Test updating non-existent stage logs warning."""
        tracker = ProgressTracker(total_stages=2)
        tracker.start_pipeline()

        # Should not raise, just log warning
        tracker.update_stage("nonexistent", 50.0)


class TestResourceLimits:
    """Test ResourceLimits dataclass."""

    def test_default_limits(self):
        """Test default resource limits."""
        limits = ResourceLimits()

        assert limits.max_gpu_memory_gb is None
        assert limits.max_cpu_memory_gb is None
        assert limits.max_models_loaded == 3
        assert limits.batch_size == 1
        assert limits.device_preference == ["cuda", "mps", "cpu"]

    def test_custom_limits(self):
        """Test custom resource limits."""
        limits = ResourceLimits(
            max_gpu_memory_gb=8.0,
            max_cpu_memory_gb=16.0,
            max_models_loaded=2,
            batch_size=4,
        )

        assert limits.max_gpu_memory_gb == 8.0
        assert limits.max_cpu_memory_gb == 16.0
        assert limits.max_models_loaded == 2
        assert limits.batch_size == 4

    def test_invalid_gpu_memory_raises(self):
        """Test invalid GPU memory raises error."""
        with pytest.raises(ValueError, match="max_gpu_memory_gb must be positive"):
            ResourceLimits(max_gpu_memory_gb=-1.0)

    def test_invalid_cpu_memory_raises(self):
        """Test invalid CPU memory raises error."""
        with pytest.raises(ValueError, match="max_cpu_memory_gb must be positive"):
            ResourceLimits(max_cpu_memory_gb=0.0)

    def test_invalid_max_models_raises(self):
        """Test invalid max_models_loaded raises error."""
        with pytest.raises(ValueError, match="max_models_loaded must be positive"):
            ResourceLimits(max_models_loaded=0)

    def test_invalid_batch_size_raises(self):
        """Test invalid batch_size raises error."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            ResourceLimits(batch_size=-1)


class TestResourceManager:
    """Test ResourceManager class."""

    def test_manager_initialization(self):
        """Test resource manager initialization."""
        limits = ResourceLimits(max_models_loaded=2)
        manager = ResourceManager(limits)

        assert manager.limits.max_models_loaded == 2
        assert len(manager._loaded_models) == 0

    def test_context_manager(self):
        """Test resource manager as context manager."""
        with ResourceManager() as rm:
            rm.register_model("test_model", MagicMock())
            assert rm.get_model("test_model") is not None

        # Models should be cleaned up after context
        assert len(rm._loaded_models) == 0

    def test_select_device_cpu_fallback(self):
        """Test device selection falls back to CPU when no GPU."""
        with patch.dict("sys.modules", {"torch": None}):
            manager = ResourceManager()
            device = manager.select_device()
            assert device == "cpu"

    def test_select_device_with_mock_cuda(self):
        """Test device selection with mocked CUDA."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value.total_memory = 16e9  # 16GB

        with patch.dict("sys.modules", {"torch": mock_torch}):
            limits = ResourceLimits(max_gpu_memory_gb=8.0)
            manager = ResourceManager(limits)

            # Mock import to return our mock
            with patch("builtins.__import__", return_value=mock_torch):
                device = manager.select_device()
                assert device == "cuda"

    def test_register_model(self):
        """Test registering a model."""
        manager = ResourceManager()
        model = MagicMock()

        manager.register_model("depth_model", model)

        assert manager.get_model("depth_model") is model
        assert "depth_model" in manager._model_load_order

    def test_register_model_fifo_eviction(self):
        """Test FIFO eviction when max models exceeded."""
        limits = ResourceLimits(max_models_loaded=2)
        manager = ResourceManager(limits)

        model1 = MagicMock()
        model2 = MagicMock()
        model3 = MagicMock()

        manager.register_model("model1", model1)
        manager.register_model("model2", model2)
        manager.register_model("model3", model3)

        # model1 should be evicted
        assert manager.get_model("model1") is None
        assert manager.get_model("model2") is model2
        assert manager.get_model("model3") is model3

    def test_unload_model(self):
        """Test unloading a model."""
        manager = ResourceManager()
        model = MagicMock()
        manager.register_model("test_model", model)

        manager.unload_model("test_model")

        assert manager.get_model("test_model") is None
        assert "test_model" not in manager._model_load_order

    def test_unload_nonexistent_model(self):
        """Test unloading non-existent model is safe."""
        manager = ResourceManager()
        # Should not raise
        manager.unload_model("nonexistent")

    def test_cleanup(self):
        """Test cleanup removes all models."""
        manager = ResourceManager()
        manager.register_model("model1", MagicMock())
        manager.register_model("model2", MagicMock())

        manager.cleanup()

        assert len(manager._loaded_models) == 0

    def test_get_memory_usage_no_torch(self):
        """Test memory usage returns 0 when torch unavailable."""
        manager = ResourceManager()

        with patch.dict("sys.modules", {"torch": None}):
            usage = manager.get_memory_usage_mb()
            assert usage == 0.0

    def test_repr(self):
        """Test string representation."""
        manager = ResourceManager()
        manager.register_model("model1", MagicMock())

        repr_str = repr(manager)
        assert "ResourceManager" in repr_str
        assert "models=1" in repr_str


class TestErrorRecoveryStrategy:
    """Test ErrorRecoveryStrategy enum."""

    def test_strategies_exist(self):
        """Test all strategies are defined."""
        assert ErrorRecoveryStrategy.FAIL_FAST.value == "fail_fast"
        assert ErrorRecoveryStrategy.RETRY.value == "retry"
        assert ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK.value == "retry_cpu_fallback"
        assert ErrorRecoveryStrategy.SKIP_STAGE.value == "skip_stage"
        assert ErrorRecoveryStrategy.RETURN_PARTIAL.value == "return_partial"


class TestPipelineError:
    """Test PipelineError exception."""

    def test_pipeline_error_creation(self):
        """Test creating pipeline error."""
        error = PipelineError(
            stage="ingest",
            message="Failed to load image",
            original_error=ValueError("bad format"),
            context={"file": "test.jpg"},
        )

        assert error.stage == "ingest"
        assert error.message == "Failed to load image"
        assert isinstance(error.original_error, ValueError)
        assert error.context["file"] == "test.jpg"

    def test_pipeline_error_str(self):
        """Test pipeline error string representation."""
        error = PipelineError(stage="segment", message="Memory error")
        assert "[segment]" in str(error)
        assert "Memory error" in str(error)

    def test_pipeline_error_repr(self):
        """Test pipeline error repr."""
        error = PipelineError(stage="segment", message="Memory error")
        repr_str = repr(error)
        assert "PipelineError" in repr_str
        assert "segment" in repr_str


class TestErrorContext:
    """Test ErrorContext dataclass."""

    def test_error_context_creation(self):
        """Test creating error context."""
        context = ErrorContext(
            stage="ingest",
            attempt=2,
            device="cuda",
            memory_mb=1024.0,
        )

        assert context.stage == "ingest"
        assert context.attempt == 2
        assert context.device == "cuda"
        assert context.memory_mb == 1024.0
        assert context.timestamp > 0


class TestErrorHandler:
    """Test ErrorHandler class."""

    def test_handler_initialization(self):
        """Test error handler initialization."""
        handler = ErrorHandler(max_retries=5, backoff_factor=3.0)

        assert handler.max_retries == 5
        assert handler.backoff_factor == 3.0

    def test_fail_fast_strategy(self):
        """Test fail-fast strategy raises immediately."""
        handler = ErrorHandler()

        def failing_func():
            raise ValueError("test error")

        with pytest.raises(PipelineError) as exc_info:
            handler.execute_with_retry(
                func=failing_func,
                stage="test_stage",
                strategy=ErrorRecoveryStrategy.FAIL_FAST,
            )

        assert exc_info.value.stage == "test_stage"

    def test_retry_strategy_success(self):
        """Test retry strategy succeeds on first try."""
        handler = ErrorHandler()

        def success_func():
            return "success"

        result = handler.execute_with_retry(
            func=success_func,
            stage="test_stage",
            strategy=ErrorRecoveryStrategy.RETRY,
        )

        assert result == "success"

    def test_retry_strategy_eventual_success(self):
        """Test retry strategy succeeds after failures."""
        handler = ErrorHandler(max_retries=3, initial_delay=0.01)
        call_count = 0

        def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary failure")
            return "success"

        result = handler.execute_with_retry(
            func=eventually_succeeds,
            stage="test_stage",
            strategy=ErrorRecoveryStrategy.RETRY,
        )

        assert result == "success"
        assert call_count == 3

    def test_retry_strategy_max_retries_exhausted(self):
        """Test retry strategy raises after max retries."""
        handler = ErrorHandler(max_retries=2, initial_delay=0.01)

        def always_fails():
            raise ValueError("always fails")

        with pytest.raises(PipelineError) as exc_info:
            handler.execute_with_retry(
                func=always_fails,
                stage="test_stage",
                strategy=ErrorRecoveryStrategy.RETRY,
            )

        assert "Failed after 2 attempts" in exc_info.value.message

    def test_skip_stage_strategy(self):
        """Test skip stage strategy returns None on error."""
        handler = ErrorHandler()

        def failing_func():
            raise ValueError("test error")

        result = handler.execute_with_retry(
            func=failing_func,
            stage="test_stage",
            strategy=ErrorRecoveryStrategy.SKIP_STAGE,
        )

        assert result is None

    def test_cpu_fallback_on_oom(self):
        """Test CPU fallback on OOM error - verifies fallback logic is triggered.

        Note: The actual error handler passes 'device' as a named parameter
        that is managed internally, not through kwargs. We test that OOM
        detection works and triggers the fallback path.
        """
        handler = ErrorHandler(max_retries=3, initial_delay=0.01)
        oom_count = [0]

        def eventually_succeeds_after_oom(**kwargs):
            oom_count[0] += 1
            if oom_count[0] <= 1:
                raise RuntimeError("CUDA out of memory")
            return "success after fallback"

        # The error handler detects OOM and switches device internally
        # After first OOM, it logs the fallback but retries
        result = handler.execute_with_retry(
            func=eventually_succeeds_after_oom,
            stage="test_stage",
            strategy=ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK,
            device="cuda",
        )

        assert result == "success after fallback"
        # Verify OOM was detected (function called at least twice)
        assert oom_count[0] >= 2

    def test_cpu_fallback_callback(self):
        """Test on_device_change callback is invoked on OOM detection."""
        handler = ErrorHandler(max_retries=3, initial_delay=0.01)
        callback_calls = []
        attempt_count = [0]

        def on_change(new_device, attempt, exc):
            callback_calls.append((new_device, attempt, type(exc).__name__))

        def oom_then_success(**kwargs):
            attempt_count[0] += 1
            if attempt_count[0] <= 1:
                raise RuntimeError("CUDA out of memory")
            return "success"

        handler.execute_with_retry(
            func=oom_then_success,
            stage="test_stage",
            strategy=ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK,
            device="cuda",
            on_device_change=on_change,
        )

        # Verify callback was invoked for the OOM event
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "cpu"
        assert callback_calls[0][2] == "RuntimeError"

    def test_is_gpu_oom(self):
        """Test GPU OOM detection."""
        handler = ErrorHandler()

        oom_errors = [
            RuntimeError("CUDA out of memory"),
            MemoryError("out of memory"),
            RuntimeError("cudnn error: allocation failed"),
        ]

        for error in oom_errors:
            assert handler._is_gpu_oom(error) is True

        non_oom_errors = [
            ValueError("invalid input"),
            RuntimeError("file not found"),
        ]

        for error in non_oom_errors:
            assert handler._is_gpu_oom(error) is False

    def test_get_error_summary(self):
        """Test getting error summary."""
        handler = ErrorHandler(max_retries=2, initial_delay=0.01)

        def failing_func():
            raise ValueError("test error")

        try:
            handler.execute_with_retry(
                func=failing_func,
                stage="test_stage",
                strategy=ErrorRecoveryStrategy.RETRY,
            )
        except PipelineError:
            pass

        summary = handler.get_error_summary()

        assert summary["total_errors"] == 2
        assert "test_stage" in summary["errors_by_stage"]
        assert len(summary["history"]) == 2

    def test_clear_history(self):
        """Test clearing error history."""
        handler = ErrorHandler()

        try:
            handler.execute_with_retry(
                func=lambda: 1 / 0,
                stage="test",
                strategy=ErrorRecoveryStrategy.FAIL_FAST,
            )
        except PipelineError:
            pass

        assert len(handler._error_history) > 0

        handler.clear_history()

        assert len(handler._error_history) == 0

    def test_unsupported_strategy_raises(self):
        """Test unsupported strategy raises ValueError."""
        handler = ErrorHandler()

        with pytest.raises(ValueError, match="Unsupported strategy"):
            handler.execute_with_retry(
                func=lambda: None,
                stage="test",
                strategy="invalid_strategy",
            )
