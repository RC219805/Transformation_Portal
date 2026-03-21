"""Unit tests for error handler (Phase 2.4).

Tests for ErrorHandler, ErrorRecoveryStrategy, PipelineError, and ErrorContext to achieve ≥85% coverage.
Covers all recovery strategies, retry logic, GPU OOM detection, error recording, and error summaries.
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

pytestmark = pytest.mark.unit


class TestErrorRecoveryStrategy:
    """Test ErrorRecoveryStrategy enum."""

    def test_all_strategies_defined(self):
        """Test all expected strategies are defined."""
        expected = {
            "FAIL_FAST",
            "RETRY",
            "RETRY_WITH_CPU_FALLBACK",
            "SKIP_STAGE",
            "RETURN_PARTIAL",
        }
        actual = {s.name for s in ErrorRecoveryStrategy}
        assert actual == expected

    def test_strategy_values(self):
        """Test strategy string values."""
        assert ErrorRecoveryStrategy.FAIL_FAST.value == "fail_fast"
        assert ErrorRecoveryStrategy.RETRY.value == "retry"
        assert ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK.value == "retry_cpu_fallback"
        assert ErrorRecoveryStrategy.SKIP_STAGE.value == "skip_stage"
        assert ErrorRecoveryStrategy.RETURN_PARTIAL.value == "return_partial"


class TestPipelineError:
    """Test PipelineError exception."""

    def test_minimal_error_creation(self):
        """Test creating error with minimal args."""
        error = PipelineError(
            stage="ingest",
            message="Failed to load image",
        )
        assert error.stage == "ingest"
        assert error.message == "Failed to load image"
        assert error.original_error is None
        assert error.context == {}
        assert str(error) == "[ingest] Failed to load image"

    def test_error_with_original_exception(self):
        """Test error wrapping original exception."""
        original = ValueError("Invalid format")
        error = PipelineError(
            stage="segment",
            message="Segmentation failed",
            original_error=original,
        )
        assert error.original_error is original
        assert isinstance(error.original_error, ValueError)

    def test_error_with_context(self):
        """Test error with additional context."""
        context = {
            "device": "cuda",
            "memory_mb": 8192.0,
            "attempt": 3,
        }
        error = PipelineError(
            stage="materials",
            message="GPU OOM",
            context=context,
        )
        assert error.context == context
        assert error.context["device"] == "cuda"
        assert error.context["memory_mb"] == 8192.0

    def test_error_repr(self):
        """Test error string representation."""
        error = PipelineError(stage="reconstruct", message="Test error")
        repr_str = repr(error)
        assert "PipelineError" in repr_str
        assert "reconstruct" in repr_str
        assert "Test error" in repr_str

    def test_error_is_exception(self):
        """Test PipelineError is an Exception subclass."""
        error = PipelineError(stage="test", message="test")
        assert isinstance(error, Exception)


class TestErrorContext:
    """Test ErrorContext dataclass."""

    def test_error_context_creation(self):
        """Test ErrorContext creation."""
        ctx = ErrorContext(
            stage="ingest",
            attempt=1,
            device="cuda",
            memory_mb=4096.0,
            traceback="Traceback...",
        )
        assert ctx.stage == "ingest"
        assert ctx.attempt == 1
        assert ctx.device == "cuda"
        assert ctx.memory_mb == 4096.0
        assert ctx.traceback == "Traceback..."
        assert isinstance(ctx.timestamp, float)

    def test_error_context_default_timestamp(self):
        """Test ErrorContext has default timestamp."""
        ctx = ErrorContext(stage="test", attempt=1, device="cpu")
        assert ctx.timestamp > 0
        assert abs(ctx.timestamp - time.time()) < 1.0

    def test_error_context_default_memory(self):
        """Test ErrorContext default memory value."""
        ctx = ErrorContext(stage="test", attempt=1, device="cpu")
        assert ctx.memory_mb == 0.0

    def test_error_context_default_traceback(self):
        """Test ErrorContext default traceback value."""
        ctx = ErrorContext(stage="test", attempt=1, device="cpu")
        assert ctx.traceback is None


class TestErrorHandlerInitialization:
    """Test ErrorHandler initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        handler = ErrorHandler()
        assert handler.max_retries == 3
        assert handler.backoff_factor == 2.0
        assert handler.initial_delay == 1.0
        assert handler.max_delay == 60.0
        assert handler._error_history == []

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        handler = ErrorHandler(
            max_retries=5,
            backoff_factor=1.5,
            initial_delay=0.5,
            max_delay=30.0,
        )
        assert handler.max_retries == 5
        assert handler.backoff_factor == 1.5
        assert handler.initial_delay == 0.5
        assert handler.max_delay == 30.0


class TestErrorHandlerFailFast:
    """Test FAIL_FAST strategy."""

    def test_fail_fast_success(self):
        """Test FAIL_FAST with successful execution."""
        handler = ErrorHandler()

        def successful_func():
            return "success"

        result = handler.execute_with_retry(
            func=successful_func,
            stage="test",
            strategy=ErrorRecoveryStrategy.FAIL_FAST,
        )
        assert result == "success"

    def test_fail_fast_failure(self):
        """Test FAIL_FAST raises immediately on error."""
        handler = ErrorHandler()

        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(PipelineError) as exc_info:
            handler.execute_with_retry(
                func=failing_func,
                stage="test_stage",
                strategy=ErrorRecoveryStrategy.FAIL_FAST,
            )

        error = exc_info.value
        assert error.stage == "test_stage"
        assert "Test error" in error.message
        assert isinstance(error.original_error, ValueError)

    def test_fail_fast_records_error(self):
        """Test FAIL_FAST records error in history."""
        handler = ErrorHandler()

        def failing_func():
            raise RuntimeError("Fail")

        try:
            handler.execute_with_retry(
                func=failing_func,
                stage="test",
                strategy=ErrorRecoveryStrategy.FAIL_FAST,
                device="cuda",
            )
        except PipelineError:
            pass

        assert len(handler._error_history) == 1
        ctx = handler._error_history[0]
        assert ctx.stage == "test"
        assert ctx.attempt == 1
        assert ctx.device == "cuda"


class TestErrorHandlerRetry:
    """Test RETRY strategy."""

    def test_retry_eventual_success(self):
        """Test RETRY succeeds after initial failures."""
        handler = ErrorHandler(max_retries=3, initial_delay=0.01)
        attempts = {"count": 0}

        def flaky_func():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("Temporary failure")
            return "success"

        with patch("time.sleep"):  # Speed up test
            result = handler.execute_with_retry(
                func=flaky_func,
                stage="test",
                strategy=ErrorRecoveryStrategy.RETRY,
            )

        assert result == "success"
        assert attempts["count"] == 3

    def test_retry_exhaustion(self):
        """Test RETRY raises after max retries."""
        handler = ErrorHandler(max_retries=3, initial_delay=0.01)

        def always_fails():
            raise RuntimeError("Permanent failure")

        with patch("time.sleep"):
            with pytest.raises(PipelineError) as exc_info:
                handler.execute_with_retry(
                    func=always_fails,
                    stage="test",
                    strategy=ErrorRecoveryStrategy.RETRY,
                )

        error = exc_info.value
        assert "Failed after 3 attempts" in error.message
        assert error.context["attempts"] == 3

    def test_retry_exponential_backoff(self):
        """Test RETRY uses exponential backoff."""
        handler = ErrorHandler(
            max_retries=3,
            initial_delay=1.0,
            backoff_factor=2.0,
        )

        def always_fails():
            raise RuntimeError("Fail")

        sleep_calls = []

        def mock_sleep(duration):
            sleep_calls.append(duration)

        with patch("time.sleep", side_effect=mock_sleep):
            try:
                handler.execute_with_retry(
                    func=always_fails,
                    stage="test",
                    strategy=ErrorRecoveryStrategy.RETRY,
                )
            except PipelineError:
                pass

        # Should have 2 sleep calls (not sleeping after last attempt)
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == 1.0  # initial_delay
        assert sleep_calls[1] == 2.0  # initial_delay * backoff_factor

    def test_retry_respects_max_delay(self):
        """Test RETRY caps delay at max_delay."""
        handler = ErrorHandler(
            max_retries=5,
            initial_delay=10.0,
            backoff_factor=3.0,
            max_delay=20.0,
        )

        def always_fails():
            raise RuntimeError("Fail")

        sleep_calls = []

        def mock_sleep(duration):
            sleep_calls.append(duration)

        with patch("time.sleep", side_effect=mock_sleep):
            try:
                handler.execute_with_retry(
                    func=always_fails,
                    stage="test",
                    strategy=ErrorRecoveryStrategy.RETRY,
                )
            except PipelineError:
                pass

        # All delays should be capped at max_delay
        for delay in sleep_calls:
            assert delay <= 20.0

    def test_retry_records_all_attempts(self):
        """Test RETRY records all error attempts."""
        handler = ErrorHandler(max_retries=3, initial_delay=0.01)

        def always_fails():
            raise RuntimeError("Fail")

        with patch("time.sleep"):
            try:
                handler.execute_with_retry(
                    func=always_fails,
                    stage="test",
                    strategy=ErrorRecoveryStrategy.RETRY,
                    device="cuda",
                )
            except PipelineError:
                pass

        # Should have 3 error records (one per attempt)
        assert len(handler._error_history) == 3
        for i, ctx in enumerate(handler._error_history, 1):
            assert ctx.attempt == i
            assert ctx.stage == "test"
            assert ctx.device == "cuda"


class TestErrorHandlerCPUFallback:
    """Test RETRY_WITH_CPU_FALLBACK strategy.

    Note: The current implementation has a limitation - CPU fallback updates
    kwargs["device"] only if "device" is already in kwargs. This means functions
    that use device as a default parameter won't see the fallback.

    The tests document this behavior. For production use, wrapped functions
    should extract device from kwargs explicitly if CPU fallback is needed.
    """

    def test_cpu_fallback_detects_oom_and_switches_device_tracker(self):
        """Test CPU fallback detects OOM and switches internal device tracking."""
        handler = ErrorHandler(max_retries=3, initial_delay=0.01)

        call_count = {"count": 0}

        def always_oom(**kwargs):
            call_count["count"] += 1
            raise RuntimeError("CUDA out of memory")

        with patch("time.sleep"):
            with pytest.raises(PipelineError) as exc_info:
                handler.execute_with_retry(
                    func=always_oom,
                    stage="test",
                    strategy=ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK,
                    device="cuda",
                    some_arg="value",  # Other kwargs pass through
                )

        # Should have attempted multiple times
        assert call_count["count"] == 3
        # Error context should reflect device="cpu" after fallback
        assert exc_info.value.context["device"] == "cpu"

    def test_cpu_fallback_oom_detection(self):
        """Test OOM detection with various error messages."""
        handler = ErrorHandler()

        oom_messages = [
            "CUDA out of memory",
            "Out of Memory",
            "allocation failed",
            "cudnn error: OOM",
        ]

        for msg in oom_messages:
            error = RuntimeError(msg)
            assert handler._is_gpu_oom(error) is True

    def test_cpu_fallback_non_oom_error(self):
        """Test non-OOM errors don't trigger device switch."""
        handler = ErrorHandler(max_retries=2, initial_delay=0.01)

        call_count = {"count": 0}

        def non_oom_error(**kwargs):
            call_count["count"] += 1
            raise RuntimeError("Invalid input")

        with patch("time.sleep"):
            with pytest.raises(PipelineError) as exc_info:
                handler.execute_with_retry(
                    func=non_oom_error,
                    stage="test",
                    strategy=ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK,
                    device="cuda",
                )

        # Device should remain cuda (no fallback)
        assert exc_info.value.context["device"] == "cuda"

    def test_cpu_fallback_already_on_cpu(self):
        """Test no device change when already on CPU."""
        handler = ErrorHandler(max_retries=2, initial_delay=0.01)

        def oom_on_cpu(**kwargs):
            raise RuntimeError("Out of memory")

        with patch("time.sleep"):
            with pytest.raises(PipelineError) as exc_info:
                handler.execute_with_retry(
                    func=oom_on_cpu,
                    stage="test",
                    strategy=ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK,
                    device="cpu",
                )

        # Should stay on CPU
        assert exc_info.value.context["device"] == "cpu"

    def test_cpu_fallback_with_kwargs_device(self):
        """Test CPU fallback behavior with device parameter.

        This documents a current limitation: the fallback only updates
        kwargs["device"] if it exists, but doesn't help with default parameters.
        """
        handler = ErrorHandler(max_retries=3, initial_delay=0.01)

        def always_oom(**kwargs):
            raise RuntimeError("CUDA out of memory")

        with patch("time.sleep"):
            with pytest.raises(PipelineError) as exc_info:
                handler.execute_with_retry(
                    func=always_oom,
                    stage="test",
                    strategy=ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK,
                    device="cuda",
                )

        # Device tracking should have switched to CPU
        assert exc_info.value.context["device"] == "cpu"
        # Error history should show OOM was detected
        assert len(handler._error_history) >= 1


class TestErrorHandlerSkipStage:
    """Test SKIP_STAGE strategy."""

    def test_skip_stage_success(self):
        """Test SKIP_STAGE returns result on success."""
        handler = ErrorHandler()

        def successful_func():
            return "success"

        result = handler.execute_with_retry(
            func=successful_func,
            stage="test",
            strategy=ErrorRecoveryStrategy.SKIP_STAGE,
        )
        assert result == "success"

    def test_skip_stage_failure_returns_none(self):
        """Test SKIP_STAGE returns None on error."""
        handler = ErrorHandler()

        def failing_func():
            raise RuntimeError("Error")

        result = handler.execute_with_retry(
            func=failing_func,
            stage="test",
            strategy=ErrorRecoveryStrategy.SKIP_STAGE,
        )
        assert result is None

    def test_skip_stage_logs_warning(self):
        """Test SKIP_STAGE logs warning on failure."""
        handler = ErrorHandler()

        def failing_func():
            raise RuntimeError("Test error")

        with patch("transformation_portal.spatial_ai.orchestration.error_handler.logger") as mock_logger:
            result = handler.execute_with_retry(
                func=failing_func,
                stage="test_stage",
                strategy=ErrorRecoveryStrategy.SKIP_STAGE,
            )

        assert result is None
        mock_logger.warning.assert_called_once()

    def test_skip_stage_records_error(self):
        """Test SKIP_STAGE records error in history."""
        handler = ErrorHandler()

        def failing_func():
            raise RuntimeError("Error")

        handler.execute_with_retry(
            func=failing_func,
            stage="test",
            strategy=ErrorRecoveryStrategy.SKIP_STAGE,
            device="cuda",
        )

        assert len(handler._error_history) == 1
        ctx = handler._error_history[0]
        assert ctx.stage == "test"
        assert ctx.device == "cuda"


class TestErrorHandlerUnsupportedStrategy:
    """Test handling of unsupported strategies."""

    def test_unsupported_strategy_raises(self):
        """Test unsupported strategy raises ValueError."""
        handler = ErrorHandler()

        # Mock an invalid strategy by using RETURN_PARTIAL
        # (not implemented in execute_with_retry)
        with pytest.raises(ValueError, match="Unsupported strategy"):
            handler.execute_with_retry(
                func=lambda: "test",
                stage="test",
                strategy=ErrorRecoveryStrategy.RETURN_PARTIAL,
            )


class TestErrorHandlerOOMDetection:
    """Test GPU OOM detection logic."""

    def test_is_gpu_oom_cuda_messages(self):
        """Test OOM detection for CUDA messages."""
        handler = ErrorHandler()

        cuda_oom_errors = [
            RuntimeError("CUDA out of memory. Tried to allocate..."),
            RuntimeError("cuda out of memory"),
            RuntimeError("CUDA OOM"),
        ]

        for error in cuda_oom_errors:
            assert handler._is_gpu_oom(error) is True

    def test_is_gpu_oom_generic_messages(self):
        """Test OOM detection for generic OOM messages."""
        handler = ErrorHandler()

        oom_errors = [
            RuntimeError("Out of memory"),
            RuntimeError("OOM at line 42"),
            RuntimeError("allocation failed"),
            RuntimeError("cudnn error occurred"),
        ]

        for error in oom_errors:
            assert handler._is_gpu_oom(error) is True

    def test_is_gpu_oom_case_insensitive(self):
        """Test OOM detection is case-insensitive."""
        handler = ErrorHandler()

        errors = [
            RuntimeError("OUT OF MEMORY"),
            RuntimeError("Oom"),
            RuntimeError("ALLOCATION FAILED"),
        ]

        for error in errors:
            assert handler._is_gpu_oom(error) is True

    def test_is_gpu_oom_non_oom_errors(self):
        """Test non-OOM errors return False."""
        handler = ErrorHandler()

        non_oom_errors = [
            RuntimeError("Invalid argument"),
            RuntimeError("File not found"),
            ValueError("Dimension mismatch"),
        ]

        for error in non_oom_errors:
            assert handler._is_gpu_oom(error) is False


class TestErrorHandlerErrorRecording:
    """Test error recording functionality."""

    def test_record_error_captures_traceback(self):
        """Test _record_error captures traceback."""
        handler = ErrorHandler()

        try:
            raise ValueError("Test error")
        except ValueError as e:
            handler._record_error("test_stage", 1, "cuda", e)

        assert len(handler._error_history) == 1
        ctx = handler._error_history[0]
        assert ctx.traceback is not None
        assert "ValueError" in ctx.traceback
        assert "Test error" in ctx.traceback

    def test_record_error_metadata(self):
        """Test _record_error stores correct metadata."""
        handler = ErrorHandler()

        error = RuntimeError("Test")
        handler._record_error("segment", 2, "mps", error)

        ctx = handler._error_history[0]
        assert ctx.stage == "segment"
        assert ctx.attempt == 2
        assert ctx.device == "mps"
        assert isinstance(ctx.timestamp, float)

    def test_multiple_error_recording(self):
        """Test recording multiple errors."""
        handler = ErrorHandler()

        handler._record_error("stage1", 1, "cuda", RuntimeError("E1"))
        handler._record_error("stage1", 2, "cuda", RuntimeError("E2"))
        handler._record_error("stage2", 1, "cpu", RuntimeError("E3"))

        assert len(handler._error_history) == 3
        assert handler._error_history[0].stage == "stage1"
        assert handler._error_history[1].stage == "stage1"
        assert handler._error_history[2].stage == "stage2"


class TestErrorHandlerErrorSummary:
    """Test error summary generation."""

    def test_get_error_summary_empty(self):
        """Test summary with no errors."""
        handler = ErrorHandler()
        summary = handler.get_error_summary()

        assert summary["total_errors"] == 0
        assert summary["errors_by_stage"] == {}
        assert summary["history"] == []

    def test_get_error_summary_single_error(self):
        """Test summary with single error."""
        handler = ErrorHandler()
        handler._record_error("ingest", 1, "cpu", RuntimeError("Error"))

        summary = handler.get_error_summary()

        assert summary["total_errors"] == 1
        assert summary["errors_by_stage"] == {"ingest": 1}
        assert len(summary["history"]) == 1
        assert summary["history"][0]["stage"] == "ingest"
        assert summary["history"][0]["attempt"] == 1
        assert summary["history"][0]["device"] == "cpu"

    def test_get_error_summary_multiple_stages(self):
        """Test summary with errors across multiple stages."""
        handler = ErrorHandler()

        # 2 errors in stage1, 3 in stage2
        handler._record_error("stage1", 1, "cuda", RuntimeError("E1"))
        handler._record_error("stage1", 2, "cuda", RuntimeError("E2"))
        handler._record_error("stage2", 1, "cuda", RuntimeError("E3"))
        handler._record_error("stage2", 2, "cpu", RuntimeError("E4"))
        handler._record_error("stage2", 3, "cpu", RuntimeError("E5"))

        summary = handler.get_error_summary()

        assert summary["total_errors"] == 5
        assert summary["errors_by_stage"] == {"stage1": 2, "stage2": 3}
        assert len(summary["history"]) == 5

    def test_get_error_summary_includes_timestamp(self):
        """Test summary includes timestamps."""
        handler = ErrorHandler()
        handler._record_error("test", 1, "cuda", RuntimeError("Error"))

        summary = handler.get_error_summary()

        assert "timestamp" in summary["history"][0]
        assert isinstance(summary["history"][0]["timestamp"], float)


class TestErrorHandlerClearHistory:
    """Test error history clearing."""

    def test_clear_history(self):
        """Test clearing error history."""
        handler = ErrorHandler()

        handler._record_error("stage1", 1, "cuda", RuntimeError("E1"))
        handler._record_error("stage2", 1, "cpu", RuntimeError("E2"))
        assert len(handler._error_history) == 2

        handler.clear_history()

        assert len(handler._error_history) == 0
        summary = handler.get_error_summary()
        assert summary["total_errors"] == 0

    def test_clear_history_idempotent(self):
        """Test clearing empty history is safe."""
        handler = ErrorHandler()
        handler.clear_history()
        handler.clear_history()
        assert len(handler._error_history) == 0


class TestErrorHandlerIntegration:
    """Integration tests for error handler."""

    def test_retry_with_backoff_integration(self):
        """Test full retry flow with exponential backoff."""
        handler = ErrorHandler(
            max_retries=3,
            initial_delay=0.01,
            backoff_factor=2.0,
        )

        attempts = {"count": 0}

        def eventually_succeeds():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("Temporary failure")
            return f"success after {attempts['count']} attempts"

        with patch("time.sleep"):
            result = handler.execute_with_retry(
                func=eventually_succeeds,
                stage="test_stage",
                strategy=ErrorRecoveryStrategy.RETRY,
                device="cuda",
            )

        assert result == "success after 3 attempts"
        assert len(handler._error_history) == 2  # 2 failures before success

        summary = handler.get_error_summary()
        assert summary["total_errors"] == 2
        assert summary["errors_by_stage"] == {"test_stage": 2}

    def test_cpu_fallback_integration(self):
        """Test CPU fallback updates device context tracking."""
        handler = ErrorHandler(max_retries=2, initial_delay=0.01)

        call_count = {"count": 0}

        def oom_on_gpu(**kwargs):
            call_count["count"] += 1
            raise RuntimeError("CUDA out of memory")

        with patch("time.sleep"):
            with pytest.raises(PipelineError) as exc_info:
                handler.execute_with_retry(
                    func=oom_on_gpu,
                    stage="segment",
                    strategy=ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK,
                    device="cuda",
                )

        # Should have detected OOM and switched to CPU tracking
        assert exc_info.value.context["device"] == "cpu"
        # Should have recorded error with original CUDA device
        assert len(handler._error_history) >= 1
        assert handler._error_history[0].device == "cuda"

    def test_mixed_strategies_workflow(self):
        """Test using different strategies for different stages."""
        handler = ErrorHandler(max_retries=2, initial_delay=0.01)

        # Stage 1: FAIL_FAST - succeeds
        result1 = handler.execute_with_retry(
            func=lambda: "stage1_success",
            stage="stage1",
            strategy=ErrorRecoveryStrategy.FAIL_FAST,
        )
        assert result1 == "stage1_success"

        # Stage 2: SKIP_STAGE - fails but skipped
        result2 = handler.execute_with_retry(
            func=lambda: 1 / 0,  # ZeroDivisionError
            stage="stage2",
            strategy=ErrorRecoveryStrategy.SKIP_STAGE,
        )
        assert result2 is None

        # Stage 3: RETRY - eventually succeeds
        attempts = {"count": 0}

        def stage3():
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise RuntimeError("Retry me")
            return "stage3_success"

        with patch("time.sleep"):
            result3 = handler.execute_with_retry(
                func=stage3,
                stage="stage3",
                strategy=ErrorRecoveryStrategy.RETRY,
            )
        assert result3 == "stage3_success"

        # Check summary
        summary = handler.get_error_summary()
        assert summary["total_errors"] == 2  # stage2 + stage3 first attempt
        assert set(summary["errors_by_stage"].keys()) == {"stage2", "stage3"}
