"""Error handling and recovery for pipeline execution (Phase 2.4).

Provides:
- Graceful degradation strategies
- Retry logic with exponential backoff
- Error context and diagnostics
- Partial result preservation

Architecture:
- Strategy pattern for error recovery
- Structured error types with context
- Automatic retry for transient failures
- CPU fallback on GPU OOM

Example:
    >>> handler = ErrorHandler(max_retries=3)
    >>> try:
    ...     result = handler.execute_with_retry(
    ...         func=process_stage,
    ...         stage="segment",
    ...         strategy=ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK
    ...     )
    ... except PipelineError as e:
    ...     print(f"Stage failed: {e.stage} - {e.message}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ErrorRecoveryStrategy(Enum):
    """Strategy for recovering from errors.

    - FAIL_FAST: Stop immediately on error
    - RETRY: Retry with exponential backoff
    - RETRY_WITH_CPU_FALLBACK: Retry, then try CPU if GPU OOM
    - SKIP_STAGE: Skip failed stage and continue
    - RETURN_PARTIAL: Return partial results up to failure point
    """

    FAIL_FAST = "fail_fast"
    RETRY = "retry"
    RETRY_WITH_CPU_FALLBACK = "retry_cpu_fallback"
    SKIP_STAGE = "skip_stage"
    RETURN_PARTIAL = "return_partial"


class PipelineError(Exception):
    """Base exception for pipeline errors.

    Attributes:
        stage: Pipeline stage where error occurred.
        message: Error message.
        original_error: Original exception that triggered this error.
        context: Additional context (device, memory, etc.).
    """

    def __init__(
        self,
        stage: str,
        message: str,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.stage = stage
        self.message = message
        self.original_error = original_error
        self.context = context or {}
        super().__init__(f"[{stage}] {message}")

    def __repr__(self) -> str:
        """String representation."""
        return f"PipelineError(stage='{self.stage}', message='{self.message}')"


@dataclass
class ErrorContext:
    """Context for an error occurrence.

    Attributes:
        stage: Stage name.
        attempt: Retry attempt number (1-indexed).
        device: Device where error occurred.
        memory_mb: Memory usage at time of error.
        timestamp: Error timestamp.
        traceback: Exception traceback.
    """

    stage: str
    attempt: int
    device: str
    memory_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)
    traceback: Optional[str] = None


class ErrorHandler:
    """Handler for pipeline errors with retry and recovery logic.

    Supports multiple recovery strategies:
    - Exponential backoff retry
    - CPU fallback on GPU OOM
    - Stage skipping
    - Partial result preservation

    Example:
        >>> handler = ErrorHandler(max_retries=3, backoff_factor=2.0)
        >>> result = handler.execute_with_retry(
        ...     func=lambda: process_image(),
        ...     stage="ingest",
        ...     strategy=ErrorRecoveryStrategy.RETRY
        ... )
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        """Initialize error handler.

        Args:
            max_retries: Maximum retry attempts per stage.
            backoff_factor: Exponential backoff multiplier.
            initial_delay: Initial retry delay in seconds.
            max_delay: Maximum retry delay in seconds.
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay

        self._error_history: List[ErrorContext] = []

    def execute_with_retry(
        self,
        func: Callable[..., Any],
        stage: str,
        strategy: ErrorRecoveryStrategy = ErrorRecoveryStrategy.RETRY,
        device: str = "cuda",
        on_device_change: Optional[Callable[[str, int, Exception], None]] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute function with retry logic.

        Args:
            func: Function to execute.
            stage: Stage name for error context.
            strategy: Recovery strategy to use.
            device: Device for execution.
            **kwargs: Arguments to pass to func.

        Returns:
            Function result.

        Raises:
            PipelineError: If all retries exhausted or strategy is FAIL_FAST.
        """
        if strategy == ErrorRecoveryStrategy.FAIL_FAST:
            return self._execute_fail_fast(func, stage, device, **kwargs)

        elif strategy == ErrorRecoveryStrategy.RETRY:
            return self._execute_with_retry(
                func,
                stage,
                device,
                allow_cpu_fallback=False,
                on_device_change=on_device_change,
                **kwargs,
            )

        elif strategy == ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK:
            return self._execute_with_retry(
                func,
                stage,
                device,
                allow_cpu_fallback=True,
                on_device_change=on_device_change,
                **kwargs,
            )

        elif strategy == ErrorRecoveryStrategy.SKIP_STAGE:
            return self._execute_skip_on_error(func, stage, device, **kwargs)

        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    def _execute_fail_fast(self, func: Callable[..., Any], stage: str, device: str, **kwargs: Any) -> Any:
        """Execute with fail-fast strategy (no retry).

        Args:
            func: Function to execute.
            stage: Stage name.
            device: Device.
            **kwargs: Function arguments.

        Returns:
            Function result.

        Raises:
            PipelineError: On any error.
        """
        try:
            return func(**kwargs)
        except Exception as e:
            self._record_error(stage, 1, device, e)
            raise PipelineError(stage=stage, message=str(e), original_error=e) from e

    def _execute_with_retry(
        self,
        func: Callable[..., Any],
        stage: str,
        device: str,
        allow_cpu_fallback: bool,
        on_device_change: Optional[Callable[[str, int, Exception], None]] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute with retry and optional CPU fallback.

        Args:
            func: Function to execute.
            stage: Stage name.
            device: Device.
            allow_cpu_fallback: Allow CPU fallback on GPU OOM.
            **kwargs: Function arguments.

        Returns:
            Function result.

        Raises:
            PipelineError: If all retries exhausted.
        """
        last_error: Optional[Exception] = None
        delay = self.initial_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                # Update device in kwargs if it's a parameter
                if "device" in kwargs:
                    kwargs["device"] = device

                return func(**kwargs)

            except Exception as e:
                last_error = e
                self._record_error(stage, attempt, device, e)

                # Check if GPU OOM and CPU fallback allowed
                is_oom = self._is_gpu_oom(e)
                if is_oom and allow_cpu_fallback and device != "cpu":
                    logger.warning("GPU OOM detected in stage '%s'; rebuilding execution on CPU", stage)
                    device = "cpu"
                    if on_device_change is not None:
                        on_device_change(device, attempt, e)
                    continue

                # If this is the last attempt, raise
                if attempt >= self.max_retries:
                    break

                # Exponential backoff
                logger.warning(
                    "Stage '%s' failed (attempt %d/%d), retrying in %.1fs: %s",
                    stage,
                    attempt,
                    self.max_retries,
                    delay,
                    e,
                )

                time.sleep(delay)
                delay = min(delay * self.backoff_factor, self.max_delay)

        # All retries exhausted
        raise PipelineError(
            stage=stage,
            message=f"Failed after {self.max_retries} attempts",
            original_error=last_error,
            context={"attempts": self.max_retries, "device": device},
        ) from last_error

    def _execute_skip_on_error(self, func: Callable[..., Any], stage: str, device: str, **kwargs: Any) -> Optional[Any]:
        """Execute and skip stage on error (return None).

        Args:
            func: Function to execute.
            stage: Stage name.
            device: Device.
            **kwargs: Function arguments.

        Returns:
            Function result or None if error.
        """
        try:
            return func(**kwargs)
        except Exception as e:
            self._record_error(stage, 1, device, e)
            logger.warning(f"Stage '{stage}' failed, skipping: {e}")
            return None

    def _is_gpu_oom(self, error: Exception) -> bool:
        """Check if error is GPU out-of-memory.

        Args:
            error: Exception to check.

        Returns:
            True if OOM error.
        """
        error_str = str(error).lower()
        oom_indicators = [
            "out of memory",
            "oom",
            "cuda out of memory",
            "cudnn error",
            "allocation failed",
        ]
        return any(indicator in error_str for indicator in oom_indicators)

    def _record_error(self, stage: str, attempt: int, device: str, error: Exception) -> None:
        """Record error in history.

        Args:
            stage: Stage name.
            attempt: Attempt number.
            device: Device.
            error: Exception.
        """
        import traceback

        context = ErrorContext(
            stage=stage,
            attempt=attempt,
            device=device,
            traceback=traceback.format_exc(),
        )
        self._error_history.append(context)
        logger.debug(f"Recorded error: {stage} attempt {attempt} on {device}")

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered.

        Returns:
            Dict with error statistics and history.
        """
        stage_errors = {}
        for ctx in self._error_history:
            if ctx.stage not in stage_errors:
                stage_errors[ctx.stage] = 0
            stage_errors[ctx.stage] += 1

        return {
            "total_errors": len(self._error_history),
            "errors_by_stage": stage_errors,
            "history": [
                {
                    "stage": ctx.stage,
                    "attempt": ctx.attempt,
                    "device": ctx.device,
                    "timestamp": ctx.timestamp,
                }
                for ctx in self._error_history
            ],
        }

    def clear_history(self) -> None:
        """Clear error history."""
        self._error_history.clear()
