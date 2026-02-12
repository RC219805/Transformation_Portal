"""Progress tracking for long-running pipeline operations (Phase 2.4).

Provides:
- Per-stage progress reporting
- Time estimates based on previous runs
- Logging integration
- Event-based progress updates

Architecture:
- Event-driven design (emit ProgressEvent)
- Stage-level granularity
- Optional time estimation from historical data
- Integration with Python logging

Example:
    >>> tracker = ProgressTracker(total_stages=4)
    >>> tracker.start_stage("ingest", "Linear ingest")
    >>> # ... do work ...
    >>> tracker.complete_stage("ingest", success=True)
    >>> print(f"Progress: {tracker.get_progress_percent():.1f}%")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ProgressEventType(Enum):
    """Type of progress event."""

    PIPELINE_START = "pipeline_start"
    PIPELINE_COMPLETE = "pipeline_complete"
    STAGE_START = "stage_start"
    STAGE_PROGRESS = "stage_progress"
    STAGE_COMPLETE = "stage_complete"
    STAGE_ERROR = "stage_error"


@dataclass
class ProgressEvent:
    """Progress event emitted during pipeline execution.

    Attributes:
        event_type: Type of event.
        stage: Stage name (e.g., "ingest", "segment", "materials").
        message: Human-readable message.
        progress_percent: Overall pipeline progress [0, 100].
        stage_progress_percent: Stage-level progress [0, 100].
        elapsed_time: Elapsed time since pipeline start (seconds).
        estimated_remaining: Estimated time remaining (seconds, if available).
        metadata: Additional event metadata.
    """

    event_type: ProgressEventType
    stage: Optional[str]
    message: str
    progress_percent: float
    stage_progress_percent: float = 0.0
    elapsed_time: float = 0.0
    estimated_remaining: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class StageMetrics:
    """Metrics for a pipeline stage.

    Attributes:
        name: Stage identifier.
        display_name: Human-readable name.
        start_time: Start timestamp (Unix time).
        end_time: End timestamp (Unix time, None if not complete).
        success: Whether stage completed successfully.
        error_message: Error message if stage failed.
    """

    name: str
    display_name: str
    start_time: float
    end_time: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds (None if not complete)."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def is_complete(self) -> bool:
        """Check if stage is complete."""
        return self.end_time is not None


class ProgressTracker:
    """Track progress of multi-stage pipeline execution.

    Emits ProgressEvent objects at key milestones and supports
    optional time estimation based on historical data.

    Example:
        >>> tracker = ProgressTracker(total_stages=3)
        >>> tracker.start_stage("ingest", "Loading image")
        >>> tracker.update_stage("ingest", 50.0)  # 50% done
        >>> tracker.complete_stage("ingest", success=True)
        >>> print(tracker.get_summary())
    """

    def __init__(
        self,
        total_stages: int,
        enable_time_estimation: bool = True,
        historical_times: Optional[Dict[str, float]] = None,
    ):
        """Initialize progress tracker.

        Args:
            total_stages: Total number of stages in pipeline.
            enable_time_estimation: Enable time estimation (requires historical_times).
            historical_times: Historical stage durations (stage_name -> seconds).
                Used for time estimation. If None, no estimates provided.
        """
        self.total_stages = total_stages
        self.enable_time_estimation = enable_time_estimation
        self.historical_times = historical_times or {}

        self._pipeline_start_time: Optional[float] = None
        self._pipeline_end_time: Optional[float] = None
        self._current_stage: Optional[str] = None
        self._stage_metrics: Dict[str, StageMetrics] = {}
        self._completed_stages: int = 0

    def start_pipeline(self) -> None:
        """Mark pipeline start."""
        self._pipeline_start_time = time.time()
        event = ProgressEvent(
            event_type=ProgressEventType.PIPELINE_START,
            stage=None,
            message=f"Starting pipeline with {self.total_stages} stages",
            progress_percent=0.0,
            elapsed_time=0.0,
        )
        logger.info(f"Pipeline started: {self.total_stages} stages")
        self._emit_event(event)

    def start_stage(self, name: str, display_name: str) -> None:
        """Mark stage start.

        Args:
            name: Stage identifier (e.g., "ingest", "segment").
            display_name: Human-readable name (e.g., "Linear Ingest").
        """
        self._current_stage = name
        metrics = StageMetrics(
            name=name,
            display_name=display_name,
            start_time=time.time(),
        )
        self._stage_metrics[name] = metrics

        # Calculate progress
        progress_percent = (self._completed_stages / self.total_stages) * 100.0
        elapsed = self._get_elapsed_time()

        # Estimate remaining time
        estimated_remaining = None
        if self.enable_time_estimation and name in self.historical_times:
            # Sum of remaining stages' historical times
            remaining_stages = self.total_stages - self._completed_stages
            if remaining_stages > 0:
                estimated_remaining = sum(
                    self.historical_times.get(s, 0) for s in self._stage_metrics if s not in self._stage_metrics
                )

        event = ProgressEvent(
            event_type=ProgressEventType.STAGE_START,
            stage=name,
            message=f"Starting stage: {display_name}",
            progress_percent=progress_percent,
            stage_progress_percent=0.0,
            elapsed_time=elapsed,
            estimated_remaining=estimated_remaining,
        )

        logger.info(f"Stage started: {display_name} ({self._completed_stages + 1}/{self.total_stages})")
        self._emit_event(event)

    def update_stage(self, name: str, stage_progress_percent: float) -> None:
        """Update stage-level progress.

        Args:
            name: Stage identifier.
            stage_progress_percent: Stage progress [0, 100].
        """
        if name not in self._stage_metrics:
            logger.warning(f"Cannot update stage '{name}' - not started")
            return

        # Calculate overall progress
        base_progress = (self._completed_stages / self.total_stages) * 100.0
        stage_contribution = (stage_progress_percent / 100.0) * (100.0 / self.total_stages)
        progress_percent = base_progress + stage_contribution

        event = ProgressEvent(
            event_type=ProgressEventType.STAGE_PROGRESS,
            stage=name,
            message=f"Stage progress: {stage_progress_percent:.1f}%",
            progress_percent=progress_percent,
            stage_progress_percent=stage_progress_percent,
            elapsed_time=self._get_elapsed_time(),
        )

        logger.debug(f"Stage progress: {name} - {stage_progress_percent:.1f}%")
        self._emit_event(event)

    def complete_stage(self, name: str, success: bool, error_message: Optional[str] = None) -> None:
        """Mark stage completion.

        Args:
            name: Stage identifier.
            success: Whether stage completed successfully.
            error_message: Error message if failed.
        """
        if name not in self._stage_metrics:
            logger.warning(f"Cannot complete stage '{name}' - not started")
            return

        metrics = self._stage_metrics[name]
        metrics.end_time = time.time()
        metrics.success = success
        metrics.error_message = error_message

        if success:
            self._completed_stages += 1

        progress_percent = (self._completed_stages / self.total_stages) * 100.0

        if success:
            event = ProgressEvent(
                event_type=ProgressEventType.STAGE_COMPLETE,
                stage=name,
                message=f"Stage completed: {metrics.display_name} ({metrics.duration:.1f}s)",
                progress_percent=progress_percent,
                stage_progress_percent=100.0,
                elapsed_time=self._get_elapsed_time(),
                metadata={"duration": metrics.duration},
            )
            logger.info(f"Stage completed: {metrics.display_name} in {metrics.duration:.1f}s")
        else:
            event = ProgressEvent(
                event_type=ProgressEventType.STAGE_ERROR,
                stage=name,
                message=f"Stage failed: {metrics.display_name} - {error_message}",
                progress_percent=progress_percent,
                stage_progress_percent=0.0,
                elapsed_time=self._get_elapsed_time(),
                metadata={"error": error_message},
            )
            logger.error(f"Stage failed: {metrics.display_name} - {error_message}")

        self._emit_event(event)
        self._current_stage = None

    def complete_pipeline(self, success: bool) -> None:
        """Mark pipeline completion.

        Args:
            success: Whether pipeline completed successfully.
        """
        self._pipeline_end_time = time.time()
        elapsed = self._get_elapsed_time()

        event = ProgressEvent(
            event_type=ProgressEventType.PIPELINE_COMPLETE,
            stage=None,
            message=f"Pipeline {'completed' if success else 'failed'} in {elapsed:.1f}s",
            progress_percent=100.0 if success else (self._completed_stages / self.total_stages) * 100.0,
            elapsed_time=elapsed,
            metadata={
                "success": success,
                "completed_stages": self._completed_stages,
                "total_stages": self.total_stages,
            },
        )

        logger.info(
            f"Pipeline {'completed' if success else 'failed'} "
            f"({self._completed_stages}/{self.total_stages} stages in {elapsed:.1f}s)"
        )
        self._emit_event(event)

    def get_progress_percent(self) -> float:
        """Get overall pipeline progress percentage.

        Returns:
            Progress [0, 100].
        """
        return (self._completed_stages / self.total_stages) * 100.0

    def get_summary(self) -> Dict:
        """Get pipeline execution summary.

        Returns:
            Dict with pipeline metrics and stage details.
        """
        return {
            "total_stages": self.total_stages,
            "completed_stages": self._completed_stages,
            "progress_percent": self.get_progress_percent(),
            "elapsed_time": self._get_elapsed_time(),
            "stages": {name: self._stage_to_dict(metrics) for name, metrics in self._stage_metrics.items()},
        }

    def _get_elapsed_time(self) -> float:
        """Get elapsed time since pipeline start.

        Returns:
            Elapsed time in seconds (0 if not started).
        """
        if self._pipeline_start_time is None:
            return 0.0

        if self._pipeline_end_time is not None:
            return self._pipeline_end_time - self._pipeline_start_time

        return time.time() - self._pipeline_start_time

    def _stage_to_dict(self, metrics: StageMetrics) -> Dict:
        """Convert StageMetrics to dict.

        Args:
            metrics: Stage metrics.

        Returns:
            Dict representation.
        """
        return {
            "name": metrics.name,
            "display_name": metrics.display_name,
            "duration": metrics.duration,
            "success": metrics.success,
            "error": metrics.error_message,
        }

    def _emit_event(self, event: ProgressEvent) -> None:
        """Emit progress event (hook for extensibility).

        Args:
            event: Progress event to emit.
        """
        # Base implementation just logs
        # Subclasses can override to send to message queues, websockets, etc.
        pass
