"""
Batch job management with checkpoint/resume.

Enables resilient batch processing with automatic recovery from failures.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Callable, Any
import json
import uuid
import time
import logging

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a job item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class JobItem:
    """
    Single item in a batch job.

    Tracks processing status and results for one input.
    """
    input_path: str
    output_path: str
    status: JobStatus = JobStatus.PENDING
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    attempt: int = 0
    metadata: Optional[dict] = None
    timing_s: Optional[dict[str, float]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> JobItem:
        """Create from dictionary."""
        data = data.copy()
        data["status"] = JobStatus(data["status"])
        return cls(**data)


@dataclass
class BatchJob:
    """
    Resumable batch processing job.

    Maintains state across processing runs, enabling recovery from failures.
    """
    job_id: str
    items: List[JobItem]
    checkpoint_path: Path
    created_at: str
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None

    def save_checkpoint(self):
        """Save job state to disk."""
        from datetime import datetime

        self.updated_at = datetime.utcnow().isoformat() + "Z"

        data = {
            "job_id": self.job_id,
            "items": [item.to_dict() for item in self.items],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at
        }

        # Ensure directory exists
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically (write to temp file, then rename)
        temp_path = self.checkpoint_path.with_suffix(".tmp")

        try:
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)

            temp_path.replace(self.checkpoint_path)
            logger.debug(f"Saved checkpoint for job {self.job_id}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()

    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path) -> BatchJob:
        """
        Load job from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            BatchJob restored from checkpoint
        """
        with open(checkpoint_path) as f:
            data = json.load(f)

        items = [JobItem.from_dict(item) for item in data["items"]]

        return cls(
            job_id=data["job_id"],
            items=items,
            checkpoint_path=checkpoint_path,
            created_at=data["created_at"],
            updated_at=data.get("updated_at"),
            completed_at=data.get("completed_at")
        )

    def get_pending_items(self) -> List[JobItem]:
        """Get items that still need processing."""
        return [
            item for item in self.items
            if item.status == JobStatus.PENDING
        ]

    def get_failed_items(self) -> List[JobItem]:
        """Get items that failed processing."""
        return [
            item for item in self.items
            if item.status == JobStatus.FAILED
        ]

    def get_completed_items(self) -> List[JobItem]:
        """Get items that completed successfully."""
        return [
            item for item in self.items
            if item.status == JobStatus.COMPLETED
        ]

    def mark_in_progress(self, item: JobItem):
        """Mark item as in progress."""
        item.status = JobStatus.IN_PROGRESS
        item.attempt += 1
        self.save_checkpoint()

    def mark_completed(self, item: JobItem, duration_ms: float, timing_s: Optional[dict[str, float]] = None):
        """Mark item as completed."""
        item.status = JobStatus.COMPLETED
        item.duration_ms = duration_ms
        item.timing_s = timing_s
        self.save_checkpoint()

    def mark_failed(self, item: JobItem, error: str):
        """Mark item as failed."""
        item.status = JobStatus.FAILED
        item.error = error
        self.save_checkpoint()

    def mark_skipped(self, item: JobItem, reason: str):
        """Mark item as skipped."""
        item.status = JobStatus.SKIPPED
        item.error = reason
        self.save_checkpoint()

    def is_complete(self) -> bool:
        """Check if all items are processed."""
        return all(
            item.status in (JobStatus.COMPLETED, JobStatus.SKIPPED, JobStatus.FAILED)
            for item in self.items
        )

    def get_stats(self) -> dict:
        """Get job statistics."""
        from collections import Counter

        status_counts = Counter(item.status for item in self.items)

        completed_items = self.get_completed_items()
        total_duration = sum(item.duration_ms or 0 for item in completed_items)
        avg_duration = total_duration / len(completed_items) if completed_items else 0

        return {
            "total": len(self.items),
            "completed": status_counts[JobStatus.COMPLETED],
            "failed": status_counts[JobStatus.FAILED],
            "pending": status_counts[JobStatus.PENDING],
            "skipped": status_counts[JobStatus.SKIPPED],
            "in_progress": status_counts[JobStatus.IN_PROGRESS],
            "avg_duration_ms": avg_duration,
            "total_duration_ms": total_duration
        }

    def print_summary(self):
        """Print job summary."""
        stats = self.get_stats()

        logger.info("=" * 60)
        logger.info(f"BATCH JOB SUMMARY: {self.job_id}")
        logger.info("=" * 60)
        logger.info(f"Total items:      {stats['total']}")
        logger.info(f"Completed:        {stats['completed']}")
        logger.info(f"Failed:           {stats['failed']}")
        logger.info(f"Pending:          {stats['pending']}")
        logger.info(f"Skipped:          {stats['skipped']}")

        if stats['completed'] > 0:
            logger.info(f"Average duration: {stats['avg_duration_ms']:.1f}ms")
            logger.info(f"Total time:       {stats['total_duration_ms']/1000:.1f}s")

        logger.info("=" * 60)


class BatchProcessor:
    """
    Batch processor with checkpoint/resume capability.

    Processes multiple inputs with automatic state tracking and recovery.
    """

    def __init__(
        self,
        processor_fn: Callable[[Path], Any],
        checkpoint_dir: Path,
        max_retries: int = 3,
        skip_existing: bool = True
    ):
        """
        Initialize batch processor.

        Args:
            processor_fn: Function to process each item (input_path) -> result
            checkpoint_dir: Directory for checkpoint files
            max_retries: Maximum retry attempts for failed items
            skip_existing: Skip items if output already exists
        """
        self.processor_fn = processor_fn
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_retries = max_retries
        self.skip_existing = skip_existing

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def process_batch(
        self,
        input_paths: List[Path],
        output_dir: Path,
        resume_from: Optional[Path] = None,
        job_id: Optional[str] = None
    ) -> BatchJob:
        """
        Process batch with checkpoint/resume.

        Args:
            input_paths: List of input file paths
            output_dir: Output directory
            resume_from: Path to checkpoint file to resume from
            job_id: Job ID (generated if not provided)

        Returns:
            BatchJob with results
        """
        from datetime import datetime

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load or create job
        if resume_from:
            job = BatchJob.load_checkpoint(resume_from)
            logger.info(f"Resuming job {job.job_id} from checkpoint")
        else:
            if job_id is None:
                job_id = str(uuid.uuid4())[:8]

            items = [
                JobItem(
                    input_path=str(p),
                    output_path=str(output_dir / p.name)
                )
                for p in input_paths
            ]

            checkpoint_path = self.checkpoint_dir / f"{job_id}.json"

            job = BatchJob(
                job_id=job_id,
                items=items,
                checkpoint_path=checkpoint_path,
                created_at=datetime.utcnow().isoformat() + "Z"
            )

            job.save_checkpoint()
            logger.info(f"Created new job {job_id} with {len(items)} items")

        # Process items
        pending = job.get_pending_items()
        logger.info(f"Processing {len(pending)} pending items")

        for i, item in enumerate(pending, 1):
            logger.info(f"Processing {i}/{len(pending)}: {Path(item.input_path).name}")

            # Check if output exists and skip_existing is enabled
            if self.skip_existing and Path(item.output_path).exists():
                logger.info("  Output exists, skipping")
                job.mark_skipped(item, "Output file already exists")
                continue

            # Process with retries
            self._process_item(job, item)

        # Mark job as complete
        if job.is_complete():
            job.completed_at = datetime.utcnow().isoformat() + "Z"
            job.save_checkpoint()
            logger.info(f"Job {job.job_id} completed")

        # Print summary
        job.print_summary()

        return job

    def _process_item(self, job: BatchJob, item: JobItem):
        """Process a single item with retry logic."""
        max_attempts = self.max_retries + 1

        for attempt in range(max_attempts):
            if attempt > 0:
                logger.info(f"  Retry attempt {attempt}/{self.max_retries}")

            try:
                job.mark_in_progress(item)

                start_time = time.perf_counter()

                # Process item
                result = self.processor_fn(Path(item.input_path))

                duration_ms = (time.perf_counter() - start_time) * 1000

                # Save result if needed
                if hasattr(result, 'save'):
                    result.save(Path(item.output_path))

                # Extract timing_s if available
                timing_s = None
                if isinstance(result, dict) and 'timing_s' in result:
                    timing_s = result['timing_s']
                elif isinstance(result, dict) and 'stage_times_sec' in result:
                    # Backward compatibility: convert stage_times_sec to timing_s
                    timing_s = result['stage_times_sec']

                # Mark success
                job.mark_completed(item, duration_ms, timing_s=timing_s)
                logger.info(f"  Completed in {duration_ms:.0f}ms")
                return

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.warning(f"  Failed: {error_msg}")

                if attempt >= self.max_retries:
                    job.mark_failed(item, error_msg)
                    logger.error("  Max retries exceeded, marking as failed")
                else:
                    # Brief pause before retry
                    time.sleep(0.5)

    def retry_failed(self, job: BatchJob) -> BatchJob:
        """
        Retry all failed items in a job.

        Args:
            job: Job with failed items

        Returns:
            Updated job
        """
        failed = job.get_failed_items()

        if not failed:
            logger.info("No failed items to retry")
            return job

        logger.info(f"Retrying {len(failed)} failed items")

        # Reset failed items to pending
        for item in failed:
            item.status = JobStatus.PENDING
            item.error = None
            item.attempt = 0

        job.save_checkpoint()

        # Process pending items
        for item in failed:
            self._process_item(job, item)

        job.print_summary()

        return job
