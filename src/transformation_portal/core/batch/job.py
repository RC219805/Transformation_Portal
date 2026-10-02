"""Batch data structures and opt-in guarded local checkpoint recovery.

Plain checkpoints remain readable snapshots. They do not prove ownership or
whether a callback already performed side effects and cannot authorize replay.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from .recovery import AttemptContext, IdempotencyContract, RecoveryIdentity

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Execution status for a batch item."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class JobItem:
    """A single unit of work within a batch job."""

    id: str  # Unique identifier (e.g., filename)
    input_path: str
    output_path: str
    status: JobStatus = JobStatus.PENDING
    error: Optional[str] = None
    execution_time: float = 0.0
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobItem":
        """Reconstruct from dictionary (JSON deserialization)."""
        return cls(**{**data, "status": JobStatus(data["status"])})


@dataclass
class BatchJob:
    """A collection of items representing a full batch workload."""

    name: str
    output_dir: str
    items: List[JobItem] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Internal map for O(1) lookups
    _item_map: Dict[str, JobItem] = field(default=None, init=False, repr=False)
    _restored_from_checkpoint: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self):
        self._rebuild_map()

    def _rebuild_map(self):
        """Rebuild internal lookup map."""
        self._item_map = {item.id: item for item in self.items}

    def add_item(self, item: JobItem):
        """Add a new item to the batch."""
        if self._item_map is None:
            self._rebuild_map()

        if item.id in self._item_map:
            logger.warning(f"Duplicate item ID {item.id} in batch {self.name}")
            return

        self.items.append(item)
        self._item_map[item.id] = item

    def get_item(self, item_id: str) -> Optional[JobItem]:
        if self._item_map is None:
            self._rebuild_map()
        return self._item_map.get(item_id)

    @property
    def progress(self) -> float:
        """Calculate percentage completion (0.0 - 1.0)."""
        if not self.items:
            return 0.0
        completed = sum(1 for i in self.items if i.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.SKIPPED))
        return completed / len(self.items)

    @property
    def stats(self) -> Dict[str, int]:
        """Get count of items by status."""
        stats = {s.value: 0 for s in JobStatus}
        for item in self.items:
            stats[item.status.value] += 1
        return stats

    def save(self, path: Union[str, Path]) -> None:
        """Durably save a plain checkpoint; persistence/ownership errors propagate.

        A plain snapshot cannot overwrite a guarded execution checkpoint.
        """
        from .recovery import BatchRecoveryRequiredError, _persist, checkpoint_ownership, read_checkpoint

        with checkpoint_ownership(Path(path)) as owned_path:
            if owned_path.exists():
                existing = read_checkpoint(owned_path)
                if isinstance(existing, dict) and "schema" in existing:
                    raise BatchRecoveryRequiredError("plain snapshots cannot overwrite guarded checkpoints")
            _persist(owned_path, self, None)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BatchJob":
        """Load job state from JSON."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        items_data = data.pop("items", [])
        job = cls(**data)

        # Reconstruct items
        job.items = [JobItem.from_dict(item) for item in items_data]
        job._rebuild_map()
        job._restored_from_checkpoint = True

        return job


class BatchProcessor:
    """Engine for executing BatchJobs."""

    def __init__(self, max_workers: int = 4, checkpoint_interval: int = 10, stop_on_errors: bool = False):
        """
        Args:
            max_workers: Number of parallel threads.
            checkpoint_interval: Retained compatibility setting; safety transitions always persist.
            stop_on_errors: If True, aborts batch on first failure.
        """
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        if type(checkpoint_interval) is not int or checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be a positive integer")
        self.max_workers = max_workers
        self.checkpoint_interval = checkpoint_interval
        self.stop_on_errors = stop_on_errors

    def process(
        self, job: BatchJob, processor_func: Callable[[JobItem], Dict[str, Any]], checkpoint_path: Union[str, Path]
    ) -> BatchJob:
        """Run fresh work only; existing checkpoints cannot authorize legacy replay.

        The path must not exist and executable items must be PENDING. Terminal-only
        jobs return unchanged. Use start_guarded/resume_guarded for explicit replay
        contracts; plain checkpoints require caller reconciliation before new work.
        """
        from .recovery import run_fresh

        return run_fresh(
            job, processor_func, Path(checkpoint_path), max_workers=self.max_workers, stop_on_errors=self.stop_on_errors
        )

    def start_guarded(
        self,
        job: BatchJob,
        processor_func: Callable[[JobItem, "AttemptContext"], Any],
        checkpoint_path: Union[str, Path],
        *,
        identity: "RecoveryIdentity",
        idempotency: Optional["IdempotencyContract"] = None,
    ) -> BatchJob:
        """Create an owned checkpoint and durably claim each item before execution.

        Identity is caller-asserted. Replay additionally requires the same explicit
        idempotency contract recorded here before any side effects. Callbacks receive
        detached items; only metadata and execution outcomes merge back into the job.
        """
        from .recovery import run_guarded

        return run_guarded(
            job,
            processor_func,
            Path(checkpoint_path),
            identity=identity,
            idempotency=idempotency,
            max_workers=self.max_workers,
            stop_on_errors=self.stop_on_errors,
        )

    def resume_guarded(
        self,
        checkpoint_path: Union[str, Path],
        processor_func: Callable[[JobItem, "AttemptContext"], Any],
        *,
        identity: "RecoveryIdentity",
        idempotency: Optional["IdempotencyContract"] = None,
    ) -> BatchJob:
        """Recover after exclusive local ownership, never from age or PID heuristics.

        Previously attempted work requires its original idempotency contract.
        Ownership does not prove external/detached side effects have stopped;
        callbacks and downstream systems must honor the stable idempotency key.
        """
        from .recovery import run_guarded

        return run_guarded(
            None,
            processor_func,
            Path(checkpoint_path),
            identity=identity,
            idempotency=idempotency,
            max_workers=self.max_workers,
            stop_on_errors=self.stop_on_errors,
        )
