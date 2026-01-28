"""
Batch Processing Engine with Checkpoint/Resume capabilities.

This module provides a robust framework for executing long-running
batch operations. It handles state persistence, error trapping,
and parallel execution.

Key Capabilities:
- Automatic crash recovery (resume from last checkpoint)
- Thread-safe state management
- Atomic checkpoint writing
- Detailed failure reporting
"""

import json
import logging
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

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
        data["status"] = JobStatus(data["status"])
        return cls(**data)


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

    def __post_init__(self):
        self._rebuild_map()

    def _rebuild_map(self):
        """Rebuild internal lookup map."""
        self._item_map = {item.id: item for item in self.items}

    def add_item(self, item: JobItem):
        """Add a new item to the batch."""
        if self._item_map is None: self._rebuild_map()
        
        if item.id in self._item_map:
            logger.warning(f"Duplicate item ID {item.id} in batch {self.name}")
            return
            
        self.items.append(item)
        self._item_map[item.id] = item

    def get_item(self, item_id: str) -> Optional[JobItem]:
        if self._item_map is None: self._rebuild_map()
        return self._item_map.get(item_id)

    @property
    def progress(self) -> float:
        """Calculate percentage completion (0.0 - 1.0)."""
        if not self.items: return 0.0
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
        """Atomically save job state to JSON."""
        path = Path(path)
        temp_path = path.with_suffix(".tmp")
        
        data = asdict(self)
        # Remove internal fields
        del data["_item_map"]
        
        try:
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
            
            # Atomic move
            shutil.move(str(temp_path), str(path))
            self.last_updated = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()

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
        
        return job


class BatchProcessor:
    """Engine for executing BatchJobs."""

    def __init__(
        self,
        max_workers: int = 4,
        checkpoint_interval: int = 10,
        stop_on_errors: bool = False
    ):
        """
        Args:
            max_workers: Number of parallel threads.
            checkpoint_interval: Save state every N completions.
            stop_on_errors: If True, aborts batch on first failure.
        """
        self.max_workers = max_workers
        self.checkpoint_interval = checkpoint_interval
        self.stop_on_errors = stop_on_errors

    def process(
        self, 
        job: BatchJob, 
        processor_func: Callable[[JobItem], Dict[str, Any]],
        checkpoint_path: Union[str, Path]
    ) -> BatchJob:
        """
        Execute the batch job.

        Args:
            job: The BatchJob object.
            processor_func: Function taking a JobItem and returning results (or raising Exception).
            checkpoint_path: Where to save the `job.json`.
        """
        logger.info(f"Starting batch '{job.name}' ({len(job.items)} items)")
        
        # Identify work
        pending_items = [
            item for item in job.items 
            if item.status in (JobStatus.PENDING, JobStatus.FAILED)
        ]
        
        if not pending_items:
            logger.info("No pending items found. Job complete.")
            return job

        logger.info(f"Resuming with {len(pending_items)} pending items...")
        
        completed_since_save = 0
        checkpoint_path = Path(checkpoint_path)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Map futures to items
            future_to_item = {
                executor.submit(self._safe_execute, processor_func, item): item 
                for item in pending_items
            }
            
            try:
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    result_item = future.result()
                    
                    # Update job state in memory
                    # (JobItem is mutable, so the list in BatchJob is updated)
                    completed_since_save += 1
                    
                    # Logging
                    if result_item.status == JobStatus.COMPLETED:
                        logger.info(f"[{job.progress:.1%}] Completed: {item.id}")
                    else:
                        logger.error(f"[{job.progress:.1%}] Failed: {item.id} - {item.error}")
                        if self.stop_on_errors:
                            logger.critical("Aborting batch due to error (stop_on_errors=True)")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break

                    # Checkpoint
                    if completed_since_save >= self.checkpoint_interval:
                        logger.debug("Saving checkpoint...")
                        job.save(checkpoint_path)
                        completed_since_save = 0
                        
            except KeyboardInterrupt:
                logger.warning("Batch interrupted by user. Saving state...")
                executor.shutdown(wait=False)
                job.save(checkpoint_path)
                raise
            finally:
                # Final save
                job.save(checkpoint_path)

        logger.info(f"Batch execution finished. Stats: {job.stats}")
        return job

    def _safe_execute(
        self, func: Callable[[JobItem], Any], item: JobItem
    ) -> JobItem:
        """Wrapper to trap errors for a single item."""
        item.status = JobStatus.RUNNING
        start = time.time()
        
        try:
            # Execute the user function
            # User function can modify item.metadata if desired
            func(item)
            item.status = JobStatus.COMPLETED
            item.error = None
            
        except Exception as e:
            item.status = JobStatus.FAILED
            item.error = str(e)
            logger.exception(f"Error processing item {item.id}")
            
        finally:
            item.execution_time = time.time() - start
            
        return item
