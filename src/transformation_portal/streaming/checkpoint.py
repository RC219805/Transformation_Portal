"""Checkpoint and resume functionality for long-running operations."""

from __future__ import annotations

import datetime
import functools
import json
import pickle
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class EvolutionaryCheckpoint:
    """Represents an evolutionary deadline for a particular workflow.

    The checkpoint keeps track of a *horizon* (a :class:`~datetime.date` after
    which a migration must be pursued) and the ``mutation_path`` that should be
    followed once the horizon has been crossed.

    The :meth:`evolve_or_alert` method returns human readable guidance that can
    be surfaced in dashboards or CI logs.  It also accepts an optional
    ``today`` override to make the class simple to test without having to rely
    on the ambient system clock.
    """

    horizon: date
    mutation_path: str

    def evolve_or_alert(self, *, today: date | None = None) -> str:
        """Return guidance about whether evolution is required.

        Parameters
        ----------
        today:
            Optional date to use instead of :func:`datetime.date.today`.  This
            is primarily useful for deterministic testing.
        """

        reference_date = today or datetime.date.today()
        if reference_date > self.horizon:
            return f"EVOLUTION REQUIRED: Migrate to {self.mutation_path}"
        return f"STABLE: Current form viable until {self.horizon.isoformat()}"


@dataclass
class Checkpoint:
    """Represents a processing checkpoint.

    Attributes:
        id: Unique checkpoint identifier
        progress: Current progress (0-100)
        state: Arbitrary state data
        timestamp: When checkpoint was created
        metadata: Additional metadata
    """
    id: str
    progress: float
    state: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any]

    def save(self, path: Path) -> None:
        """Save checkpoint to file.

        Args:
            path: Path to checkpoint file
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            'id': self.id,
            'progress': self.progress,
            'state': self.state,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
        }

        with open(path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'Checkpoint':
        """Load checkpoint from file.

        Args:
            path: Path to checkpoint file

        Returns:
            Checkpoint instance
        """
        with open(path) as f:
            data = json.load(f)

        return cls(**data)


class CheckpointManager:
    """Manage checkpoints for resumable operations.

    Example:
        >>> manager = CheckpointManager("batch_process")
        >>>
        >>> # Save checkpoint
        >>> checkpoint = manager.create_checkpoint(
        ...     progress=50.0,
        ...     state={'current_file': 'image_50.jpg', 'batch': 2}
        ... )
        >>> manager.save(checkpoint)
        >>>
        >>> # Resume from checkpoint
        >>> last_checkpoint = manager.get_latest()
        >>> if last_checkpoint:
        ...     resume_from(last_checkpoint.state)
    """

    def __init__(self, operation_id: str, checkpoint_dir: Optional[Path] = None):
        """Initialize checkpoint manager.

        Args:
            operation_id: Unique identifier for operation
            checkpoint_dir: Directory for checkpoints (defaults to .checkpoints/)
        """
        self.operation_id = operation_id
        self.checkpoint_dir = checkpoint_dir or Path('.checkpoints') / operation_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def create_checkpoint(
        self,
        progress: float,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """Create a new checkpoint.

        Args:
            progress: Progress percentage (0-100)
            state: State data to save
            metadata: Optional metadata

        Returns:
            Checkpoint instance
        """
        checkpoint_id = f"{self.operation_id}_{int(time.time())}"

        return Checkpoint(
            id=checkpoint_id,
            progress=progress,
            state=state,
            timestamp=time.time(),
            metadata=metadata or {}
        )

    def save(self, checkpoint: Checkpoint) -> Path:
        """Save checkpoint to disk.

        Args:
            checkpoint: Checkpoint to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint.id}.json"
        checkpoint.save(checkpoint_path)
        return checkpoint_path

    def get_latest(self) -> Optional[Checkpoint]:
        """Get the most recent checkpoint.

        Returns:
            Latest checkpoint or None if no checkpoints exist
        """
        checkpoints = list(self.checkpoint_dir.glob('*.json'))

        if not checkpoints:
            return None

        # Sort by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return Checkpoint.load(latest)

    def list_checkpoints(self) -> list[Checkpoint]:
        """List all checkpoints for this operation.

        Returns:
            List of Checkpoint instances, sorted by timestamp
        """
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob('*.json'):
            try:
                checkpoint = Checkpoint.load(checkpoint_file)
                checkpoints.append(checkpoint)
            except Exception as e:
                print(f"Failed to load checkpoint {checkpoint_file}: {e}")

        return sorted(checkpoints, key=lambda c: c.timestamp)

    def clear(self) -> None:
        """Delete all checkpoints for this operation."""
        for checkpoint_file in self.checkpoint_dir.glob('*.json'):
            checkpoint_file.unlink()

        # Remove directory if empty
        try:
            self.checkpoint_dir.rmdir()
        except OSError:
            pass


def checkpoint(
    operation_id: str,
    checkpoint_interval: int = 10,
    checkpoint_dir: Optional[Path] = None
):
    """Decorator to add automatic checkpointing to a function.

    The decorated function should yield (progress, state) tuples during execution.

    Args:
        operation_id: Unique operation identifier
        checkpoint_interval: Save checkpoint every N iterations
        checkpoint_dir: Directory for checkpoints

    Example:
        >>> @checkpoint(operation_id="batch_process", checkpoint_interval=10)
        ... def process_batch(files):
        ...     for i, file in enumerate(files):
        ...         result = process_file(file)
        ...         progress = (i + 1) / len(files) * 100
        ...         state = {'current_index': i, 'file': file}
        ...         yield progress, state, result
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = CheckpointManager(operation_id, checkpoint_dir)
            iteration = 0

            for result in func(*args, **kwargs):
                # Expect (progress, state, actual_result) tuples
                if isinstance(result, tuple) and len(result) == 3:
                    progress, state, actual_result = result

                    # Save checkpoint at intervals
                    if iteration % checkpoint_interval == 0:
                        checkpoint_obj = manager.create_checkpoint(
                            progress=progress,
                            state=state
                        )
                        manager.save(checkpoint_obj)

                    iteration += 1
                    yield actual_result
                else:
                    yield result

        return wrapper

    return decorator


def resume_from_checkpoint(
    operation_id: str,
    checkpoint_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Resume operation from last checkpoint.

    Args:
        operation_id: Operation identifier
        checkpoint_dir: Directory containing checkpoints

    Returns:
        State dictionary from checkpoint, or None if no checkpoint exists

    Example:
        >>> state = resume_from_checkpoint("batch_process")
        >>> if state:
        ...     start_index = state.get('current_index', 0)
        ...     process_from_index(start_index)
    """
    manager = CheckpointManager(operation_id, checkpoint_dir)
    checkpoint = manager.get_latest()

    if checkpoint:
        return checkpoint.state

    return None
