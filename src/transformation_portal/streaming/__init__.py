"""Real-time streaming and progress tracking for Transformation Portal.

Provides live progress updates, streaming results, and checkpoint/resume
capabilities for long-running operations.

Example:
    >>> from transformation_portal.streaming import ProgressTracker, StreamingProcessor
    >>> 
    >>> tracker = ProgressTracker(total=100)
    >>> for i in range(100):
    ...     result = process_item(i)
    ...     tracker.update(1, message=f"Processed {i}")
"""

from .progress import (
    ProgressTracker,
    ProgressBar,
    MultiProgress,
    create_progress,
)
from .checkpoint import (
    Checkpoint,
    CheckpointManager,
    checkpoint,
)
from .streaming import (
    StreamingProcessor,
    stream_results,
    batch_stream,
)

__all__ = [
    'ProgressTracker',
    'ProgressBar',
    'MultiProgress',
    'create_progress',
    'Checkpoint',
    'CheckpointManager',
    'checkpoint',
    'StreamingProcessor',
    'stream_results',
    'batch_stream',
]

__version__ = '1.0.0'
