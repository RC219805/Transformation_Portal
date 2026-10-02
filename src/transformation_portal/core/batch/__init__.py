"""
Batch processing with checkpoint/resume capability.

Recovery is explicit and requires local ownership plus caller-owned idempotency.
Compatibility note: retained as an internal/shared helper surface with
direct smoke coverage, but it currently has no production imports.
"""

from .job import BatchJob, BatchProcessor, JobItem, JobStatus
from .recovery import (
    AttemptContext,
    BatchCheckpointError,
    BatchOwnershipError,
    BatchRecoveryRequiredError,
    IdempotencyContract,
    RecoveryIdentity,
)

__all__ = [
    "BatchJob",
    "JobItem",
    "JobStatus",
    "BatchProcessor",
    "RecoveryIdentity",
    "IdempotencyContract",
    "AttemptContext",
    "BatchOwnershipError",
    "BatchRecoveryRequiredError",
    "BatchCheckpointError",
]
