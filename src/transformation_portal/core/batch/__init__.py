"""
Batch processing with checkpoint/resume capability.

Provides robust batch processing with automatic recovery.
Compatibility note: retained as an internal/shared helper surface with
direct smoke coverage, but it currently has no production imports.
"""

from .job import BatchJob, BatchProcessor, JobItem, JobStatus

__all__ = [
    "BatchJob",
    "JobItem",
    "JobStatus",
    "BatchProcessor",
]
