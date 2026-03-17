"""Runtime utilities for GPU resource management and process isolation.

This package provides:
- GPU semaphore for exclusive device access
- Spawn-safe worker utilities
- GPU session pool for persistent workers with warm models
- CUDA IPC for efficient tensor sharing
"""

from transformation_portal.runtime.gpu_semaphore import (
    GPUSemaphore,
    GPUSemaphoreError,
    GPUSlot,
)
from transformation_portal.runtime.gpu_session_pool import (
    GPUSessionPool,
    SessionPoolError,
    SessionTask,
)
from transformation_portal.runtime.worker import (
    SpawnError,
    run_spawned,
    run_with_gpu,
)

__all__ = [
    # GPU Semaphore
    "GPUSemaphore",
    "GPUSemaphoreError",
    "GPUSlot",
    # GPU Session Pool
    "GPUSessionPool",
    "SessionPoolError",
    "SessionTask",
    # Worker utilities
    "SpawnError",
    "run_spawned",
    "run_with_gpu",
]
