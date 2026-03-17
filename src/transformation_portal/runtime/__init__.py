"""Runtime utilities for GPU resource management and process isolation.

This package provides:
- GPU semaphore for exclusive device access
- GPU pool for deterministic leasing
- Spawn-safe worker utilities
- GPU session pool for persistent workers with warm models
- CUDA IPC for efficient tensor sharing
- Execution sandbox for isolated, deterministic node execution
- Sandbox executor for running DAG nodes with CAS-only IO
- Process executor for spawn-safe isolation
- Execution engine with Merkle DAG provenance
"""

from transformation_portal.runtime.engine import (
    EngineConfig,
    ExecutionEngine,
    ExecutionEngineError,
    ExecutionRecord,
)
from transformation_portal.runtime.gpu_pool import (
    GPULease,
    GPUPool,
    GPUPoolError,
)
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
from transformation_portal.runtime.process_executor import (
    ProcessExecutor,
    ProcessExecutorError,
    ProcessResult,
    ProcessTask,
)
from transformation_portal.runtime.sandbox import (
    Sandbox,
    SandboxConfig,
    SandboxError,
    SandboxMetrics,
)
from transformation_portal.runtime.sandbox_executor import (
    DAGNodeProtocol,
    ExecutionResult,
    ExecutorConfig,
    SandboxExecutor,
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
    # GPU Pool
    "GPUPool",
    "GPUPoolError",
    "GPULease",
    # GPU Session Pool
    "GPUSessionPool",
    "SessionPoolError",
    "SessionTask",
    # Sandbox
    "Sandbox",
    "SandboxConfig",
    "SandboxError",
    "SandboxMetrics",
    # Sandbox Executor
    "SandboxExecutor",
    "ExecutorConfig",
    "ExecutionResult",
    "DAGNodeProtocol",
    # Process Executor
    "ProcessExecutor",
    "ProcessExecutorError",
    "ProcessTask",
    "ProcessResult",
    # Execution Engine
    "ExecutionEngine",
    "EngineConfig",
    "ExecutionEngineError",
    "ExecutionRecord",
    # Worker utilities
    "SpawnError",
    "run_spawned",
    "run_with_gpu",
]
