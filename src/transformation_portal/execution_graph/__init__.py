"""Execution graph utilities for DAG-based pipeline orchestration.

This package provides:
- DAG nodes for pipeline stages
- Priority-based scheduler with resource awareness
- Distributed executor with Ray and local backends
"""

from transformation_portal.execution_graph.nodes.base import DAGNode, NodeResult, PassthroughNode
from transformation_portal.execution_graph.scheduler import (
    PriorityDAGScheduler,
    ResourceRequirements,
    ScheduledNode,
    SchedulerError,
)
from transformation_portal.execution_graph.distributed_executor import (
    DistributedDAGExecutor,
    DistributedExecutorError,
    ExecutionConfig,
    create_executor,
)

__all__ = [
    # Nodes
    "DAGNode",
    "NodeResult",
    "PassthroughNode",
    # Scheduler
    "PriorityDAGScheduler",
    "ResourceRequirements",
    "ScheduledNode",
    "SchedulerError",
    # Distributed executor
    "DistributedDAGExecutor",
    "DistributedExecutorError",
    "ExecutionConfig",
    "create_executor",
]
