"""Pipeline execution manager with async execution and event streaming.

This module provides async pipeline execution with real-time status
updates streamed via WebSocket. It integrates with the DAG scheduler,
Merkle DAG for lineage, and CAS for artifact storage.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, List, Optional

if TYPE_CHECKING:
    from transformation_portal.storage.cas_store import ArtifactStore
    from transformation_portal.storage.merkle_dag import MerkleDAG

logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    """Node execution status."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    SKIPPED = "skipped"


class RunStatus(str, Enum):
    """Pipeline run status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class NodeState:
    """State of a single node in a run."""

    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    progress: int = 0
    logs: List[str] = field(default_factory=list)
    merkle_hash: Optional[str] = None


@dataclass
class RunState:
    """State of a pipeline run."""

    run_id: str
    status: RunStatus = RunStatus.PENDING
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    nodes: Dict[str, NodeState] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# Type alias for broadcast function
BroadcastFn = Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]


class ExecutionManager:
    """Manages async pipeline execution with event streaming.

    This manager:
    - Executes pipelines asynchronously
    - Streams node-level status updates
    - Tracks run state and history
    - Integrates with Merkle DAG for lineage

    Example:
        >>> manager = ExecutionManager()
        >>>
        >>> async def broadcast(msg):
        ...     for client in websocket_clients:
        ...         await client.send_json(msg)
        >>>
        >>> run_id = await manager.run_pipeline(pipeline_json, broadcast)
    """

    def __init__(
        self,
        *,
        merkle_dag: Optional["MerkleDAG"] = None,  # type: ignore
        cas: Optional["ArtifactStore"] = None,  # type: ignore
        node_registry: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize execution manager.

        Args:
            merkle_dag: Optional MerkleDAG for lineage tracking
            cas: Optional ArtifactStore for artifacts
            node_registry: Map of node type -> node implementation class
        """
        self.merkle_dag = merkle_dag
        self.cas = cas
        self.node_registry = node_registry or {}
        self.active_runs: Dict[str, RunState] = {}
        self.run_history: List[str] = []
        self._max_history = 100

    def _now(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def _resolve_node_impl(self, node_def: Dict[str, Any]) -> Any:
        """Resolve node implementation from definition.

        Args:
            node_def: Node definition from pipeline JSON

        Returns:
            Node implementation instance
        """
        node_type = node_def.get("type", "passthrough")

        if node_type in self.node_registry:
            # Instantiate from registry
            impl_class = self.node_registry[node_type]
            config = node_def.get("config", {})
            return impl_class(**config)

        # Fallback: use PassthroughNode
        from transformation_portal.execution_graph.nodes.base import PassthroughNode

        return PassthroughNode()

    def _get_node_deps(
        self,
        node_id: str,
        edges: List[Dict[str, Any]],
    ) -> List[str]:
        """Extract dependencies for a node from edges.

        Args:
            node_id: Target node ID
            edges: List of edge definitions

        Returns:
            List of dependency node IDs
        """
        deps = []
        for edge in edges:
            if edge.get("target") == node_id:
                source = edge.get("source")
                if source:
                    deps.append(source)
        return deps

    def allocate_run_id(self) -> str:
        """Allocate a new run ID without starting execution.

        Returns:
            Unique run ID that does not collide with active or historical runs
        """
        # Use a short UUID prefix but ensure it does not collide with existing runs.
        run_id = str(uuid.uuid4())[:8]
        while run_id in self.active_runs or run_id in self.run_history:
            run_id = str(uuid.uuid4())[:8]
        return run_id

    def start_pipeline_background(
        self,
        run_id: str,
        pipeline: Dict[str, Any],
        broadcast: BroadcastFn,
    ) -> asyncio.Task[None]:
        """Start pipeline execution in a background task.

        This immediately returns a Task that executes the pipeline.
        Use this when you need the HTTP response to return immediately.

        A done-callback is attached to log exceptions, ensuring they are
        observed and do not surface as "Task exception was never retrieved".

        Args:
            run_id: Pre-allocated run ID (from allocate_run_id)
            pipeline: Pipeline definition with nodes and edges
            broadcast: Async function to broadcast events

        Returns:
            asyncio.Task running the pipeline execution
        """
        task = asyncio.create_task(
            self._execute_pipeline(run_id, pipeline, broadcast),
            name=f"pipeline-{run_id}",
        )

        def _log_task_result(t: asyncio.Task[None]) -> None:
            try:
                # Accessing result() ensures exceptions are observed and do not
                # surface as "Task exception was never retrieved".
                t.result()
            except asyncio.CancelledError:
                logger.info("Pipeline run %s was cancelled", run_id)
            except Exception:
                logger.error("Unhandled exception in pipeline run %s", run_id, exc_info=True)

        task.add_done_callback(_log_task_result)
        return task

    async def run_pipeline(
        self,
        pipeline: Dict[str, Any],
        broadcast: BroadcastFn,
    ) -> str:
        """Execute a pipeline asynchronously with event streaming.

        Note: This method awaits the full execution. For immediate return,
        use allocate_run_id() + start_pipeline_background() instead.

        Args:
            pipeline: Pipeline definition with nodes and edges
            broadcast: Async function to broadcast events

        Returns:
            Run ID (after execution completes)
        """
        run_id = self.allocate_run_id()
        await self._execute_pipeline(run_id, pipeline, broadcast)
        return run_id

    async def _execute_pipeline(
        self,
        run_id: str,
        pipeline: Dict[str, Any],
        broadcast: BroadcastFn,
    ) -> None:
        """Internal pipeline execution implementation.

        Args:
            run_id: Run ID for this execution
            pipeline: Pipeline definition with nodes and edges
            broadcast: Async function to broadcast events
        """
        nodes = pipeline.get("nodes", [])
        edges = pipeline.get("edges", [])

        # Initialize run state
        run_state = RunState(
            run_id=run_id,
            status=RunStatus.RUNNING,
            start_time=self._now(),
        )

        for node_def in nodes:
            node_id = node_def.get("id", "")
            run_state.nodes[node_id] = NodeState(node_id=node_id)

        self.active_runs[run_id] = run_state
        self.run_history.append(run_id)

        # Trim history
        while len(self.run_history) > self._max_history:
            old_id = self.run_history.pop(0)
            if old_id in self.active_runs:
                del self.active_runs[old_id]

        # Broadcast run started
        await broadcast(
            {
                "type": "run_started",
                "run_id": run_id,
                "node_count": len(nodes),
                "timestamp": self._now(),
            }
        )

        try:
            # Build scheduler
            from transformation_portal.execution_graph.scheduler import (
                PriorityDAGScheduler,
                ResourceRequirements,
            )

            scheduler = PriorityDAGScheduler()

            for node_def in nodes:
                node_id = node_def["id"]
                node_impl = self._resolve_node_impl(node_def)
                deps = self._get_node_deps(node_id, edges)

                scheduler.add_node(
                    node_id=node_id,
                    node=node_impl,
                    deps=deps,
                    priority=node_def.get("priority", 0),
                    resources=ResourceRequirements(
                        gpu=node_def.get("gpu", False),
                    ),
                )

                # Mark as queued
                run_state.nodes[node_id].status = NodeStatus.QUEUED

            # Get execution order
            execution_order = scheduler.get_execution_order()

            await broadcast(
                {
                    "type": "execution_plan",
                    "run_id": run_id,
                    "order": execution_order,
                }
            )

            # Execute nodes in order
            results: Dict[str, Any] = {}

            for node_id in execution_order:
                scheduled_node = scheduler.nodes[node_id]
                node_state = run_state.nodes[node_id]

                # Mark running
                node_state.status = NodeStatus.RUNNING
                node_state.start_time = self._now()

                await broadcast(
                    {
                        "type": "node_start",
                        "run_id": run_id,
                        "node": node_id,
                        "timestamp": node_state.start_time,
                    }
                )

                try:
                    # Gather inputs from dependencies
                    inputs = {dep: results.get(dep, {}) for dep in scheduled_node.deps}

                    # Log start
                    node_state.logs.append(f"Starting execution with {len(inputs)} inputs")

                    await broadcast(
                        {
                            "type": "log",
                            "run_id": run_id,
                            "node": node_id,
                            "message": f"Executing with inputs: {list(inputs.keys())}",
                        }
                    )

                    # Execute node (with small delay for UI visibility)
                    await asyncio.sleep(0.1)

                    result = scheduled_node.node.run(**inputs)

                    # Extract outputs
                    if hasattr(result, "outputs"):
                        outputs = result.outputs
                    elif isinstance(result, dict):
                        outputs = result
                    else:
                        outputs = {"result": result}

                    results[node_id] = outputs
                    node_state.outputs = outputs
                    node_state.status = NodeStatus.COMPLETE
                    node_state.end_time = self._now()
                    node_state.progress = 100

                    # Record in Merkle DAG
                    if self.merkle_dag:
                        dep_hashes = [
                            run_state.nodes[dep].merkle_hash for dep in scheduled_node.deps if run_state.nodes[dep].merkle_hash
                        ]
                        node_state.merkle_hash = self.merkle_dag.add_computation(
                            node_id=node_id,
                            inputs=dep_hashes,
                            outputs=outputs,
                            metadata={"run_id": run_id},
                        )

                    await broadcast(
                        {
                            "type": "node_complete",
                            "run_id": run_id,
                            "node": node_id,
                            "outputs": outputs,
                            "merkle_hash": node_state.merkle_hash,
                            "timestamp": node_state.end_time,
                        }
                    )

                    node_state.logs.append(f"Completed with {len(outputs)} outputs")

                except Exception as exc:
                    error_msg = f"{type(exc).__name__}: {exc}"
                    node_state.status = NodeStatus.ERROR
                    node_state.error = error_msg
                    node_state.end_time = self._now()
                    node_state.logs.append(f"Error: {error_msg}")

                    await broadcast(
                        {
                            "type": "node_error",
                            "run_id": run_id,
                            "node": node_id,
                            "error": error_msg,
                            "timestamp": node_state.end_time,
                        }
                    )

                    # Continue with other nodes or fail?
                    # For now, continue but mark results as None
                    results[node_id] = None

            # Run complete
            run_state.status = RunStatus.COMPLETE
            run_state.end_time = self._now()
            run_state.results = results

            await broadcast(
                {
                    "type": "run_complete",
                    "run_id": run_id,
                    "results": {k: v for k, v in results.items() if v is not None},
                    "timestamp": run_state.end_time,
                }
            )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            run_state.status = RunStatus.ERROR
            run_state.error = error_msg
            run_state.end_time = self._now()

            logger.exception("Pipeline execution failed: %s", run_id)

            await broadcast(
                {
                    "type": "run_error",
                    "run_id": run_id,
                    "error": error_msg,
                    "traceback": traceback.format_exc(),
                    "timestamp": run_state.end_time,
                }
            )

    def get_run_state(self, run_id: str) -> Optional[RunState]:
        """Get state of a run.

        Args:
            run_id: Run ID

        Returns:
            RunState if found, None otherwise
        """
        return self.active_runs.get(run_id)

    def get_active_runs(self) -> List[Dict[str, Any]]:
        """Get list of active runs.

        Returns:
            List of run summaries
        """
        return [
            {
                "run_id": run.run_id,
                "status": run.status.value,
                "start_time": run.start_time,
                "node_count": len(run.nodes),
            }
            for run in self.active_runs.values()
        ]

    async def cancel_run(self, run_id: str, broadcast: BroadcastFn) -> bool:
        """Cancel a running pipeline.

        Args:
            run_id: Run ID to cancel
            broadcast: Broadcast function

        Returns:
            True if cancelled, False if not found
        """
        run = self.active_runs.get(run_id)
        if run is None:
            return False

        if run.status == RunStatus.RUNNING:
            run.status = RunStatus.CANCELLED
            run.end_time = self._now()

            await broadcast(
                {
                    "type": "run_cancelled",
                    "run_id": run_id,
                    "timestamp": run.end_time,
                }
            )

        return True
