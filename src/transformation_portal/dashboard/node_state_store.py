"""Node state store for tracking per-node execution details.

This module provides persistent storage for node execution state,
enabling inspection of inputs, outputs, artifacts, and logs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NodeExecutionState:
    """Detailed execution state for a single node.

    Attributes:
        node_id: Node identifier
        status: Current status (idle, queued, running, complete, error)
        inputs: Resolved input data from dependencies
        outputs: Node output data
        artifacts: Map of artifact name -> CAS hash
        logs: Execution log messages
        metrics: Performance metrics (timing, memory, etc.)
        error: Error message if failed
        start_time: Execution start time
        end_time: Execution end time
        merkle_hash: Merkle DAG node hash for lineage
    """

    node_id: str
    status: str = "idle"
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    merkle_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "node_id": self.node_id,
            "status": self.status,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "artifacts": self.artifacts,
            "logs": self.logs,
            "metrics": self.metrics,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "merkle_hash": self.merkle_hash,
        }


@dataclass
class RunExecutionState:
    """Execution state for an entire pipeline run.

    Attributes:
        run_id: Run identifier
        status: Overall run status
        nodes: Map of node_id -> NodeExecutionState
        start_time: Run start time
        end_time: Run end time
        config: Run configuration
    """

    run_id: str
    status: str = "pending"
    nodes: Dict[str, NodeExecutionState] = field(default_factory=dict)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


class NodeStateStore:
    """Centralized store for node execution state.

    Tracks detailed state for all nodes across all runs, enabling
    inspection and debugging through the UI.

    Example:
        >>> store = NodeStateStore()
        >>> store.init_run("run_123", ["ingest", "segment", "export"])
        >>> store.set_status("run_123", "ingest", "running")
        >>> store.update_inputs("run_123", "ingest", {"image": "path/to/img"})
        >>> store.update_outputs("run_123", "ingest", {"rgb": array})
        >>> store.add_log("run_123", "ingest", "Loaded image successfully")
        >>> store.set_status("run_123", "ingest", "complete")
    """

    def __init__(self, max_runs: int = 100) -> None:
        """Initialize store.

        Args:
            max_runs: Maximum number of runs to keep in memory
        """
        self.runs: Dict[str, RunExecutionState] = {}
        self.run_history: List[str] = []
        self.max_runs = max_runs

    def _now(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def _trim_history(self) -> None:
        """Remove old runs if over limit."""
        while len(self.run_history) > self.max_runs:
            old_id = self.run_history.pop(0)
            if old_id in self.runs:
                del self.runs[old_id]

    def init_run(
        self,
        run_id: str,
        node_ids: List[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize state for a new run.

        Args:
            run_id: Run identifier
            node_ids: List of node IDs in the run
            config: Optional run configuration
        """
        self.runs[run_id] = RunExecutionState(
            run_id=run_id,
            status="running",
            start_time=self._now(),
            config=config or {},
            nodes={node_id: NodeExecutionState(node_id=node_id) for node_id in node_ids},
        )
        self.run_history.append(run_id)
        self._trim_history()
        logger.debug("Initialized run %s with %d nodes", run_id, len(node_ids))

    def get_run(self, run_id: str) -> Optional[RunExecutionState]:
        """Get run state.

        Args:
            run_id: Run identifier

        Returns:
            RunExecutionState if found, None otherwise
        """
        return self.runs.get(run_id)

    def get_node(
        self,
        run_id: str,
        node_id: str,
    ) -> Optional[NodeExecutionState]:
        """Get node state.

        Args:
            run_id: Run identifier
            node_id: Node identifier

        Returns:
            NodeExecutionState if found, None otherwise
        """
        run = self.runs.get(run_id)
        if run is None:
            return None
        return run.nodes.get(node_id)

    def set_status(self, run_id: str, node_id: str, status: str) -> None:
        """Update node status.

        Args:
            run_id: Run identifier
            node_id: Node identifier
            status: New status
        """
        node = self.get_node(run_id, node_id)
        if node is None:
            return

        node.status = status

        if status == "running" and node.start_time is None:
            node.start_time = self._now()
        elif status in ("complete", "error") and node.end_time is None:
            node.end_time = self._now()

    def update_inputs(
        self,
        run_id: str,
        node_id: str,
        inputs: Dict[str, Any],
    ) -> None:
        """Update node inputs.

        Args:
            run_id: Run identifier
            node_id: Node identifier
            inputs: Input data dictionary
        """
        node = self.get_node(run_id, node_id)
        if node:
            node.inputs = inputs

    def update_outputs(
        self,
        run_id: str,
        node_id: str,
        outputs: Dict[str, Any],
    ) -> None:
        """Update node outputs.

        Args:
            run_id: Run identifier
            node_id: Node identifier
            outputs: Output data dictionary
        """
        node = self.get_node(run_id, node_id)
        if node:
            node.outputs = outputs

    def update_artifacts(
        self,
        run_id: str,
        node_id: str,
        artifacts: Dict[str, str],
    ) -> None:
        """Update node artifacts.

        Args:
            run_id: Run identifier
            node_id: Node identifier
            artifacts: Map of artifact name -> CAS hash
        """
        node = self.get_node(run_id, node_id)
        if node:
            node.artifacts.update(artifacts)

    def add_artifact(
        self,
        run_id: str,
        node_id: str,
        name: str,
        hash: str,
    ) -> None:
        """Add a single artifact.

        Args:
            run_id: Run identifier
            node_id: Node identifier
            name: Artifact name
            hash: CAS hash
        """
        node = self.get_node(run_id, node_id)
        if node:
            node.artifacts[name] = hash

    def add_log(
        self,
        run_id: str,
        node_id: str,
        message: str,
    ) -> None:
        """Add a log message.

        Args:
            run_id: Run identifier
            node_id: Node identifier
            message: Log message
        """
        node = self.get_node(run_id, node_id)
        if node:
            timestamp = self._now()
            node.logs.append(f"[{timestamp}] {message}")

    def update_metrics(
        self,
        run_id: str,
        node_id: str,
        metrics: Dict[str, Any],
    ) -> None:
        """Update node metrics.

        Args:
            run_id: Run identifier
            node_id: Node identifier
            metrics: Metrics dictionary
        """
        node = self.get_node(run_id, node_id)
        if node:
            node.metrics.update(metrics)

    def set_error(
        self,
        run_id: str,
        node_id: str,
        error: str,
    ) -> None:
        """Set node error.

        Args:
            run_id: Run identifier
            node_id: Node identifier
            error: Error message
        """
        node = self.get_node(run_id, node_id)
        if node:
            node.error = error
            node.status = "error"
            if node.end_time is None:
                node.end_time = self._now()

    def set_merkle_hash(
        self,
        run_id: str,
        node_id: str,
        hash: str,
    ) -> None:
        """Set Merkle DAG hash for lineage.

        Args:
            run_id: Run identifier
            node_id: Node identifier
            hash: Merkle node hash
        """
        node = self.get_node(run_id, node_id)
        if node:
            node.merkle_hash = hash

    def complete_run(
        self,
        run_id: str,
        status: str = "complete",
    ) -> None:
        """Mark run as complete.

        Args:
            run_id: Run identifier
            status: Final status
        """
        run = self.get_run(run_id)
        if run:
            run.status = status
            run.end_time = self._now()

    def get_all_runs(self) -> List[Dict[str, Any]]:
        """Get summary of all runs.

        Returns:
            List of run summaries
        """
        return [
            {
                "run_id": run.run_id,
                "status": run.status,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "node_count": len(run.nodes),
            }
            for run in self.runs.values()
        ]


# Global store instance
_global_store: Optional[NodeStateStore] = None


def get_store() -> NodeStateStore:
    """Get or create the global node state store."""
    global _global_store
    if _global_store is None:
        _global_store = NodeStateStore()
    return _global_store


def set_store(store: NodeStateStore) -> None:
    """Set the global node state store."""
    global _global_store
    _global_store = store
