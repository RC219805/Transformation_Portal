"""Deterministic Replay Engine.

This module provides replay capability for DAG nodes using their
Merkle hash. A node can be re-executed exactly if:
- Inputs are CAS hashes (deterministic)
- Code version is pinned
- Model revision is pinned
- Environment is stable

Example:
    >>> engine = ExecutionEngine(config)
    >>> dag = engine.dag
    >>>
    >>> # Original execution
    >>> node_hash, outputs1 = engine.run_node(MyNode, inputs={"sha": "abc..."})
    >>>
    >>> # Later: replay from hash alone
    >>> replay = ReplayEngine(dag, engine)
    >>> outputs2 = replay.replay(node_hash)
    >>>
    >>> assert outputs1 == outputs2  # Deterministic!
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

from transformation_portal.storage.merkle_dag import MerkleDAG, MerkleNode

logger = logging.getLogger(__name__)


class ReplayError(RuntimeError):
    """Raised for replay failures."""


@dataclass(frozen=True)
class ReplayConfig:
    """Configuration for replay engine.

    Attributes:
        workspace_root: Root for replay workspaces
        cas_root: CAS storage root
        verify_outputs: Whether to verify outputs match original
    """

    workspace_root: Path
    cas_root: Path
    verify_outputs: bool = True


@dataclass
class ReplayResult:
    """Result of a replay execution.

    Attributes:
        original_hash: Original Merkle hash
        replay_hash: New Merkle hash from replay
        outputs: Replay outputs
        outputs_match: Whether outputs match original
        duration_seconds: Replay duration
    """

    original_hash: str
    replay_hash: Optional[str]
    outputs: Dict[str, Any]
    outputs_match: bool
    duration_seconds: float


class NodeRegistry:
    """Registry for node classes by name.

    Allows replay engine to resolve node types from stored names.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Type] = {}

    def register(self, name: str, node_cls: Type) -> None:
        """Register a node class.

        Args:
            name: Unique name for the node type
            node_cls: Node class
        """
        self._registry[name] = node_cls
        logger.debug("Registered node type: %s", name)

    def get(self, name: str) -> Optional[Type]:
        """Get a node class by name.

        Args:
            name: Node type name

        Returns:
            Node class or None if not found
        """
        return self._registry.get(name)

    def resolve(self, name: str) -> Type:
        """Resolve a node class by name.

        Args:
            name: Node type name

        Returns:
            Node class

        Raises:
            ReplayError: If node type not found
        """
        cls = self.get(name)
        if cls is None:
            raise ReplayError(f"Unknown node type: {name}")
        return cls

    def list_types(self) -> list[str]:
        """List all registered node types."""
        return list(self._registry.keys())


# Global node registry
NODE_REGISTRY = NodeRegistry()


def register_node(name: str) -> Callable[[Type], Type]:
    """Decorator to register a node class.

    Example:
        >>> @register_node("process_image")
        ... class ProcessImageNode:
        ...     def run(self, *, sandbox, **inputs):
        ...         ...
    """

    def decorator(cls: Type) -> Type:
        NODE_REGISTRY.register(name, cls)
        return cls

    return decorator


class ReplayEngine:
    """Deterministic replay engine for DAG nodes.

    Re-runs nodes using only their Merkle hash. The hash encodes
    all inputs, so replay produces identical outputs (assuming
    deterministic execution).

    Example:
        >>> replay = ReplayEngine(dag, engine, registry)
        >>>
        >>> # Replay a single node
        >>> result = replay.replay(node_hash)
        >>> print(f"Outputs match: {result.outputs_match}")
        >>>
        >>> # Replay entire lineage
        >>> results = replay.replay_lineage(final_hash)
    """

    def __init__(
        self,
        dag: MerkleDAG,
        engine: "ExecutionEngine",
        registry: Optional[NodeRegistry] = None,
    ) -> None:
        """Initialize replay engine.

        Args:
            dag: Merkle DAG with node records
            engine: Execution engine for running nodes
            registry: Node registry for resolving types
        """
        self.dag = dag
        self.engine = engine
        self.registry = registry or NODE_REGISTRY
        self._replay_count = 0

    def replay(
        self,
        node_hash: str,
        *,
        verify: bool = True,
    ) -> ReplayResult:
        """Replay a node from its Merkle hash.

        Args:
            node_hash: Merkle hash of the node to replay
            verify: Whether to verify outputs match original

        Returns:
            ReplayResult with replay outputs

        Raises:
            ReplayError: If node not found or replay fails
        """
        import time

        if node_hash not in self.dag.nodes:
            raise ReplayError(f"Node not found in DAG: {node_hash}")

        node = self.dag.nodes[node_hash]

        if node.node_type != "computation":
            raise ReplayError(f"Cannot replay non-computation node: {node.node_type}")

        # Get node class
        node_type_name = node.metadata.get("node_type", node.metadata.get("node_id", ""))
        node_cls = self.registry.resolve(node_type_name)

        # Extract inputs from metadata/outputs
        # The original inputs are stored in metadata
        original_inputs = node.metadata.get("original_inputs", {})

        self._replay_count += 1
        replay_id = f"replay_{node_hash[:8]}_{self._replay_count}"

        start_time = time.time()

        try:
            replay_hash, outputs = self.engine.run_node(
                node_cls,
                inputs=original_inputs,
                node_id=replay_id,
            )
        except Exception as e:
            raise ReplayError(f"Replay execution failed: {e}")

        duration = time.time() - start_time

        # Verify outputs match
        outputs_match = True
        if verify:
            original_outputs = node.outputs
            outputs_match = self._compare_outputs(original_outputs, outputs)

            if not outputs_match:
                logger.warning(
                    "Replay outputs differ: original=%s, replay=%s",
                    original_outputs,
                    outputs,
                )

        logger.info(
            "Replayed node %s -> %s (match=%s, duration=%.2fs)",
            node_hash[:8],
            replay_hash[:8] if replay_hash else "N/A",
            outputs_match,
            duration,
        )

        return ReplayResult(
            original_hash=node_hash,
            replay_hash=replay_hash,
            outputs=outputs,
            outputs_match=outputs_match,
            duration_seconds=duration,
        )

    def replay_lineage(
        self,
        node_hash: str,
        *,
        verify: bool = True,
    ) -> list[ReplayResult]:
        """Replay entire lineage of a node.

        Replays all ancestor nodes in topological order.

        Args:
            node_hash: Merkle hash of the final node
            verify: Whether to verify outputs

        Returns:
            List of ReplayResults in execution order
        """
        lineage = self.dag.get_lineage(node_hash)

        results = []
        for node in lineage:
            if node.node_type == "computation":
                result = self.replay(node.hash, verify=verify)
                results.append(result)

        return results

    def _compare_outputs(
        self,
        original: Dict[str, Any],
        replay: Dict[str, Any],
    ) -> bool:
        """Compare original and replay outputs.

        Args:
            original: Original outputs
            replay: Replay outputs

        Returns:
            True if outputs match
        """
        if set(original.keys()) != set(replay.keys()):
            return False

        for key in original:
            if original[key] != replay[key]:
                return False

        return True

    def verify_reproducibility(self, node_hash: str) -> bool:
        """Verify a node is reproducible.

        Replays the node and checks outputs match.

        Args:
            node_hash: Node to verify

        Returns:
            True if reproducible
        """
        try:
            result = self.replay(node_hash, verify=True)
            return result.outputs_match
        except ReplayError:
            return False

    @property
    def replay_count(self) -> int:
        """Total number of replays executed."""
        return self._replay_count


# Import here to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformation_portal.runtime.engine import ExecutionEngine
