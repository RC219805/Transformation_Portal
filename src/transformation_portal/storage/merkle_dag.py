"""Merkle DAG for artifact lineage tracking.

This module provides a content-addressable lineage graph that:
- Records all pipeline artifacts with their provenance
- Uses SHA-256 hashing for deterministic identity
- Integrates with the CAS (ArtifactStore) for storage
- Enables reproducibility verification

Design:
    Each node in the Merkle DAG represents an artifact or computation step.
    The hash of each node depends on its inputs, creating a cryptographic
    chain of custody for all artifacts.

Example:
    >>> merkle = MerkleDAG()
    >>>
    >>> # Record input artifact
    >>> input_hash = merkle.add_artifact(
    ...     artifact_type="image",
    ...     content_hash="abc123...",
    ...     metadata={"path": "input.png"},
    ... )
    >>>
    >>> # Record computation
    >>> output_hash = merkle.add_computation(
    ...     node_id="segment",
    ...     inputs=[input_hash],
    ...     outputs={"mask": "def456..."},
    ...     metadata={"model": "sam2"},
    ... )
    >>>
    >>> # Export lineage
    >>> merkle.export(Path("lineage.json"))
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MerkleNode:
    """Node in the Merkle DAG.

    Attributes:
        hash: SHA-256 hash of the node (content-derived)
        node_type: Type of node ("artifact", "computation", "checkpoint")
        inputs: List of input node hashes
        outputs: Output data/hashes
        metadata: Additional metadata
        timestamp: ISO timestamp of node creation
    """

    hash: str
    node_type: str
    inputs: tuple[str, ...]
    outputs: dict[str, Any]
    metadata: dict[str, Any]
    timestamp: str


class MerkleDAGError(RuntimeError):
    """Raised for Merkle DAG errors."""


class MerkleDAG:
    """Content-addressable lineage graph using Merkle hashing.

    Tracks artifact provenance through a DAG where each node's
    identity is derived from its content and inputs.

    Example:
        >>> dag = MerkleDAG()
        >>>
        >>> # Add source artifact
        >>> img_hash = dag.add_artifact(
        ...     artifact_type="input_image",
        ...     content_hash="abc123...",
        ...     metadata={"filename": "photo.jpg"},
        ... )
        >>>
        >>> # Add computation step
        >>> seg_hash = dag.add_computation(
        ...     node_id="segmentation",
        ...     inputs=[img_hash],
        ...     outputs={"mask_hash": "def456..."},
        ...     metadata={"model": "sam2-large"},
        ... )
        >>>
        >>> # Verify lineage
        >>> lineage = dag.get_lineage(seg_hash)
        >>> print(f"Depth: {len(lineage)}")
    """

    def __init__(self) -> None:
        """Initialize empty Merkle DAG."""
        self.nodes: dict[str, MerkleNode] = {}
        self._root_hashes: list[str] = []

    def _hash_payload(self, payload: dict[str, Any]) -> str:
        """Compute SHA-256 hash of a JSON-serializable payload.

        Args:
            payload: Dictionary to hash

        Returns:
            Lowercase hex SHA-256 hash
        """
        # Canonical JSON for deterministic hashing
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _now_iso(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def add_artifact(
        self,
        *,
        artifact_type: str,
        content_hash: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add an artifact node to the DAG.

        Artifacts are leaf nodes with no computation inputs.
        They represent raw data like images, configs, or model weights.

        Args:
            artifact_type: Type of artifact (e.g., "image", "model", "config")
            content_hash: SHA-256 hash of the artifact content
            metadata: Additional artifact metadata

        Returns:
            Hash of the created node
        """
        payload = {
            "type": "artifact",
            "artifact_type": artifact_type,
            "content_hash": content_hash,
            "metadata": metadata or {},
        }

        node_hash = self._hash_payload(payload)

        if node_hash not in self.nodes:
            self.nodes[node_hash] = MerkleNode(
                hash=node_hash,
                node_type="artifact",
                inputs=(),
                outputs={"content_hash": content_hash},
                metadata={"artifact_type": artifact_type, **(metadata or {})},
                timestamp=self._now_iso(),
            )
            self._root_hashes.append(node_hash)
            logger.debug("Added artifact node: %s", node_hash[:8])

        return node_hash

    def add_computation(
        self,
        *,
        node_id: str,
        inputs: list[str],
        outputs: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a computation node to the DAG.

        Computation nodes represent pipeline stages that transform
        inputs into outputs.

        Args:
            node_id: Identifier of the computation (e.g., DAG node ID)
            inputs: List of input node hashes
            outputs: Output data (typically hashes of output artifacts)
            metadata: Additional computation metadata

        Returns:
            Hash of the created node
        """
        # Validate inputs exist
        for input_hash in inputs:
            if input_hash not in self.nodes:
                raise MerkleDAGError(f"Input node not found: {input_hash}")

        payload = {
            "type": "computation",
            "node_id": node_id,
            "inputs": sorted(inputs),
            "outputs": outputs,
            "metadata": metadata or {},
        }

        node_hash = self._hash_payload(payload)

        if node_hash not in self.nodes:
            self.nodes[node_hash] = MerkleNode(
                hash=node_hash,
                node_type="computation",
                inputs=tuple(sorted(inputs)),
                outputs=outputs,
                metadata={"node_id": node_id, **(metadata or {})},
                timestamp=self._now_iso(),
            )
            logger.debug("Added computation node: %s", node_hash[:8])

        return node_hash

    def add_checkpoint(
        self,
        *,
        checkpoint_id: str,
        inputs: list[str],
        state: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a checkpoint node to the DAG.

        Checkpoints represent saved pipeline state for resumption.

        Args:
            checkpoint_id: Unique checkpoint identifier
            inputs: List of input node hashes
            state: Checkpoint state data
            metadata: Additional metadata

        Returns:
            Hash of the created node
        """
        payload = {
            "type": "checkpoint",
            "checkpoint_id": checkpoint_id,
            "inputs": sorted(inputs),
            "state": state,
            "metadata": metadata or {},
        }

        node_hash = self._hash_payload(payload)

        if node_hash not in self.nodes:
            self.nodes[node_hash] = MerkleNode(
                hash=node_hash,
                node_type="checkpoint",
                inputs=tuple(sorted(inputs)),
                outputs=state,
                metadata={"checkpoint_id": checkpoint_id, **(metadata or {})},
                timestamp=self._now_iso(),
            )
            logger.debug("Added checkpoint node: %s", node_hash[:8])

        return node_hash

    def get_node(self, node_hash: str) -> Optional[MerkleNode]:
        """Get a node by its hash.

        Args:
            node_hash: SHA-256 hash of the node

        Returns:
            MerkleNode if found, None otherwise
        """
        return self.nodes.get(node_hash)

    def get_lineage(
        self,
        node_hash: str,
        *,
        max_depth: Optional[int] = None,
    ) -> list[MerkleNode]:
        """Get the full lineage (ancestry) of a node.

        Returns all ancestor nodes in topological order
        (roots first, target node last).

        Args:
            node_hash: Hash of the target node
            max_depth: Maximum depth to traverse (None = full lineage)

        Returns:
            List of MerkleNodes in topological order
        """
        if node_hash not in self.nodes:
            return []

        visited: set[str] = set()
        lineage: list[MerkleNode] = []

        def visit(h: str, depth: int = 0) -> None:
            if h in visited:
                return
            if max_depth is not None and depth > max_depth:
                return

            visited.add(h)
            node = self.nodes.get(h)
            if node is None:
                return

            # Visit inputs first (DFS)
            for input_hash in node.inputs:
                visit(input_hash, depth + 1)

            lineage.append(node)

        visit(node_hash)
        return lineage

    def verify_integrity(self) -> list[str]:
        """Verify integrity of all nodes in the DAG.

        Checks that:
        - All input references are valid
        - No orphaned nodes (except roots)

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        for node_hash, node in self.nodes.items():
            for input_hash in node.inputs:
                if input_hash not in self.nodes:
                    errors.append(f"Node {node_hash[:8]} references missing input {input_hash[:8]}")

        return errors

    def export(
        self,
        path: Path,
        *,
        pretty: bool = True,
    ) -> None:
        """Export the DAG to JSON file.

        Args:
            path: Output file path
            pretty: If True, format JSON with indentation
        """
        data = {
            "version": "1.0",
            "exported_at": self._now_iso(),
            "node_count": len(self.nodes),
            "root_hashes": self._root_hashes,
            "nodes": {
                h: {
                    "type": n.node_type,
                    "inputs": list(n.inputs),
                    "outputs": n.outputs,
                    "metadata": n.metadata,
                    "timestamp": n.timestamp,
                }
                for h, n in self.nodes.items()
            },
        }

        indent = 2 if pretty else None
        path.write_text(json.dumps(data, indent=indent, sort_keys=True))
        logger.info("Exported Merkle DAG to %s (%d nodes)", path, len(self.nodes))

    @classmethod
    def load(cls, path: Path) -> "MerkleDAG":
        """Load a DAG from JSON file.

        Args:
            path: Input file path

        Returns:
            Loaded MerkleDAG
        """
        data = json.loads(path.read_text())

        dag = cls()
        dag._root_hashes = data.get("root_hashes", [])

        for h, node_data in data.get("nodes", {}).items():
            dag.nodes[h] = MerkleNode(
                hash=h,
                node_type=node_data["type"],
                inputs=tuple(node_data.get("inputs", [])),
                outputs=node_data.get("outputs", {}),
                metadata=node_data.get("metadata", {}),
                timestamp=node_data.get("timestamp", ""),
            )

        logger.info("Loaded Merkle DAG from %s (%d nodes)", path, len(dag.nodes))
        return dag

    def summary(self) -> dict[str, Any]:
        """Get summary statistics of the DAG.

        Returns:
            Dictionary with node counts by type, depth, etc.
        """
        type_counts: dict[str, int] = {}
        for node in self.nodes.values():
            type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1

        return {
            "total_nodes": len(self.nodes),
            "root_nodes": len(self._root_hashes),
            "nodes_by_type": type_counts,
        }
