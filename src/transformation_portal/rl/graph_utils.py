"""Graph utilities for building DAG representations from pipelines.

This module provides utilities to convert pipeline configurations
into graph representations suitable for Graph Attention Networks.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy torch import
_torch = None


def _get_torch():
    """Lazy import torch."""
    global _torch
    if _torch is None:
        try:
            import torch

            _torch = torch
        except ImportError:
            raise ImportError("PyTorch required for graph utilities")
    return _torch


def build_edge_index(pipeline: dict[str, Any]) -> Any:
    """Build edge index tensor from pipeline dependencies.

    Converts pipeline node dependencies into a graph edge index tensor
    suitable for Graph Neural Networks. Each edge represents a
    dependency relationship (source -> target means source is a
    dependency of target).

    Args:
        pipeline: Pipeline configuration with "nodes" list.
                  Each node should have "id" and optional "deps" list.

    Returns:
        Edge index tensor [2, num_edges] where:
        - edge_index[0] = source node indices
        - edge_index[1] = destination node indices

    Example:
        >>> pipeline = {
        ...     "nodes": [
        ...         {"id": "depth", "deps": []},
        ...         {"id": "sam2", "deps": ["depth"]},
        ...         {"id": "nvdiffrec", "deps": ["depth", "sam2"]},
        ...     ]
        ... }
        >>> edges = build_edge_index(pipeline)
        >>> print(edges.shape)  # [2, 3]
    """
    torch = _get_torch()

    nodes = pipeline.get("nodes", [])
    if not nodes:
        return torch.zeros((2, 0), dtype=torch.long)

    # Build node ID to index mapping
    node_ids = [n.get("id", f"node_{i}") for i, n in enumerate(nodes)]
    idx_map = {nid: i for i, nid in enumerate(node_ids)}

    # Collect edges: dep -> node (source -> target)
    edges: list[tuple[int, int]] = []

    for node in nodes:
        node_id = node.get("id")
        if node_id is None:
            continue

        node_idx = idx_map.get(node_id)
        if node_idx is None:
            continue

        deps = node.get("deps", [])
        for dep_id in deps:
            dep_idx = idx_map.get(dep_id)
            if dep_idx is not None:
                edges.append((dep_idx, node_idx))

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)

    src, dst = zip(*edges)
    return torch.tensor([src, dst], dtype=torch.long)


def build_adjacency_matrix(pipeline: dict[str, Any]) -> Any:
    """Build adjacency matrix from pipeline dependencies.

    Args:
        pipeline: Pipeline configuration

    Returns:
        Adjacency matrix [num_nodes, num_nodes]
    """
    torch = _get_torch()

    nodes = pipeline.get("nodes", [])
    n = len(nodes)

    if n == 0:
        return torch.zeros((0, 0), dtype=torch.float32)

    adj = torch.zeros((n, n), dtype=torch.float32)
    edge_index = build_edge_index(pipeline)

    if edge_index.numel() > 0:
        src, dst = edge_index
        adj[src, dst] = 1.0

    return adj


def get_node_ordering(pipeline: dict[str, Any]) -> list[str]:
    """Get topological ordering of nodes.

    Args:
        pipeline: Pipeline configuration

    Returns:
        List of node IDs in topological order
    """
    nodes = pipeline.get("nodes", [])
    node_ids = [n.get("id", f"node_{i}") for i, n in enumerate(nodes)]

    # Build dependency graph
    deps_map: dict[str, set[str]] = {}
    for node in nodes:
        nid = node.get("id")
        if nid:
            deps_map[nid] = set(node.get("deps", []))

    # Kahn's algorithm for topological sort
    in_degree = {nid: len(deps) for nid, deps in deps_map.items()}
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    result = []

    while queue:
        node = queue.pop(0)
        result.append(node)

        # Reduce in-degree of dependents
        for nid, deps in deps_map.items():
            if node in deps:
                in_degree[nid] -= 1
                if in_degree[nid] == 0:
                    queue.append(nid)

    # Add any remaining nodes (cycles or disconnected)
    for nid in node_ids:
        if nid not in result:
            result.append(nid)

    return result


def get_node_features(
    pipeline: dict[str, Any],
    metrics: dict[str, float],
    diff: dict[str, Any],
) -> Any:
    """Extract node features from pipeline for GNN.

    Args:
        pipeline: Pipeline configuration
        metrics: Evaluation metrics
        diff: Semantic diff result

    Returns:
        Node feature tensor [num_nodes, feature_dim]
    """
    torch = _get_torch()
    from transformation_portal.rl.ma_state import encode_local, get_local_dim

    nodes = pipeline.get("nodes", [])
    n = len(nodes)

    if n == 0:
        return torch.zeros((0, get_local_dim()), dtype=torch.float32)

    features = []
    for node in nodes:
        node_id = node.get("id", "unknown")
        node_cfg = node.get("config", {})
        feat = encode_local(node_cfg, node_id)
        features.append(torch.tensor(feat, dtype=torch.float32))

    return torch.stack(features, dim=0)


def validate_dag(pipeline: dict[str, Any]) -> tuple[bool, str]:
    """Validate that pipeline forms a valid DAG.

    Checks for:
    - No cycles
    - All dependencies exist
    - No self-loops

    Args:
        pipeline: Pipeline configuration

    Returns:
        Tuple of (is_valid, error_message)
    """
    nodes = pipeline.get("nodes", [])
    node_ids = {n.get("id") for n in nodes if n.get("id")}

    # Check dependencies exist
    for node in nodes:
        nid = node.get("id")
        for dep in node.get("deps", []):
            if dep not in node_ids:
                return False, f"Node {nid} has missing dependency: {dep}"
            if dep == nid:
                return False, f"Node {nid} has self-loop"

    # Check for cycles using DFS
    visited: set[str] = set()
    rec_stack: set[str] = set()

    def has_cycle(nid: str) -> bool:
        visited.add(nid)
        rec_stack.add(nid)

        # Find node
        node = next((n for n in nodes if n.get("id") == nid), None)
        if node is None:
            return False

        for dep in node.get("deps", []):
            if dep not in visited:
                if has_cycle(dep):
                    return True
            elif dep in rec_stack:
                return True

        rec_stack.remove(nid)
        return False

    for node in nodes:
        nid = node.get("id")
        if nid and nid not in visited:
            if has_cycle(nid):
                return False, f"Cycle detected involving node {nid}"

    return True, "Valid DAG"
