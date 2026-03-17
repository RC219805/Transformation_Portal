"""Base classes for DAG execution nodes.

This module provides the base abstractions for nodes in the execution
graph. Nodes are composable units of computation that:
- Accept typed inputs
- Produce typed outputs
- Can be wired together in a DAG

Design aligned with spatial_ai/orchestration/graph/execution_graph.py
but simplified for standalone node execution and testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class NodeResult:
    """Result from a DAG node execution.

    Attributes:
        outputs: Dictionary of named outputs from the node
        metadata: Optional execution metadata (timing, resource usage, etc.)
        error: Error message if execution failed
    """

    outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if execution succeeded."""
        return self.error is None


class DAGNode:
    """Base class for DAG execution nodes.

    Subclasses should implement the run() method to perform
    their specific computation.

    Example:
        >>> class MyNode(DAGNode):
        ...     def run(self, *, input_data: list[int]) -> NodeResult:
        ...         total = sum(input_data)
        ...         return NodeResult(outputs={"total": total})
        >>>
        >>> node = MyNode()
        >>> result = node.run(input_data=[1, 2, 3])
        >>> assert result.outputs["total"] == 6
    """

    def run(self, **inputs: Any) -> NodeResult:
        """Execute the node with given inputs.

        Args:
            **inputs: Named inputs to the node

        Returns:
            NodeResult with outputs and metadata

        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError("Subclasses must implement run()")

    def validate_inputs(self, **inputs: Any) -> Optional[str]:
        """Validate inputs before execution.

        Override this method to add input validation logic.

        Args:
            **inputs: Named inputs to validate

        Returns:
            Error message if validation fails, None if valid
        """
        return None


class PassthroughNode(DAGNode):
    """Node that passes inputs through as outputs.

    Useful for testing and as a placeholder in graphs.
    """

    def run(self, **inputs: Any) -> NodeResult:
        """Pass all inputs through as outputs."""
        return NodeResult(outputs=dict(inputs))
