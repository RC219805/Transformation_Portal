"""Unit tests for PipelineDefinition validation behavior.

Tests cover:
1. dict -> PipelineNode/PipelineEdge coercion (backward compatibility)
2. Rejection of invalid payload shapes (non-list inputs)
3. Type-safe passthrough of already-typed objects
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from transformation_portal.dashboard.dag_editor_api import (
    PipelineDefinition,
    PipelineEdge,
    PipelineNode,
)

pytestmark = pytest.mark.unit


class TestPipelineDefinitionNodeCoercion:
    """Tests for nodes field coercion in PipelineDefinition."""

    def test_coerce_dict_to_pipeline_node(self) -> None:
        """Test that raw dicts are coerced to PipelineNode objects."""
        pipeline = PipelineDefinition(
            name="test",
            nodes=[
                {
                    "id": "node-1",
                    "type": "input",
                    "label": "Input Node",
                    "position": {"x": 0.0, "y": 0.0},
                }
            ],
        )
        assert len(pipeline.nodes) == 1
        assert isinstance(pipeline.nodes[0], PipelineNode)
        assert pipeline.nodes[0].id == "node-1"
        assert pipeline.nodes[0].type == "input"
        assert pipeline.nodes[0].label == "Input Node"

    def test_passthrough_typed_pipeline_node(self) -> None:
        """Test that typed PipelineNode objects pass through unchanged."""
        node = PipelineNode(
            id="node-1",
            type="processor",
            label="Processor",
            position={"x": 100.0, "y": 100.0},
        )
        pipeline = PipelineDefinition(name="test", nodes=[node])
        assert pipeline.nodes[0] is node

    def test_mixed_dict_and_typed_nodes(self) -> None:
        """Test mixing typed nodes and dicts in the same list."""
        typed_node = PipelineNode(
            id="typed-1",
            type="typed",
            label="Typed Node",
            position={"x": 0.0, "y": 0.0},
        )
        pipeline = PipelineDefinition(
            name="test",
            nodes=[
                typed_node,
                {
                    "id": "dict-1",
                    "type": "dict",
                    "label": "Dict Node",
                    "position": {"x": 50.0, "y": 50.0},
                },
            ],
        )
        assert len(pipeline.nodes) == 2
        assert pipeline.nodes[0] is typed_node
        assert isinstance(pipeline.nodes[1], PipelineNode)
        assert pipeline.nodes[1].id == "dict-1"

    def test_none_nodes_coerced_to_empty_list(self) -> None:
        """Test that None for nodes is coerced to empty list."""
        pipeline = PipelineDefinition(name="test", nodes=None)
        assert pipeline.nodes == []

    def test_empty_list_nodes_accepted(self) -> None:
        """Test that empty list for nodes is accepted as-is."""
        pipeline = PipelineDefinition(name="test", nodes=[])
        assert pipeline.nodes == []

    def test_reject_string_nodes(self) -> None:
        """Test that string input for nodes raises ValidationError."""
        with pytest.raises(ValidationError, match="nodes"):
            PipelineDefinition(name="test", nodes="invalid")

    def test_reject_dict_nodes(self) -> None:
        """Test that single dict (not list) for nodes raises ValidationError."""
        with pytest.raises(ValidationError, match="nodes"):
            PipelineDefinition(
                name="test",
                nodes={"id": "node-1", "type": "test", "label": "Test", "position": {"x": 0, "y": 0}},
            )

    def test_reject_integer_nodes(self) -> None:
        """Test that integer input for nodes raises ValidationError."""
        with pytest.raises(ValidationError, match="nodes"):
            PipelineDefinition(name="test", nodes=123)


class TestPipelineDefinitionEdgeCoercion:
    """Tests for edges field coercion in PipelineDefinition."""

    def test_coerce_dict_to_pipeline_edge(self) -> None:
        """Test that raw dicts are coerced to PipelineEdge objects."""
        pipeline = PipelineDefinition(
            name="test",
            edges=[
                {
                    "id": "edge-1",
                    "source": "node-1",
                    "target": "node-2",
                }
            ],
        )
        assert len(pipeline.edges) == 1
        assert isinstance(pipeline.edges[0], PipelineEdge)
        assert pipeline.edges[0].id == "edge-1"
        assert pipeline.edges[0].source == "node-1"
        assert pipeline.edges[0].target == "node-2"

    def test_passthrough_typed_pipeline_edge(self) -> None:
        """Test that typed PipelineEdge objects pass through unchanged."""
        edge = PipelineEdge(
            id="edge-1",
            source="node-1",
            target="node-2",
        )
        pipeline = PipelineDefinition(name="test", edges=[edge])
        assert pipeline.edges[0] is edge

    def test_mixed_dict_and_typed_edges(self) -> None:
        """Test mixing typed edges and dicts in the same list."""
        typed_edge = PipelineEdge(
            id="typed-edge",
            source="a",
            target="b",
        )
        pipeline = PipelineDefinition(
            name="test",
            edges=[
                typed_edge,
                {
                    "id": "dict-edge",
                    "source": "c",
                    "target": "d",
                },
            ],
        )
        assert len(pipeline.edges) == 2
        assert pipeline.edges[0] is typed_edge
        assert isinstance(pipeline.edges[1], PipelineEdge)
        assert pipeline.edges[1].id == "dict-edge"

    def test_none_edges_coerced_to_empty_list(self) -> None:
        """Test that None for edges is coerced to empty list."""
        pipeline = PipelineDefinition(name="test", edges=None)
        assert pipeline.edges == []

    def test_empty_list_edges_accepted(self) -> None:
        """Test that empty list for edges is accepted as-is."""
        pipeline = PipelineDefinition(name="test", edges=[])
        assert pipeline.edges == []

    def test_reject_string_edges(self) -> None:
        """Test that string input for edges raises ValidationError."""
        with pytest.raises(ValidationError, match="edges"):
            PipelineDefinition(name="test", edges="invalid")

    def test_reject_dict_edges(self) -> None:
        """Test that single dict (not list) for edges raises ValidationError."""
        with pytest.raises(ValidationError, match="edges"):
            PipelineDefinition(
                name="test",
                edges={"id": "edge-1", "source": "a", "target": "b"},
            )

    def test_reject_integer_edges(self) -> None:
        """Test that integer input for edges raises ValidationError."""
        with pytest.raises(ValidationError, match="edges"):
            PipelineDefinition(name="test", edges=456)


class TestPipelineDefinitionComplete:
    """Tests for complete PipelineDefinition with nodes and edges."""

    def test_complete_pipeline_with_coercion(self) -> None:
        """Test a complete pipeline with dict coercion for both nodes and edges."""
        pipeline = PipelineDefinition(
            name="complete-test",
            nodes=[
                {
                    "id": "input",
                    "type": "source",
                    "label": "Input",
                    "position": {"x": 0.0, "y": 0.0},
                },
                {
                    "id": "output",
                    "type": "sink",
                    "label": "Output",
                    "position": {"x": 200.0, "y": 0.0},
                },
            ],
            edges=[
                {
                    "id": "e1",
                    "source": "input",
                    "target": "output",
                }
            ],
            metadata={"version": "1.0"},
        )
        assert pipeline.name == "complete-test"
        assert len(pipeline.nodes) == 2
        assert len(pipeline.edges) == 1
        assert all(isinstance(n, PipelineNode) for n in pipeline.nodes)
        assert all(isinstance(e, PipelineEdge) for e in pipeline.edges)
        assert pipeline.metadata == {"version": "1.0"}

    def test_empty_pipeline(self) -> None:
        """Test that a pipeline with all defaults is valid."""
        pipeline = PipelineDefinition()
        assert pipeline.name == ""
        assert pipeline.nodes == []
        assert pipeline.edges == []
        assert pipeline.metadata == {}
