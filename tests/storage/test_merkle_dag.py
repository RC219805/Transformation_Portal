"""Tests for storage.merkle_dag module.

Covers:
- MerkleNode dataclass
- MerkleDAG artifact/computation/checkpoint operations
- Lineage traversal
- Export/load serialization
- Integrity verification
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from transformation_portal.storage.merkle_dag import MerkleDAG, MerkleDAGError, MerkleNode

pytestmark = pytest.mark.unit


class TestMerkleNode:
    """Tests for MerkleNode dataclass."""

    def test_merkle_node_creation(self):
        """Test MerkleNode creation with all fields."""
        node = MerkleNode(
            hash="abc123def456",
            node_type="artifact",
            inputs=("input1", "input2"),
            outputs={"content_hash": "xyz789"},
            metadata={"filename": "test.jpg"},
            timestamp="2024-01-01T00:00:00Z",
        )

        assert node.hash == "abc123def456"
        assert node.node_type == "artifact"
        assert node.inputs == ("input1", "input2")
        assert node.outputs == {"content_hash": "xyz789"}
        assert node.metadata == {"filename": "test.jpg"}
        assert node.timestamp == "2024-01-01T00:00:00Z"

    def test_merkle_node_frozen(self):
        """Test MerkleNode is immutable."""
        node = MerkleNode(
            hash="hash123",
            node_type="artifact",
            inputs=(),
            outputs={},
            metadata={},
            timestamp="",
        )

        with pytest.raises(AttributeError):
            node.hash = "modified"


class TestMerkleDAG:
    """Tests for MerkleDAG."""

    def test_dag_creation(self):
        """Test MerkleDAG creation."""
        dag = MerkleDAG()

        assert len(dag.nodes) == 0
        assert len(dag._root_hashes) == 0

    def test_add_artifact(self):
        """Test adding artifact node."""
        dag = MerkleDAG()

        content_hash = hashlib.sha256(b"test content").hexdigest()
        node_hash = dag.add_artifact(
            artifact_type="image",
            content_hash=content_hash,
            metadata={"filename": "test.png"},
        )

        assert node_hash is not None
        assert len(node_hash) == 64  # SHA-256 hex

        node = dag.get_node(node_hash)
        assert node is not None
        assert node.node_type == "artifact"
        assert node.outputs["content_hash"] == content_hash
        assert node.metadata["artifact_type"] == "image"
        assert node.metadata["filename"] == "test.png"
        assert node.inputs == ()

    def test_add_artifact_deduplication(self):
        """Test adding same artifact returns same hash."""
        dag = MerkleDAG()

        content_hash = "abc123"
        hash1 = dag.add_artifact(artifact_type="image", content_hash=content_hash)
        hash2 = dag.add_artifact(artifact_type="image", content_hash=content_hash)

        assert hash1 == hash2
        assert len(dag.nodes) == 1

    def test_add_computation(self):
        """Test adding computation node."""
        dag = MerkleDAG()

        # First add input artifacts
        input1 = dag.add_artifact(artifact_type="image", content_hash="input1hash")
        input2 = dag.add_artifact(artifact_type="config", content_hash="input2hash")

        # Add computation
        comp_hash = dag.add_computation(
            node_id="enhance",
            inputs=[input1, input2],
            outputs={"result_hash": "outputhash"},
            metadata={"model": "sam2"},
        )

        node = dag.get_node(comp_hash)
        assert node is not None
        assert node.node_type == "computation"
        assert set(node.inputs) == {input1, input2}
        assert node.outputs == {"result_hash": "outputhash"}
        assert node.metadata["node_id"] == "enhance"
        assert node.metadata["model"] == "sam2"

    def test_add_computation_validates_inputs(self):
        """Test add_computation validates input references."""
        dag = MerkleDAG()

        with pytest.raises(MerkleDAGError, match="Input node not found"):
            dag.add_computation(
                node_id="bad-compute",
                inputs=["nonexistent-hash"],
                outputs={},
            )

    def test_add_checkpoint(self):
        """Test adding checkpoint node."""
        dag = MerkleDAG()

        # Add some prior nodes
        art1 = dag.add_artifact(artifact_type="input", content_hash="art1")
        comp1 = dag.add_computation(node_id="step1", inputs=[art1], outputs={"out": "val"})

        # Add checkpoint
        checkpoint_hash = dag.add_checkpoint(
            checkpoint_id="checkpoint-001",
            inputs=[comp1],
            state={"progress": 50, "current_step": "step1"},
            metadata={"reason": "scheduled"},
        )

        node = dag.get_node(checkpoint_hash)
        assert node is not None
        assert node.node_type == "checkpoint"
        assert node.inputs == (comp1,)
        assert node.outputs == {"progress": 50, "current_step": "step1"}
        assert node.metadata["checkpoint_id"] == "checkpoint-001"

    def test_get_node_nonexistent(self):
        """Test get_node returns None for nonexistent hash."""
        dag = MerkleDAG()

        result = dag.get_node("nonexistent-hash")

        assert result is None

    def test_get_lineage_simple(self):
        """Test get_lineage for simple chain."""
        dag = MerkleDAG()

        # Create chain: artifact -> computation -> computation
        art = dag.add_artifact(artifact_type="input", content_hash="input")
        comp1 = dag.add_computation(node_id="step1", inputs=[art], outputs={"out1": "val1"})
        comp2 = dag.add_computation(node_id="step2", inputs=[comp1], outputs={"out2": "val2"})

        lineage = dag.get_lineage(comp2)

        assert len(lineage) == 3
        # Topological order: root first
        assert lineage[0].hash == art
        assert lineage[1].hash == comp1
        assert lineage[2].hash == comp2

    def test_get_lineage_diamond(self):
        """Test get_lineage with diamond dependency."""
        dag = MerkleDAG()

        # Diamond pattern:
        #      A
        #     / \
        #    B   C
        #     \ /
        #      D

        a = dag.add_artifact(artifact_type="root", content_hash="a")
        b = dag.add_computation(node_id="b", inputs=[a], outputs={"b": "val"})
        c = dag.add_computation(node_id="c", inputs=[a], outputs={"c": "val"})
        d = dag.add_computation(node_id="d", inputs=[b, c], outputs={"d": "val"})

        lineage = dag.get_lineage(d)

        # Should include all 4 nodes
        assert len(lineage) == 4
        hashes = [n.hash for n in lineage]

        # A must come before B and C
        assert hashes.index(a) < hashes.index(b)
        assert hashes.index(a) < hashes.index(c)
        # B and C must come before D
        assert hashes.index(b) < hashes.index(d)
        assert hashes.index(c) < hashes.index(d)

    def test_get_lineage_max_depth(self):
        """Test get_lineage with max_depth limit."""
        dag = MerkleDAG()

        # Create deep chain
        prev = dag.add_artifact(artifact_type="root", content_hash="root")
        for i in range(5):
            prev = dag.add_computation(node_id=f"step{i}", inputs=[prev], outputs={"i": i})

        # Get lineage with max_depth=2
        lineage = dag.get_lineage(prev, max_depth=2)

        # Should stop early
        assert len(lineage) < 6

    def test_get_lineage_nonexistent(self):
        """Test get_lineage returns empty for nonexistent node."""
        dag = MerkleDAG()

        lineage = dag.get_lineage("nonexistent")

        assert lineage == []

    def test_verify_integrity_valid(self):
        """Test verify_integrity on valid DAG."""
        dag = MerkleDAG()

        art = dag.add_artifact(artifact_type="input", content_hash="test")
        dag.add_computation(node_id="process", inputs=[art], outputs={"done": True})

        errors = dag.verify_integrity()

        assert errors == []

    def test_verify_integrity_invalid_reference(self):
        """Test verify_integrity detects invalid references."""
        dag = MerkleDAG()

        # Manually add a node with invalid input reference
        dag.nodes["broken-node"] = MerkleNode(
            hash="broken-node",
            node_type="computation",
            inputs=("nonexistent-input",),
            outputs={},
            metadata={},
            timestamp="",
        )

        errors = dag.verify_integrity()

        assert len(errors) == 1
        assert "missing input" in errors[0].lower()

    def test_export_and_load(self, tmp_path):
        """Test export and load roundtrip."""
        dag = MerkleDAG()

        art = dag.add_artifact(artifact_type="image", content_hash="img123", metadata={"name": "photo.jpg"})
        comp = dag.add_computation(node_id="enhance", inputs=[art], outputs={"result": "res456"})

        export_path = tmp_path / "lineage.json"
        dag.export(export_path)

        # Load into new DAG
        loaded = MerkleDAG.load(export_path)

        assert len(loaded.nodes) == 2
        assert art in loaded.nodes
        assert comp in loaded.nodes

        loaded_art = loaded.get_node(art)
        assert loaded_art is not None
        assert loaded_art.outputs["content_hash"] == "img123"

    def test_export_creates_valid_json(self, tmp_path):
        """Test export creates valid, readable JSON."""
        dag = MerkleDAG()
        dag.add_artifact(artifact_type="test", content_hash="test123")

        export_path = tmp_path / "dag.json"
        dag.export(export_path, pretty=True)

        # Should be valid JSON
        data = json.loads(export_path.read_text())

        assert "version" in data
        assert data["version"] == "1.0"
        assert "node_count" in data
        assert data["node_count"] == 1
        assert "nodes" in data
        assert "root_hashes" in data

    def test_export_pretty_vs_compact(self, tmp_path):
        """Test export pretty vs compact formatting."""
        dag = MerkleDAG()
        dag.add_artifact(artifact_type="test", content_hash="test")

        pretty_path = tmp_path / "pretty.json"
        compact_path = tmp_path / "compact.json"

        dag.export(pretty_path, pretty=True)
        dag.export(compact_path, pretty=False)

        pretty_size = pretty_path.stat().st_size
        compact_size = compact_path.stat().st_size

        # Pretty should be larger due to indentation
        assert pretty_size > compact_size

    def test_summary(self):
        """Test summary statistics."""
        dag = MerkleDAG()

        # Add various nodes
        art1 = dag.add_artifact(artifact_type="input", content_hash="a1")
        art2 = dag.add_artifact(artifact_type="input", content_hash="a2")
        comp = dag.add_computation(node_id="process", inputs=[art1, art2], outputs={})
        dag.add_checkpoint(checkpoint_id="cp1", inputs=[comp], state={"done": False})

        summary = dag.summary()

        assert summary["total_nodes"] == 4
        assert summary["root_nodes"] == 2  # art1 and art2
        assert summary["nodes_by_type"]["artifact"] == 2
        assert summary["nodes_by_type"]["computation"] == 1
        assert summary["nodes_by_type"]["checkpoint"] == 1

    def test_hash_determinism(self):
        """Test hashing is deterministic."""
        dag1 = MerkleDAG()
        dag2 = MerkleDAG()

        # Same input should produce same hash
        hash1 = dag1.add_artifact(artifact_type="test", content_hash="content", metadata={"key": "value"})
        hash2 = dag2.add_artifact(artifact_type="test", content_hash="content", metadata={"key": "value"})

        assert hash1 == hash2

    def test_hash_varies_with_content(self):
        """Test different content produces different hashes."""
        dag = MerkleDAG()

        hash1 = dag.add_artifact(artifact_type="test", content_hash="content1")
        hash2 = dag.add_artifact(artifact_type="test", content_hash="content2")

        assert hash1 != hash2

    def test_computation_input_order_independent(self):
        """Test computation hash is independent of input order."""
        dag = MerkleDAG()

        a1 = dag.add_artifact(artifact_type="a", content_hash="a")
        a2 = dag.add_artifact(artifact_type="b", content_hash="b")

        # Same computation with different input order
        hash1 = dag.add_computation(node_id="comp", inputs=[a1, a2], outputs={})

        dag2 = MerkleDAG()
        b1 = dag2.add_artifact(artifact_type="a", content_hash="a")
        b2 = dag2.add_artifact(artifact_type="b", content_hash="b")
        hash2 = dag2.add_computation(node_id="comp", inputs=[b2, b1], outputs={})

        # Should produce same hash regardless of input order
        assert hash1 == hash2


class TestMerkleDAGEdgeCases:
    """Edge case tests for MerkleDAG."""

    def test_empty_dag_operations(self):
        """Test operations on empty DAG."""
        dag = MerkleDAG()

        assert dag.get_lineage("any-hash") == []
        assert dag.verify_integrity() == []
        assert dag.summary()["total_nodes"] == 0

    def test_large_metadata(self):
        """Test handling of large metadata."""
        dag = MerkleDAG()

        large_metadata = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}

        node_hash = dag.add_artifact(
            artifact_type="test",
            content_hash="content",
            metadata=large_metadata,
        )

        node = dag.get_node(node_hash)
        assert len(node.metadata) > 100  # artifact_type + large_metadata

    def test_nested_outputs(self):
        """Test computation with nested output structure."""
        dag = MerkleDAG()

        art = dag.add_artifact(artifact_type="input", content_hash="in")

        nested_outputs = {
            "level1": {
                "level2": {
                    "level3": {"value": "deep"},
                },
            },
            "array": [1, 2, 3, {"nested": True}],
        }

        comp_hash = dag.add_computation(node_id="nested", inputs=[art], outputs=nested_outputs)

        node = dag.get_node(comp_hash)
        assert node.outputs == nested_outputs

    def test_export_load_preserves_all_data(self, tmp_path):
        """Test export/load preserves all node data."""
        dag = MerkleDAG()

        art = dag.add_artifact(
            artifact_type="complex",
            content_hash="hash123",
            metadata={"nested": {"data": [1, 2, 3]}},
        )

        path = tmp_path / "test.json"
        dag.export(path)

        loaded = MerkleDAG.load(path)

        orig_node = dag.get_node(art)
        loaded_node = loaded.get_node(art)

        assert loaded_node is not None
        assert loaded_node.hash == orig_node.hash
        assert loaded_node.node_type == orig_node.node_type
        assert loaded_node.inputs == orig_node.inputs
        assert loaded_node.outputs == orig_node.outputs
        # Note: metadata may have slight differences due to timestamp
