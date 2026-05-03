"""Tests for ComfyUI integration module.

Tests custom node registry, node base classes, workflow builder,
and workflow serialization/deserialization.

Note: The comfyui module has heavy dependencies (torch, cv2, atmosphere)
that are not available in the core test environment. Tests will skip
when these dependencies are not available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Check if all comfyui dependencies are available
_comfyui_available = False
_skip_reason = "comfyui dependencies not available"

try:
    # The comfyui module imports torch, cv2, and atmosphere components
    from transformation_portal.comfyui import custom_nodes, workflow_builder

    _comfyui_available = True
except (ImportError, OSError, RuntimeError) as e:
    # Catch broader exceptions: ImportError for missing deps, OSError/RuntimeError
    # for broken wheels or missing shared libraries (consistent with conftest.py patterns)
    _skip_reason = f"comfyui module not importable: {e}"

# Pytest markers - ML marker required because comfyui has heavy deps
pytestmark = [
    pytest.mark.unit,
    pytest.mark.ml,
    pytest.mark.skipif(not _comfyui_available, reason=_skip_reason),
]


class TestCustomNodeRegistry:
    """Tests for CustomNodeRegistry."""

    def test_registry_starts_with_registered_nodes(self):
        """Registry should have pre-registered nodes after import."""
        from transformation_portal.comfyui.custom_nodes import CustomNodeRegistry

        nodes = CustomNodeRegistry.list_nodes()
        # At minimum, we have FluxEnhancementNode, SkyGANNode, SceneAnalysisNode
        assert len(nodes) >= 3

    def test_get_node_returns_registered_node(self):
        """get_node should return class for registered nodes."""
        from transformation_portal.comfyui.custom_nodes import (
            CustomNodeRegistry,
            FluxEnhancementNode,
        )

        node_class = CustomNodeRegistry.get_node("FluxEnhancementNode")
        assert node_class is FluxEnhancementNode

    def test_get_node_returns_none_for_unknown(self):
        """get_node should return None for unregistered nodes."""
        from transformation_portal.comfyui.custom_nodes import CustomNodeRegistry

        node_class = CustomNodeRegistry.get_node("NonExistentNode")
        assert node_class is None

    def test_list_nodes_returns_list(self):
        """list_nodes should return a list of strings."""
        from transformation_portal.comfyui.custom_nodes import CustomNodeRegistry

        nodes = CustomNodeRegistry.list_nodes()
        assert isinstance(nodes, list)
        assert all(isinstance(name, str) for name in nodes)

    def test_register_decorator_adds_to_registry(self):
        """@register decorator should add node to registry."""
        from transformation_portal.comfyui.custom_nodes import BaseNode, CustomNodeRegistry

        # Define a test node
        @CustomNodeRegistry.register
        class TestRegisteredNode(BaseNode):
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}

            @classmethod
            def RETURN_TYPES(cls):
                return ()

            def execute(self):
                return ()

        # Should now be in registry
        assert CustomNodeRegistry.get_node("TestRegisteredNode") is TestRegisteredNode

        # Clean up - remove from registry to avoid test pollution
        # Note: Accessing _nodes directly is intentional for test cleanup since
        # CustomNodeRegistry doesn't expose an unregister() method. This is
        # acceptable in test code to prevent state leakage between tests.
        CustomNodeRegistry._nodes.pop("TestRegisteredNode", None)


class TestBaseNode:
    """Tests for BaseNode abstract base class."""

    def test_base_node_has_category(self):
        """BaseNode should have CATEGORY attribute."""
        from transformation_portal.comfyui.custom_nodes import BaseNode

        assert hasattr(BaseNode, "CATEGORY")
        assert BaseNode.CATEGORY == "Transformation Portal"

    def test_base_node_abstract_methods(self):
        """BaseNode should require INPUT_TYPES, RETURN_TYPES, execute."""
        from transformation_portal.comfyui.custom_nodes import BaseNode

        assert hasattr(BaseNode, "INPUT_TYPES")
        assert hasattr(BaseNode, "RETURN_TYPES")
        assert hasattr(BaseNode, "execute")

        assert getattr(BaseNode.INPUT_TYPES, "__isabstractmethod__", False)
        assert getattr(BaseNode.RETURN_TYPES, "__isabstractmethod__", False)
        assert getattr(BaseNode.execute, "__isabstractmethod__", False)

        with pytest.raises(TypeError, match="abstract"):
            BaseNode()  # type: ignore[abstract]

    def test_concrete_node_implements_interface(self):
        """Concrete nodes should implement required methods."""
        from transformation_portal.comfyui.custom_nodes import FluxEnhancementNode

        # Should have INPUT_TYPES
        inputs = FluxEnhancementNode.INPUT_TYPES()
        assert isinstance(inputs, dict)
        assert "required" in inputs or "optional" in inputs

        # Should have RETURN_TYPES
        returns = FluxEnhancementNode.RETURN_TYPES()
        assert isinstance(returns, tuple)

    def test_to_numpy_helper_with_tensor(self):
        """_to_numpy should convert torch tensor to numpy."""
        import torch

        from transformation_portal.comfyui.custom_nodes import BaseNode

        # Create a concrete instance for testing helpers
        class TestNode(BaseNode):
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}

            @classmethod
            def RETURN_TYPES(cls):
                return ()

            def execute(self):
                return ()

        node = TestNode()

        # Create (B,H,W,C) tensor as ComfyUI passes
        tensor = torch.zeros(1, 64, 64, 3)
        result = node._to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64, 3)

    def test_to_numpy_helper_with_array(self):
        """_to_numpy should pass through numpy arrays."""
        from transformation_portal.comfyui.custom_nodes import BaseNode

        class TestNode(BaseNode):
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}

            @classmethod
            def RETURN_TYPES(cls):
                return ()

            def execute(self):
                return ()

        node = TestNode()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        result = node._to_numpy(arr)

        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64, 3)

    def test_to_tensor_helper(self):
        """_to_tensor should convert numpy to torch tensor with batch dim."""
        import torch

        from transformation_portal.comfyui.custom_nodes import BaseNode

        class TestNode(BaseNode):
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}

            @classmethod
            def RETURN_TYPES(cls):
                return ()

            def execute(self):
                return ()

        node = TestNode()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        result = node._to_tensor(arr)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 64, 64, 3)  # Batch dim added
        assert result.dtype == torch.float32


class TestFluxEnhancementNode:
    """Tests for FluxEnhancementNode."""

    def test_input_types_structure(self):
        """INPUT_TYPES should have required and optional fields."""
        from transformation_portal.comfyui.custom_nodes import FluxEnhancementNode

        inputs = FluxEnhancementNode.INPUT_TYPES()

        assert "required" in inputs
        assert "optional" in inputs

        # Required inputs
        required = inputs["required"]
        assert "image" in required
        assert "strength" in required
        assert "num_steps" in required
        assert "guidance_scale" in required
        assert "variant" in required

        # Optional inputs
        optional = inputs["optional"]
        assert "prompt" in optional
        assert "seed" in optional

    def test_return_types_is_image(self):
        """RETURN_TYPES should return IMAGE tuple."""
        from transformation_portal.comfyui.custom_nodes import FluxEnhancementNode

        returns = FluxEnhancementNode.RETURN_TYPES()
        assert returns == ("IMAGE",)

    def test_category(self):
        """FluxEnhancementNode should have Enhancement category."""
        from transformation_portal.comfyui.custom_nodes import FluxEnhancementNode

        assert FluxEnhancementNode.CATEGORY == "Transformation Portal/Enhancement"

    def test_execute_rejects_unknown_variant_before_image_conversion(self, monkeypatch):
        """Unknown variant must fail before image/pipeline setup."""
        from transformation_portal.comfyui.custom_nodes import FluxEnhancementNode

        node = FluxEnhancementNode()

        def _conversion_must_not_run(_image):
            raise RuntimeError("image conversion should not run before variant validation")

        monkeypatch.setattr(node, "_to_numpy", _conversion_must_not_run)

        with pytest.raises(ValueError, match="Unknown variant") as exc_info:
            node.execute(
                image=np.zeros((2, 2, 3), dtype=np.float32),
                strength=0.45,
                num_steps=4,
                guidance_scale=3.5,
                variant="not_a_real_variant",
            )

        assert "dev" in str(exc_info.value)
        assert "schnell" in str(exc_info.value)


class TestSkyGANNode:
    """Tests for SkyGANNode."""

    def test_input_types_structure(self):
        """INPUT_TYPES should have required atmospheric parameters."""
        from transformation_portal.comfyui.custom_nodes import SkyGANNode

        inputs = SkyGANNode.INPUT_TYPES()

        assert "required" in inputs
        required = inputs["required"]

        # Required atmospheric parameters
        assert "image" in required
        assert "location" in required
        assert "season" in required
        assert "time_of_day" in required
        assert "cloud_coverage" in required

        # Physics guardrail controls
        assert "auto_correct" in required
        assert "strict_physics" in required

    def test_return_types(self):
        """SkyGANNode should return IMAGE, IMAGE (mask), STRING (report)."""
        from transformation_portal.comfyui.custom_nodes import SkyGANNode

        returns = SkyGANNode.RETURN_TYPES()
        assert returns == ("IMAGE", "IMAGE", "STRING")

    def test_category(self):
        """SkyGANNode should have Atmospheric category."""
        from transformation_portal.comfyui.custom_nodes import SkyGANNode

        assert SkyGANNode.CATEGORY == "Transformation Portal/Atmospheric"

    def test_time_of_day_mapping_covers_dropdown_choices(self):
        """Dropdown choices and mapping stay aligned in order, length, and uniqueness."""
        from transformation_portal.comfyui.custom_nodes import SkyGANNode

        choices = SkyGANNode.INPUT_TYPES()["required"]["time_of_day"][0]
        mapping = SkyGANNode._TIME_OF_DAY_HOURS

        # Exact sequence equality also catches reordering and duplicate entries
        # in the dropdown that a set comparison would silently mask.
        assert list(choices) == list(mapping.keys())
        assert len(choices) == len(set(choices))

        for label, hour in mapping.items():
            assert 0.0 <= hour < 24.0, f"{label} -> {hour} out of range"

        # Sanity: ordering matches a normal day progression
        assert (
            mapping["sunrise"]
            < mapping["morning"]
            < mapping["midday"]
            < mapping["golden_hour"]
            < mapping["sunset"]
            < mapping["twilight"]
        )

    def test_execute_rejects_unknown_time_of_day(self, monkeypatch):
        """Unknown time_of_day must fail fast before any preset work."""
        from transformation_portal.comfyui.custom_nodes import SkyGANNode

        # Booby-trap LocationPresets at the import site so that any preset
        # construction trips a RuntimeError. This enforces the ordering
        # guarantee — LocationPresets falls back silently to "montecito" for
        # unknown locations, so we cannot rely on a bad location to surface
        # a preset-side error if validation regressed.
        def _no_presets(*args, **kwargs):
            raise RuntimeError("LocationPresets must not be constructed before time_of_day validation")

        monkeypatch.setattr("transformation_portal.comfyui.custom_nodes.LocationPresets", _no_presets)

        node = SkyGANNode()
        dummy_image = np.zeros((4, 4, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="Unknown time_of_day") as exc_info:
            node.execute(
                image=dummy_image,
                location="montecito",
                season="summer",
                time_of_day="not_a_real_slot",
                cloud_coverage=0.3,
                auto_correct=True,
                strict_physics=False,
            )

        # Error message preserves dropdown order (insertion order), not
        # alphabetical order, so it matches the UI for easier debugging.
        expected_choices = list(SkyGANNode._TIME_OF_DAY_HOURS)
        assert str(expected_choices) in str(exc_info.value)


class TestSceneAnalysisNode:
    """Tests for SceneAnalysisNode."""

    def test_input_types_structure(self):
        """INPUT_TYPES should have image and detailed flag."""
        from transformation_portal.comfyui.custom_nodes import SceneAnalysisNode

        inputs = SceneAnalysisNode.INPUT_TYPES()

        assert "required" in inputs
        required = inputs["required"]
        assert "image" in required
        assert "detailed" in required

    def test_return_types(self):
        """SceneAnalysisNode should return STRING (JSON)."""
        from transformation_portal.comfyui.custom_nodes import SceneAnalysisNode

        returns = SceneAnalysisNode.RETURN_TYPES()
        assert returns == ("STRING",)

    def test_category(self):
        """SceneAnalysisNode should have Analysis category."""
        from transformation_portal.comfyui.custom_nodes import SceneAnalysisNode

        assert SceneAnalysisNode.CATEGORY == "Transformation Portal/Analysis"


class TestNodeType:
    """Tests for NodeType enum."""

    def test_node_type_values(self):
        """NodeType values should match ComfyUI class names."""
        from transformation_portal.comfyui.workflow_builder import NodeType

        # Verify key mappings
        assert NodeType.INPUT.value == "LoadImage"
        assert NodeType.OUTPUT.value == "SaveImage"
        assert NodeType.FLUX_ENHANCEMENT.value == "FluxEnhancementNode"
        assert NodeType.SKYGAN_SKY.value == "SkyGANNode"
        assert NodeType.SCENE_ANALYSIS.value == "SceneAnalysisNode"

    def test_all_node_types_have_string_values(self):
        """All NodeType values should be strings."""
        from transformation_portal.comfyui.workflow_builder import NodeType

        for node_type in NodeType:
            assert isinstance(node_type.value, str)


class TestNodeConnection:
    """Tests for NodeConnection dataclass."""

    def test_connection_creation(self):
        """Should create connection with required fields."""
        from transformation_portal.comfyui.workflow_builder import NodeConnection

        conn = NodeConnection(
            source_node_id="node_1",
            source_output="IMAGE",
            target_node_id="node_2",
            target_input="image",
        )

        assert conn.source_node_id == "node_1"
        assert conn.source_output == "IMAGE"
        assert conn.target_node_id == "node_2"
        assert conn.target_input == "image"

    def test_to_comfyui_format_image(self):
        """IMAGE output should map to slot 0."""
        from transformation_portal.comfyui.workflow_builder import NodeConnection

        conn = NodeConnection(
            source_node_id="node_1",
            source_output="IMAGE",
            target_node_id="node_2",
            target_input="image",
        )

        result = conn.to_comfyui_format()
        assert result == ["node_1", 0]

    def test_to_comfyui_format_mask(self):
        """MASK output should map to slot 1."""
        from transformation_portal.comfyui.workflow_builder import NodeConnection

        conn = NodeConnection(
            source_node_id="node_1",
            source_output="MASK",
            target_node_id="node_2",
            target_input="mask",
        )

        result = conn.to_comfyui_format()
        assert result == ["node_1", 1]


class TestNode:
    """Tests for Node dataclass."""

    def test_node_creation(self):
        """Should create node with required fields."""
        from transformation_portal.comfyui.workflow_builder import Node, NodeType

        node = Node(
            node_id="test_1",
            node_type=NodeType.FLUX_ENHANCEMENT,
            parameters={"strength": 0.5},
        )

        assert node.node_id == "test_1"
        assert node.node_type == NodeType.FLUX_ENHANCEMENT
        assert node.parameters["strength"] == 0.5

    def test_to_comfyui_format(self):
        """Node should convert to ComfyUI JSON format."""
        from transformation_portal.comfyui.workflow_builder import Node, NodeType

        node = Node(
            node_id="test_1",
            node_type=NodeType.FLUX_ENHANCEMENT,
            parameters={"strength": 0.5, "num_steps": 4},
        )

        result = node.to_comfyui_format()

        assert result["class_type"] == "FluxEnhancementNode"
        assert result["inputs"]["strength"] == 0.5
        assert result["inputs"]["num_steps"] == 4
        assert "_meta" in result


class TestWorkflow:
    """Tests for Workflow dataclass."""

    def test_empty_workflow(self):
        """Empty workflow should have no nodes or connections."""
        from transformation_portal.comfyui.workflow_builder import Workflow

        wf = Workflow()
        assert len(wf.nodes) == 0
        assert len(wf.connections) == 0

    def test_to_comfyui_format(self):
        """Workflow should convert to ComfyUI JSON format."""
        from transformation_portal.comfyui.workflow_builder import Node, NodeType, Workflow

        wf = Workflow()
        wf.nodes["node_1"] = Node(
            node_id="node_1",
            node_type=NodeType.INPUT,
            parameters={"image": "test.jpg"},
        )

        result = wf.to_comfyui_format()

        assert "node_1" in result
        assert result["node_1"]["class_type"] == "LoadImage"

    def test_save_and_load(self, tmp_path):
        """Workflow should save and load from JSON file."""
        from transformation_portal.comfyui.workflow_builder import Node, NodeType, Workflow

        # Create workflow
        wf = Workflow()
        wf.metadata = {"name": "Test Workflow"}
        wf.nodes["node_1"] = Node(
            node_id="node_1",
            node_type=NodeType.INPUT,
            parameters={"image": "test.jpg"},
        )

        # Save
        save_path = tmp_path / "test_workflow.json"
        wf.save(save_path)

        assert save_path.exists()

        # Load
        loaded = Workflow.load(save_path)
        assert loaded.metadata["name"] == "Test Workflow"
        assert "node_1" in loaded.nodes


class TestWorkflowBuilder:
    """Tests for WorkflowBuilder fluent API."""

    def test_builder_creates_empty_workflow(self):
        """New builder should create empty workflow with metadata."""
        from transformation_portal.comfyui.workflow_builder import WorkflowBuilder

        builder = WorkflowBuilder(name="Test Workflow")
        assert len(builder.workflow.nodes) == 0
        assert builder.workflow.metadata["name"] == "Test Workflow"

    def test_add_input(self):
        """add_input should add LoadImage node."""
        from transformation_portal.comfyui.workflow_builder import NodeType, WorkflowBuilder

        builder = WorkflowBuilder()
        builder.add_input("test.jpg")

        assert len(builder.workflow.nodes) == 1
        node = list(builder.workflow.nodes.values())[0]
        assert node.node_type == NodeType.INPUT
        assert node.parameters["image"] == "test.jpg"

    def test_add_flux_enhancement(self):
        """add_flux_enhancement should add FluxEnhancementNode."""
        from transformation_portal.comfyui.workflow_builder import NodeType, WorkflowBuilder

        builder = WorkflowBuilder()
        builder.add_input("test.jpg")
        builder.add_flux_enhancement(strength=0.6, num_steps=8)

        # Should have 2 nodes
        assert len(builder.workflow.nodes) == 2

        # Check FLUX node parameters
        nodes = list(builder.workflow.nodes.values())
        flux_node = next(n for n in nodes if n.node_type == NodeType.FLUX_ENHANCEMENT)
        assert flux_node.parameters["strength"] == 0.6
        assert flux_node.parameters["num_steps"] == 8

    def test_add_skygan_sky(self):
        """add_skygan_sky should add SkyGANNode with physics controls."""
        from transformation_portal.comfyui.workflow_builder import NodeType, WorkflowBuilder

        builder = WorkflowBuilder()
        builder.add_input("test.jpg")
        builder.add_skygan_sky(
            location="montecito",
            time_of_day="golden_hour",
            auto_correct=True,
            strict_physics=False,
        )

        # Check SkyGAN node parameters
        nodes = list(builder.workflow.nodes.values())
        skygan_node = next(n for n in nodes if n.node_type == NodeType.SKYGAN_SKY)
        assert skygan_node.parameters["location"] == "montecito"
        assert skygan_node.parameters["time_of_day"] == "golden_hour"
        assert skygan_node.parameters["auto_correct"] is True
        assert skygan_node.parameters["strict_physics"] is False

    def test_add_output(self):
        """add_output should add SaveImage node."""
        from transformation_portal.comfyui.workflow_builder import NodeType, WorkflowBuilder

        builder = WorkflowBuilder()
        builder.add_input("test.jpg")
        builder.add_output("output.jpg", format="jpg", quality=90)

        # Check output node
        nodes = list(builder.workflow.nodes.values())
        output_node = next(n for n in nodes if n.node_type == NodeType.OUTPUT)
        assert output_node.parameters["filename"] == "output.jpg"
        assert output_node.parameters["format"] == "jpg"
        assert output_node.parameters["quality"] == 90

    def test_fluent_api_chaining(self):
        """Builder should support method chaining."""
        from transformation_portal.comfyui.workflow_builder import WorkflowBuilder

        builder = WorkflowBuilder()

        # Should be able to chain all methods
        result = (
            builder.add_input("test.jpg")
            .add_flux_enhancement(strength=0.5)
            .add_skygan_sky(location="montecito")
            .add_output("output.jpg")
        )

        # Chain should return builder
        assert result is builder
        assert len(builder.workflow.nodes) == 4

    def test_build_returns_workflow(self):
        """build() should return completed Workflow."""
        from transformation_portal.comfyui.workflow_builder import Workflow, WorkflowBuilder

        builder = WorkflowBuilder()
        builder.add_input("test.jpg")

        result = builder.build()

        assert isinstance(result, Workflow)
        assert len(result.nodes) == 1

    def test_connections_created_between_nodes(self):
        """Builder should create connections between consecutive nodes."""
        from transformation_portal.comfyui.workflow_builder import WorkflowBuilder

        builder = WorkflowBuilder()
        builder.add_input("test.jpg")
        builder.add_flux_enhancement(strength=0.5)

        # Should have 1 connection
        assert len(builder.workflow.connections) == 1

        conn = builder.workflow.connections[0]
        # Connection should link input to flux
        assert "loadimage" in conn.source_node_id.lower()
        assert "flux" in conn.target_node_id.lower()

    def test_repr(self):
        """WorkflowBuilder __repr__ should be informative."""
        from transformation_portal.comfyui.workflow_builder import WorkflowBuilder

        builder = WorkflowBuilder()
        builder.add_input("test.jpg")
        builder.add_flux_enhancement()

        repr_str = repr(builder)
        assert "WorkflowBuilder" in repr_str
        assert "2" in repr_str  # 2 nodes


class TestWorkflowIntegration:
    """Integration tests for workflow building and serialization."""

    def test_full_workflow_roundtrip(self, tmp_path):
        """Build workflow, save, and reload should preserve structure."""
        from transformation_portal.comfyui.workflow_builder import WorkflowBuilder

        # Build workflow
        builder = WorkflowBuilder(name="Integration Test")
        workflow = (
            builder.add_input("input.jpg")
            .add_flux_enhancement(strength=0.45, num_steps=4)
            .add_skygan_sky(location="montecito", auto_correct=True)
            .add_output("output.jpg")
            .build()
        )

        # Save
        save_path = tmp_path / "integration_test.json"
        workflow.save(save_path)

        # Verify file content
        with open(save_path) as f:
            data = json.load(f)

        assert data["metadata"]["name"] == "Integration Test"
        assert len(data["nodes"]) == 4

    def test_comfyui_format_is_valid_json(self):
        """to_comfyui_format should produce valid JSON-serializable dict."""
        from transformation_portal.comfyui.workflow_builder import WorkflowBuilder

        builder = WorkflowBuilder()
        workflow = builder.add_input("test.jpg").add_flux_enhancement(strength=0.5).add_output("output.jpg").build()

        comfyui_format = workflow.to_comfyui_format()

        # Should be JSON-serializable without error
        json_str = json.dumps(comfyui_format)
        assert len(json_str) > 0

        # Should be deserializable
        parsed = json.loads(json_str)
        assert len(parsed) == 3  # 3 nodes
