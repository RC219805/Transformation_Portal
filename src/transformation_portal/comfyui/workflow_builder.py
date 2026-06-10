"""ComfyUI workflow builder for programmatic pipeline construction.

Provides a fluent API for building complex enhancement workflows that can be:
- Exported as ComfyUI JSON workflows
- Executed programmatically
- Visualized in ComfyUI interface
- Version controlled and shared

Example:
    >>> builder = WorkflowBuilder()
    >>> workflow = (builder
    ...     .add_input("image.jpg")
    ...     .add_scene_analysis()
    ...     .add_flux_enhancement(strength=0.45)
    ...     # The NEW Paradigm: Intelligent Physics
    ...     .add_skygan_sky(
    ...         location="montecito",
    ...         time_of_day="golden_hour",
    ...         auto_correct=True
    ...     )
    ...     .add_quality_validation(pass_threshold=7.0)
    ...     .add_output("enhanced.jpg")
    ...     .build()
    ... )
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """ComfyUI node types.

    CRITICAL: Values must match class names in CustomNodeRegistry (custom_nodes.py).
    """

    INPUT = "LoadImage"
    OUTPUT = "SaveImage"
    FLUX_ENHANCEMENT = "FluxEnhancementNode"
    SKYGAN_SKY = "SkyGANNode"
    SCENE_ANALYSIS = "SceneAnalysisNode"
    MATERIAL_SEGMENTATION = "MaterialSegmentationNode"
    NEUROAESTHETICS = "NeuroaestheticsNode"
    QUALITY_VALIDATION = "QualityValidationNode"
    CONTROLNET = "ControlNetPreprocessor"
    IMAGE_RESIZE = "ImageResize"
    IMAGE_BLEND = "ImageBlend"
    ATMOSPHERIC_MODEL = "AtmosphericModelNode"
    COLOR_CORRECTION = "ColorCorrection"


@dataclass
class NodeConnection:
    """Connection between nodes."""

    source_node_id: str
    source_output: str
    target_node_id: str
    target_input: str

    def to_comfyui_format(self) -> List[Union[str, int]]:
        """Convert to ComfyUI connection format [node_id, output_slot]."""
        # Note: ComfyUI expects output_slot as int, usually 0
        slot_idx = 0
        if isinstance(self.source_output, int):
            slot_idx = self.source_output
        # Basic mapping for multi-output nodes
        elif self.source_output == "IMAGE":
            slot_idx = 0
        elif self.source_output == "MASK":
            slot_idx = 1
        elif self.source_output == "REPORT":
            slot_idx = 2

        return [self.source_node_id, slot_idx]


@dataclass
class Node:
    """Workflow node representation."""

    node_id: str
    node_type: NodeType
    parameters: Dict[str, Any] = field(default_factory=dict)
    position: Tuple[int, int] = (0, 0)
    inputs: Dict[str, Any] = field(default_factory=dict)

    def to_comfyui_format(self) -> Dict[str, Any]:
        """Convert to ComfyUI node format."""
        return {
            "class_type": self.node_type.value,
            "inputs": {**self.parameters, **self.inputs},
            "_meta": {
                "title": self.node_type.value,
            },
        }


@dataclass
class Workflow:
    """Complete workflow representation."""

    nodes: Dict[str, Node] = field(default_factory=dict)
    connections: List[NodeConnection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_comfyui_format(self) -> Dict[str, Any]:
        """Convert to ComfyUI workflow JSON format."""
        comfyui_workflow = {}

        # Add nodes
        for node_id, node in self.nodes.items():
            comfyui_workflow[node_id] = node.to_comfyui_format()

        # Add connections as inputs
        for conn in self.connections:
            target_node = comfyui_workflow.get(conn.target_node_id)
            if target_node:
                target_node["inputs"][conn.target_input] = conn.to_comfyui_format()

        return comfyui_workflow

    def save(self, path: Union[str, Path]) -> None:
        """Save workflow to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        workflow_data = {
            "last_node_id": len(self.nodes),
            "last_link_id": len(self.connections),
            "nodes": self.to_comfyui_format(),
            "metadata": self.metadata,
        }

        with open(path, "w") as f:
            json.dump(workflow_data, f, indent=2)

        logger.info(f"Workflow saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Workflow":
        """Load workflow from JSON file."""
        path = Path(path)

        with open(path, "r") as f:
            workflow_data = json.load(f)

        workflow = cls()
        workflow.metadata = workflow_data.get("metadata", {})

        # Parse nodes
        nodes_data = workflow_data.get("nodes", {})
        for node_id, node_data in nodes_data.items():
            node_type_str = node_data.get("class_type")
            try:
                # Reverse lookup enum from value
                node_type = next(t for t in NodeType if t.value == node_type_str)
            except StopIteration:
                logger.warning(f"Unknown node type: {node_type_str}, skipping")
                continue

            node = Node(
                node_id=node_id,
                node_type=node_type,
                parameters=node_data.get("inputs", {}),
            )
            workflow.nodes[node_id] = node

        logger.info(f"Workflow loaded from {path}")
        return workflow


class WorkflowBuilder:
    """Fluent API for building ComfyUI workflows."""

    def __init__(self, name: str = "Transformation Portal Workflow"):
        self.workflow = Workflow()
        self.workflow.metadata = {
            "name": name,
            "version": "2.0",  # Bumped version for Paradigm Shift
            "description": "Generated by Transformation Portal WorkflowBuilder",
        }
        self._node_counter = 0
        self._last_node_id: Optional[str] = None
        self._last_output = "IMAGE"  # Default output type

        logger.info(f"Initialized WorkflowBuilder: {name}")

    def _generate_node_id(self, prefix: str = "node") -> str:
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    def _add_node(
        self,
        node_type: NodeType,
        parameters: Optional[Dict[str, Any]] = None,
        connect_to_previous: bool = True,
        input_name: str = "image",
    ) -> str:
        node_id = self._generate_node_id(node_type.value.lower())

        node = Node(
            node_id=node_id,
            node_type=node_type,
            parameters=parameters or {},
            position=(self._node_counter * 200, 0),
        )

        self.workflow.nodes[node_id] = node

        if connect_to_previous and self._last_node_id:
            connection = NodeConnection(
                source_node_id=self._last_node_id,
                source_output=self._last_output,
                target_node_id=node_id,
                target_input=input_name,
            )
            self.workflow.connections.append(connection)

        self._last_node_id = node_id
        return node_id

    def add_input(self, image_path: str, node_id: Optional[str] = None) -> "WorkflowBuilder":
        if node_id:
            self._last_node_id = node_id
        else:
            self._add_node(
                NodeType.INPUT,
                parameters={"image": image_path},
                connect_to_previous=False,
            )

        self._last_output = "IMAGE"
        logger.info(f"Added input node: {image_path}")
        return self

    def add_scene_analysis(self, detailed: bool = True) -> "WorkflowBuilder":
        previous_node_id = self._last_node_id
        previous_output = self._last_output

        self._add_node(
            NodeType.SCENE_ANALYSIS,
            parameters={"detailed": detailed},
        )

        # Scene analysis reads the current image as a sidecar. Preserve the
        # image chain so downstream image-processing nodes do not consume the
        # analysis report as their input image.
        self._last_node_id = previous_node_id
        self._last_output = previous_output

        logger.info("Added scene analysis node")
        return self

    def add_flux_enhancement(
        self,
        prompt: Optional[str] = None,
        strength: float = 0.45,
        num_steps: int = 4,
        guidance_scale: float = 3.5,
        variant: str = "dev",
        use_controlnet: bool = False,
        controlnet_types: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        params = {
            "strength": strength,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "variant": variant,
            "use_controlnet": use_controlnet,
        }
        if prompt:
            params["prompt"] = prompt
        if controlnet_types is not None:
            params["controlnet_types"] = controlnet_types

        self._add_node(NodeType.FLUX_ENHANCEMENT, parameters=params)
        self._last_output = "IMAGE"
        return self

    def add_material_segmentation(
        self,
        materials: Optional[List[str]] = None,
        filter_by_area: bool = True,
        min_area: int = 500,
    ) -> "WorkflowBuilder":
        params: Dict[str, Any] = {
            "filter_by_area": filter_by_area,
            "min_area": min_area,
        }
        if materials is not None:
            params["materials"] = materials

        self._add_node(NodeType.MATERIAL_SEGMENTATION, parameters=params)
        self._last_output = "IMAGE"
        return self

    def add_skygan_sky(
        self,
        location: str = "montecito",
        season: str = "summer",
        time_of_day: str = "golden_hour",
        cloud_coverage: float = 0.3,
        auto_correct: bool = True,
        strict_physics: bool = False,
        update_reflections: bool = True,
        sun_azimuth: Optional[float] = None,
        sun_elevation: Optional[float] = None,
        turbidity: Optional[float] = None,
    ) -> "WorkflowBuilder":
        """Add SkyGAN atmospheric rendering node.

        UPGRADED: Now exposes physics guardrails.
        """
        params = {
            "location": location,
            "season": season,
            "time_of_day": time_of_day,
            "cloud_coverage": cloud_coverage,
            "auto_correct": auto_correct,
            "strict_physics": strict_physics,
            "update_reflections": update_reflections,
        }

        if sun_azimuth is not None:
            params["sun_azimuth"] = sun_azimuth
        if sun_elevation is not None:
            params["sun_elevation"] = sun_elevation
        if turbidity is not None:
            params["turbidity"] = turbidity

        self._add_node(NodeType.SKYGAN_SKY, parameters=params)
        self._last_output = "IMAGE"  # The node returns IMAGE, MASK, REPORT
        logger.info(f"Added SkyGAN node (location={location}, auto_correct={auto_correct})")
        return self

    def add_neuroaesthetics_optimization(
        self,
        emotional_target: str = "luxury",
        optimize_composition: bool = True,
        optimize_color_harmony: bool = True,
        optimize_spatial_frequency: bool = True,
    ) -> "WorkflowBuilder":
        self._add_node(
            NodeType.NEUROAESTHETICS,
            parameters={
                "emotional_target": emotional_target,
                "optimize_composition": optimize_composition,
                "optimize_color_harmony": optimize_color_harmony,
                "optimize_spatial_frequency": optimize_spatial_frequency,
            },
        )
        self._last_output = "IMAGE"
        return self

    def add_atmospheric_model(
        self,
        apply_aerial_perspective: bool = True,
        marine_layer: bool = False,
        max_distance: float = 1000.0,
    ) -> "WorkflowBuilder":
        self._add_node(
            NodeType.ATMOSPHERIC_MODEL,
            parameters={
                "apply_aerial_perspective": apply_aerial_perspective,
                "marine_layer": marine_layer,
                "max_distance": max_distance,
            },
        )
        self._last_output = "IMAGE"
        return self

    def add_quality_validation(
        self,
        pass_threshold: float = 7.0,
        warning_threshold: float = 5.0,
        check_realism: bool = True,
        check_structural_accuracy: bool = True,
        check_material_consistency: bool = False,
    ) -> "WorkflowBuilder":
        previous_node_id = self._last_node_id
        previous_output = self._last_output

        params = {
            "pass_threshold": pass_threshold,
            "warning_threshold": warning_threshold,
            "check_realism": check_realism,
            "check_structural_accuracy": check_structural_accuracy,
            "check_material_consistency": check_material_consistency,
        }
        self._add_node(NodeType.QUALITY_VALIDATION, parameters=params)

        # Quality validation emits a report, but downstream image nodes should
        # keep consuming the image that was validated.
        self._last_node_id = previous_node_id
        self._last_output = previous_output

        return self

    def add_output(self, output_path: str, format: str = "jpg", quality: int = 95) -> "WorkflowBuilder":
        self._add_node(
            NodeType.OUTPUT,
            parameters={"filename": output_path, "format": format, "quality": quality},
        )
        return self

    def build(self) -> Workflow:
        logger.info(f"Built workflow with {len(self.workflow.nodes)} nodes")
        return self.workflow

    def __repr__(self) -> str:
        return f"WorkflowBuilder(nodes={len(self.workflow.nodes)})"
