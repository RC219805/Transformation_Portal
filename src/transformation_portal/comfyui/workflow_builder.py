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
    ...     .add_material_segmentation()
    ...     .add_flux_enhancement(strength=0.45)
    ...     .add_skygan_sky(location="montecito", time_of_day="golden_hour")
    ...     .add_quality_validation(pass_threshold=7.0)
    ...     .add_output("enhanced.jpg")
    ...     .build()
    ... )
    >>> workflow.save("luxury_estate_pipeline.json")
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum


logger = logging.getLogger(__name__)


class NodeType(Enum):
    """ComfyUI node types."""
    INPUT = "LoadImage"
    OUTPUT = "SaveImage"
    FLUX_ENHANCEMENT = "FluxEnhancement"
    SKYGAN_SKY = "SkyGANGenerator"
    SCENE_ANALYSIS = "SceneAnalysis"
    MATERIAL_SEGMENTATION = "MaterialSegmentation"
    NEUROAESTHETICS = "NeuroaestheticsOptimization"
    QUALITY_VALIDATION = "QualityValidation"
    CONTROLNET = "ControlNetPreprocessor"
    IMAGE_RESIZE = "ImageResize"
    IMAGE_BLEND = "ImageBlend"
    ATMOSPHERIC_MODEL = "AtmosphericModel"
    COLOR_CORRECTION = "ColorCorrection"


@dataclass
class NodeConnection:
    """Connection between nodes."""
    source_node_id: str
    source_output: str
    target_node_id: str
    target_input: str

    def to_comfyui_format(self) -> List[str]:
        """Convert to ComfyUI connection format [node_id, output_slot]."""
        return [self.source_node_id, int(self.source_output)]


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
            }
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
        """Save workflow to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        workflow_data = {
            "last_node_id": len(self.nodes),
            "last_link_id": len(self.connections),
            "nodes": self.to_comfyui_format(),
            "metadata": self.metadata
        }

        with open(path, 'w') as f:
            json.dump(workflow_data, f, indent=2)

        logger.info(f"Workflow saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Workflow':
        """Load workflow from JSON file.

        Args:
            path: Input file path

        Returns:
            Loaded Workflow instance
        """
        path = Path(path)

        with open(path, 'r') as f:
            workflow_data = json.load(f)

        workflow = cls()
        workflow.metadata = workflow_data.get("metadata", {})

        # Parse nodes
        nodes_data = workflow_data.get("nodes", {})
        for node_id, node_data in nodes_data.items():
            node_type_str = node_data.get("class_type")
            try:
                node_type = NodeType(node_type_str)
            except ValueError:
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
    """Fluent API for building ComfyUI workflows.

    Provides a chainable interface for constructing complex enhancement
    pipelines that integrate all Transformation Portal components.

    Example:
        >>> builder = WorkflowBuilder(name="Luxury Estate Enhancement")
        >>> workflow = (builder
        ...     .add_input("estate.jpg")
        ...     .add_scene_analysis()
        ...     .add_flux_enhancement(strength=0.45, num_steps=4)
        ...     .add_quality_validation()
        ...     .add_output("enhanced.jpg")
        ...     .build()
        ... )
    """

    def __init__(self, name: str = "Transformation Portal Workflow"):
        """Initialize workflow builder.

        Args:
            name: Workflow name for metadata
        """
        self.workflow = Workflow()
        self.workflow.metadata = {
            "name": name,
            "version": "1.0",
            "description": "Generated by Transformation Portal WorkflowBuilder"
        }
        self._node_counter = 0
        self._last_node_id: Optional[str] = None
        self._last_output = "IMAGE"  # Default output type

        logger.info(f"Initialized WorkflowBuilder: {name}")

    def _generate_node_id(self, prefix: str = "node") -> str:
        """Generate unique node ID.

        Args:
            prefix: Node ID prefix

        Returns:
            Unique node ID
        """
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    def _add_node(
        self,
        node_type: NodeType,
        parameters: Optional[Dict[str, Any]] = None,
        connect_to_previous: bool = True,
        input_name: str = "image"
    ) -> str:
        """Add node to workflow.

        Args:
            node_type: Type of node to add
            parameters: Node parameters
            connect_to_previous: Whether to connect to previous node
            input_name: Input parameter name for connection

        Returns:
            Generated node ID
        """
        node_id = self._generate_node_id(node_type.value.lower())

        node = Node(
            node_id=node_id,
            node_type=node_type,
            parameters=parameters or {},
            position=(self._node_counter * 200, 0)
        )

        self.workflow.nodes[node_id] = node

        # Connect to previous node if requested
        if connect_to_previous and self._last_node_id:
            connection = NodeConnection(
                source_node_id=self._last_node_id,
                source_output=self._last_output,
                target_node_id=node_id,
                target_input=input_name
            )
            self.workflow.connections.append(connection)

        self._last_node_id = node_id
        return node_id

    def add_input(
        self,
        image_path: str,
        node_id: Optional[str] = None
    ) -> 'WorkflowBuilder':
        """Add input image node.

        Args:
            image_path: Path to input image
            node_id: Optional custom node ID

        Returns:
            Self for chaining
        """
        if node_id:
            self._last_node_id = node_id
        else:
            self._add_node(
                NodeType.INPUT,
                parameters={"image": image_path},
                connect_to_previous=False
            )

        self._last_output = "IMAGE"
        logger.info(f"Added input node: {image_path}")
        return self

    def add_scene_analysis(
        self,
        detailed: bool = True
    ) -> 'WorkflowBuilder':
        """Add scene analysis node using VLM.

        Args:
            detailed: Whether to perform detailed analysis

        Returns:
            Self for chaining
        """
        self._add_node(
            NodeType.SCENE_ANALYSIS,
            parameters={
                "detailed": detailed,
                "analyze_materials": True,
                "analyze_style": True,
                "analyze_lighting": True
            }
        )
        self._last_output = "SCENE_ANALYSIS"
        logger.info("Added scene analysis node")
        return self

    def add_material_segmentation(
        self,
        materials: Optional[List[str]] = None
    ) -> 'WorkflowBuilder':
        """Add material segmentation node using SAM + CLIP.

        Args:
            materials: Optional list of materials to segment

        Returns:
            Self for chaining
        """
        params = {
            "use_sam": True,
            "use_clip": True,
            "filter_by_area": True,
            "min_area": 500
        }

        if materials:
            params["materials"] = materials

        self._add_node(
            NodeType.MATERIAL_SEGMENTATION,
            parameters=params
        )
        self._last_output = "SEGMENTATION"
        logger.info("Added material segmentation node")
        return self

    def add_flux_enhancement(
        self,
        prompt: Optional[str] = None,
        strength: float = 0.45,
        num_steps: int = 4,
        guidance_scale: float = 3.5,
        variant: str = "dev",
        use_controlnet: bool = False,
        controlnet_types: Optional[List[str]] = None
    ) -> 'WorkflowBuilder':
        """Add FLUX enhancement node.

        Args:
            prompt: Enhancement prompt (auto-generated if None)
            strength: Enhancement strength (0-1)
            num_steps: Number of diffusion steps
            guidance_scale: CFG scale
            variant: FLUX variant ("dev" or "schnell")
            use_controlnet: Whether to use ControlNet
            controlnet_types: ControlNet types if enabled

        Returns:
            Self for chaining
        """
        params = {
            "strength": strength,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "variant": variant,
            "use_controlnet": use_controlnet
        }

        if prompt:
            params["prompt"] = prompt

        if use_controlnet and controlnet_types:
            params["controlnet_types"] = controlnet_types

        self._add_node(
            NodeType.FLUX_ENHANCEMENT,
            parameters=params
        )
        self._last_output = "IMAGE"
        logger.info(f"Added FLUX enhancement node (variant={variant})")
        return self

    def add_skygan_sky(
        self,
        location: str = "montecito",
        season: str = "summer",
        time_of_day: str = "golden_hour",
        sun_azimuth: Optional[float] = None,
        sun_elevation: Optional[float] = None,
        turbidity: Optional[float] = None,
        cloud_coverage: float = 0.3,
        update_reflections: bool = True
    ) -> 'WorkflowBuilder':
        """Add SkyGAN atmospheric rendering node.

        Args:
            location: Location preset name
            season: Season for atmospheric parameters
            time_of_day: Time of day preset
            sun_azimuth: Optional sun azimuth override
            sun_elevation: Optional sun elevation override
            turbidity: Optional turbidity override
            cloud_coverage: Cloud coverage amount (0-1)
            update_reflections: Whether to update water/glass reflections

        Returns:
            Self for chaining
        """
        params = {
            "location": location,
            "season": season,
            "time_of_day": time_of_day,
            "cloud_coverage": cloud_coverage,
            "update_reflections": update_reflections
        }

        if sun_azimuth is not None:
            params["sun_azimuth"] = sun_azimuth

        if sun_elevation is not None:
            params["sun_elevation"] = sun_elevation

        if turbidity is not None:
            params["turbidity"] = turbidity

        self._add_node(
            NodeType.SKYGAN_SKY,
            parameters=params
        )
        self._last_output = "IMAGE"
        logger.info(f"Added SkyGAN node (location={location}, time={time_of_day})")
        return self

    def add_neuroaesthetics_optimization(
        self,
        emotional_target: str = "luxury",
        optimize_composition: bool = True,
        optimize_color_harmony: bool = True,
        optimize_spatial_frequency: bool = True
    ) -> 'WorkflowBuilder':
        """Add neuroaesthetics optimization node.

        Args:
            emotional_target: Target emotion (luxury, aspiration, etc.)
            optimize_composition: Enable golden ratio optimization
            optimize_color_harmony: Enable color harmony optimization
            optimize_spatial_frequency: Enable spatial frequency optimization

        Returns:
            Self for chaining
        """
        params = {
            "emotional_target": emotional_target,
            "optimize_composition": optimize_composition,
            "optimize_color_harmony": optimize_color_harmony,
            "optimize_spatial_frequency": optimize_spatial_frequency
        }

        self._add_node(
            NodeType.NEUROAESTHETICS,
            parameters=params
        )
        self._last_output = "IMAGE"
        logger.info(f"Added neuroaesthetics optimization (target={emotional_target})")
        return self

    def add_quality_validation(
        self,
        pass_threshold: float = 7.0,
        warning_threshold: float = 5.0,
        check_realism: bool = True,
        check_structural_accuracy: bool = True,
        check_material_consistency: bool = True
    ) -> 'WorkflowBuilder':
        """Add quality validation node using VLM.

        Args:
            pass_threshold: Minimum score to pass (0-10)
            warning_threshold: Warning threshold
            check_realism: Validate photorealism
            check_structural_accuracy: Validate architectural accuracy
            check_material_consistency: Validate material rendering

        Returns:
            Self for chaining
        """
        params = {
            "pass_threshold": pass_threshold,
            "warning_threshold": warning_threshold,
            "check_realism": check_realism,
            "check_structural_accuracy": check_structural_accuracy,
            "check_material_consistency": check_material_consistency
        }

        self._add_node(
            NodeType.QUALITY_VALIDATION,
            parameters=params
        )
        self._last_output = "VALIDATION_REPORT"
        logger.info("Added quality validation node")
        return self

    def add_atmospheric_model(
        self,
        apply_aerial_perspective: bool = True,
        marine_layer: bool = False,
        max_distance: float = 1000.0
    ) -> 'WorkflowBuilder':
        """Add atmospheric model node for depth-based effects.

        Args:
            apply_aerial_perspective: Apply depth-based atmospheric effects
            marine_layer: Simulate coastal fog
            max_distance: Maximum distance for atmospheric effects (meters)

        Returns:
            Self for chaining
        """
        params = {
            "apply_aerial_perspective": apply_aerial_perspective,
            "marine_layer": marine_layer,
            "max_distance": max_distance
        }

        self._add_node(
            NodeType.ATMOSPHERIC_MODEL,
            parameters=params
        )
        self._last_output = "IMAGE"
        logger.info("Added atmospheric model node")
        return self

    def add_output(
        self,
        output_path: str,
        format: str = "jpg",
        quality: int = 95
    ) -> 'WorkflowBuilder':
        """Add output image node.

        Args:
            output_path: Output file path
            format: Image format (jpg, png, etc.)
            quality: Output quality (1-100)

        Returns:
            Self for chaining
        """
        self._add_node(
            NodeType.OUTPUT,
            parameters={
                "filename": output_path,
                "format": format,
                "quality": quality
            }
        )
        logger.info(f"Added output node: {output_path}")
        return self

    def build(self) -> Workflow:
        """Build and return the complete workflow.

        Returns:
            Constructed Workflow instance
        """
        logger.info(f"Built workflow with {len(self.workflow.nodes)} nodes")
        return self.workflow

    def __repr__(self) -> str:
        return (
            f"WorkflowBuilder(nodes={len(self.workflow.nodes)}, "
            f"connections={len(self.workflow.connections)})"
        )
