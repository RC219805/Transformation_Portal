"""Workflow executor for running ComfyUI workflows programmatically.

Executes workflows built with WorkflowBuilder without requiring the
ComfyUI GUI. Useful for:
- Batch processing
- Automated pipelines
- Testing and validation
- CLI tools
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

# Import the World-Class components
from transformation_portal.atmosphere import AtmosphericParameters, LocationPresets, SkyBlender, SkyGANGenerator, SkyParameters
from transformation_portal.comfyui.workflow_builder import Node, NodeType, Workflow

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Context for workflow execution."""

    node_outputs: Dict[str, Any]
    execution_times: Dict[str, float]
    errors: List[str]

    def get_output(self, node_id: str, output_name: str = "IMAGE") -> Any:
        outputs = self.node_outputs.get(node_id, {})
        return outputs.get(output_name)

    def set_output(self, node_id: str, outputs: Dict[str, Any]) -> None:
        self.node_outputs[node_id] = outputs

    def add_error(self, node_id: str, error: str) -> None:
        self.errors.append(f"{node_id}: {error}")


class WorkflowExecutor:
    """Executes ComfyUI workflows programmatically."""

    def __init__(self, cache_models: bool = True, verbose: bool = False):
        self.cache_models = cache_models
        self.verbose = verbose
        self._model_cache: Dict[str, Any] = {}
        self._total_executions = 0
        self._total_time = 0.0

        # Initialize the heavy engine once if caching is on
        if self.cache_models:
            self._sky_blender = SkyBlender()
        else:
            self._sky_blender = None

        logger.info("WorkflowExecutor initialized")

    @property
    def sky_blender(self) -> SkyBlender:
        """Lazy-load the blender."""
        if self._sky_blender is None:
            self._sky_blender = SkyBlender()
        return self._sky_blender

    def execute(self, workflow: Workflow, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Execute a workflow (Topological Sort)."""
        start_time = time.time()

        # [Topological Sort Logic - Preserved from your original code]
        execution_order = self._build_execution_order(workflow)
        context = ExecutionContext(node_outputs={}, execution_times={}, errors=[])

        logger.info(f"Execution order: {execution_order}")

        for node_id in execution_order:
            node = workflow.nodes[node_id]
            try:
                self._execute_node(node, workflow, context)
            except Exception as e:
                logger.error(f"Error executing {node_id}: {str(e)}")
                context.add_error(node_id, str(e))
                break

        # [Stats Collection - Preserved]
        execution_time = time.time() - start_time
        self._total_executions += 1
        self._total_time += execution_time

        return {
            "success": len(context.errors) == 0,
            "node_outputs": context.node_outputs,
            "errors": context.errors,
            "execution_time": execution_time,
        }

    def _build_execution_order(self, workflow: Workflow) -> List[str]:
        """Build topological execution order for nodes."""
        dependencies: Dict[str, List[str]] = {nid: [] for nid in workflow.nodes}
        for conn in workflow.connections:
            dependencies[conn.target_node_id].append(conn.source_node_id)

        visited = set()
        order = []

        def visit(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            for dep in dependencies[node_id]:
                visit(dep)
            order.append(node_id)

        for node_id in workflow.nodes:
            visit(node_id)
        return order

    def _execute_node(self, node: Node, workflow: Workflow, context: ExecutionContext) -> None:
        """Dispatcher."""
        node_start = time.time()
        logger.info(f"Executing node: {node.node_id} ({node.node_type.value})")

        inputs = self._get_node_inputs(node, workflow, context)

        method_name = f"_execute_{node.node_type.name.lower()}_node"
        handler = getattr(self, method_name, None)

        if handler is None and node.node_type == NodeType.SKYGAN_SKY:
            handler = self._execute_skygan_node
        elif handler is None and node.node_type == NodeType.ATMOSPHERIC_MODEL:
            handler = self._execute_atmospheric_model_node

        if handler is None:
            raise NotImplementedError(
                f"No executor implementation for node type {node.node_type.name} " f"({node.node_type.value})"
            )

        outputs = handler(node, inputs)

        context.set_output(node.node_id, outputs)

        if self.verbose:
            logger.info(f"Node {node.node_id} finished in {time.time()-node_start:.2f}s")

    def _get_node_inputs(self, node: Node, workflow: Workflow, context: ExecutionContext) -> Dict[str, Any]:
        """Resolve connections to values."""
        inputs = {}
        for conn in workflow.connections:
            if conn.target_node_id == node.node_id:
                inputs[conn.target_input] = context.get_output(conn.source_node_id, conn.source_output)
        inputs.update(node.parameters)
        return inputs

    # --- SPECIFIC NODE EXECUTORS ---

    def _execute_skygan_node(self, node: Node, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SkyGAN using the SMART PIPELINE (Paradigm Shift)."""
        image = inputs.get("image")
        if image is None:
            raise ValueError("SkyGAN missing image")

        # 1. Map Inputs to Parameters
        presets = LocationPresets()
        location = inputs.get("location", "montecito")
        season = inputs.get("season", "summer")
        time_str = inputs.get("time_of_day", "golden_hour")

        # Get base physics data
        atmo_params = presets.get_atmospheric_parameters(location, season)
        sky_params = presets.get_sky_parameters(location, season=season)  # Defaults

        # Apply user overrides
        if "cloud_coverage" in inputs:
            sky_params.cloud_coverage = inputs["cloud_coverage"]

        # 2. Execute via SkyBlender
        # This handles Shadow Analysis, Auto-Correction, and Volumetric Blending
        enhanced_image, report = self.sky_blender.smart_render(
            source_image=np.array(image),
            sky_params=sky_params,
            atmo_params=atmo_params,
            auto_correct=inputs.get("auto_correct", True),
            strict_physics=inputs.get("strict_physics", False),
        )

        return {"IMAGE": enhanced_image, "REPORT": report.message}

    def _execute_atmospheric_model_node(self, node: Node, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Physics Model with Depth Awareness."""
        image = inputs.get("image")
        if image is None:
            raise ValueError("Atmospheric Model missing image")

        # 1. Ensure Depth Map
        # If not provided by upstream node, we generate it locally
        depth_map = inputs.get("depth_map")
        if depth_map is None:
            depth_map = self.sky_blender._estimate_depth(np.array(image))

        # 2. Prepare Parameters
        params = AtmosphericParameters(
            turbidity=inputs.get("turbidity", 2.0),
            visibility=inputs.get("visibility", 30.0),
            marine_influence=inputs.get("marine_influence", 0.6),
        )

        # 3. Apply Physics
        # Note: We access the low-level model inside the blender
        processed = self.sky_blender.atmosphere.apply_aerial_perspective(
            image=np.array(image), depth_map=depth_map, params=params
        )

        return {"IMAGE": processed}

    # ... (Other executors like _execute_flux_node, _execute_input_node remain similar to your draft)
