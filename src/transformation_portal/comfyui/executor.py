"""Workflow executor for running ComfyUI workflows programmatically.

Executes workflows built with WorkflowBuilder without requiring the
ComfyUI GUI. Useful for:
- Batch processing
- Automated pipelines
- Testing and validation
- CLI tools

Example:
    >>> executor = WorkflowExecutor()
    >>> workflow = WorkflowTemplates.full_luxury_estate_pipeline(
    ...     input_path="estate.jpg",
    ...     output_path="enhanced.jpg"
    ... )
    >>> results = executor.execute(workflow)
    >>> print(f"Enhancement complete: {results['output']}")
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import time

import numpy as np
from PIL import Image

from transformation_portal.comfyui.workflow_builder import Workflow, Node, NodeType


logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Context for workflow execution.

    Maintains state during workflow execution including intermediate
    results and node outputs.
    """
    node_outputs: Dict[str, Any]
    execution_times: Dict[str, float]
    errors: List[str]

    def get_output(self, node_id: str, output_name: str = "IMAGE") -> Any:
        """Get output from a node.

        Args:
            node_id: Node ID
            output_name: Output name

        Returns:
            Node output value
        """
        outputs = self.node_outputs.get(node_id, {})
        return outputs.get(output_name)

    def set_output(self, node_id: str, outputs: Dict[str, Any]) -> None:
        """Set outputs for a node.

        Args:
            node_id: Node ID
            outputs: Dictionary of output values
        """
        self.node_outputs[node_id] = outputs

    def add_error(self, node_id: str, error: str) -> None:
        """Add execution error.

        Args:
            node_id: Node ID where error occurred
            error: Error message
        """
        self.errors.append(f"{node_id}: {error}")


class WorkflowExecutor:
    """Executes ComfyUI workflows programmatically.

    Processes workflows node-by-node, respecting dependencies and
    connections. Supports execution monitoring and error handling.

    Example:
        >>> executor = WorkflowExecutor(verbose=True)
        >>> workflow = Workflow.load("my_workflow.json")
        >>> results = executor.execute(workflow)
    """

    def __init__(
        self,
        cache_models: bool = True,
        verbose: bool = False
    ):
        """Initialize workflow executor.

        Args:
            cache_models: Cache loaded models between executions
            verbose: Enable verbose logging
        """
        self.cache_models = cache_models
        self.verbose = verbose

        # Model cache
        self._model_cache: Dict[str, Any] = {}

        # Execution stats
        self._total_executions = 0
        self._total_time = 0.0

        logger.info("WorkflowExecutor initialized")

    def execute(
        self,
        workflow: Workflow,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Execute a workflow.

        Args:
            workflow: Workflow to execute
            output_dir: Optional output directory for results

        Returns:
            Dictionary of execution results
        """
        start_time = time.time()

        logger.info(f"Executing workflow: {workflow.metadata.get('name', 'Unnamed')}")
        logger.info(f"Nodes: {len(workflow.nodes)}, Connections: {len(workflow.connections)}")

        # Initialize execution context
        context = ExecutionContext(
            node_outputs={},
            execution_times={},
            errors=[]
        )

        # Build execution order (topological sort)
        execution_order = self._build_execution_order(workflow)

        logger.info(f"Execution order: {execution_order}")

        # Execute nodes in order
        for node_id in execution_order:
            node = workflow.nodes[node_id]

            try:
                self._execute_node(node, workflow, context)
            except Exception as e:
                error_msg = f"Error executing {node_id}: {str(e)}"
                logger.error(error_msg)
                context.add_error(node_id, str(e))

                # Stop execution on error
                break

        # Collect results
        execution_time = time.time() - start_time
        self._total_executions += 1
        self._total_time += execution_time

        results = {
            "success": len(context.errors) == 0,
            "execution_time": execution_time,
            "node_outputs": context.node_outputs,
            "execution_times": context.execution_times,
            "errors": context.errors,
            "workflow_name": workflow.metadata.get("name", "Unnamed")
        }

        logger.info(f"Workflow execution completed in {execution_time:.2f}s")
        if context.errors:
            logger.error(f"Execution failed with {len(context.errors)} errors")
        else:
            logger.info("Execution successful")

        return results

    def _build_execution_order(self, workflow: Workflow) -> List[str]:
        """Build topological execution order for nodes.

        Args:
            workflow: Workflow to analyze

        Returns:
            List of node IDs in execution order
        """
        # Build dependency graph
        dependencies: Dict[str, List[str]] = {
            node_id: [] for node_id in workflow.nodes.keys()
        }

        for conn in workflow.connections:
            dependencies[conn.target_node_id].append(conn.source_node_id)

        # Topological sort
        visited = set()
        order = []

        def visit(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)

            # Visit dependencies first
            for dep in dependencies[node_id]:
                visit(dep)

            order.append(node_id)

        # Visit all nodes
        for node_id in workflow.nodes.keys():
            visit(node_id)

        return order

    def _execute_node(
        self,
        node: Node,
        workflow: Workflow,
        context: ExecutionContext
    ) -> None:
        """Execute a single node.

        Args:
            node: Node to execute
            workflow: Parent workflow
            context: Execution context
        """
        node_start = time.time()

        logger.info(f"Executing node: {node.node_id} ({node.node_type.value})")

        # Get inputs from connections
        inputs = self._get_node_inputs(node, workflow, context)

        # Execute based on node type
        if node.node_type == NodeType.INPUT:
            outputs = self._execute_input_node(node, inputs)
        elif node.node_type == NodeType.OUTPUT:
            outputs = self._execute_output_node(node, inputs)
        elif node.node_type == NodeType.FLUX_ENHANCEMENT:
            outputs = self._execute_flux_node(node, inputs)
        elif node.node_type == NodeType.SKYGAN_SKY:
            outputs = self._execute_skygan_node(node, inputs)
        elif node.node_type == NodeType.SCENE_ANALYSIS:
            outputs = self._execute_scene_analysis_node(node, inputs)
        elif node.node_type == NodeType.MATERIAL_SEGMENTATION:
            outputs = self._execute_material_segmentation_node(node, inputs)
        elif node.node_type == NodeType.NEUROAESTHETICS:
            outputs = self._execute_neuroaesthetics_node(node, inputs)
        elif node.node_type == NodeType.QUALITY_VALIDATION:
            outputs = self._execute_quality_validation_node(node, inputs)
        elif node.node_type == NodeType.ATMOSPHERIC_MODEL:
            outputs = self._execute_atmospheric_model_node(node, inputs)
        else:
            logger.warning(f"Unknown node type: {node.node_type.value}")
            outputs = inputs  # Pass through

        # Store outputs
        context.set_output(node.node_id, outputs)

        # Record execution time
        node_time = time.time() - node_start
        context.execution_times[node.node_id] = node_time

        if self.verbose:
            logger.info(f"Node {node.node_id} completed in {node_time:.2f}s")

    def _get_node_inputs(
        self,
        node: Node,
        workflow: Workflow,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Get inputs for a node from connections.

        Args:
            node: Target node
            workflow: Parent workflow
            context: Execution context

        Returns:
            Dictionary of input values
        """
        inputs = {}

        # Get from connections
        for conn in workflow.connections:
            if conn.target_node_id == node.node_id:
                source_output = context.get_output(
                    conn.source_node_id,
                    conn.source_output
                )
                inputs[conn.target_input] = source_output

        # Merge with node parameters
        inputs.update(node.parameters)

        return inputs

    def _execute_input_node(
        self,
        node: Node,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute input image loading node.

        Args:
            node: Input node
            inputs: Node inputs

        Returns:
            Dictionary with loaded image
        """
        image_path = inputs.get("image")
        if not image_path:
            raise ValueError("Input node missing 'image' parameter")

        logger.info(f"Loading image: {image_path}")

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        return {"IMAGE": image_array}

    def _execute_output_node(
        self,
        node: Node,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute output image saving node.

        Args:
            node: Output node
            inputs: Node inputs with image

        Returns:
            Dictionary with output path
        """
        image_array = inputs.get("image")
        output_path = inputs.get("filename")
        quality = inputs.get("quality", 95)

        if image_array is None:
            raise ValueError("Output node missing image input")

        logger.info(f"Saving image: {output_path}")

        # Convert to PIL and save
        image = Image.fromarray(image_array.astype(np.uint8))
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=quality)

        return {"output_path": str(output_path)}

    def _execute_flux_node(
        self,
        node: Node,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute FLUX enhancement node.

        Args:
            node: FLUX node
            inputs: Node inputs

        Returns:
            Dictionary with enhanced image
        """
        from transformation_portal.diffusion import FLUXPipeline

        image = inputs.get("image")
        if image is None:
            raise ValueError("FLUX node missing image input")

        # Get parameters
        strength = inputs.get("strength", 0.45)
        num_steps = inputs.get("num_steps", 4)
        variant = inputs.get("variant", "dev")

        # Initialize or get from cache
        cache_key = f"flux_{variant}"
        if self.cache_models and cache_key in self._model_cache:
            pipeline = self._model_cache[cache_key]
        else:
            pipeline = FLUXPipeline(variant=variant)
            if self.cache_models:
                self._model_cache[cache_key] = pipeline

        # Enhance
        enhanced = pipeline.enhance(
            image=image,
            strength=strength,
            num_steps=num_steps
        )

        return {"IMAGE": np.array(enhanced)}

    def _execute_skygan_node(
        self,
        node: Node,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute SkyGAN node.

        Args:
            node: SkyGAN node
            inputs: Node inputs

        Returns:
            Dictionary with enhanced image
        """
        from transformation_portal.atmosphere import (
            SkyGANGenerator,
            LocationPresets,
            SkyBlender
        )

        image = inputs.get("image")
        if image is None:
            raise ValueError("SkyGAN node missing image input")

        # Get parameters
        location = inputs.get("location", "montecito")
        season = inputs.get("season", "summer")
        time_of_day = inputs.get("time_of_day", "golden_hour")

        # Get preset
        preset = LocationPresets.get_preset(location, season)
        time_params = LocationPresets.get_time_of_day(location, time_of_day, season)

        # Generate and blend sky
        generator = SkyGANGenerator()
        sky = generator.generate_sky(
            sun_azimuth=time_params.sun_azimuth,
            sun_elevation=time_params.sun_elevation,
            turbidity=preset.turbidity,
            atmospheric_params=preset
        )

        blender = SkyBlender()
        enhanced, _ = blender.blend_sky(image=image, sky=sky, return_mask=True)

        return {"IMAGE": np.array(enhanced)}

    def _execute_scene_analysis_node(
        self,
        node: Node,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute scene analysis node.

        Args:
            node: Scene analysis node
            inputs: Node inputs

        Returns:
            Dictionary with analysis results
        """
        from transformation_portal.vlm import SceneAnalyzer

        image = inputs.get("image")
        if image is None:
            raise ValueError("Scene analysis node missing image input")

        analyzer = SceneAnalyzer()
        analysis = analyzer.analyze_scene(image)

        return {"SCENE_ANALYSIS": analysis, "IMAGE": image}

    def _execute_material_segmentation_node(
        self,
        node: Node,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute material segmentation node.

        Args:
            node: Material segmentation node
            inputs: Node inputs

        Returns:
            Dictionary with segmentation results
        """
        from transformation_portal.segmentation import MaterialSegmenter

        image = inputs.get("image")
        if image is None:
            raise ValueError("Material segmentation node missing image input")

        segmenter = MaterialSegmenter()
        segments = segmenter.segment_materials(image)

        return {"SEGMENTATION": segments, "IMAGE": image}

    def _execute_neuroaesthetics_node(
        self,
        node: Node,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute neuroaesthetics optimization node.

        Args:
            node: Neuroaesthetics node
            inputs: Node inputs

        Returns:
            Dictionary with optimized image
        """
        from transformation_portal.neuroaesthetics import EmotionalOptimizer

        image = inputs.get("image")
        if image is None:
            raise ValueError("Neuroaesthetics node missing image input")

        emotional_target = inputs.get("emotional_target", "luxury")

        optimizer = EmotionalOptimizer()
        result = optimizer.optimize_for_emotion(image, emotional_target)

        return {"IMAGE": np.array(result["optimized_image"])}

    def _execute_quality_validation_node(
        self,
        node: Node,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute quality validation node.

        Args:
            node: Quality validation node
            inputs: Node inputs

        Returns:
            Dictionary with validation results
        """
        from transformation_portal.vlm import QualityValidator

        image = inputs.get("image")
        if image is None:
            raise ValueError("Quality validation node missing image input")

        pass_threshold = inputs.get("pass_threshold", 7.0)

        validator = QualityValidator(pass_threshold=pass_threshold)
        validation = validator.validate(image)

        return {
            "VALIDATION_REPORT": validation,
            "IMAGE": image,
            "passed": validation.passed
        }

    def _execute_atmospheric_model_node(
        self,
        node: Node,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute atmospheric model node.

        Args:
            node: Atmospheric model node
            inputs: Node inputs

        Returns:
            Dictionary with processed image
        """

        image = inputs.get("image")
        if image is None:
            raise ValueError("Atmospheric model node missing image input")

        # This would need depth map input - simplified for now
        return {"IMAGE": image}

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics.

        Returns:
            Dictionary with execution statistics
        """
        return {
            "total_executions": self._total_executions,
            "total_time": self._total_time,
            "average_time": self._total_time / max(1, self._total_executions),
            "cached_models": list(self._model_cache.keys())
        }

    def clear_cache(self) -> None:
        """Clear model cache to free memory."""
        self._model_cache.clear()
        logger.info("Model cache cleared")

    def __repr__(self) -> str:
        return (
            f"WorkflowExecutor(executions={self._total_executions}, "
            f"cached_models={len(self._model_cache)})"
        )


# Export
__all__ = ['WorkflowExecutor', 'ExecutionContext']
