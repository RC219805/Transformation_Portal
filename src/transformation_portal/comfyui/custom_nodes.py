"""Custom ComfyUI nodes for Transformation Portal components.

Provides node implementations that wrap Transformation Portal functionality
for use in ComfyUI workflows. Each node exposes inputs/outputs compatible
with ComfyUI's execution engine.

Node Categories:
- Analysis: Scene analysis, material segmentation
- Enhancement: FLUX diffusion, neuroaesthetics
- Atmospheric: SkyGAN, atmospheric modeling
- Validation: Quality validation, metrics
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


class CustomNodeRegistry:
    """Registry for custom Transformation Portal nodes.

    Maintains a mapping of node types to their implementations
    for ComfyUI integration.
    """

    _nodes: Dict[str, type] = {}

    @classmethod
    def register(cls, node_class: type) -> type:
        """Register a custom node class.

        Args:
            node_class: Node class to register

        Returns:
            The registered class (for use as decorator)
        """
        node_name = node_class.__name__
        cls._nodes[node_name] = node_class
        logger.debug(f"Registered custom node: {node_name}")
        return node_class

    @classmethod
    def get_node(cls, node_name: str) -> Optional[type]:
        """Get registered node class by name.

        Args:
            node_name: Name of the node class

        Returns:
            Node class if found, None otherwise
        """
        return cls._nodes.get(node_name)

    @classmethod
    def list_nodes(cls) -> List[str]:
        """List all registered node names.

        Returns:
            List of registered node names
        """
        return list(cls._nodes.keys())


class BaseNode:
    """Base class for custom nodes.

    Provides common functionality and interface for all custom nodes.
    """

    CATEGORY = "Transformation Portal"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define node input types.

        Must be implemented by subclasses.

        Returns:
            Dictionary describing input types and parameters
        """
        raise NotImplementedError

    @classmethod
    def RETURN_TYPES(cls) -> Tuple[str, ...]:
        """Define node output types.

        Must be implemented by subclasses.

        Returns:
            Tuple of output type names
        """
        raise NotImplementedError

    def execute(self, **kwargs) -> Tuple[Any, ...]:
        """Execute node logic.

        Must be implemented by subclasses.

        Args:
            **kwargs: Input parameters

        Returns:
            Tuple of output values
        """
        raise NotImplementedError


@CustomNodeRegistry.register
class FluxEnhancementNode(BaseNode):
    """FLUX diffusion enhancement node.

    Wraps FLUXPipeline for ComfyUI integration.
    """

    CATEGORY = "Transformation Portal/Enhancement"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "num_steps": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 50,
                    "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5
                }),
                "variant": (["dev", "schnell"],),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": -1}),
                "use_controlnet": ("BOOLEAN", {"default": False}),
            }
        }

    @classmethod
    def RETURN_TYPES(cls):
        return ("IMAGE",)

    def execute(
        self,
        image: np.ndarray,
        strength: float,
        num_steps: int,
        guidance_scale: float,
        variant: str,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: int = -1,
        use_controlnet: bool = False
    ) -> Tuple[np.ndarray]:
        """Execute FLUX enhancement.

        Args:
            image: Input image array
            strength: Enhancement strength
            num_steps: Number of diffusion steps
            guidance_scale: CFG scale
            variant: FLUX variant
            prompt: Optional enhancement prompt
            negative_prompt: Optional negative prompt
            seed: Random seed (-1 for random)
            use_controlnet: Whether to use ControlNet

        Returns:
            Tuple containing enhanced image
        """
        from transformation_portal.diffusion import FLUXPipeline

        logger.info(f"Executing FLUX enhancement (variant={variant}, strength={strength})")

        # Initialize pipeline
        pipeline = FLUXPipeline(variant=variant)

        # Convert seed
        seed_value = None if seed == -1 else seed

        # Enhance image
        enhanced = pipeline.enhance(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed_value
        )

        # Convert to numpy array
        enhanced_array = np.array(enhanced)

        return (enhanced_array,)


@CustomNodeRegistry.register
class SkyGANNode(BaseNode):
    """SkyGAN atmospheric rendering node.

    Wraps SkyGANGenerator for ComfyUI integration.
    """

    CATEGORY = "Transformation Portal/Atmospheric"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "location": (["montecito", "santa_barbara", "hope_ranch", "riviera"],),
                "season": (["spring", "summer", "fall", "winter"],),
                "time_of_day": (["sunrise", "morning", "midday", "golden_hour", "sunset", "twilight"],),
                "cloud_coverage": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "sun_azimuth": ("FLOAT", {"default": -1.0}),
                "sun_elevation": ("FLOAT", {"default": -1.0}),
                "turbidity": ("FLOAT", {"default": -1.0}),
                "update_reflections": ("BOOLEAN", {"default": True}),
            }
        }

    @classmethod
    def RETURN_TYPES(cls):
        return ("IMAGE", "IMAGE")  # Enhanced image, Sky mask

    def execute(
        self,
        image: np.ndarray,
        location: str,
        season: str,
        time_of_day: str,
        cloud_coverage: float,
        sun_azimuth: float = -1.0,
        sun_elevation: float = -1.0,
        turbidity: float = -1.0,
        update_reflections: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute SkyGAN sky replacement.

        Args:
            image: Input image array
            location: Location preset
            season: Season for atmospheric parameters
            time_of_day: Time of day preset
            cloud_coverage: Cloud coverage amount
            sun_azimuth: Optional sun azimuth override
            sun_elevation: Optional sun elevation override
            turbidity: Optional turbidity override
            update_reflections: Whether to update reflections

        Returns:
            Tuple of (enhanced image, sky mask)
        """
        from transformation_portal.atmosphere import (
            SkyGANGenerator,
            LocationPresets,
            SkyBlender
        )

        logger.info(f"Executing SkyGAN (location={location}, time={time_of_day})")

        # Get location preset
        location_preset = LocationPresets.get_preset(location, season)

        # Get time of day parameters
        time_params = LocationPresets.get_time_of_day(location, time_of_day, season)

        # Override if specified
        if sun_azimuth >= 0:
            time_params.sun_azimuth = sun_azimuth
        if sun_elevation >= 0:
            time_params.sun_elevation = sun_elevation
        if turbidity >= 0:
            location_preset.turbidity = turbidity

        # Generate sky
        generator = SkyGANGenerator()
        sky = generator.generate_sky(
            sun_azimuth=time_params.sun_azimuth,
            sun_elevation=time_params.sun_elevation,
            turbidity=location_preset.turbidity,
            cloud_coverage=cloud_coverage,
            atmospheric_params=location_preset
        )

        # Blend sky
        blender = SkyBlender()
        enhanced, mask = blender.blend_sky(
            image=image,
            sky=sky,
            update_reflections=update_reflections,
            return_mask=True
        )

        # Convert to numpy arrays
        enhanced_array = np.array(enhanced)
        mask_array = np.array(mask)

        return (enhanced_array, mask_array)


@CustomNodeRegistry.register
class SceneAnalysisNode(BaseNode):
    """Scene analysis node using VLM.

    Wraps SceneAnalyzer for ComfyUI integration.
    """

    CATEGORY = "Transformation Portal/Analysis"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detailed": ("BOOLEAN", {"default": True}),
            },
        }

    @classmethod
    def RETURN_TYPES(cls):
        return ("SCENE_ANALYSIS", "STRING")  # Analysis object, JSON string

    def execute(
        self,
        image: np.ndarray,
        detailed: bool
    ) -> Tuple[Dict[str, Any], str]:
        """Execute scene analysis.

        Args:
            image: Input image array
            detailed: Whether to perform detailed analysis

        Returns:
            Tuple of (analysis dict, JSON string)
        """
        from transformation_portal.vlm import SceneAnalyzer
        import json

        logger.info("Executing scene analysis")

        # Initialize analyzer
        analyzer = SceneAnalyzer()

        # Analyze scene
        analysis = analyzer.analyze_scene(image, detailed=detailed)

        # Convert to JSON string
        analysis_json = json.dumps(analysis, indent=2)

        return (analysis, analysis_json)


@CustomNodeRegistry.register
class MaterialSegmentationNode(BaseNode):
    """Material segmentation node using SAM + CLIP.

    Wraps MaterialSegmenter for ComfyUI integration.
    """

    CATEGORY = "Transformation Portal/Analysis"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filter_by_area": ("BOOLEAN", {"default": True}),
                "min_area": ("INT", {
                    "default": 500,
                    "min": 100,
                    "max": 10000,
                    "step": 100
                }),
            },
        }

    @classmethod
    def RETURN_TYPES(cls):
        return ("SEGMENTATION", "IMAGE")  # Segmentation data, Visualization

    def execute(
        self,
        image: np.ndarray,
        filter_by_area: bool,
        min_area: int
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Execute material segmentation.

        Args:
            image: Input image array
            filter_by_area: Whether to filter small segments
            min_area: Minimum segment area

        Returns:
            Tuple of (segmentation data, visualization)
        """
        from transformation_portal.segmentation import MaterialSegmenter

        logger.info("Executing material segmentation")

        # Initialize segmenter
        segmenter = MaterialSegmenter()

        # Segment materials
        segments = segmenter.segment_materials(
            image=image,
            filter_by_area=filter_by_area,
            min_area=min_area
        )

        # Create visualization
        viz = segmenter.visualize_segmentation(image, segments)
        viz_array = np.array(viz)

        return (segments, viz_array)


@CustomNodeRegistry.register
class NeuroaestheticsNode(BaseNode):
    """Neuroaesthetics optimization node.

    Wraps EmotionalOptimizer for ComfyUI integration.
    """

    CATEGORY = "Transformation Portal/Enhancement"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "emotional_target": ([
                    "luxury", "aspiration", "desire", "nostalgia",
                    "comfort", "serenity", "energy"
                ],),
                "optimize_composition": ("BOOLEAN", {"default": True}),
                "optimize_color_harmony": ("BOOLEAN", {"default": True}),
                "optimize_spatial_frequency": ("BOOLEAN", {"default": True}),
            },
        }

    @classmethod
    def RETURN_TYPES(cls):
        return ("IMAGE", "STRING")  # Optimized image, Analysis report

    def execute(
        self,
        image: np.ndarray,
        emotional_target: str,
        optimize_composition: bool,
        optimize_color_harmony: bool,
        optimize_spatial_frequency: bool
    ) -> Tuple[np.ndarray, str]:
        """Execute neuroaesthetics optimization.

        Args:
            image: Input image array
            emotional_target: Target emotion
            optimize_composition: Enable composition optimization
            optimize_color_harmony: Enable color optimization
            optimize_spatial_frequency: Enable frequency optimization

        Returns:
            Tuple of (optimized image, analysis report)
        """
        from transformation_portal.neuroaesthetics import EmotionalOptimizer
        import json

        logger.info(f"Executing neuroaesthetics optimization (target={emotional_target})")

        # Initialize optimizer
        optimizer = EmotionalOptimizer()

        # Optimize image
        result = optimizer.optimize_for_emotion(
            image=image,
            target_emotion=emotional_target,
            optimize_composition=optimize_composition,
            optimize_color_harmony=optimize_color_harmony,
            optimize_spatial_frequency=optimize_spatial_frequency
        )

        # Create report
        report = json.dumps({
            "emotional_target": emotional_target,
            "composition_score": result.get("composition_score", 0.0),
            "color_harmony_score": result.get("color_harmony_score", 0.0),
            "spatial_frequency_score": result.get("spatial_frequency_score", 0.0),
            "overall_score": result.get("overall_score", 0.0)
        }, indent=2)

        optimized_array = np.array(result["optimized_image"])

        return (optimized_array, report)


@CustomNodeRegistry.register
class QualityValidationNode(BaseNode):
    """Quality validation node using VLM.

    Wraps QualityValidator for ComfyUI integration.
    """

    CATEGORY = "Transformation Portal/Validation"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pass_threshold": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5
                }),
                "warning_threshold": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5
                }),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            }
        }

    @classmethod
    def RETURN_TYPES(cls):
        return ("BOOLEAN", "STRING", "FLOAT")  # Passed, Report, Overall score

    def execute(
        self,
        image: np.ndarray,
        pass_threshold: float,
        warning_threshold: float,
        reference_image: Optional[np.ndarray] = None
    ) -> Tuple[bool, str, float]:
        """Execute quality validation.

        Args:
            image: Input image array
            pass_threshold: Minimum score to pass
            warning_threshold: Warning threshold
            reference_image: Optional reference for comparison

        Returns:
            Tuple of (passed, report JSON, overall score)
        """
        from transformation_portal.vlm import QualityValidator
        import json

        logger.info("Executing quality validation")

        # Initialize validator
        validator = QualityValidator(
            pass_threshold=pass_threshold,
            warning_threshold=warning_threshold
        )

        # Validate image
        validation = validator.validate(image, detailed=True)

        # Create report
        report = json.dumps({
            "passed": validation.passed,
            "overall_score": validation.overall_score,
            "aspects": [
                {
                    "aspect": aspect.aspect,
                    "score": aspect.score,
                    "feedback": aspect.feedback
                }
                for aspect in validation.scores
            ],
            "recommendations": validation.recommendations
        }, indent=2)

        return (validation.passed, report, validation.overall_score)


# Export all nodes
__all__ = [
    'CustomNodeRegistry',
    'BaseNode',
    'FluxEnhancementNode',
    'SkyGANNode',
    'SceneAnalysisNode',
    'MaterialSegmentationNode',
    'NeuroaestheticsNode',
    'QualityValidationNode',
]
