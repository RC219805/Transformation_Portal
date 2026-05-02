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

import json
import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Import the Paradigm Shift components
from transformation_portal.atmosphere import AtmosphericParameters, LocationPresets, SkyBlender, SkyGANGenerator, SkyParameters

logger = logging.getLogger(__name__)


class CustomNodeRegistry:
    """Registry for custom Transformation Portal nodes."""

    _nodes: Dict[str, type] = {}

    @classmethod
    def register(cls, node_class: type) -> type:
        node_name = node_class.__name__
        cls._nodes[node_name] = node_class
        logger.debug(f"Registered custom node: {node_name}")
        return node_class

    @classmethod
    def get_node(cls, node_name: str) -> Optional[type]:
        return cls._nodes.get(node_name)

    @classmethod
    def list_nodes(cls) -> List[str]:
        return list(cls._nodes.keys())


class BaseNode(ABC):
    """Base class for custom nodes.

    All ComfyUI nodes must implement INPUT_TYPES, RETURN_TYPES, and execute.
    """

    CATEGORY = "Transformation Portal"

    @classmethod
    @abstractmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input schema for this node."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def RETURN_TYPES(cls) -> Tuple[str, ...]:
        """Define output types for this node."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, **kwargs) -> Tuple[Any, ...]:
        """Execute the node's processing logic."""
        raise NotImplementedError

    def _to_numpy(self, image: Any) -> np.ndarray:
        """Helper to ensure image is numpy array (H,W,3)."""
        if isinstance(image, torch.Tensor):
            # ComfyUI often passes (B,H,W,C) tensors
            return image.cpu().numpy()[0]
        return np.array(image)

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Helper to convert back to ComfyUI tensor format."""
        # Ensure float32 0-1 range
        if array.dtype == np.uint8:
            array = array.astype(np.float32) / 255.0
        # Add batch dimension (1,H,W,C)
        return torch.from_numpy(array)[None, ...]


@CustomNodeRegistry.register
class FluxEnhancementNode(BaseNode):
    """FLUX diffusion enhancement node."""

    CATEGORY = "Transformation Portal/Enhancement"
    _VALID_VARIANTS = ("dev", "schnell")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05}),
                "num_steps": ("INT", {"default": 4, "min": 1, "max": 50, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.5}),
                "variant": (["dev", "schnell"],),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": -1}),
                "use_controlnet": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def RETURN_TYPES(cls):
        return ("IMAGE",)

    @classmethod
    def _validate_inputs(cls, strength: float, num_steps: int, guidance_scale: float, variant: str) -> None:
        if variant not in cls._VALID_VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of {list(cls._VALID_VARIANTS)}")
        numeric_checks = (
            ("strength", strength, 0.0, 1.0),
            ("guidance_scale", guidance_scale, 1.0, 20.0),
        )
        for name, value, minimum, maximum in numeric_checks:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(numeric_value) or numeric_value < minimum or numeric_value > maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        try:
            step_count = int(num_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("num_steps must be an integer") from exc
        if step_count < 1 or step_count > 50:
            raise ValueError("num_steps must be between 1 and 50")

    def execute(
        self,
        image: Any,
        strength: float,
        num_steps: int,
        guidance_scale: float,
        variant: str,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: int = -1,
        use_controlnet: bool = False,
    ) -> Tuple[Any]:
        self._validate_inputs(strength=strength, num_steps=num_steps, guidance_scale=guidance_scale, variant=variant)

        from transformation_portal.diffusion import FLUXPipeline

        logger.info(f"Executing FLUX enhancement (variant={variant}, strength={strength})")

        # Convert input
        img_np = self._to_numpy(image)

        pipeline = FLUXPipeline(variant=variant)
        seed_value = None if seed == -1 else seed

        enhanced = pipeline.enhance(
            image=img_np,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed_value,
        )

        return (self._to_tensor(np.array(enhanced)),)


@CustomNodeRegistry.register
class SkyGANNode(BaseNode):
    """SkyGAN atmospheric rendering node.

    INTELLIGENT UPDATE:
    Now supports Physics Guardrails via SkyBlender.smart_render().
    Returns the analysis report to the user.
    """

    CATEGORY = "Transformation Portal/Atmospheric"

    # Approximate hour-of-day for each named slot. Used to resolve the
    # ComfyUI dropdown selection into the float hour expected by
    # LocationPresets.get_sky_parameters(). Users can still override the
    # derived sun_azimuth/sun_elevation via the optional inputs below.
    _TIME_OF_DAY_HOURS: Dict[str, float] = {
        "sunrise": 6.5,
        "morning": 9.0,
        "midday": 12.0,
        "golden_hour": 17.0,
        "sunset": 18.5,
        "twilight": 19.5,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "location": (["montecito", "santa_barbara", "hope_ranch", "riviera"],),
                "season": (["spring", "summer", "fall", "winter"],),
                "time_of_day": (["sunrise", "morning", "midday", "golden_hour", "sunset", "twilight"],),
                "cloud_coverage": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                # THE BRAIN: New Controls for Physics Engine
                "auto_correct": ("BOOLEAN", {"default": True, "label": "Auto-Fix Shadows"}),
                "strict_physics": ("BOOLEAN", {"default": False, "label": "Strict Mode"}),
            },
            "optional": {
                "sun_azimuth": ("FLOAT", {"default": -1.0}),
                "sun_elevation": ("FLOAT", {"default": -1.0}),
                "turbidity": ("FLOAT", {"default": -1.0}),
            },
        }

    @classmethod
    def RETURN_TYPES(cls):
        # Returns: Enhanced Image, Sky Mask (Alpha), Analysis Report (String)
        return ("IMAGE", "IMAGE", "STRING")

    def execute(
        self,
        image: Any,
        location: str,
        season: str,
        time_of_day: str,
        cloud_coverage: float,
        auto_correct: bool,
        strict_physics: bool,
        sun_azimuth: float = -1.0,
        sun_elevation: float = -1.0,
        turbidity: float = -1.0,
    ) -> Tuple[Any, Any, str]:

        logger.info(f"Executing SkyGAN Smart Render (Auto-Correct: {auto_correct})")

        # Validate user-controlled inputs before doing any preset work, so that
        # malformed values fail fast instead of being masked by unrelated
        # preset/atmosphere errors. Use the mapping's insertion order in the
        # error message so it matches the ComfyUI dropdown order.
        try:
            hour_of_day = self._TIME_OF_DAY_HOURS[time_of_day]
        except KeyError:
            raise ValueError(f"Unknown time_of_day {time_of_day!r}; expected one of {list(self._TIME_OF_DAY_HOURS)}") from None

        # 1. Prepare Data
        img_np = self._to_numpy(image)

        # Data Layer: Get presets
        presets = LocationPresets()
        location_preset = presets.get_atmospheric_parameters(location, season)
        time_params = presets.get_sky_parameters(
            location=location,
            season=season,
            time_of_day=hour_of_day,
        )

        # Apply Overrides
        if sun_azimuth >= 0:
            time_params.sun_azimuth = sun_azimuth
        if sun_elevation >= 0:
            time_params.sun_elevation = sun_elevation
        if turbidity >= 0:
            location_preset.turbidity = turbidity
        time_params.cloud_coverage = cloud_coverage

        # 2. Execute The Paradigm Shift
        # We use the intelligent 'smart_render' which performs shadow analysis
        blender = SkyBlender()

        # Note: smart_render returns (image, CorrectionSuggestion)
        # It handles the sky generation internally now
        try:
            enhanced_np, suggestion = blender.smart_render(
                source_image=img_np,
                sky_params=time_params,
                atmo_params=location_preset,
                auto_correct=auto_correct,
                strict_physics=strict_physics,
            )
        except ValueError as e:
            # Catch PhysicsViolationError from strict mode
            raise ValueError(f"Physics Guardrail blocked render: {e}")

        # 3. Recover the Mask (for downstream compositing)
        # Since smart_render composes the final image, we re-run the fast segmentation
        # if the user needs the mask separately.
        mask_np = blender._segment_sky(img_np)

        # 4. Format Output
        report = f"CONFIDENCE: {suggestion.confidence:.2f}\n"
        report += f"ANALYSIS: {suggestion.message}"

        return (self._to_tensor(enhanced_np), self._to_tensor(mask_np), report)


@CustomNodeRegistry.register
class SceneAnalysisNode(BaseNode):
    """Scene analysis node using VLM."""

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
        return ("STRING",)  # Returning JSON string is safest for ComfyUI

    def execute(self, image: Any, detailed: bool) -> Tuple[str]:
        from transformation_portal.vlm import SceneAnalyzer

        img_np = self._to_numpy(image)
        analyzer = SceneAnalyzer()
        analysis = analyzer.analyze_scene(img_np, detailed=detailed)

        return (json.dumps(analysis, indent=2),)


# ... (MaterialSegmentationNode, NeuroaestheticsNode, QualityValidationNode would follow similar patterns using _to_numpy/_to_tensor)

# Export all nodes
__all__ = [
    "CustomNodeRegistry",
    "BaseNode",
    "FluxEnhancementNode",
    "SkyGANNode",
    "SceneAnalysisNode",
]
