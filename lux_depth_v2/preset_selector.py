"""
Scene-aware preset auto-selection for Lux Depth V2.

Maps CLIP scene classification results to optimal quality tier presets:
    - Scene type: interior / exterior
    - Scene subtype: kitchen, pool, etc.
    - Quality tier: standard / max / apex

Enables `--auto-preset` CLI flag for intelligent preset selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    from transformation_portal.segmentation.clip_classifier import CLIPClassifier
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

from lux_depth_v2.config import Preset


logger = logging.getLogger(__name__)


class SceneType(str, Enum):
    """Primary scene classification."""
    INTERIOR = "interior"
    EXTERIOR = "exterior"
    UNKNOWN = "unknown"


class QualityTier(str, Enum):
    """Quality tier selection."""
    STANDARD = "standard"
    MAX = "max"
    APEX = "apex"


@dataclass
class SceneClassification:
    """Scene classification result."""
    scene_type: SceneType
    scene_subtype: str  # e.g., "kitchen", "pool", "bathroom"
    confidence: float
    
    @property
    def is_confident(self) -> bool:
        """Check if classification confidence is acceptable."""
        return self.confidence >= 0.5


@dataclass
class PresetRecommendation:
    """Preset selection recommendation."""
    preset: Preset
    scene: SceneClassification
    fallback_used: bool
    reason: str


class PresetSelector:
    """
    Intelligent preset selection based on CLIP scene classification.
    
    Maps scene understanding to optimal quality presets:
        - Interior Kitchen + APEX → interior_luxury_apex_quality
        - Exterior Pool + APEX → exterior_pool_apex_quality
        - Interior + Max → interior_luxury_max_quality
        - Exterior + Standard → exterior_showcase
    
    Usage:
        >>> selector = PresetSelector()
        >>> recommendation = selector.select_preset(
        ...     image_path="kitchen.jpg",
        ...     quality_tier=QualityTier.APEX
        ... )
        >>> print(recommendation.preset)  # Preset.INTERIOR_LUXURY_APEX_QUALITY
        >>> print(recommendation.reason)  # "Interior kitchen scene detected (confidence: 0.87)"
    """
    
    # Scene type detection prompts
    SCENE_TYPE_PROMPTS = {
        SceneType.INTERIOR: [
            "interior room",
            "indoor space",
            "inside a building",
            "interior design"
        ],
        SceneType.EXTERIOR: [
            "exterior view",
            "outdoor space",
            "outside building",
            "landscape architecture"
        ]
    }
    
    # Scene subtype prompts (room types)
    SCENE_SUBTYPE_PROMPTS = {
        "kitchen": ["kitchen", "culinary space", "cooking area"],
        "bathroom": ["bathroom", "powder room", "spa bathroom"],
        "bedroom": ["bedroom", "master suite", "sleeping quarters"],
        "living_room": ["living room", "great room", "lounge"],
        "pool": ["swimming pool", "pool area", "pool deck"],
        "courtyard": ["courtyard", "patio", "outdoor living space"],
        "facade": ["building facade", "exterior architecture", "building exterior"],
    }
    
    # Preset mapping: (scene_type, subtype, tier) → Preset
    PRESET_MAP: Dict[Tuple[SceneType, str, QualityTier], Preset] = {
        # Interior APEX
        (SceneType.INTERIOR, "kitchen", QualityTier.APEX): Preset.INTERIOR_LUXURY_APEX_QUALITY,
        (SceneType.INTERIOR, "bathroom", QualityTier.APEX): Preset.INTERIOR_LUXURY_APEX_QUALITY,
        (SceneType.INTERIOR, "bedroom", QualityTier.APEX): Preset.INTERIOR_LUXURY_APEX_QUALITY,
        (SceneType.INTERIOR, "living_room", QualityTier.APEX): Preset.INTERIOR_LUXURY_APEX_QUALITY,
        
        # Interior Max
        (SceneType.INTERIOR, "kitchen", QualityTier.MAX): Preset.INTERIOR_LUXURY_MAX_QUALITY,
        (SceneType.INTERIOR, "bathroom", QualityTier.MAX): Preset.INTERIOR_LUXURY_MAX_QUALITY,
        (SceneType.INTERIOR, "bedroom", QualityTier.MAX): Preset.INTERIOR_LUXURY_MAX_QUALITY,
        (SceneType.INTERIOR, "living_room", QualityTier.MAX): Preset.INTERIOR_LUXURY_MAX_QUALITY,
        
        # Interior Standard
        (SceneType.INTERIOR, "kitchen", QualityTier.STANDARD): Preset.INTERIOR_LUXURY,
        (SceneType.INTERIOR, "bathroom", QualityTier.STANDARD): Preset.INTERIOR_LUXURY,
        (SceneType.INTERIOR, "bedroom", QualityTier.STANDARD): Preset.INTERIOR_LUXURY,
        (SceneType.INTERIOR, "living_room", QualityTier.STANDARD): Preset.INTERIOR_LUXURY,
        
        # Exterior Pool APEX
        (SceneType.EXTERIOR, "pool", QualityTier.APEX): Preset.EXTERIOR_POOL_APEX_QUALITY,
        (SceneType.EXTERIOR, "courtyard", QualityTier.APEX): Preset.EXTERIOR_POOL_APEX_QUALITY,
        
        # Exterior Pool Max/Standard (use showcase preset)
        (SceneType.EXTERIOR, "pool", QualityTier.MAX): Preset.EXTERIOR_SHOWCASE,
        (SceneType.EXTERIOR, "pool", QualityTier.STANDARD): Preset.EXTERIOR_SHOWCASE,
        (SceneType.EXTERIOR, "courtyard", QualityTier.MAX): Preset.EXTERIOR_SHOWCASE,
        (SceneType.EXTERIOR, "courtyard", QualityTier.STANDARD): Preset.EXTERIOR_SHOWCASE,
        (SceneType.EXTERIOR, "facade", QualityTier.MAX): Preset.EXTERIOR_SHOWCASE,
        (SceneType.EXTERIOR, "facade", QualityTier.STANDARD): Preset.EXTERIOR_SHOWCASE,
    }
    
    # Fallback presets per tier
    FALLBACK_PRESETS = {
        (SceneType.INTERIOR, QualityTier.APEX): Preset.INTERIOR_LUXURY_APEX_QUALITY,
        (SceneType.INTERIOR, QualityTier.MAX): Preset.INTERIOR_LUXURY_MAX_QUALITY,
        (SceneType.INTERIOR, QualityTier.STANDARD): Preset.INTERIOR_LUXURY,
        (SceneType.EXTERIOR, QualityTier.APEX): Preset.EXTERIOR_SHOWCASE,  # Conservative fallback
        (SceneType.EXTERIOR, QualityTier.MAX): Preset.EXTERIOR_SHOWCASE,
        (SceneType.EXTERIOR, QualityTier.STANDARD): Preset.EXTERIOR_SHOWCASE,
        (SceneType.UNKNOWN, QualityTier.APEX): Preset.INTERIOR_LUXURY_APEX_QUALITY,  # Most conservative
        (SceneType.UNKNOWN, QualityTier.MAX): Preset.INTERIOR_LUXURY_MAX_QUALITY,
        (SceneType.UNKNOWN, QualityTier.STANDARD): Preset.INTERIOR_LUXURY,
    }
    
    def __init__(
        self,
        clip_model: str = "openai/clip-vit-base-patch32",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Initialize preset selector.
        
        Args:
            clip_model: CLIP model name (base model for speed)
            confidence_threshold: Minimum confidence for scene classification
            device: Computation device (auto-detected if None)
        """
        if not CLIP_AVAILABLE:
            raise ImportError(
                "CLIP classifier not available. Install with: "
                "pip install transformers torch"
            )
        
        self.confidence_threshold = confidence_threshold
        self.clip = CLIPClassifier(model_name=clip_model, device=device)
        logger.info(f"PresetSelector initialized with {clip_model}")
    
    def classify_scene(self, image: Union[str, Path, Image.Image]) -> SceneClassification:
        """
        Classify scene type and subtype using CLIP.
        
        Args:
            image: Input image (path or PIL Image)
        
        Returns:
            Scene classification with confidence
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Step 1: Detect scene type (interior vs exterior)
        type_prompts = []
        type_labels = []
        for scene_type, prompts in self.SCENE_TYPE_PROMPTS.items():
            type_prompts.extend(prompts)
            type_labels.extend([scene_type] * len(prompts))
        
        type_probs = self.clip.classify_image(image, type_prompts)
        
        # Average probabilities per scene type
        type_scores = {}
        for label in [SceneType.INTERIOR, SceneType.EXTERIOR]:
            indices = [i for i, l in enumerate(type_labels) if l == label]
            type_scores[label] = np.mean([type_probs[i] for i in indices])
        
        # Select scene type with highest score
        scene_type = max(type_scores, key=type_scores.get)
        type_confidence = type_scores[scene_type]
        
        # Step 2: Detect scene subtype (room/area type)
        subtype_prompts = []
        subtype_labels = []
        for subtype, prompts in self.SCENE_SUBTYPE_PROMPTS.items():
            subtype_prompts.extend(prompts)
            subtype_labels.extend([subtype] * len(prompts))
        
        subtype_probs = self.clip.classify_image(image, subtype_prompts)
        
        # Average probabilities per subtype
        subtype_scores = {}
        for subtype in self.SCENE_SUBTYPE_PROMPTS.keys():
            indices = [i for i, l in enumerate(subtype_labels) if l == subtype]
            subtype_scores[subtype] = np.mean([subtype_probs[i] for i in indices])
        
        scene_subtype = max(subtype_scores, key=subtype_scores.get)
        subtype_confidence = subtype_scores[scene_subtype]
        
        # Combined confidence (conservative: use minimum)
        overall_confidence = min(type_confidence, subtype_confidence)
        
        logger.info(
            f"Scene classified: {scene_type.value} {scene_subtype} "
            f"(type: {type_confidence:.3f}, subtype: {subtype_confidence:.3f})"
        )
        
        return SceneClassification(
            scene_type=scene_type,
            scene_subtype=scene_subtype,
            confidence=overall_confidence
        )
    
    def select_preset(
        self,
        image: Union[str, Path, Image.Image],
        quality_tier: QualityTier = QualityTier.MAX
    ) -> PresetRecommendation:
        """
        Select optimal preset for image based on scene and quality tier.
        
        Args:
            image: Input image (path or PIL Image)
            quality_tier: Desired quality tier
        
        Returns:
            Preset recommendation with reasoning
        """
        # Classify scene
        scene = self.classify_scene(image)
        
        # Look up preset in mapping
        lookup_key = (scene.scene_type, scene.scene_subtype, quality_tier)
        preset = self.PRESET_MAP.get(lookup_key)
        fallback_used = False
        
        # Fallback if low confidence or no exact match
        if preset is None or not scene.is_confident:
            fallback_key = (scene.scene_type, quality_tier)
            preset = self.FALLBACK_PRESETS.get(
                fallback_key,
                Preset.INTERIOR_LUXURY_MAX_QUALITY  # Ultimate fallback
            )
            fallback_used = True
            reason = (
                f"Fallback preset for {scene.scene_type.value} + {quality_tier.value} "
                f"(confidence: {scene.confidence:.2f} < {self.confidence_threshold})"
            )
        else:
            reason = (
                f"{scene.scene_type.value.capitalize()} {scene.scene_subtype} scene detected "
                f"(confidence: {scene.confidence:.2f})"
            )
        
        logger.info(f"Selected preset: {preset.value} | {reason}")
        
        return PresetRecommendation(
            preset=preset,
            scene=scene,
            fallback_used=fallback_used,
            reason=reason
        )
    
    def select_preset_from_path(
        self,
        image_path: Union[str, Path],
        quality_tier: QualityTier = QualityTier.MAX
    ) -> Preset:
        """
        Simplified interface: return just the preset.
        
        Args:
            image_path: Path to input image
            quality_tier: Desired quality tier
        
        Returns:
            Selected preset
        """
        recommendation = self.select_preset(image_path, quality_tier)
        return recommendation.preset


def auto_select_preset(
    image_path: Union[str, Path],
    quality_tier: str = "max",
    confidence_threshold: float = 0.5
) -> Preset:
    """
    Convenience function for auto preset selection.
    
    Args:
        image_path: Path to input image
        quality_tier: Quality tier ("standard", "max", or "apex")
        confidence_threshold: Minimum confidence for classification
    
    Returns:
        Selected preset
    
    Example:
        >>> preset = auto_select_preset("kitchen.jpg", quality_tier="apex")
        >>> print(preset)  # Preset.INTERIOR_LUXURY_APEX_QUALITY
    """
    tier_map = {
        "standard": QualityTier.STANDARD,
        "max": QualityTier.MAX,
        "apex": QualityTier.APEX
    }
    
    tier = tier_map.get(quality_tier.lower(), QualityTier.MAX)
    selector = PresetSelector(confidence_threshold=confidence_threshold)
    return selector.select_preset_from_path(image_path, tier)
