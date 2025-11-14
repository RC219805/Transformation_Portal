"""Emotional optimization for luxury real estate imagery.

Integrates all neuroaesthetics modules to optimize for specific emotional responses:

1. **Nostalgia** (hippocampus-ventral striatum circuits)
   - Warm color palettes
   - Soft golden-hour lighting
   - Natural materials (wood, stone)
   - Heritage architectural details

2. **Aspiration** (ventral striatum, ventromedial PFC)
   - High spatial quality
   - Abundant natural light
   - Open flowing spaces
   - Golden ratio proportions

3. **Desire** (caudate nucleus, reward pathways)
   - Quality craftsmanship visibility
   - Premium material textures
   - Exclusive features
   - Believable luxury (not too perfect = skepticism)

Combines golden ratio, color harmony, and spatial frequency analysis
for scientifically-informed aesthetic optimization.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from transformation_portal.neuroaesthetics.golden_ratio import (
    GoldenRatioAnalyzer,
    GoldenRatioAnalysis
)
from transformation_portal.neuroaesthetics.color_harmony import (
    ColorHarmonyAnalyzer,
    HarmonyAnalysis
)
from transformation_portal.neuroaesthetics.spatial_frequency import (
    SpatialFrequencyAnalyzer,
    SpatialFrequencyAnalysis
)


logger = logging.getLogger(__name__)


class EmotionalTarget(Enum):
    """Target emotional response."""
    NOSTALGIA = "nostalgia"
    ASPIRATION = "aspiration"
    DESIRE = "desire"
    LUXURY = "luxury"
    COMFORT = "comfort"
    ENERGY = "energy"
    SERENITY = "serenity"


@dataclass
class EmotionalProfile:
    """Complete emotional and aesthetic profile.

    Attributes:
        golden_ratio_analysis: Compositional analysis
        color_harmony_analysis: Color harmony analysis
        spatial_frequency_analysis: Spatial frequency analysis
        emotional_scores: Scores for each emotional target (0-1)
        overall_quality: Overall neuroaesthetic quality (0-1)
        optimization_priority: Ordered list of improvements
        enhancement_strategy: Specific enhancement recommendations
    """
    golden_ratio_analysis: GoldenRatioAnalysis
    color_harmony_analysis: HarmonyAnalysis
    spatial_frequency_analysis: SpatialFrequencyAnalysis
    emotional_scores: Dict[str, float]
    overall_quality: float
    optimization_priority: List[Tuple[str, float]]  # (aspect, importance)
    enhancement_strategy: Dict[str, Any]


class EmotionalOptimizer:
    """Optimize imagery for specific emotional responses.

    Integrates compositional, color, and spatial frequency analysis
    to create scientifically-informed enhancement strategies.

    Example:
        >>> optimizer = EmotionalOptimizer()
        >>>
        >>> # Analyze current emotional profile
        >>> profile = optimizer.analyze("luxury_property.jpg")
        >>> print(f"Overall quality: {profile.overall_quality:.2f}")
        >>> print(f"Emotional scores: {profile.emotional_scores}")
        >>>
        >>> # Optimize for specific emotion
        >>> strategy = optimizer.optimize_for_emotion(
        ...     "luxury_property.jpg",
        ...     target_emotion=EmotionalTarget.ASPIRATION
        ... )
        >>> print(f"Enhancement strategy: {strategy}")
    """

    # Emotional target definitions
    EMOTIONAL_REQUIREMENTS = {
        EmotionalTarget.NOSTALGIA: {
            "temperature": 0.5,  # Warm
            "golden_ratio_min": 0.6,
            "visual_comfort_min": 0.7,
            "preferred_harmony": ["analogous", "warm"]
        },
        EmotionalTarget.ASPIRATION: {
            "golden_ratio_min": 0.75,
            "avg_lightness_min": 60,
            "visual_comfort_min": 0.75,
            "spatial_quality": "high"
        },
        EmotionalTarget.DESIRE: {
            "golden_ratio_min": 0.7,
            "color_harmony_min": 0.7,
            "visual_comfort_min": 0.75,
            "material_quality": "premium"
        },
        EmotionalTarget.LUXURY: {
            "avg_saturation_max": 30,  # Sophisticated = low saturation
            "avg_lightness_min": 50,
            "golden_ratio_min": 0.7,
            "visual_comfort_min": 0.8
        },
        EmotionalTarget.COMFORT: {
            "temperature": 0.3,  # Warm
            "preferred_harmony": ["analogous"],
            "visual_comfort_min": 0.8
        },
        EmotionalTarget.SERENITY: {
            "temperature": -0.3,  # Cool
            "preferred_harmony": ["analogous", "cool"],
            "visual_comfort_min": 0.85,
            "balance_score_min": 0.75
        },
        EmotionalTarget.ENERGY: {
            "avg_saturation_min": 40,
            "preferred_harmony": ["complementary"],
            "hsf_min": 0.15  # More detail = energy
        }
    }

    def __init__(self):
        """Initialize emotional optimizer."""
        self.golden_ratio_analyzer = GoldenRatioAnalyzer()
        self.color_harmony_analyzer = ColorHarmonyAnalyzer()
        self.spatial_frequency_analyzer = SpatialFrequencyAnalyzer()

        logger.info("EmotionalOptimizer initialized")

    def analyze(
        self,
        image: Union[str, np.ndarray, Image.Image]
    ) -> EmotionalProfile:
        """Analyze complete emotional and aesthetic profile.

        Args:
            image: Input image

        Returns:
            Complete emotional profile
        """
        logger.info("Analyzing emotional profile...")

        # Run all analyses
        golden_ratio_analysis = self.golden_ratio_analyzer.analyze(image)
        color_harmony_analysis = self.color_harmony_analyzer.analyze(image)
        spatial_frequency_analysis = self.spatial_frequency_analyzer.analyze(image)

        # Calculate emotional scores
        emotional_scores = self._calculate_emotional_scores(
            golden_ratio_analysis,
            color_harmony_analysis,
            spatial_frequency_analysis
        )

        # Calculate overall quality
        overall_quality = self._calculate_overall_quality(
            golden_ratio_analysis,
            color_harmony_analysis,
            spatial_frequency_analysis
        )

        # Determine optimization priorities
        optimization_priority = self._determine_optimization_priority(
            golden_ratio_analysis,
            color_harmony_analysis,
            spatial_frequency_analysis
        )

        # Generate enhancement strategy
        enhancement_strategy = self._generate_enhancement_strategy(
            golden_ratio_analysis,
            color_harmony_analysis,
            spatial_frequency_analysis,
            emotional_scores
        )

        return EmotionalProfile(
            golden_ratio_analysis=golden_ratio_analysis,
            color_harmony_analysis=color_harmony_analysis,
            spatial_frequency_analysis=spatial_frequency_analysis,
            emotional_scores=emotional_scores,
            overall_quality=overall_quality,
            optimization_priority=optimization_priority,
            enhancement_strategy=enhancement_strategy
        )

    def optimize_for_emotion(
        self,
        image: Union[str, np.ndarray, Image.Image],
        target_emotion: EmotionalTarget
    ) -> Dict[str, Any]:
        """Generate optimization strategy for target emotion.

        Args:
            image: Input image
            target_emotion: Desired emotional response

        Returns:
            Dictionary with specific enhancement recommendations
        """
        # Analyze current state
        profile = self.analyze(image)

        # Get requirements for target emotion
        requirements = self.EMOTIONAL_REQUIREMENTS[target_emotion]

        # Identify gaps
        gaps = self._identify_gaps(profile, requirements)

        # Generate targeted strategy
        strategy = {
            "target_emotion": target_emotion.value,
            "current_score": profile.emotional_scores.get(target_emotion.value, 0.0),
            "gaps": gaps,
            "recommended_adjustments": self._generate_adjustments(gaps, requirements),
            "processing_parameters": self._get_processing_parameters(requirements)
        }

        return strategy

    def _calculate_emotional_scores(
        self,
        golden_ratio: GoldenRatioAnalysis,
        color_harmony: HarmonyAnalysis,
        spatial_frequency: SpatialFrequencyAnalysis
    ) -> Dict[str, float]:
        """Calculate scores for each emotional target.

        Args:
            golden_ratio: Golden ratio analysis
            color_harmony: Color harmony analysis
            spatial_frequency: Spatial frequency analysis

        Returns:
            Dictionary mapping emotions to scores (0-1)
        """
        # Get color emotional profile as base
        scores = color_harmony.emotional_profile.copy()

        # Adjust based on composition
        if golden_ratio.score > 0.75:
            scores["aspiration"] = max(scores.get("aspiration", 0), 0.8)

        # Adjust based on visual comfort
        if spatial_frequency.visual_comfort_score > 0.8:
            scores["luxury"] = max(scores.get("luxury", 0), 0.75)

        # Overall quality affects luxury and aspiration
        overall = (golden_ratio.score + color_harmony.harmony_score +
                   spatial_frequency.visual_comfort_score) / 3

        if overall > 0.8:
            scores["luxury"] = max(scores.get("luxury", 0), 0.85)
            scores["aspiration"] = max(scores.get("aspiration", 0), 0.8)

        return scores

    def _calculate_overall_quality(
        self,
        golden_ratio: GoldenRatioAnalysis,
        color_harmony: HarmonyAnalysis,
        spatial_frequency: SpatialFrequencyAnalysis
    ) -> float:
        """Calculate overall neuroaesthetic quality.

        Args:
            golden_ratio: Golden ratio analysis
            color_harmony: Color harmony analysis
            spatial_frequency: Spatial frequency analysis

        Returns:
            Overall quality score (0-1)
        """
        # Weighted average
        weights = {
            "golden_ratio": 0.3,
            "color_harmony": 0.35,
            "spatial_frequency": 0.35
        }

        quality = (
            golden_ratio.score * weights["golden_ratio"] +
            color_harmony.harmony_score * weights["color_harmony"] +
            spatial_frequency.visual_comfort_score * weights["spatial_frequency"]
        )

        return quality

    def _determine_optimization_priority(
        self,
        golden_ratio: GoldenRatioAnalysis,
        color_harmony: HarmonyAnalysis,
        spatial_frequency: SpatialFrequencyAnalysis
    ) -> List[Tuple[str, float]]:
        """Determine which aspects need optimization most.

        Args:
            golden_ratio: Golden ratio analysis
            color_harmony: Color harmony analysis
            spatial_frequency: Spatial frequency analysis

        Returns:
            List of (aspect, importance) tuples, ordered by importance
        """
        priorities = []

        # Composition priority
        if golden_ratio.score < 0.7:
            importance = (0.7 - golden_ratio.score) * 1.5
            priorities.append(("composition", importance))

        # Color harmony priority
        if color_harmony.harmony_score < 0.7:
            importance = (0.7 - color_harmony.harmony_score) * 1.3
            priorities.append(("color_harmony", importance))

        # Spatial frequency priority
        if spatial_frequency.visual_comfort_score < 0.7:
            importance = (0.7 - spatial_frequency.visual_comfort_score) * 1.2
            priorities.append(("spatial_frequency", importance))

        # Sort by importance
        priorities.sort(key=lambda x: x[1], reverse=True)

        return priorities

    def _generate_enhancement_strategy(
        self,
        golden_ratio: GoldenRatioAnalysis,
        color_harmony: HarmonyAnalysis,
        spatial_frequency: SpatialFrequencyAnalysis,
        emotional_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate comprehensive enhancement strategy.

        Args:
            golden_ratio: Golden ratio analysis
            color_harmony: Color harmony analysis
            spatial_frequency: Spatial frequency analysis
            emotional_scores: Emotional scores

        Returns:
            Enhancement strategy dictionary
        """
        strategy = {
            "crop_adjustment": None,
            "color_adjustments": {},
            "frequency_adjustments": {},
            "processing_order": []
        }

        # Crop recommendation based on golden ratio
        if golden_ratio.score < 0.7:
            strategy["crop_adjustment"] = {
                "recommended": True,
                "target_aspect_ratio": 1.618,  # Golden ratio
                "reason": "Improve compositional balance"
            }
            strategy["processing_order"].append("crop")

        # Color adjustments
        if color_harmony.harmony_score < 0.7:
            strategy["color_adjustments"] = {
                "target_temperature": color_harmony.temperature,
                "harmony_type": color_harmony.harmony_type.value,
                "adjustments": color_harmony.recommendations
            }
            strategy["processing_order"].append("color_harmony")

        # Spatial frequency adjustments
        if spatial_frequency.visual_comfort_score < 0.7:
            strategy["frequency_adjustments"] = {
                "reduce_hsf": spatial_frequency.hsf_energy > 0.25,
                "enhance_lsf": spatial_frequency.lsf_energy < 0.35,
                "adjustments": spatial_frequency.recommendations
            }
            strategy["processing_order"].append("spatial_frequency")

        # Add general enhancement if all scores are good
        if not strategy["processing_order"]:
            strategy["processing_order"].append("fine_tuning")

        return strategy

    def _identify_gaps(
        self,
        profile: EmotionalProfile,
        requirements: Dict
    ) -> List[Dict[str, Any]]:
        """Identify gaps between current state and requirements.

        Args:
            profile: Current emotional profile
            requirements: Target requirements

        Returns:
            List of gap descriptions
        """
        gaps = []

        # Check golden ratio
        if "golden_ratio_min" in requirements:
            if profile.golden_ratio_analysis.score < requirements["golden_ratio_min"]:
                gaps.append({
                    "aspect": "composition",
                    "current": profile.golden_ratio_analysis.score,
                    "target": requirements["golden_ratio_min"],
                    "gap": requirements["golden_ratio_min"] - profile.golden_ratio_analysis.score
                })

        # Check color harmony
        if "color_harmony_min" in requirements:
            if profile.color_harmony_analysis.harmony_score < requirements["color_harmony_min"]:
                gaps.append({
                    "aspect": "color_harmony",
                    "current": profile.color_harmony_analysis.harmony_score,
                    "target": requirements["color_harmony_min"],
                    "gap": requirements["color_harmony_min"] - profile.color_harmony_analysis.harmony_score
                })

        # Check visual comfort
        if "visual_comfort_min" in requirements:
            if profile.spatial_frequency_analysis.visual_comfort_score < requirements["visual_comfort_min"]:
                gaps.append({
                    "aspect": "visual_comfort",
                    "current": profile.spatial_frequency_analysis.visual_comfort_score,
                    "target": requirements["visual_comfort_min"],
                    "gap": requirements["visual_comfort_min"] - profile.spatial_frequency_analysis.visual_comfort_score
                })

        # Sort by gap size
        gaps.sort(key=lambda x: x["gap"], reverse=True)

        return gaps

    def _generate_adjustments(
        self,
        gaps: List[Dict],
        requirements: Dict
    ) -> List[str]:
        """Generate specific adjustment recommendations.

        Args:
            gaps: Identified gaps
            requirements: Target requirements

        Returns:
            List of actionable recommendations
        """
        adjustments = []

        for gap in gaps:
            if gap["aspect"] == "composition":
                adjustments.append(
                    f"Crop or reframe to improve golden ratio score from "
                    f"{gap['current']:.2f} to {gap['target']:.2f}"
                )

            elif gap["aspect"] == "color_harmony":
                if "temperature" in requirements:
                    target_temp = requirements["temperature"]
                    if target_temp > 0:
                        adjustments.append("Shift color palette toward warm tones")
                    else:
                        adjustments.append("Shift color palette toward cool tones")

            elif gap["aspect"] == "visual_comfort":
                adjustments.append(
                    "Optimize spatial frequency balance: "
                    "reduce high-frequency noise, enhance structural clarity"
                )

        return adjustments

    def _get_processing_parameters(
        self,
        requirements: Dict
    ) -> Dict[str, Any]:
        """Get specific processing parameters for requirements.

        Args:
            requirements: Target requirements

        Returns:
            Processing parameters dictionary
        """
        params = {
            "clarity": 0.5,
            "saturation": 1.0,
            "temperature_shift": 0.0,
            "sharpness": 0.5,
            "smoothing": 0.0
        }

        # Temperature adjustments
        if "temperature" in requirements:
            params["temperature_shift"] = requirements["temperature"] * 10  # Scale for processing

        # Saturation adjustments
        if "avg_saturation_max" in requirements:
            params["saturation"] = 0.85  # Reduce saturation for luxury

        if "avg_saturation_min" in requirements:
            params["saturation"] = 1.15  # Increase saturation for energy

        # Sharpness adjustments
        if "hsf_min" in requirements:
            params["sharpness"] = 0.7  # Increase detail

        # Smoothing for visual comfort
        if "visual_comfort_min" in requirements and requirements["visual_comfort_min"] > 0.8:
            params["smoothing"] = 0.3  # Gentle smoothing

        return params

    def __repr__(self) -> str:
        return "EmotionalOptimizer()"
