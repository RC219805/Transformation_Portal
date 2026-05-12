"""Scene analysis for luxury architectural imagery.

Provides structured scene understanding using LLaVA-1.5:
- Room type and space classification
- Architectural style recognition
- Material identification
- Luxury feature detection
- Lighting condition analysis
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from transformation_portal.vlm.llava import LLaVAProcessor

LLaVAProcessor: Any = None

logger = logging.getLogger(__name__)


def _resolve_llava_processor_class() -> Any:
    global LLaVAProcessor
    if LLaVAProcessor is None:
        from transformation_portal.vlm.llava import LLaVAProcessor as resolved_processor

        LLaVAProcessor = resolved_processor
    return LLaVAProcessor


class SpaceType(Enum):
    """Architectural space types."""

    INTERIOR = "interior"
    EXTERIOR = "exterior"
    AERIAL = "aerial"
    UNKNOWN = "unknown"


class RoomType(Enum):
    """Interior room types."""

    LIVING = "living_room"
    KITCHEN = "kitchen"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    DINING = "dining_room"
    OFFICE = "office"
    POOL_AREA = "pool_area"
    COURTYARD = "courtyard"
    ENTRY = "entry"
    UNKNOWN = "unknown"


class ArchitecturalStyle(Enum):
    """Architectural styles."""

    MODERN = "modern"
    CONTEMPORARY = "contemporary"
    TRADITIONAL = "traditional"
    MEDITERRANEAN = "mediterranean"
    COASTAL = "coastal"
    TRANSITIONAL = "transitional"
    INDUSTRIAL = "industrial"
    MINIMALIST = "minimalist"
    LUXURY_ESTATE = "luxury_estate"
    UNKNOWN = "unknown"


@dataclass
class SceneAnalysis:
    """Structured scene analysis results.

    Attributes:
        space_type: Interior, exterior, or aerial
        room_type: Specific room classification (if interior)
        architectural_style: Detected architectural style
        materials: List of identified materials
        luxury_features: List of luxury elements
        lighting_conditions: Lighting description
        confidence: Confidence score (0-1) if available
        raw_analysis: Full LLaVA response text
    """

    space_type: SpaceType
    room_type: Optional[RoomType]
    architectural_style: ArchitecturalStyle
    materials: List[str]
    luxury_features: List[str]
    lighting_conditions: str
    confidence: float
    raw_analysis: str


class SceneAnalyzer:
    """Analyze architectural scenes using LLaVA-1.5.

    Provides structured scene understanding for luxury real estate imagery.

    Example:
        >>> analyzer = SceneAnalyzer()
        >>> analysis = analyzer.analyze("luxury_kitchen.jpg")
        >>> print(f"Room: {analysis.room_type}, Style: {analysis.architectural_style}")
        >>> print(f"Materials: {', '.join(analysis.materials)}")
    """

    STRUCTURED_ANALYSIS_PROMPT = """Analyze this architectural photograph systematically:

1. SPACE TYPE: Is this an interior, exterior, or aerial view?

2. ROOM TYPE (if interior): What type of room or space is this?
   (living room, kitchen, bedroom, bathroom, dining room, office, pool area, courtyard, entry, etc.)

3. ARCHITECTURAL STYLE: What is the predominant architectural style?
   (modern, contemporary, traditional, mediterranean, coastal, transitional, industrial, minimalist, luxury estate, etc.)

4. MATERIALS: List all visible materials in order of prominence.
   (marble, granite, wood, metal, glass, stone, fabric, tile, concrete, etc.)

5. LUXURY FEATURES: Identify premium or luxury elements.
   (high ceilings, custom finishes, designer fixtures, premium appliances, water features, etc.)

6. LIGHTING: Describe the lighting conditions.
   (natural light, artificial, golden hour, blue hour, mixed, dramatic, soft, etc.)

Provide your analysis in this exact format with clear sections."""

    def __init__(self, llava_processor: Optional["LLaVAProcessor"] = None, **llava_kwargs):
        """Initialize scene analyzer.

        Args:
            llava_processor: Existing LLaVA processor (creates new if None)
            **llava_kwargs: Arguments passed to LLaVAProcessor if creating new
        """
        if llava_processor is not None:
            self.processor = llava_processor
        else:
            self.processor = _resolve_llava_processor_class()(**llava_kwargs)

        logger.info("SceneAnalyzer initialized")

    def analyze(self, image: Union[str, Path, Image.Image, np.ndarray], detailed: bool = True) -> SceneAnalysis:
        """Analyze architectural scene.

        Args:
            image: Input image
            detailed: Whether to perform detailed analysis

        Returns:
            Structured SceneAnalysis object
        """
        # Get raw analysis from LLaVA
        raw_analysis = self.processor.analyze_image(
            image,
            prompt=self.STRUCTURED_ANALYSIS_PROMPT if detailed else None,
            temperature=0.1,  # Low temperature for consistent classification
        )

        # Parse structured response
        space_type = self._extract_space_type(raw_analysis)
        room_type = self._extract_room_type(raw_analysis) if space_type == SpaceType.INTERIOR else None
        architectural_style = self._extract_style(raw_analysis)
        materials = self._extract_materials(raw_analysis)
        luxury_features = self._extract_luxury_features(raw_analysis)
        lighting = self._extract_lighting(raw_analysis)

        # Confidence estimation (could be enhanced with entropy-based scoring)
        confidence = 0.85  # Placeholder - LLaVA doesn't provide native confidence scores

        return SceneAnalysis(
            space_type=space_type,
            room_type=room_type,
            architectural_style=architectural_style,
            materials=materials,
            luxury_features=luxury_features,
            lighting_conditions=lighting,
            confidence=confidence,
            raw_analysis=raw_analysis,
        )

    def _extract_space_type(self, text: str) -> SpaceType:
        """Extract space type from analysis text."""
        text_lower = text.lower()

        if "aerial" in text_lower or "overhead" in text_lower or "drone" in text_lower:
            return SpaceType.AERIAL
        elif "exterior" in text_lower or "outdoor" in text_lower or "facade" in text_lower:
            return SpaceType.EXTERIOR
        elif "interior" in text_lower or "indoor" in text_lower or "room" in text_lower:
            return SpaceType.INTERIOR

        return SpaceType.UNKNOWN

    def _extract_room_type(self, text: str) -> RoomType:
        """Extract room type from analysis text."""
        text_lower = text.lower()

        # Room type detection patterns
        room_patterns = {
            RoomType.KITCHEN: ["kitchen", "culinary"],
            RoomType.BATHROOM: ["bathroom", "bath", "powder room"],
            RoomType.BEDROOM: ["bedroom", "master suite", "sleeping"],
            RoomType.LIVING: ["living room", "living space", "lounge", "great room"],
            RoomType.DINING: ["dining room", "dining area"],
            RoomType.OFFICE: ["office", "study", "library"],
            RoomType.POOL_AREA: ["pool", "pool area", "poolside"],
            RoomType.COURTYARD: ["courtyard", "patio", "terrace"],
            RoomType.ENTRY: ["entry", "foyer", "entrance", "entryway"],
        }

        for room_type, keywords in room_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return room_type

        return RoomType.UNKNOWN

    def _extract_style(self, text: str) -> ArchitecturalStyle:
        """Extract architectural style from analysis text."""
        text_lower = text.lower()

        style_patterns = {
            ArchitecturalStyle.MEDITERRANEAN: ["mediterranean", "spanish", "tuscan"],
            ArchitecturalStyle.MODERN: ["modern", "mid-century"],
            ArchitecturalStyle.CONTEMPORARY: ["contemporary", "current"],
            ArchitecturalStyle.TRADITIONAL: ["traditional", "classic"],
            ArchitecturalStyle.COASTAL: ["coastal", "beach", "seaside"],
            ArchitecturalStyle.TRANSITIONAL: ["transitional", "blend"],
            ArchitecturalStyle.INDUSTRIAL: ["industrial", "loft"],
            ArchitecturalStyle.MINIMALIST: ["minimalist", "minimal", "zen"],
            ArchitecturalStyle.LUXURY_ESTATE: ["luxury estate", "grand", "palatial"],
        }

        for style, keywords in style_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return style

        return ArchitecturalStyle.UNKNOWN

    def _extract_materials(self, text: str) -> List[str]:
        """Extract materials from analysis text."""
        # Common luxury materials
        materials = [
            "marble",
            "granite",
            "quartz",
            "wood",
            "hardwood",
            "oak",
            "walnut",
            "glass",
            "metal",
            "stainless steel",
            "bronze",
            "brass",
            "copper",
            "stone",
            "limestone",
            "travertine",
            "slate",
            "tile",
            "porcelain",
            "fabric",
            "leather",
            "concrete",
            "plaster",
            "brick",
        ]

        text_lower = text.lower()
        found_materials = []

        for material in materials:
            if material in text_lower:
                found_materials.append(material)

        return found_materials

    def _extract_luxury_features(self, text: str) -> List[str]:
        """Extract luxury features from analysis text."""
        # Luxury feature keywords
        features = [
            "high ceiling",
            "vaulted ceiling",
            "custom",
            "designer",
            "premium",
            "luxury",
            "high-end",
            "statement",
            "chandelier",
            "water feature",
            "fountain",
            "fireplace",
            "built-in",
            "wine cellar",
            "home theater",
            "spa",
            "infinity pool",
            "smart home",
            "automation",
            "panoramic view",
            "ocean view",
        ]

        text_lower = text.lower()
        found_features = []

        for feature in features:
            if feature in text_lower:
                found_features.append(feature)

        return found_features

    def _extract_lighting(self, text: str) -> str:
        """Extract lighting description from analysis text."""
        text_lower = text.lower()

        # Look for lighting section
        if "lighting:" in text_lower:
            # Extract text after "lighting:" label
            lighting_section = text_lower.split("lighting:")[-1].split("\n")[0]
            return lighting_section.strip()

        # Fallback: look for common lighting terms
        lighting_terms = [
            "natural light",
            "golden hour",
            "blue hour",
            "soft light",
            "dramatic light",
            "ambient",
            "artificial",
            "mixed lighting",
            "bright",
            "dim",
            "moody",
        ]

        for term in lighting_terms:
            if term in text_lower:
                return term

        return "standard lighting"

    def get_processing_recommendations(self, analysis: SceneAnalysis) -> Dict[str, Any]:
        """Get processing recommendations based on scene analysis.

        Returns recommended settings for enhancement based on detected scene characteristics.

        Args:
            analysis: Scene analysis results

        Returns:
            Dictionary with recommended processing parameters
        """
        recommendations = {
            "scene_type": analysis.space_type.value,
            "suggested_preset": None,
            "enhancement_strength": 0.5,
            "material_response_strength": 0.7,
            "depth_processing": True,
            "atmospheric_effects": False,
        }

        # Room-specific recommendations
        if analysis.room_type == RoomType.KITCHEN:
            recommendations["suggested_preset"] = "kitchen-bright"
            recommendations["enhancement_strength"] = 0.45
            recommendations["material_response_strength"] = 0.75

        elif analysis.room_type == RoomType.BATHROOM:
            recommendations["suggested_preset"] = "bathroom-spa"
            recommendations["enhancement_strength"] = 0.4
            recommendations["material_response_strength"] = 0.7

        elif analysis.room_type == RoomType.BEDROOM:
            recommendations["suggested_preset"] = "bedroom-cozy"
            recommendations["enhancement_strength"] = 0.35

        elif analysis.room_type == RoomType.POOL_AREA:
            recommendations["suggested_preset"] = "pool-luxury"
            recommendations["enhancement_strength"] = 0.5
            recommendations["atmospheric_effects"] = True

        # Style-specific adjustments
        if analysis.architectural_style == ArchitecturalStyle.MODERN:
            recommendations["enhancement_strength"] *= 0.9  # More subtle for modern

        elif analysis.architectural_style == ArchitecturalStyle.MEDITERRANEAN:
            recommendations["atmospheric_effects"] = True  # Enhance coastal atmosphere

        # Lighting-based adjustments
        if "golden hour" in analysis.lighting_conditions.lower():
            recommendations["color_grading"] = "california-golden-hour"
        elif "natural" in analysis.lighting_conditions.lower():
            recommendations["preserve_lighting"] = True

        # Aerial-specific
        if analysis.space_type == SpaceType.AERIAL:
            recommendations["suggested_preset"] = "aerial-estate"
            recommendations["atmospheric_effects"] = True

        return recommendations
