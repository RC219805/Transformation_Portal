"""Intelligent architectural prompt generation for FLUX.

Builds optimal prompts based on:
- Scene understanding (room type, style, materials)
- Target emotional response (nostalgia, aspiration, luxury)
- Material emphasis (marble, wood, glass, water)
- Location characteristics (coastal, Mediterranean, traditional)

Prompt engineering principles:
- Specific architectural terminology
- Professional photography descriptors
- Material and finish details
- Lighting conditions
- Quality markers (8k, sharp, detailed)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RoomType(Enum):
    """Room types for architectural prompting."""
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    BEDROOM = "bedroom"
    LIVING = "living room"
    DINING = "dining room"
    OFFICE = "office"
    POOL_AREA = "pool area"
    EXTERIOR = "exterior"
    COURTYARD = "courtyard"
    ENTRY = "entry foyer"


class ArchitecturalStyle(Enum):
    """Architectural styles."""
    MODERN = "modern"
    CONTEMPORARY = "contemporary"
    TRADITIONAL = "traditional"
    MEDITERRANEAN = "Mediterranean"
    COASTAL = "coastal"
    LUXURY_ESTATE = "luxury estate"
    TRANSITIONAL = "transitional"


class EmotionalTarget(Enum):
    """Target emotional responses."""
    NOSTALGIA = "nostalgia"
    ASPIRATION = "aspiration"
    LUXURY = "luxury"
    COMFORT = "comfort"
    SERENITY = "serenity"


@dataclass
class PromptComponents:
    """Structured prompt components.

    Attributes:
        subject: Main subject description
        style_modifiers: Architectural style descriptors
        material_details: Material and finish specifications
        lighting: Lighting conditions
        quality_tags: Quality and technical markers
        atmosphere: Atmospheric and emotional descriptors
        negative_elements: Things to avoid
    """
    subject: str
    style_modifiers: List[str]
    material_details: List[str]
    lighting: List[str]
    quality_tags: List[str]
    atmosphere: List[str]
    negative_elements: List[str]


class ArchitecturalPromptBuilder:
    """Build optimized prompts for architectural enhancement.

    Uses scene understanding and target emotions to generate
    prompts that maximize FLUX's architectural capabilities.

    Example:
        >>> builder = ArchitecturalPromptBuilder()
        >>> prompt = builder.build_prompt(
        ...     room_type=RoomType.KITCHEN,
        ...     style=ArchitecturalStyle.MODERN,
        ...     materials=["marble", "stainless steel", "glass"],
        ...     emotional_target=EmotionalTarget.ASPIRATION
        ... )
        >>> print(prompt)
        "modern luxury kitchen with marble countertops, stainless steel appliances,
        glass backsplash, natural light, professional architectural photography,
        8k, highly detailed, photorealistic"
    """

    # Base quality descriptors
    BASE_QUALITY = [
        "professional architectural photography",
        "8k resolution",
        "highly detailed",
        "sharp focus",
        "photorealistic"
    ]

    # Lighting descriptors
    LIGHTING_DESCRIPTORS = {
        "natural": ["natural light", "abundant daylight", "window light"],
        "golden_hour": ["golden hour light", "warm sunset glow", "soft golden light"],
        "bright": ["bright lighting", "well-lit", "luminous"],
        "dramatic": ["dramatic lighting", "moody atmosphere", "sculptural lighting"],
        "soft": ["soft diffused light", "gentle illumination", "ambient lighting"]
    }

    # Material descriptors
    MATERIAL_DESCRIPTORS = {
        "marble": ["marble countertops", "marble surfaces", "polished marble", "natural stone veining"],
        "granite": ["granite", "natural stone", "textured stone"],
        "wood": ["hardwood floors", "wood cabinetry", "natural wood grain", "warm wood tones"],
        "glass": ["glass", "transparent surfaces", "reflective glass", "modern glazing"],
        "metal": ["metal fixtures", "metallic accents"],
        "stainless steel": ["stainless steel appliances", "brushed metal", "contemporary fixtures"],
        "water": ["water feature", "reflective water", "pool", "fountain"],
        "tile": ["tile work", "ceramic tile", "designer tile"],
        "stone": ["natural stone", "stone features", "textured stonework"]
    }

    # Emotional atmosphere descriptors
    EMOTIONAL_DESCRIPTORS = {
        EmotionalTarget.NOSTALGIA: [
            "warm atmosphere", "inviting", "heritage details",
            "classic elegance", "timeless design"
        ],
        EmotionalTarget.ASPIRATION: [
            "aspirational", "sophisticated", "high-end",
            "elegant", "refined luxury"
        ],
        EmotionalTarget.LUXURY: [
            "luxury", "premium finishes", "bespoke details",
            "custom craftsmanship", "exclusive design"
        ],
        EmotionalTarget.COMFORT: [
            "comfortable", "welcoming", "cozy elegance",
            "livable luxury", "relaxed sophistication"
        ],
        EmotionalTarget.SERENITY: [
            "serene", "peaceful", "tranquil atmosphere",
            "calm elegance", "zen-like quality"
        ]
    }

    # Style-specific descriptors
    STYLE_DESCRIPTORS = {
        ArchitecturalStyle.MODERN: [
            "modern", "minimalist", "clean lines",
            "contemporary design", "sleek"
        ],
        ArchitecturalStyle.CONTEMPORARY: [
            "contemporary", "current design",
            "sophisticated", "urban chic"
        ],
        ArchitecturalStyle.TRADITIONAL: [
            "traditional", "classic", "timeless",
            "elegant details", "refined craftsmanship"
        ],
        ArchitecturalStyle.MEDITERRANEAN: [
            "Mediterranean", "warm tones", "natural materials",
            "arched details", "textured walls"
        ],
        ArchitecturalStyle.COASTAL: [
            "coastal", "beach house", "ocean-inspired",
            "breezy", "light and airy"
        ],
        ArchitecturalStyle.LUXURY_ESTATE: [
            "luxury estate", "grand", "palatial",
            "impressive scale", "resort-style"
        ],
        ArchitecturalStyle.TRANSITIONAL: [
            "transitional", "blend of styles",
            "balanced design", "modern traditional"
        ]
    }

    # Default negative prompt elements
    DEFAULT_NEGATIVE = [
        "oversaturated", "artificial", "fake", "CGI", "unrealistic",
        "distorted", "low quality", "blurry", "noise", "artifacts",
        "overexposed", "underexposed", "cluttered", "messy",
        "poor composition", "amateur"
    ]

    def __init__(self):
        """Initialize prompt builder."""
        logger.info("ArchitecturalPromptBuilder initialized")

    def build_prompt(
        self,
        room_type: Optional[RoomType] = None,
        style: Optional[ArchitecturalStyle] = None,
        materials: Optional[List[str]] = None,
        emotional_target: Optional[EmotionalTarget] = None,
        lighting: str = "natural",
        custom_elements: Optional[List[str]] = None,
        include_quality_tags: bool = True
    ) -> str:
        """Build complete architectural prompt.

        Args:
            room_type: Type of room/space
            style: Architectural style
            materials: List of materials to emphasize
            emotional_target: Target emotional response
            lighting: Lighting condition
            custom_elements: Additional custom descriptors
            include_quality_tags: Include quality markers

        Returns:
            Complete prompt string
        """
        components = []

        # Add style descriptors
        if style is not None:
            style_desc = self.STYLE_DESCRIPTORS.get(style, [])
            if style_desc:
                components.append(style_desc[0])  # Primary style descriptor

        # Add luxury prefix
        if emotional_target == EmotionalTarget.LUXURY:
            components.append("luxury")

        # Add room type
        if room_type is not None:
            components.append(room_type.value)

        # Add material details
        if materials:
            material_desc = self._get_material_descriptions(materials)
            if material_desc:
                components.append("with " + ", ".join(material_desc[:3]))  # Top 3

        # Add lighting
        lighting_desc = self.LIGHTING_DESCRIPTORS.get(lighting, [])
        if lighting_desc:
            components.append(lighting_desc[0])

        # Add emotional atmosphere
        if emotional_target is not None:
            emotional_desc = self.EMOTIONAL_DESCRIPTORS.get(emotional_target, [])
            if emotional_desc:
                components.extend(emotional_desc[:2])  # Top 2

        # Add custom elements
        if custom_elements:
            components.extend(custom_elements)

        # Add quality tags
        if include_quality_tags:
            components.extend(self.BASE_QUALITY)

        # Join components
        prompt = ", ".join(components)

        logger.debug(f"Built prompt: {prompt}")

        return prompt

    def build_negative_prompt(
        self,
        custom_negatives: Optional[List[str]] = None
    ) -> str:
        """Build negative prompt.

        Args:
            custom_negatives: Additional elements to avoid

        Returns:
            Negative prompt string
        """
        negatives = self.DEFAULT_NEGATIVE.copy()

        if custom_negatives:
            negatives.extend(custom_negatives)

        return ", ".join(negatives)

    def build_from_scene_analysis(
        self,
        scene_analysis: Dict[str, any],
        emotional_target: Optional[EmotionalTarget] = None
    ) -> Dict[str, str]:
        """Build prompts from scene analysis results.

        Integrates with transformation_portal.vlm.SceneAnalyzer output.

        Args:
            scene_analysis: Scene analysis dictionary with keys:
                - room_type: str
                - architectural_style: str
                - materials: List[str]
                - lighting_conditions: str
            emotional_target: Override emotional target

        Returns:
            Dictionary with 'prompt' and 'negative_prompt'
        """
        # Map string values to enums
        room_type = None
        if "room_type" in scene_analysis:
            room_str = scene_analysis["room_type"]
            room_type = self._map_room_type(room_str)

        style = None
        if "architectural_style" in scene_analysis:
            style_str = scene_analysis["architectural_style"]
            style = self._map_style(style_str)

        materials = scene_analysis.get("materials", [])

        # Infer lighting from conditions
        lighting = "natural"
        if "lighting_conditions" in scene_analysis:
            lighting_str = scene_analysis["lighting_conditions"].lower()
            if "golden" in lighting_str:
                lighting = "golden_hour"
            elif "dramatic" in lighting_str:
                lighting = "dramatic"
            elif "soft" in lighting_str:
                lighting = "soft"

        # Build prompt
        prompt = self.build_prompt(
            room_type=room_type,
            style=style,
            materials=materials,
            emotional_target=emotional_target,
            lighting=lighting
        )

        negative_prompt = self.build_negative_prompt()

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt
        }

    def build_progressive_prompts(
        self,
        base_prompt: str,
        num_variations: int = 3
    ) -> List[str]:
        """Generate progressive prompt variations.

        Creates variations with increasing detail levels for
        iterative refinement.

        Args:
            base_prompt: Base prompt
            num_variations: Number of variations

        Returns:
            List of progressively detailed prompts
        """
        variations = [base_prompt]

        # Add detail modifiers progressively
        detail_levels = [
            ["refined details", "enhanced clarity"],
            ["intricate details", "premium finishes", "exceptional quality"],
            ["ultra-detailed", "masterful craftsmanship", "museum quality"]
        ]

        for i in range(min(num_variations - 1, len(detail_levels))):
            enhanced = base_prompt + ", " + ", ".join(detail_levels[i])
            variations.append(enhanced)

        return variations

    def _get_material_descriptions(
        self,
        materials: List[str]
    ) -> List[str]:
        """Get descriptive phrases for materials.

        Args:
            materials: List of material names

        Returns:
            List of descriptive phrases
        """
        descriptions = []

        for material in materials:
            material_lower = material.lower()
            if material_lower in self.MATERIAL_DESCRIPTORS:
                desc_list = self.MATERIAL_DESCRIPTORS[material_lower]
                descriptions.append(desc_list[0])  # Primary descriptor

        return descriptions

    def _map_room_type(self, room_str: str) -> Optional[RoomType]:
        """Map string to RoomType enum."""
        room_str_lower = room_str.lower()

        mapping = {
            "kitchen": RoomType.KITCHEN,
            "bathroom": RoomType.BATHROOM,
            "bath": RoomType.BATHROOM,
            "bedroom": RoomType.BEDROOM,
            "living": RoomType.LIVING,
            "living room": RoomType.LIVING,
            "dining": RoomType.DINING,
            "dining room": RoomType.DINING,
            "office": RoomType.OFFICE,
            "pool": RoomType.POOL_AREA,
            "exterior": RoomType.EXTERIOR,
            "courtyard": RoomType.COURTYARD,
            "entry": RoomType.ENTRY,
            "foyer": RoomType.ENTRY
        }

        for key, value in mapping.items():
            if key in room_str_lower:
                return value

        return None

    def _map_style(self, style_str: str) -> Optional[ArchitecturalStyle]:
        """Map string to ArchitecturalStyle enum."""
        style_str_lower = style_str.lower()

        mapping = {
            "modern": ArchitecturalStyle.MODERN,
            "contemporary": ArchitecturalStyle.CONTEMPORARY,
            "traditional": ArchitecturalStyle.TRADITIONAL,
            "mediterranean": ArchitecturalStyle.MEDITERRANEAN,
            "coastal": ArchitecturalStyle.COASTAL,
            "luxury": ArchitecturalStyle.LUXURY_ESTATE,
            "transitional": ArchitecturalStyle.TRANSITIONAL
        }

        for key, value in mapping.items():
            if key in style_str_lower:
                return value

        return None

    def __repr__(self) -> str:
        return "ArchitecturalPromptBuilder()"
