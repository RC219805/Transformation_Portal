"""Pre-configured architectural photography style presets.

Curated collection of professional architectural photography styles
from magazines, award-winning photographers, and signature aesthetics.

Style Categories:
- Editorial: Architectural Digest, Dwell, Elle Decor
- Luxury: High-end real estate photography
- Minimalist: Clean, Scandinavian, Japanese aesthetics
- Warm/Inviting: Residential, lifestyle photography
- Dramatic: High-contrast, moody lighting

Each preset includes:
- Reference image path
- Style description
- Recommended strength
- Target prompt
- Color grading notes
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class StylePreset:
    """Style preset configuration.

    Attributes:
        name: Preset name
        description: Style description
        reference_image: Path to reference image
        prompt: Recommended text prompt
        strength: Recommended style strength (0-1)
        category: Style category
        color_temp: Color temperature (warm/neutral/cool)
        contrast: Contrast level (low/medium/high)
        tags: Searchable tags
    """
    name: str
    description: str
    reference_image: str
    prompt: str
    strength: float = 0.7
    category: str = "general"
    color_temp: str = "neutral"
    contrast: str = "medium"
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ArchitecturalStylePresets:
    """Pre-configured architectural photography styles.

    Provides curated style presets for common architectural photography
    aesthetics. Each preset is based on professional reference images
    and optimized parameters.

    Example:
        >>> presets = ArchitecturalStylePresets()
        >>> preset = presets.get_preset("architectural_digest")
        >>> print(preset.description)
    """

    # Note: In production, these would point to actual curated reference images
    # For now, these are placeholder paths showing the structure
    PRESETS: Dict[str, StylePreset] = {
        "architectural_digest": StylePreset(
            name="Architectural Digest",
            description=(
                "Editorial luxury aesthetic with warm, inviting tones. "
                "Perfect balance of aspiration and livability. Rich colors, "
                "sophisticated lighting, and impeccable composition."
            ),
            reference_image="references/editorial/architectural_digest.jpg",
            prompt=(
                "architectural digest magazine photography, editorial luxury, "
                "warm sophisticated lighting, rich colors, professional interior "
                "photography, 8k, high detail"
            ),
            strength=0.75,
            category="editorial",
            color_temp="warm",
            contrast="medium",
            tags=["luxury", "editorial", "warm", "sophisticated", "magazine"]
        ),

        "dwell_modern": StylePreset(
            name="Dwell Modern",
            description=(
                "Clean, modern aesthetic with natural light emphasis. "
                "Scandinavian-influenced with muted tones and minimalist "
                "composition. Celebrates simplicity and functionality."
            ),
            reference_image="references/editorial/dwell_modern.jpg",
            prompt=(
                "dwell magazine modern architecture, clean lines, natural light, "
                "scandinavian aesthetic, minimalist, neutral tones, professional "
                "architectural photography"
            ),
            strength=0.70,
            category="editorial",
            color_temp="cool",
            contrast="low",
            tags=["modern", "minimalist", "scandinavian", "natural light", "clean"]
        ),

        "elle_decor_glamorous": StylePreset(
            name="Elle Decor Glamorous",
            description=(
                "Glamorous, high-style aesthetic with dramatic lighting. "
                "Bold colors, luxurious textures, and fashion-forward "
                "composition. Emphasizes drama and opulence."
            ),
            reference_image="references/editorial/elle_decor_glamorous.jpg",
            prompt=(
                "elle decor magazine photography, glamorous luxury interior, "
                "dramatic lighting, bold colors, fashion-forward, high style, "
                "professional editorial photography"
            ),
            strength=0.80,
            category="editorial",
            color_temp="warm",
            contrast="high",
            tags=["glamorous", "dramatic", "bold", "luxury", "high-style"]
        ),

        "luxury_real_estate": StylePreset(
            name="Luxury Real Estate",
            description=(
                "Premium real estate photography optimized for high-end "
                "property marketing. Bright, inviting, aspirational. "
                "Balances realism with enhancement."
            ),
            reference_image="references/luxury/real_estate_premium.jpg",
            prompt=(
                "luxury real estate photography, professional property marketing, "
                "bright inviting lighting, aspirational, photorealistic, "
                "high-end residential, 8k"
            ),
            strength=0.65,
            category="luxury",
            color_temp="warm",
            contrast="medium",
            tags=["real estate", "luxury", "bright", "aspirational", "marketing"]
        ),

        "coastal_luxury": StylePreset(
            name="Coastal Luxury",
            description=(
                "Sophisticated coastal aesthetic with bright, airy feel. "
                "Emphasis on natural light, ocean views, and indoor-outdoor "
                "connection. Fresh, serene, upscale."
            ),
            reference_image="references/luxury/coastal_luxury.jpg",
            prompt=(
                "luxury coastal architecture photography, bright airy interior, "
                "ocean view, natural light, indoor-outdoor living, fresh serene, "
                "professional architectural photography"
            ),
            strength=0.70,
            category="luxury",
            color_temp="cool",
            contrast="low",
            tags=["coastal", "airy", "bright", "ocean", "serene", "luxury"]
        ),

        "minimalist_zen": StylePreset(
            name="Minimalist Zen",
            description=(
                "Japanese-influenced minimalism with emphasis on negative space, "
                "natural materials, and tranquility. Muted palette, soft light, "
                "meditative quality."
            ),
            reference_image="references/minimalist/zen_aesthetic.jpg",
            prompt=(
                "minimalist zen interior photography, japanese aesthetic, "
                "negative space, natural materials, tranquil, soft diffuse light, "
                "meditative, clean composition"
            ),
            strength=0.75,
            category="minimalist",
            color_temp="neutral",
            contrast="low",
            tags=["minimalist", "zen", "japanese", "tranquil", "natural"]
        ),

        "scandinavian_hygge": StylePreset(
            name="Scandinavian Hygge",
            description=(
                "Warm, inviting Scandinavian aesthetic with cozy textures "
                "and natural light. Emphasis on comfort, natural materials, "
                "and lived-in warmth."
            ),
            reference_image="references/minimalist/scandinavian_hygge.jpg",
            prompt=(
                "scandinavian hygge interior photography, cozy warm aesthetic, "
                "natural light, wood textures, inviting comfortable, "
                "professional residential photography"
            ),
            strength=0.70,
            category="minimalist",
            color_temp="warm",
            contrast="low",
            tags=["scandinavian", "hygge", "cozy", "warm", "natural", "comfortable"]
        ),

        "industrial_loft": StylePreset(
            name="Industrial Loft",
            description=(
                "Urban industrial aesthetic with exposed materials, "
                "high ceilings, and dramatic natural light. Raw textures "
                "balanced with refined furnishings."
            ),
            reference_image="references/modern/industrial_loft.jpg",
            prompt=(
                "industrial loft photography, exposed brick and beams, "
                "dramatic natural light, urban aesthetic, high ceilings, "
                "raw refined, professional architectural photography"
            ),
            strength=0.70,
            category="modern",
            color_temp="cool",
            contrast="high",
            tags=["industrial", "loft", "urban", "dramatic", "raw", "modern"]
        ),

        "mid_century_modern": StylePreset(
            name="Mid-Century Modern",
            description=(
                "Classic mid-century modern aesthetic with period-accurate "
                "colors and lighting. Warm tones, clean lines, and iconic "
                "furniture emphasized."
            ),
            reference_image="references/modern/mid_century.jpg",
            prompt=(
                "mid-century modern interior photography, period authentic, "
                "warm retro tones, clean lines, iconic furniture, "
                "professional architectural photography, vintage aesthetic"
            ),
            strength=0.75,
            category="modern",
            color_temp="warm",
            contrast="medium",
            tags=["mid-century", "modern", "retro", "vintage", "classic"]
        ),

        "contemporary_sleek": StylePreset(
            name="Contemporary Sleek",
            description=(
                "Ultra-modern contemporary with sleek surfaces and "
                "sophisticated lighting. High-tech, precise, and refined. "
                "Emphasizes geometry and materials."
            ),
            reference_image="references/modern/contemporary_sleek.jpg",
            prompt=(
                "contemporary interior photography, sleek modern, sophisticated "
                "lighting, high-tech refined, geometric composition, "
                "professional architectural photography, ultra modern"
            ),
            strength=0.70,
            category="modern",
            color_temp="cool",
            contrast="medium",
            tags=["contemporary", "sleek", "modern", "high-tech", "geometric"]
        ),

        "dramatic_moody": StylePreset(
            name="Dramatic Moody",
            description=(
                "High-contrast, moody aesthetic with dramatic shadows "
                "and selective lighting. Cinema-quality with rich blacks "
                "and glowing highlights."
            ),
            reference_image="references/dramatic/moody_interior.jpg",
            prompt=(
                "dramatic moody interior photography, high contrast lighting, "
                "rich shadows, selective illumination, cinematic quality, "
                "professional architectural photography, atmospheric"
            ),
            strength=0.80,
            category="dramatic",
            color_temp="warm",
            contrast="high",
            tags=["dramatic", "moody", "high-contrast", "cinematic", "atmospheric"]
        ),

        "golden_hour_glow": StylePreset(
            name="Golden Hour Glow",
            description=(
                "Warm golden hour lighting with sun-washed interiors. "
                "Romantic, ethereal quality with soft shadows and "
                "glowing highlights."
            ),
            reference_image="references/dramatic/golden_hour.jpg",
            prompt=(
                "golden hour interior photography, warm sun-washed lighting, "
                "romantic ethereal quality, soft shadows, glowing highlights, "
                "professional architectural photography, magic hour"
            ),
            strength=0.75,
            category="dramatic",
            color_temp="warm",
            contrast="medium",
            tags=["golden hour", "warm", "romantic", "ethereal", "sun-washed"]
        ),

        "twilight_blue_hour": StylePreset(
            name="Twilight Blue Hour",
            description=(
                "Blue hour twilight aesthetic with interior lights glowing "
                "against dusky blue sky. Sophisticated balance of interior "
                "and exterior lighting."
            ),
            reference_image="references/dramatic/twilight_blue.jpg",
            prompt=(
                "twilight blue hour architecture photography, interior lights "
                "glowing, dusky blue sky, sophisticated lighting balance, "
                "professional architectural photography, magic hour"
            ),
            strength=0.75,
            category="dramatic",
            color_temp="cool",
            contrast="medium",
            tags=["twilight", "blue hour", "evening", "sophisticated", "balanced"]
        ),

        "natural_organic": StylePreset(
            name="Natural Organic",
            description=(
                "Emphasis on natural materials, organic textures, and "
                "connection to nature. Earthy tones, soft natural light, "
                "biophilic design principles."
            ),
            reference_image="references/natural/organic_natural.jpg",
            prompt=(
                "natural organic interior photography, earth tones, natural "
                "materials, soft natural light, biophilic design, connection "
                "to nature, professional architectural photography"
            ),
            strength=0.70,
            category="natural",
            color_temp="warm",
            contrast="low",
            tags=["natural", "organic", "earthy", "biophilic", "sustainable"]
        ),

        "bright_airy_residential": StylePreset(
            name="Bright Airy Residential",
            description=(
                "Light, bright, and welcoming residential aesthetic. "
                "Optimized for family homes with emphasis on livability "
                "and warmth. Fresh, clean, inviting."
            ),
            reference_image="references/residential/bright_airy.jpg",
            prompt=(
                "bright airy residential photography, welcoming family home, "
                "fresh clean aesthetic, warm inviting lighting, livable spaces, "
                "professional real estate photography"
            ),
            strength=0.65,
            category="residential",
            color_temp="warm",
            contrast="low",
            tags=["bright", "airy", "residential", "family", "welcoming", "livable"]
        ),
    }

    @classmethod
    def get_preset(cls, name: str) -> Dict[str, any]:
        """Get style preset by name.

        Args:
            name: Preset name

        Returns:
            Preset configuration dictionary

        Raises:
            ValueError: If preset not found
        """
        if name not in cls.PRESETS:
            available = list(cls.PRESETS.keys())
            raise ValueError(
                f"Preset '{name}' not found. Available presets: {available}"
            )

        preset = cls.PRESETS[name]

        return {
            "name": preset.name,
            "description": preset.description,
            "reference_image": preset.reference_image,
            "prompt": preset.prompt,
            "strength": preset.strength,
            "category": preset.category,
            "color_temp": preset.color_temp,
            "contrast": preset.contrast,
            "tags": preset.tags
        }

    @classmethod
    def list_presets(
        cls,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[str]:
        """List available presets, optionally filtered.

        Args:
            category: Filter by category
            tags: Filter by tags (any match)

        Returns:
            List of preset names
        """
        presets = cls.PRESETS.values()

        if category:
            presets = [p for p in presets if p.category == category]

        if tags:
            presets = [
                p for p in presets
                if any(tag in p.tags for tag in tags)
            ]

        return [p.name for p in presets]

    @classmethod
    def get_categories(cls) -> List[str]:
        """Get all unique categories.

        Returns:
            List of category names
        """
        categories = set(p.category for p in cls.PRESETS.values())
        return sorted(categories)

    @classmethod
    def search_presets(cls, query: str) -> List[str]:
        """Search presets by name, description, or tags.

        Args:
            query: Search query (case-insensitive)

        Returns:
            List of matching preset names
        """
        query = query.lower()
        matches = []

        for name, preset in cls.PRESETS.items():
            if (
                query in preset.name.lower() or
                query in preset.description.lower() or
                any(query in tag.lower() for tag in preset.tags)
            ):
                matches.append(name)

        return matches

    @classmethod
    def get_preset_info(cls) -> Dict[str, Dict[str, any]]:
        """Get all preset information.

        Returns:
            Dictionary mapping preset names to configurations
        """
        return {
            name: cls.get_preset(name)
            for name in cls.PRESETS.keys()
        }


# Export
__all__ = ['StylePreset', 'ArchitecturalStylePresets']
