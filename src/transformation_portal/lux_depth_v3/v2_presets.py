"""V2 Enhancement Presets.

Defines preset configurations for V2 depth-aware enhancement.
Following architectural guidance from V2_ENHANCEMENT_ARCHITECTURAL_GUIDANCE.md

Presets provide tested parameter combinations for common use cases:
- default: Balanced enhancement for general use
- luxury_estate: Premium marketing aesthetic
- architectural: Technical visualization with minimal atmosphere
- none: Skip V2 enhancement (passthrough)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class V2EnhancementConfig:
    """Configuration for V2 enhancement.

    Attributes:
        preset: Preset name
        enhancement_strength: Global enhancement strength [0, 1]
        clarity_strength: Clarity enhancement strength [0, 1]
        material_strength: Material-specific processing strength [0, 1]
        depth_aware_tone_mapping: RESERVED - Currently always enabled when depth map provided
        atmospheric_effects: RESERVED - Currently not separately controllable
        version: Stage version for cache invalidation

    Note:
        The depth_aware_tone_mapping and atmospheric_effects flags are reserved for
        future use. Current implementation always applies depth-aware tone mapping when
        a depth map is provided. To disable depth effects, use preset="none" or omit
        the depth map.
    """

    preset: str = "default"
    enhancement_strength: float = 0.7
    clarity_strength: float = 0.5
    material_strength: float = 0.6
    depth_aware_tone_mapping: bool = True
    atmospheric_effects: bool = True
    version: str = "1.0.0"

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.enhancement_strength <= 1.0:
            raise ValueError(f"enhancement_strength must be in [0, 1], got {self.enhancement_strength}")
        if not 0.0 <= self.clarity_strength <= 1.0:
            raise ValueError(f"clarity_strength must be in [0, 1], got {self.clarity_strength}")
        if not 0.0 <= self.material_strength <= 1.0:
            raise ValueError(f"material_strength must be in [0, 1], got {self.material_strength}")

    @classmethod
    def from_preset(cls, preset: str) -> V2EnhancementConfig:
        """Load configuration from preset name.

        Args:
            preset: Preset name (default, luxury_estate, architectural, none)

        Returns:
            V2EnhancementConfig instance

        Raises:
            ValueError: If preset is unknown
        """
        if preset not in PRESETS:
            available = ", ".join(PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset}'. Available presets: {available}")

        preset_config = PRESETS[preset]
        return cls(
            preset=preset,
            enhancement_strength=preset_config["enhancement_strength"],
            clarity_strength=preset_config["clarity_strength"],
            material_strength=preset_config["material_strength"],
            depth_aware_tone_mapping=preset_config["depth_aware_tone_mapping"],
            atmospheric_effects=preset_config["atmospheric_effects"],
        )

    def to_dict(self) -> Dict[str, any]:
        """Convert configuration to dictionary.

        Returns:
            Dict representation of configuration
        """
        return {
            "preset": self.preset,
            "enhancement_strength": self.enhancement_strength,
            "clarity_strength": self.clarity_strength,
            "material_strength": self.material_strength,
            "depth_aware_tone_mapping": self.depth_aware_tone_mapping,
            "atmospheric_effects": self.atmospheric_effects,
            "version": self.version,
        }


# Preset Definitions
# Based on V2_ENHANCEMENT_ARCHITECTURAL_GUIDANCE.md Section 4

PRESETS: Dict[str, Dict[str, any]] = {
    "default": {
        "description": "Balanced enhancement for general use",
        "enhancement_strength": 0.7,
        "clarity_strength": 0.5,
        "material_strength": 0.6,
        "depth_aware_tone_mapping": True,
        "atmospheric_effects": True,
        "use_case": "General real estate photography",
    },
    "luxury_estate": {
        "description": "Premium marketing aesthetic",
        "enhancement_strength": 0.8,
        "clarity_strength": 0.6,
        "material_strength": 0.7,
        "depth_aware_tone_mapping": True,
        "atmospheric_effects": True,
        "use_case": "High-end luxury real estate marketing",
    },
    "architectural": {
        "description": "Technical visualization with minimal atmosphere",
        "enhancement_strength": 0.6,
        "clarity_strength": 0.7,
        "material_strength": 0.5,
        "depth_aware_tone_mapping": True,
        "atmospheric_effects": False,
        "use_case": "Architectural visualization and technical documentation",
    },
    "none": {
        "description": "Skip V2 enhancement (passthrough)",
        "enhancement_strength": 0.0,
        "clarity_strength": 0.0,
        "material_strength": 0.0,
        "depth_aware_tone_mapping": False,
        "atmospheric_effects": False,
        "use_case": "Disable V2 enhancement (PBR-only mode)",
    },
}


def get_preset_description(preset: str) -> Optional[str]:
    """Get description for a preset.

    Args:
        preset: Preset name

    Returns:
        Preset description or None if preset not found
    """
    if preset in PRESETS:
        return PRESETS[preset]["description"]
    return None


def list_presets() -> Dict[str, str]:
    """List all available presets with descriptions.

    Returns:
        Dict mapping preset names to descriptions
    """
    return {name: config["description"] for name, config in PRESETS.items()}
