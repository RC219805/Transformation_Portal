"""
Configuration Presets Registry.

Manages named configuration profiles (e.g., 'production', 'preview').
"""

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class Preset:
    """A named configuration profile."""

    name: str
    description: str
    overrides: Dict[str, Any]  # Dictionary matching schema structure


class PresetRegistry:
    """Central store for configuration presets."""

    _presets: Dict[str, Preset] = {}

    @classmethod
    def register(cls, preset: Preset):
        cls._presets[preset.name] = preset
        logger.debug(f"Registered preset: {preset.name}")

    @classmethod
    def get(cls, name: str) -> Preset:
        if name not in cls._presets:
            raise KeyError(f"Preset '{name}' not found. Available: {list(cls._presets.keys())}")
        return cls._presets[name]

    @classmethod
    def list(cls) -> List[str]:
        return list(cls._presets.keys())


# --- Standard Presets ---


def _register_defaults():
    # 1. Fast Preview (Speed over Quality)
    PresetRegistry.register(
        Preset(
            name="fast_preview",
            description="Low resolution, high speed for rapid iteration.",
            overrides={
                "performance": {"batch_size": 4, "tile_size": 0},  # No tiling
                "device": {"precision": "fp16"},
                "output": {"format": "jpg", "quality": 85},
                "validation": {"enabled": False},
            },
        )
    )

    # 2. Production (Balanced)
    PresetRegistry.register(
        Preset(
            name="production",
            description="Standard delivery settings for high-end web usage.",
            overrides={
                "performance": {"batch_size": 1, "tile_size": 1024},
                "device": {"precision": "fp16"},
                "output": {"format": "jpg", "quality": 92},
                "validation": {"enabled": True, "check_blur": True},
            },
        )
    )

    # 3. Archival (Maximum Quality)
    PresetRegistry.register(
        Preset(
            name="archival",
            description="Maximum fidelity for print/archive. 16-bit processing.",
            overrides={
                "performance": {"batch_size": 1, "tile_size": 512, "tile_overlap": 128},
                "device": {"precision": "fp32"},
                "output": {"format": "tiff", "quality": 100},
                "validation": {"enabled": True, "min_resolution": 2048},
            },
        )
    )


# Initialize defaults on import
_register_defaults()


# Public API aliases
register_preset = PresetRegistry.register
list_presets = PresetRegistry.list


def load_preset(name: str) -> Dict[str, Any]:
    """Resolve a preset into a configuration dictionary."""
    preset = PresetRegistry.get(name)
    return deepcopy(preset.overrides)
