"""
Preset management system.

Provides a registry for configuration presets that can be shared
across all pipelines.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import yaml
import json


class Preset(str, Enum):
    """Standard preset names across all pipelines."""
    # Quality presets
    PHOTO_REALISTIC = "photo_realistic"
    ARCHIVAL_QUALITY = "archival_quality"
    
    # Scene type presets
    INTERIOR_LUXURY = "interior_luxury"
    EXTERIOR_SHOWCASE = "exterior_showcase"
    ARCHITECTURAL = "architectural"
    
    # Specialized presets
    SIGNATURE_ESTATE = "signature_estate"
    GOLDEN_HOUR = "golden_hour"
    HDR_MASTERING = "hdr_mastering"
    
    # Performance presets
    FAST_PREVIEW = "fast_preview"
    BALANCED = "balanced"
    MAXIMUM_QUALITY = "maximum_quality"


class PresetRegistry:
    """
    Registry for configuration presets.
    
    Allows pipelines to register their preset configurations
    and load them by name.
    """
    
    _presets: Dict[str, Dict[str, Any]] = {}
    _preset_factories: Dict[str, Callable[[], Dict[str, Any]]] = {}
    
    @classmethod
    def register(cls, name: str, config: Dict[str, Any]) -> None:
        """
        Register a preset configuration.
        
        Args:
            name: Preset name
            config: Configuration dictionary
        """
        cls._presets[name] = config
    
    @classmethod
    def register_factory(cls, name: str, factory: Callable[[], Dict[str, Any]]) -> None:
        """
        Register a preset factory function.
        
        Args:
            name: Preset name
            factory: Function that returns configuration dictionary
        """
        cls._preset_factories[name] = factory
    
    @classmethod
    def get(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a preset configuration by name.
        
        Args:
            name: Preset name
            
        Returns:
            Configuration dictionary or None if not found
        """
        # Try direct preset first
        if name in cls._presets:
            return cls._presets[name].copy()
        
        # Try factory
        if name in cls._preset_factories:
            return cls._preset_factories[name]()
        
        return None
    
    @classmethod
    def list_presets(cls) -> list[str]:
        """
        List all registered preset names.
        
        Returns:
            List of preset names
        """
        return sorted(set(cls._presets.keys()) | set(cls._preset_factories.keys()))
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered presets (useful for testing)."""
        cls._presets.clear()
        cls._preset_factories.clear()


def load_preset(name: str) -> Optional[Dict[str, Any]]:
    """
    Load a preset by name.
    
    Args:
        name: Preset name (can be enum or string)
        
    Returns:
        Configuration dictionary or None if not found
    """
    if isinstance(name, Enum):
        name = name.value
    
    return PresetRegistry.get(name)


def register_preset(name: str, config: Dict[str, Any]) -> None:
    """
    Register a new preset.
    
    Args:
        name: Preset name
        config: Configuration dictionary
    """
    PresetRegistry.register(name, config)


def list_presets() -> list[str]:
    """
    List all available presets.
    
    Returns:
        List of preset names
    """
    return PresetRegistry.list_presets()


def load_preset_from_file(path: Path) -> Dict[str, Any]:
    """
    Load preset from YAML or JSON file.
    
    Args:
        path: Path to preset file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Preset file not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix in (".yaml", ".yml"):
        with open(path) as f:
            return yaml.safe_load(f)
    elif suffix == ".json":
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported preset file format: {suffix}")


# Register default presets
def _register_defaults():
    """Register default presets."""
    
    # Photo realistic preset (balanced quality)
    PresetRegistry.register("photo_realistic", {
        "performance": {
            "batch_size": 1,
            "tile_size": 2048,
            "tile_overlap": 64,
            "enable_tiling": True,
        },
        "output": {
            "save_master": True,
            "save_preview": True,
            "compression": "lzw",
        },
        "extras": {
            "material_strength": 0.70,
            "clarity": 0.18,
            "detail_strength": 0.65,
        }
    })
    
    # Interior luxury preset (high quality, enhanced materials)
    PresetRegistry.register("interior_luxury", {
        "performance": {
            "batch_size": 1,
            "tile_size": 2048,
            "tile_overlap": 64,
            "enable_tiling": True,
        },
        "output": {
            "save_master": True,
            "save_preview": True,
            "compression": "lzw",
        },
        "extras": {
            "material_strength": 0.90,
            "clarity": 0.20,
            "detail_strength": 0.70,
            "saturation": 1.045,
        }
    })
    
    # Fast preview preset (speed over quality)
    PresetRegistry.register("fast_preview", {
        "performance": {
            "batch_size": 4,
            "tile_size": 512,
            "tile_overlap": 32,
            "enable_tiling": True,
            "enable_caching": True,
        },
        "output": {
            "save_master": False,
            "save_preview": True,
            "compression": None,
        },
        "extras": {
            "material_strength": 0.50,
            "clarity": 0.10,
            "detail_strength": 0.50,
        }
    })


# Initialize defaults on import
_register_defaults()
