#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Material Response Profiles for Transformation Portal.

Defines preset material profiles for different surface types and use cases.
Each profile provides optimized parameters for the MaterialResponseEngine.

Available Profiles:
    - luxury_interior: Default profile for luxury interior spaces
    - wood_floor_oak: Optimized for oak and hardwood floors
    - marble_stone: Settings for marble and stone surfaces
    - textile_linen: Enhanced textile and fabric rendering
    - metal_brushed: Brushed metal and stainless steel
    - glass_window: Glass and window reflections
    - exterior_courtyard: Outdoor courtyard and patio spaces
    - aerial_estate: Aerial/drone photography optimization

Example:
    from transformation_portal.processors.material_response.profiles import (
        get_profile,
        list_profiles,
        PROFILES
    )

    # Get a specific profile
    profile = get_profile('luxury_interior')

    # List all available profiles
    for name in list_profiles():
        print(f"Profile: {name}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MaterialProfile:
    """Material profile definition.

    Attributes:
        name: Profile identifier.
        display_name: Human-readable profile name.
        description: Profile description.
        texture_boost: High-frequency texture enhancement.
        ambient_occlusion: Contact shadow intensity.
        highlight_warmth: Warm highlight mix.
        haze_strength: Volumetric haze blend.
        haze_tint: RGB haze tint values.
        floor_plank_contrast: Wood floor definition.
        floor_specular: Specular streak intensity on flooring.
        textile_contrast: Linen/fabric separation.
        leather_sheen: Leather surface sheen.
        window_light_wrap: Window light wrap intensity.
        window_reflection: Window reflection on floors.
        wall_texture: Wall surface texture.
    """
    name: str
    display_name: str
    description: str
    texture_boost: float = 0.25
    ambient_occlusion: float = 0.12
    highlight_warmth: float = 0.08
    haze_strength: float = 0.06
    haze_tint: tuple = (0.82, 0.88, 0.96)
    floor_plank_contrast: float = 0.12
    floor_specular: float = 0.18
    textile_contrast: float = 0.18
    leather_sheen: float = 0.16
    window_light_wrap: float = 0.14
    window_reflection: float = 0.12
    wall_texture: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'texture_boost': self.texture_boost,
            'ambient_occlusion': self.ambient_occlusion,
            'highlight_warmth': self.highlight_warmth,
            'haze_strength': self.haze_strength,
            'haze_tint': self.haze_tint,
            'floor_plank_contrast': self.floor_plank_contrast,
            'floor_specular': self.floor_specular,
            'textile_contrast': self.textile_contrast,
            'leather_sheen': self.leather_sheen,
            'window_light_wrap': self.window_light_wrap,
            'window_reflection': self.window_reflection,
            'wall_texture': self.wall_texture,
        }


# ============================================================================
# PROFILE DEFINITIONS
# ============================================================================

PROFILES: Dict[str, MaterialProfile] = {
    # Default luxury interior profile
    'luxury_interior': MaterialProfile(
        name='luxury_interior',
        display_name='Luxury Interior',
        description='Default profile for luxury interior spaces with balanced enhancement',
        texture_boost=0.25,
        ambient_occlusion=0.12,
        highlight_warmth=0.08,
        haze_strength=0.06,
        haze_tint=(0.82, 0.88, 0.96),
        floor_plank_contrast=0.12,
        floor_specular=0.18,
        textile_contrast=0.18,
        leather_sheen=0.16,
        window_light_wrap=0.14,
        window_reflection=0.12,
        wall_texture=0.1,
    ),

    # Oak hardwood floor optimization
    'wood_floor_oak': MaterialProfile(
        name='wood_floor_oak',
        display_name='Wood Floor - Oak',
        description='Optimized for oak and hardwood floor enhancement with warm grain definition',
        texture_boost=0.30,
        ambient_occlusion=0.15,
        highlight_warmth=0.12,
        haze_strength=0.04,
        haze_tint=(0.85, 0.82, 0.78),  # Warm oak tint
        floor_plank_contrast=0.22,  # Strong grain definition
        floor_specular=0.25,  # Polished wood sheen
        textile_contrast=0.10,  # Reduced for floor focus
        leather_sheen=0.10,
        window_light_wrap=0.16,
        window_reflection=0.20,  # Strong floor reflections
        wall_texture=0.08,
    ),

    # Marble and stone surfaces
    'marble_stone': MaterialProfile(
        name='marble_stone',
        display_name='Marble & Stone',
        description='Settings for marble, granite, and natural stone surfaces',
        texture_boost=0.18,  # Subtle texture for polished surfaces
        ambient_occlusion=0.08,  # Minimal shadowing
        highlight_warmth=0.04,  # Cool stone appearance
        haze_strength=0.03,
        haze_tint=(0.92, 0.94, 0.96),  # Cool neutral
        floor_plank_contrast=0.08,  # Minimal - no planks
        floor_specular=0.28,  # High polish
        textile_contrast=0.12,
        leather_sheen=0.14,
        window_light_wrap=0.10,
        window_reflection=0.25,  # Strong reflection on polished stone
        wall_texture=0.06,
    ),

    # Textile and linen enhancement
    'textile_linen': MaterialProfile(
        name='textile_linen',
        display_name='Textile & Linen',
        description='Enhanced textile and fabric rendering for upholstery and bedding',
        texture_boost=0.35,  # Strong texture for fabric weave
        ambient_occlusion=0.14,
        highlight_warmth=0.06,
        haze_strength=0.05,
        haze_tint=(0.88, 0.88, 0.90),  # Neutral soft
        floor_plank_contrast=0.08,
        floor_specular=0.10,
        textile_contrast=0.28,  # Strong fabric separation
        leather_sheen=0.22,  # Enhanced leather
        window_light_wrap=0.18,  # Soft light wrap
        window_reflection=0.08,
        wall_texture=0.12,
    ),

    # Brushed metal and stainless steel
    'metal_brushed': MaterialProfile(
        name='metal_brushed',
        display_name='Metal - Brushed',
        description='Brushed metal and stainless steel surface enhancement',
        texture_boost=0.22,
        ambient_occlusion=0.10,
        highlight_warmth=0.02,  # Cool metallic
        haze_strength=0.02,  # Minimal haze
        haze_tint=(0.95, 0.96, 0.98),  # Cool steel
        floor_plank_contrast=0.06,
        floor_specular=0.15,
        textile_contrast=0.08,
        leather_sheen=0.08,
        window_light_wrap=0.08,
        window_reflection=0.10,
        wall_texture=0.04,
    ),

    # Glass and window surfaces
    'glass_window': MaterialProfile(
        name='glass_window',
        display_name='Glass & Windows',
        description='Glass and window reflection optimization',
        texture_boost=0.12,  # Minimal texture for glass
        ambient_occlusion=0.06,
        highlight_warmth=0.04,
        haze_strength=0.08,  # Atmospheric haze for depth
        haze_tint=(0.85, 0.90, 0.95),  # Sky tint
        floor_plank_contrast=0.10,
        floor_specular=0.20,
        textile_contrast=0.14,
        leather_sheen=0.12,
        window_light_wrap=0.22,  # Strong light wrap
        window_reflection=0.18,
        wall_texture=0.08,
    ),

    # Exterior courtyard and patio
    'exterior_courtyard': MaterialProfile(
        name='exterior_courtyard',
        display_name='Exterior Courtyard',
        description='Outdoor courtyard and patio spaces with natural lighting',
        texture_boost=0.28,
        ambient_occlusion=0.16,  # Strong shadows outdoors
        highlight_warmth=0.12,  # Warm sunlight
        haze_strength=0.10,  # Atmospheric perspective
        haze_tint=(0.78, 0.85, 0.92),  # Sky blue haze
        floor_plank_contrast=0.14,
        floor_specular=0.12,
        textile_contrast=0.16,
        leather_sheen=0.14,
        window_light_wrap=0.06,
        window_reflection=0.08,
        wall_texture=0.14,  # Outdoor texture
    ),

    # Aerial and drone photography
    'aerial_estate': MaterialProfile(
        name='aerial_estate',
        display_name='Aerial Estate',
        description='Aerial and drone photography optimization for estate views',
        texture_boost=0.20,
        ambient_occlusion=0.08,
        highlight_warmth=0.10,
        haze_strength=0.15,  # Strong atmospheric perspective
        haze_tint=(0.75, 0.85, 0.95),  # Blue sky haze
        floor_plank_contrast=0.06,  # Reduced for distance
        floor_specular=0.08,
        textile_contrast=0.10,
        leather_sheen=0.08,
        window_light_wrap=0.04,
        window_reflection=0.06,
        wall_texture=0.10,
    ),
}


def get_profile(name: str) -> Dict[str, Any]:
    """Get a material profile by name.

    Args:
        name: Profile name (e.g., 'luxury_interior').

    Returns:
        Profile configuration dictionary.

    Raises:
        KeyError: If profile name is not found.
    """
    if name not in PROFILES:
        available = ', '.join(sorted(PROFILES.keys()))
        raise KeyError(f"Unknown profile '{name}'. Available: {available}")

    return PROFILES[name].to_dict()


def list_profiles() -> List[str]:
    """List all available profile names.

    Returns:
        List of profile names.
    """
    return list(PROFILES.keys())


def get_profile_info(name: str) -> Dict[str, str]:
    """Get display information for a profile.

    Args:
        name: Profile name.

    Returns:
        Dictionary with display_name and description.
    """
    profile = PROFILES.get(name)
    if profile is None:
        return {'display_name': name, 'description': 'Unknown profile'}

    return {
        'name': profile.name,
        'display_name': profile.display_name,
        'description': profile.description,
    }


def get_all_profiles() -> Dict[str, Dict[str, Any]]:
    """Get all profiles as dictionaries.

    Returns:
        Dictionary mapping profile names to their configurations.
    """
    return {name: profile.to_dict() for name, profile in PROFILES.items()}


__all__ = [
    'MaterialProfile',
    'PROFILES',
    'get_profile',
    'list_profiles',
    'get_profile_info',
    'get_all_profiles',
]
