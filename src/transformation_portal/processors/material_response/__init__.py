"""Material Response System - Physics-based surface enhancement.

This module provides physics-based surface enhancement for luxury real estate
and architectural visualization. It implements the three core Material Response tenets:

1. Respect energy conservation in highlights (preserve specular sheen)
2. Preserve midtone texture (keep materials tactile and dimensional)
3. Blend transitions between materials (authored, not procedural)

Components:
    - core: Marketing principle documentation and validation
    - engine: Main processing engine with configurable parameters
    - profiles: Preset material profiles for different surface types

Example:
    from transformation_portal.processors.material_response import (
        MaterialResponseEngine,
        get_profile,
        list_profiles,
    )

    engine = MaterialResponseEngine.from_config({
        'profile': 'luxury_interior',
        'texture_boost': 0.25,
    })
    result = engine.apply(image)
"""

from .engine import MaterialResponseEngine, MaterialResponseConfig, MaterialMask
from .profiles import (
    MaterialProfile,
    PROFILES,
    get_profile,
    list_profiles,
    get_profile_info,
    get_all_profiles,
)

__all__ = [
    # Engine
    'MaterialResponseEngine',
    'MaterialResponseConfig',
    'MaterialMask',
    # Profiles
    'MaterialProfile',
    'PROFILES',
    'get_profile',
    'list_profiles',
    'get_profile_info',
    'get_all_profiles',
]

