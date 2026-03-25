"""
Scene Type Taxonomy for Transformation Portal.

Provides canonical scene types, normalization, and validation for consistent
labeling across run cards, RAG retrieval, and quality analysis.
"""

from typing import Dict, List, Optional

SCENE_TYPES: Dict[str, Dict[str, any]] = {
    # Interior Spaces
    "interior_bedroom": {
        "aliases": ["bedroom", "bed", "master", "guest_room", "suite"],
        "description": "Bedrooms, master suites, guest rooms",
    },
    "interior_great_room": {
        "aliases": ["great", "living", "family", "lounge", "great_room", "greatroom"],
        "description": "Great rooms, living rooms, family rooms",
    },
    "interior_kitchen": {
        "aliases": ["kitchen", "kit", "pantry", "butler"],
        "description": "Kitchens, butler's pantry",
    },
    "interior_bathroom": {
        "aliases": ["bathroom", "bath", "powder", "powder_room"],
        "description": "Bathrooms, powder rooms",
    },
    "interior_dining_room": {
        "aliases": ["dining", "breakfast", "dining_room"],
        "description": "Dining rooms, breakfast nooks",
    },
    "interior_office": {
        "aliases": ["office", "study", "library", "den"],
        "description": "Offices, studies, libraries, dens",
    },
    "interior_closet": {
        "aliases": ["closet", "wardrobe", "dressing", "walk_in"],
        "description": "Walk-in closets, dressing rooms",
    },
    "interior_hallway": {
        "aliases": ["hallway", "corridor", "foyer", "entry"],
        "description": "Hallways, corridors, foyers",
    },
    # Exterior Spaces
    "exterior_pool": {
        "aliases": ["pool", "spa", "water", "jacuzzi", "hot_tub"],
        "description": "Pools, spas, water features",
    },
    "exterior_garden": {
        "aliases": ["garden", "yard", "landscape", "landscaping"],
        "description": "Gardens, yards, landscaping",
    },
    "exterior_courtyard": {
        "aliases": ["courtyard", "patio", "terrace", "deck"],
        "description": "Courtyards, patios, terraces, decks",
    },
    "exterior_facade": {
        "aliases": ["facade", "front", "entry", "exterior", "elevation"],
        "description": "Building facades, entries, elevations",
    },
    "aerial_exterior": {
        "aliases": ["aerial", "drone", "overhead", "bird", "birds_eye"],
        "description": "Aerial views, drone shots",
    },
    # Special Conditions
    "twilight_exterior": {
        "aliases": ["twilight", "dusk", "blue_hour", "golden_hour"],
        "description": "Twilight exterior shots, dusk, golden hour",
    },
    "night_interior": {
        "aliases": ["night", "evening_interior", "evening"],
        "description": "Night interior shots",
    },
    "night_exterior": {
        "aliases": ["night_exterior", "nighttime_exterior"],
        "description": "Night exterior shots",
    },
}


def normalize_scene_type(raw_input: str) -> str:
    """
    Convert any alias to canonical scene type.

    Args:
        raw_input: Raw scene type string (from filename, folder, or user input)

    Returns:
        Canonical scene type string

    Raises:
        ValueError: If scene type is not recognized

    Examples:
        >>> normalize_scene_type("master")
        'interior_bedroom'
        >>> normalize_scene_type("pool")
        'exterior_pool'
        >>> normalize_scene_type("drone")
        'aerial_exterior'
    """
    raw_lower = raw_input.lower().strip()

    # Check exact match first
    if raw_lower in SCENE_TYPES:
        return raw_lower

    # Check aliases (exact match only to avoid substring collisions)
    for canonical, config in SCENE_TYPES.items():
        if raw_lower in config["aliases"]:
            return canonical

    # Not found
    valid_types = ", ".join(SCENE_TYPES.keys())
    raise ValueError(f"Unknown scene type: '{raw_input}'. " f"Valid types: {valid_types}")


def validate_scene_type(scene_type: str) -> bool:
    """
    Check if scene type is in canonical taxonomy.

    Args:
        scene_type: Scene type string to validate

    Returns:
        True if valid, False otherwise

    Examples:
        >>> validate_scene_type("interior_kitchen")
        True
        >>> validate_scene_type("invalid_type")
        False
    """
    return scene_type in SCENE_TYPES


def get_scene_type_description(scene_type: str) -> Optional[str]:
    """
    Get description for a canonical scene type.

    Args:
        scene_type: Canonical scene type

    Returns:
        Description string, or None if not found

    Examples:
        >>> get_scene_type_description("interior_kitchen")
        "Kitchens, butler's pantry"
    """
    return SCENE_TYPES.get(scene_type, {}).get("description")


def list_scene_types() -> List[str]:
    """
    Get list of all canonical scene types.

    Returns:
        List of canonical scene type strings

    Examples:
        >>> types = list_scene_types()
        >>> "interior_kitchen" in types
        True
    """
    return list(SCENE_TYPES.keys())


def get_all_aliases(scene_type: str) -> List[str]:
    """
    Get all aliases for a canonical scene type.

    Args:
        scene_type: Canonical scene type

    Returns:
        List of alias strings

    Examples:
        >>> get_all_aliases("interior_bedroom")
        ['bedroom', 'bed', 'master', 'guest_room', 'suite']
    """
    return SCENE_TYPES.get(scene_type, {}).get("aliases", [])
