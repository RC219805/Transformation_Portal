"""Materials V3 taxonomy normalization.

Fixes the "water/pool_water/ocean" and "foliage/tree/vegetation" identity issues
that appeared in Stage 6 results.

Key design:
- Canonical material keys (string-stable, lowercase snake_case)
- Semantic → canonical mapping (many-to-one)
- Per-material metadata (thresholds, refinement strategy, response params)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Canonical Material Keys
# ---------------------------------------------------------------------------

CANONICAL_MATERIALS = {
    # Base materials (SegFormer ADE-derived)
    "wood",
    "metal",
    "glass",
    "stone",
    "water",
    "fabric",
    "foliage",
    "sky",
    "ground",
    "wall",
    "floor",
    "ceiling",
    
    # Expanded (Materials V3)
    "concrete",
    "brick",
    "tile",
    "marble",
    "granite",
    "stucco",
    "polished",  # polished metal/stone
    "painted",
    "raw_wood",
    "stained_wood",
    "laminate",
    "stainless",
    "aluminum",
    "chrome",
    "brass",
    "copper",
    "clear_glass",
    "frosted_glass",
    "mirror",
    "pool_water",
    "ocean_water",
    "still_water",
    "grass",
    "tree",
    "shrub",
    "flowers",
}


# ---------------------------------------------------------------------------
# Semantic → Canonical Mapping
# ---------------------------------------------------------------------------

SEMANTIC_TO_CANONICAL: Dict[str, str] = {
    # Water variants
    "water": "water",
    "pool_water": "water",
    "pool": "water",
    "ocean": "water",
    "ocean_water": "water",
    "sea": "water",
    "lake": "water",
    "pond": "water",
    "still_water": "water",
    "water_surface": "water",
    
    # Foliage variants
    "foliage": "foliage",
    "tree": "foliage",
    "trees": "foliage",
    "vegetation": "foliage",
    "plant": "foliage",
    "plants": "foliage",
    "shrub": "foliage",
    "shrubs": "foliage",
    "bush": "foliage",
    "bushes": "foliage",
    "grass": "foliage",
    "lawn": "foliage",
    "leaves": "foliage",
    "canopy": "foliage",
    
    # Glass variants
    "glass": "glass",
    "window": "glass",
    "windows": "glass",
    "clear_glass": "glass",
    "frosted_glass": "glass",
    "mirror": "glass",
    "mirrors": "glass",
    "glazing": "glass",
    
    # Wood variants
    "wood": "wood",
    "wooden": "wood",
    "timber": "wood",
    "lumber": "wood",
    "raw_wood": "wood",
    "stained_wood": "wood",
    "painted_wood": "wood",
    "laminate": "wood",
    "plywood": "wood",
    "mdf": "wood",
    
    # Metal variants
    "metal": "metal",
    "steel": "metal",
    "iron": "metal",
    "stainless": "metal",
    "stainless_steel": "metal",
    "aluminum": "metal",
    "aluminium": "metal",
    "chrome": "metal",
    "brass": "metal",
    "copper": "metal",
    "bronze": "metal",
    "polished_metal": "metal",
    
    # Stone variants
    "stone": "stone",
    "rock": "stone",
    "marble": "stone",
    "granite": "stone",
    "limestone": "stone",
    "sandstone": "stone",
    "slate": "stone",
    "travertine": "stone",
    "quartz": "stone",
    "paver": "stone",
    "pavers": "stone",
    "flagstone": "stone",
    "cobblestone": "stone",
    
    # Fabric variants
    "fabric": "fabric",
    "textile": "fabric",
    "cloth": "fabric",
    "upholstery": "fabric",
    "curtain": "fabric",
    "curtains": "fabric",
    "drapes": "fabric",
    "linen": "fabric",
    "cotton": "fabric",
    "velvet": "fabric",
    "silk": "fabric",
    
    # Architectural surfaces
    "wall": "wall",
    "walls": "wall",
    "ceiling": "ceiling",
    "ceilings": "ceiling",
    "floor": "floor",
    "floors": "floor",
    "flooring": "floor",
    "ground": "ground",
    "earth": "ground",
    "soil": "ground",
    "dirt": "ground",
    "sand": "ground",
    
    # Other common
    "sky": "sky",
    "clouds": "sky",
    "concrete": "stone",  # map to stone for response purposes
    "brick": "stone",
    "tile": "stone",
    "stucco": "wall",
    "painted": "wall",
    "drywall": "wall",
    "plaster": "wall",
}


@dataclass
class MaterialMetadata:
    """Per-material metadata for V3 processing.
    
    Attributes
    ----------
    canonical_key : str
        Canonical material key
    confidence_threshold : float
        Default confidence threshold
    edge_threshold : float
        Threshold for edge pixels
    refinement_priority : int
        0-10 (10 = always refine, 0 = never)
    response_strength : float
        Material response strength multiplier
    specular_sensitive : bool
        Whether material has strong highlights
    """
    
    canonical_key: str
    confidence_threshold: float = 0.50
    edge_threshold: float = 0.30
    refinement_priority: int = 5
    response_strength: float = 1.0
    specular_sensitive: bool = False
    
    # Refinement hints
    benefits_from_effsam: bool = False
    typical_coverage_range: tuple[float, float] = (0.01, 0.80)


# Default metadata per canonical material
DEFAULT_MATERIAL_METADATA: Dict[str, MaterialMetadata] = {
    "glass": MaterialMetadata(
        canonical_key="glass",
        confidence_threshold=0.40,
        edge_threshold=0.25,
        refinement_priority=10,
        benefits_from_effsam=True,
        specular_sensitive=True,
    ),
    "water": MaterialMetadata(
        canonical_key="water",
        confidence_threshold=0.35,
        edge_threshold=0.20,
        refinement_priority=9,
        benefits_from_effsam=True,
        specular_sensitive=True,
    ),
    "foliage": MaterialMetadata(
        canonical_key="foliage",
        confidence_threshold=0.45,
        edge_threshold=0.30,
        refinement_priority=8,
        benefits_from_effsam=True,
        specular_sensitive=False,
    ),
    "metal": MaterialMetadata(
        canonical_key="metal",
        confidence_threshold=0.60,
        edge_threshold=0.40,
        refinement_priority=6,
        benefits_from_effsam=False,
        specular_sensitive=True,
    ),
    "wood": MaterialMetadata(
        canonical_key="wood",
        confidence_threshold=0.65,
        edge_threshold=0.45,
        refinement_priority=4,
        benefits_from_effsam=False,
        specular_sensitive=False,
    ),
    "stone": MaterialMetadata(
        canonical_key="stone",
        confidence_threshold=0.65,
        edge_threshold=0.45,
        refinement_priority=3,
        benefits_from_effsam=False,
        specular_sensitive=False,
    ),
    "fabric": MaterialMetadata(
        canonical_key="fabric",
        confidence_threshold=0.55,
        edge_threshold=0.35,
        refinement_priority=2,
        benefits_from_effsam=False,
        specular_sensitive=False,
    ),
    "sky": MaterialMetadata(
        canonical_key="sky",
        confidence_threshold=0.70,
        edge_threshold=0.50,
        refinement_priority=0,  # Never refine sky
        benefits_from_effsam=False,
        specular_sensitive=False,
    ),
}


def normalize_material_name(semantic_name: str) -> str:
    """Normalize a semantic material name to canonical key.
    
    Parameters
    ----------
    semantic_name : str
        Input material name (e.g., 'pool_water', 'window', 'tree')
    
    Returns
    -------
    str
        Canonical material key (e.g., 'water', 'glass', 'foliage')
    
    Examples
    --------
    >>> normalize_material_name("pool_water")
    'water'
    >>> normalize_material_name("window")
    'glass'
    >>> normalize_material_name("tree")
    'foliage'
    """
    name_lower = semantic_name.lower().strip().replace("-", "_")
    canonical = SEMANTIC_TO_CANONICAL.get(name_lower, name_lower)
    
    # If still not in canonical set, pass through but log warning
    if canonical not in CANONICAL_MATERIALS:
        canonical = name_lower
    
    return canonical


def get_material_metadata(material_name: str) -> MaterialMetadata:
    """Get metadata for a material (normalizes name first).
    
    Parameters
    ----------
    material_name : str
        Material name (semantic or canonical)
    
    Returns
    -------
    MaterialMetadata
        Metadata object with thresholds and refinement hints
    """
    canonical = normalize_material_name(material_name)
    
    # Return known metadata or construct default
    if canonical in DEFAULT_MATERIAL_METADATA:
        return DEFAULT_MATERIAL_METADATA[canonical]
    
    # Fallback: generic metadata
    return MaterialMetadata(
        canonical_key=canonical,
        confidence_threshold=0.50,
        edge_threshold=0.30,
        refinement_priority=5,
    )


def should_refine_material(
    material_name: str,
    *,
    refinement_strategy: str = "canary",
    force_list: Optional[Set[str]] = None,
) -> bool:
    """Decide if a material should use EfficientSAM refinement.
    
    Parameters
    ----------
    material_name : str
        Material name (will be normalized)
    refinement_strategy : str
        'off', 'canary', 'selective', 'aggressive'
    force_list : Optional[Set[str]]
        Optional explicit list of materials to refine
    
    Returns
    -------
    bool
        True if material should be refined
    """
    canonical = normalize_material_name(material_name)
    metadata = get_material_metadata(canonical)
    
    if force_list is not None:
        return canonical in force_list
    
    if refinement_strategy == "off":
        return False
    
    if refinement_strategy == "canary":
        # Stage 6 validated list
        return canonical in {"glass", "water", "foliage"}
    
    if refinement_strategy == "selective":
        # Use refinement_priority >= 6
        return metadata.refinement_priority >= 6
    
    if refinement_strategy == "aggressive":
        # Refine everything except sky/ground
        return canonical not in {"sky", "ground", "ceiling"}
    
    return False


def normalize_material_dict(
    material_dict: Dict[str, any],
) -> Dict[str, any]:
    """Normalize all keys in a material dictionary to canonical keys.
    
    Parameters
    ----------
    material_dict : Dict[str, any]
        Dictionary with semantic material keys
    
    Returns
    -------
    Dict[str, any]
        Dictionary with canonical material keys
    
    Examples
    --------
    >>> d = {"pool_water": 0.8, "window": 0.6, "tree": 0.5}
    >>> normalize_material_dict(d)
    {'water': 0.8, 'glass': 0.6, 'foliage': 0.5}
    """
    normalized = {}
    for key, value in material_dict.items():
        canonical = normalize_material_name(key)
        # If multiple semantic keys map to same canonical, keep highest value
        if canonical in normalized:
            if hasattr(value, "__gt__"):  # numeric-like
                normalized[canonical] = max(normalized[canonical], value)
        else:
            normalized[canonical] = value
    return normalized
