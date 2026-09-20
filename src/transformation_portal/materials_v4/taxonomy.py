"""Exact, versioned Materials V4 label normalization.

Scene regions and surface materials share the compatibility label namespace;
the namespace does not imply that a photographic edit is implemented.
"""

from __future__ import annotations

from types import MappingProxyType

TAXONOMY_VERSION = "tp.materials.taxonomy.v1"
MATERIAL_LABELS = frozenset(
    {
        "sky",
        "glass",
        "water",
        "foliage",
        "wood",
        "stone",
        "metal",
        "fabric",
        "stucco",
        "paint",
        "ceramic",
        "leather",
        "unknown",
        "mixed",
    }
)
LABEL_ALIASES = MappingProxyType(
    {
        "cloud": "sky",
        "window": "glass",
        "pool": "water",
        "ocean": "water",
        "sea": "water",
        "plant": "foliage",
        "tree": "foliage",
        "leaf": "foliage",
        "marble": "stone",
        "granite": "stone",
        "limestone": "stone",
        "travertine": "stone",
        "concrete": "stone",
        "plaster": "stucco",
        "wood floor": "wood",
        "wood panel": "wood",
        "marble surface": "stone",
        "granite stone": "stone",
        "limestone wall": "stone",
        "travertine tile": "stone",
        "brushed steel": "metal",
        "polished brass": "metal",
        "copper metal": "metal",
        "aluminum frame": "metal",
        "clear glass": "glass",
        "frosted glass": "glass",
        "tinted glass": "glass",
        "linen fabric": "fabric",
        "velvet upholstery": "fabric",
        "silk curtain": "fabric",
        "cotton textile": "fabric",
        "concrete wall": "stone",
        "concrete floor": "stone",
        "leather furniture": "leather",
        "painted wall": "paint",
        "plaster ceiling": "stucco",
        "ceramic tile": "ceramic",
        "porcelain surface": "ceramic",
    }
)


def canonical_label(value: str) -> str:
    """Map exact aliases only; unrelated or composite labels remain unknown."""
    if not isinstance(value, str) or not value.strip() or len(value) > 128 or not value.isascii():
        raise ValueError("Material label must be bounded, nonempty ASCII text")
    normalized = value.strip().lower()
    if any(ord(char) < 32 or ord(char) == 127 for char in normalized):
        raise ValueError("Material label cannot contain control characters")
    return normalized if normalized in MATERIAL_LABELS else LABEL_ALIASES.get(normalized, "unknown")
