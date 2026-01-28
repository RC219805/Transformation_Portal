"""Materials V3 Taxonomy definitions."""
from enum import Enum

class RefinementStrategy(Enum):
    CANARY = "canary"
    ALL = "all"
    NONE = "none"

# Expanded Metadata (PR-4C)
DEFAULT_MATERIAL_METADATA = {
    "glass": {"priority": 10, "threshold": 0.40, "canary": True},
    "water": {"priority": 9, "threshold": 0.35, "canary": True},
    "foliage": {"priority": 5, "threshold": 0.50, "canary": True},
    "wood": {"priority": 3, "threshold": 0.60, "canary": False},
    "stone": {"priority": 3, "threshold": 0.60, "canary": False},
    "metal": {"priority": 4, "threshold": 0.55, "canary": False},
    "fabric": {"priority": 2, "threshold": 0.60, "canary": False},
    "stucco": {"priority": 1, "threshold": 0.65, "canary": False},
}
