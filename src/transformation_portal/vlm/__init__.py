"""Vision-Language Model integration for scene understanding and quality validation.

This module provides VLM capabilities for:
- Scene understanding and classification
- Quality assessment and validation
- Architectural element detection
- Material recognition
- Realism verification
"""

from importlib import import_module
from typing import Any

__all__ = [
    "LLaVAProcessor",
    "SceneAnalyzer",
    "QualityValidator",
]

_LAZY_EXPORTS = {
    "LLaVAProcessor": "transformation_portal.vlm.llava",
    "SceneAnalyzer": "transformation_portal.vlm.scene_analyzer",
    "QualityValidator": "transformation_portal.vlm.quality_validator",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
