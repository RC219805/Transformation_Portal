"""Vision-Language Model integration for scene understanding and quality validation.

This module provides VLM capabilities for:
- Scene understanding and classification
- Quality assessment and validation
- Architectural element detection
- Material recognition
- Realism verification
"""

from transformation_portal.vlm.llava import LLaVAProcessor
from transformation_portal.vlm.scene_analyzer import SceneAnalyzer
from transformation_portal.vlm.quality_validator import QualityValidator

__all__ = [
    'LLaVAProcessor',
    'SceneAnalyzer',
    'QualityValidator',
]
