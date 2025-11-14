"""Semantic segmentation for architectural imagery.

Provides intelligent scene understanding using:
- SAM (Segment Anything Model) for universal segmentation
- CLIP for zero-shot material classification
- Material-aware enhancement strategies

Enables context-aware processing that respects semantic boundaries.
"""

from transformation_portal.segmentation.sam_segmenter import SAMSegmenter
from transformation_portal.segmentation.clip_classifier import CLIPClassifier
from transformation_portal.segmentation.material_segmenter import MaterialSegmenter

__all__ = [
    'SAMSegmenter',
    'CLIPClassifier',
    'MaterialSegmenter',
]
