"""
Transformation Portal - Interface Contracts

This package defines abstract base classes (ABCs) that establish contracts
for all major architectural layers. All implementations should conform to
these interfaces to ensure consistency, testability, and maintainability.

Module Organization:
- processor.py: Base interface for image/video processors
- pipeline.py: Base interface for multi-stage pipelines

Design Principles:
1. Explicit over implicit - Clear method signatures and return types
2. Fail fast - Validation in interface base methods
3. Serializable config - All configuration must be JSON-serializable
4. Stateless preferred - Document if stateful behavior required
5. Type hints - Full type annotations for all methods

Usage Example:
    >>> from transformation_portal.interfaces import ImageProcessor
    >>>
    >>> class MyProcessor(ImageProcessor):
    ...     def process(self, image, **kwargs):
    ...         return image * 1.2  # Brighten
    ...
    ...     def get_config(self):
    ...         return {"brightness": 1.2}

See Also:
- docs/architecture/adr/ADR-001-module-interface-contracts.md
- docs/ARCHITECTURE.md
"""

from transformation_portal.interfaces.enhancer import (
    AdaptiveEnhancer,
    EnhancementError,
    Enhancer,
)
from transformation_portal.interfaces.estimator import (
    DepthEstimator,
    EstimationError,
    NormalEstimator,
    UnifiedEstimator,
)
from transformation_portal.interfaces.pipeline import (
    BatchPipeline,
    Pipeline,
    PipelineError,
    PipelineStage,
)
from transformation_portal.interfaces.processor import (
    ImageProcessor,
    ProcessingError,
    VideoProcessor,
)
from transformation_portal.interfaces.segmenter import (
    MaterialSegmenter,
    MaterialType,
    SegmentationError,
    Segmenter,
    SemanticSegmenter,
)

__all__ = [
    # Processor interfaces
    "ImageProcessor",
    "VideoProcessor",
    "ProcessingError",
    # Pipeline interfaces
    "Pipeline",
    "PipelineStage",
    "BatchPipeline",
    "PipelineError",
    # Enhancer interfaces
    "Enhancer",
    "AdaptiveEnhancer",
    "EnhancementError",
    # Segmenter interfaces
    "Segmenter",
    "MaterialSegmenter",
    "SemanticSegmenter",
    "MaterialType",
    "SegmentationError",
    # Estimator interfaces
    "DepthEstimator",
    "NormalEstimator",
    "UnifiedEstimator",
    "EstimationError",
]

__version__ = "0.2.0"
