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

from transformation_portal.interfaces.processor import ImageProcessor, VideoProcessor
from transformation_portal.interfaces.pipeline import Pipeline, PipelineStage, BatchPipeline

__all__ = [
    'ImageProcessor',
    'VideoProcessor',
    'Pipeline',
    'PipelineStage',
    'BatchPipeline',
]

__version__ = '0.1.0'
