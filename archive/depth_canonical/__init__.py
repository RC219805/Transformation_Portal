"""Canonical depth processing module for Transformation Portal.

This module consolidates all depth-related functionality into a single,
well-organized package with PBR (Physically Based Rendering) map generation.

Public API:
    Configuration:
        - UnifiedDepthConfig: Main configuration class
        - ModelConfig: Model selection and device configuration
        - ProcessingConfig: Processing pipeline configuration
        - PBRConfig: PBR map generation configuration
        - IOConfig: I/O and caching configuration
        - SecurityConfig: Security and validation configuration

    Enumerations:
        - DeviceType: CPU, CUDA, MPS, CoreML
        - ModelVariant: Depth Anything V2/V3 variants

    Pipeline:
        - DepthPipeline: Main orchestrator for depth processing
        - DepthPipelineResult: Result container

    Processing:
        - generate_pbr_maps: Generate normal, roughness, AO maps from depth
        - write_pbr_maps: Atomic write of PBR maps to disk

    Models:
        - ModelRegistry: Model management and loading

Example:
    >>> from transformation_portal.depth_canonical import (
    ...     UnifiedDepthConfig,
    ...     ProcessingConfig,
    ...     PBRConfig,
    ...     DepthPipeline,
    ... )
    >>>
    >>> # Configure pipeline with PBR enabled
    >>> config = UnifiedDepthConfig(
    ...     processing=ProcessingConfig(
    ...         pbr=PBRConfig(enabled=True, normal_strength=1.2)
    ...     )
    ... )
    >>>
    >>> # Create pipeline
    >>> pipeline = DepthPipeline(config)
    >>>
    >>> # Process with pre-computed depth (Phase 1)
    >>> result = pipeline.process(
    ...     depth_map=my_depth_array,
    ...     output_dir="output/",
    ...     basename="render_001"
    ... )
    >>>
    >>> # Access results
    >>> print(result.pbr_paths)  # {"normal": Path(...), "roughness": Path(...), "ao": Path(...)}
"""

from .config import (
    DeviceType,
    IOConfig,
    ModelConfig,
    ModelVariant,
    PBRConfig,
    ProcessingConfig,
    SecurityConfig,
    UnifiedDepthConfig,
)
from .io import write_pbr_maps
from .models import ModelRegistry
from .pipeline import DepthPipeline, DepthPipelineResult
from .processing import generate_pbr_maps

__version__ = "1.0.0"

__all__ = [
    # Configuration
    "UnifiedDepthConfig",
    "ModelConfig",
    "ProcessingConfig",
    "PBRConfig",
    "IOConfig",
    "SecurityConfig",
    # Enumerations
    "DeviceType",
    "ModelVariant",
    # Pipeline
    "DepthPipeline",
    "DepthPipelineResult",
    # Processing
    "generate_pbr_maps",
    "write_pbr_maps",
    # Models
    "ModelRegistry",
]
