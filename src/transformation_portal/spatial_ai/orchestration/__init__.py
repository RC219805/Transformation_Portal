"""Orchestration layer for Spatial AI pipeline (Phase 2.4).

This module provides end-to-end pipeline orchestration, tying together:
- Phase 1: Linear ingest
- Phase 2.1: SAM2 segmentation
- Phase 2.2: PBR materials
- Phase 2.3: 3D reconstruction

Public API:
    - SpatialAIPipeline: Main orchestrator
    - PipelineResult: Output dataclass
    - ResourceManager: GPU memory management
    - ProgressTracker: Progress reporting
    - PipelineConfig: Configuration schema

Example:
    >>> from transformation_portal.spatial_ai.orchestration import SpatialAIPipeline
    >>> pipeline = SpatialAIPipeline.from_preset("spatial_ai_standard")
    >>> result = pipeline.process(
    ...     input_path="scene.tiff",
    ...     output_dir="output/"
    ... )
    >>> print(f"Completed {len(result.stages_completed)} stages in {result.execution_time:.1f}s")

Architecture (ADR-027, ADR-028):
- Phased execution with resource management
- Graceful degradation and error recovery
- Progress tracking and provenance logging
- Tier enforcement (3DGS requires research tier)
- Contract validation at phase boundaries
"""

from __future__ import annotations

from .config import PipelineConfig
from .error_handler import ErrorHandler, ErrorRecoveryStrategy, PipelineError
from .pipeline import PipelineResult, SpatialAIPipeline
from .progress_tracker import ProgressEvent, ProgressTracker
from .resource_manager import ResourceLimits, ResourceManager

__all__ = [
    "SpatialAIPipeline",
    "PipelineConfig",
    "PipelineResult",
    "ResourceManager",
    "ResourceLimits",
    "ProgressTracker",
    "ProgressEvent",
    "ErrorHandler",
    "ErrorRecoveryStrategy",
    "PipelineError",
]
