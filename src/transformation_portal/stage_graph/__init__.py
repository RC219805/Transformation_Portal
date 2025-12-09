"""
Stage Graph Architecture for Transformation Portal.

Provides cacheable, measurable pipeline stages with intelligent routing
and policy-based processing decisions.

Key Features:
- Content-addressed caching for 10-20x speedups
- Stage-level dependency tracking
- Policy engine for context-aware routing
- Parallel execution where possible
- Full observability and profiling
"""

from .stage import Stage, StageResult, StageContext, StageStatus
from .graph import StageGraph, GraphBuilder, GraphExecution
from .policy import (
    ProcessingPolicy,
    DevicePolicy,
    QualityPolicy,
    CachingPolicy,
    PolicyEngine,
    SceneType,
    QualityPreset,
)
from .stages import (
    DepthEstimationStage,
    MaterialSegmentationStage,
    EnhancementStage,
    UpscalingStage,
)

__all__ = [
    # Core abstractions
    "Stage",
    "StageResult",
    "StageContext",
    "StageStatus",
    # Graph
    "StageGraph",
    "GraphBuilder",
    "GraphExecution",
    # Policy
    "ProcessingPolicy",
    "DevicePolicy",
    "QualityPolicy",
    "CachingPolicy",
    "PolicyEngine",
    "SceneType",
    "QualityPreset",
    # Concrete stages
    "DepthEstimationStage",
    "MaterialSegmentationStage",
    "EnhancementStage",
    "UpscalingStage",
]
