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

from .graph import GraphBuilder, GraphExecution, StageGraph
from .policy import (
    CachingPolicy,
    DevicePolicy,
    PolicyEngine,
    ProcessingPolicy,
    QualityPolicy,
    QualityPreset,
    SceneType,
)
from .stage import Stage, StageContext, StageResult, StageStatus
from .stages import (
    DepthEstimationStage,
    EnhancementStage,
    MaterialSegmentationStage,
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
