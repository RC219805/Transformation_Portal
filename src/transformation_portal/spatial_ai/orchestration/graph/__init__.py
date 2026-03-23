"""Execution graph infrastructure for Spatial AI orchestration (Phase 3 L1).

This package implements the core execution graph abstraction for spatial AI pipelines:
- Explicit DAG modeling (not implicit sequencing)
- Content-addressed caching (deterministic, hash-based)
- Automatic provenance tracking (input fingerprints + model revisions + timestamps)
- Fail-fast resource budgeting (validate before execution, not during)

Architecture (ADR-029):
- Domain-specific (not a generic workflow engine)
- Deterministic by default (same inputs → same outputs)
- Introspectable (can query stages, dependencies, resource requirements)
- Contract-driven (explicit input/output mappings)

Modules:
- stage: Core abstraction for execution stages (Stage protocol, StageMetadata)
- execution_graph: DAG composition (ExecutionGraph, topological sort, validation)
- artifact_store: Content-addressed cache with provenance (ArtifactStore)
- executor: Runtime orchestration (Executor, sequential execution, caching)

Example:
    >>> from transformation_portal.spatial_ai.orchestration.graph import (
    ...     ExecutionGraph,
    ...     Executor,
    ...     ArtifactStore,
    ... )
    >>>
    >>> # Build graph
    >>> graph = ExecutionGraph()
    >>> graph.add_stage("ingest", IngestStage(), inputs={})
    >>> graph.add_stage("segment", SAM2Stage(), inputs={"linear_rgb": "ingest.linear_rgb"})
    >>>
    >>> # Execute with caching
    >>> store = ArtifactStore(cache_dir=".cache/spatial_ai")
    >>> executor = Executor(artifact_store=store)
    >>> result = executor.execute(graph, inputs={"input_path": "scene.tiff"})
    >>>
    >>> print(f"Executed: {result.stages_executed}, Cached: {result.stages_cached}")
"""

from __future__ import annotations

from .artifact_store import ArtifactStore, CacheLockTimeout, ProvenanceMetadata
from .execution_graph import ExecutionGraph, ExecutionPlan, StageNode
from .executor import ExecutionContext, ExecutionResult, Executor
from .stage import ResourceRequirements, Stage, StageMetadata
from .stage_adapters import (
    IngestStage,
    IngestStageConfig,
    MaterialsStage,
    SegmentationStage,
    build_spatial_ai_graph,
)

__all__ = [
    # Stage protocol
    "Stage",
    "StageMetadata",
    "ResourceRequirements",
    # Execution graph
    "ExecutionGraph",
    "StageNode",
    "ExecutionPlan",
    # Artifact store
    "ArtifactStore",
    "ProvenanceMetadata",
    "CacheLockTimeout",
    # Executor
    "Executor",
    "ExecutionContext",
    "ExecutionResult",
    # Stage adapters (ADR-029 integration)
    "IngestStage",
    "IngestStageConfig",
    "SegmentationStage",
    "MaterialsStage",
    "build_spatial_ai_graph",
]
