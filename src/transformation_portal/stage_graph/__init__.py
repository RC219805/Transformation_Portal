"""
Stage Graph Architecture for Transformation Portal.

Provides trusted in-process target-executor primitives. ADR-051 does not
authorize canonical plans to use this package for production execution yet.

Keep this package initializer lazy: the canonical execution-plan parser uses
the static registry and must not import graph executors, concrete stages, or
their optional numerical/ML dependencies merely to validate a plan.

Key Features:
- Experimental content-addressed stage caching
- Stage-level dependency tracking
- Policy engine for context-aware routing
- Parallel execution where possible
- Full observability and profiling
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Core abstractions
    "Stage": (".stage", "Stage"),
    "StageResult": (".stage", "StageResult"),
    "StageContext": (".stage", "StageContext"),
    "StageStatus": (".stage", "StageStatus"),
    # Graph
    "StageGraph": (".graph", "StageGraph"),
    "GraphBuilder": (".graph", "GraphBuilder"),
    "GraphExecution": (".graph", "GraphExecution"),
    # Policy
    "ProcessingPolicy": (".policy", "ProcessingPolicy"),
    "DevicePolicy": (".policy", "DevicePolicy"),
    "QualityPolicy": (".policy", "QualityPolicy"),
    "CachingPolicy": (".policy", "CachingPolicy"),
    "PolicyEngine": (".policy", "PolicyEngine"),
    "SceneType": (".policy", "SceneType"),
    "QualityPreset": (".policy", "QualityPreset"),
    # Static semantic registry
    "OUTPUT_DEFINITIONS": (".registry", "OUTPUT_DEFINITIONS"),
    "STAGE_DEFINITIONS": (".registry", "STAGE_DEFINITIONS"),
    "OutputCardinality": (".registry", "OutputCardinality"),
    "OutputKindDefinition": (".registry", "OutputKindDefinition"),
    "OutputScope": (".registry", "OutputScope"),
    "StageDefinition": (".registry", "StageDefinition"),
    "StageRegistryIdentifier": (".registry", "StageRegistryIdentifier"),
    "UnknownOutputKind": (".registry", "UnknownOutputKind"),
    "UnknownStageRegistryIdentifier": (".registry", "UnknownStageRegistryIdentifier"),
    "get_output_definition": (".registry", "get_output_definition"),
    "get_stage_definition": (".registry", "get_stage_definition"),
    "stage_registry_identifiers": (".registry", "stage_registry_identifiers"),
    # Concrete stages
    "DepthEstimationStage": (".stages", "DepthEstimationStage"),
    "MaterialSegmentationStage": (".stages", "MaterialSegmentationStage"),
    "EnhancementStage": (".stages", "EnhancementStage"),
    "UpscalingStage": (".stages", "UpscalingStage"),
}

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
    # Static semantic registry (does not construct stages)
    "OUTPUT_DEFINITIONS",
    "STAGE_DEFINITIONS",
    "OutputCardinality",
    "OutputKindDefinition",
    "OutputScope",
    "StageDefinition",
    "StageRegistryIdentifier",
    "UnknownOutputKind",
    "UnknownStageRegistryIdentifier",
    "get_output_definition",
    "get_stage_definition",
    "stage_registry_identifiers",
    # Concrete stages
    "DepthEstimationStage",
    "MaterialSegmentationStage",
    "EnhancementStage",
    "UpscalingStage",
]


def __getattr__(name: str) -> object:
    """Resolve the unchanged public API without eager optional imports."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
