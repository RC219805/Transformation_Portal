"""ComfyUI declarative workflow integration.

The package-level import intentionally exposes only pure workflow construction
primitives. Runtime custom nodes and the executor are loaded lazily because they
depend on optional ML/runtime packages that are not part of the core install.
"""

from typing import Any

from transformation_portal.comfyui.workflow_builder import Node, NodeConnection, NodeType, Workflow, WorkflowBuilder
from transformation_portal.comfyui.workflow_templates import WorkflowTemplates

__all__ = [
    "Node",
    "NodeConnection",
    "NodeType",
    "Workflow",
    "WorkflowBuilder",
    "WorkflowTemplates",
    "WorkflowExecutor",
    "CustomNodeRegistry",
    "BaseNode",
    "FluxEnhancementNode",
    "SkyGANNode",
    "SceneAnalysisNode",
]


def __getattr__(name: str) -> Any:
    if name == "WorkflowExecutor":
        from transformation_portal.comfyui.executor import WorkflowExecutor

        return WorkflowExecutor

    if name in {
        "CustomNodeRegistry",
        "BaseNode",
        "FluxEnhancementNode",
        "SkyGANNode",
        "SceneAnalysisNode",
    }:
        from transformation_portal.comfyui import custom_nodes

        return getattr(custom_nodes, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
