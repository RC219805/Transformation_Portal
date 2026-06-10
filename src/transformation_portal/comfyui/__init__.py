"""ComfyUI workflow integration for visual pipeline orchestration.

ComfyUI provides node-based workflow design for complex enhancement pipelines,
enabling:
- Visual pipeline composition
- Modular component integration
- Parameter experimentation
- Reproducible enhancement workflows

Integrates all Transformation Portal components:
- FLUX diffusion for AI enhancement
- SkyGAN for atmospheric rendering
- VLM for scene analysis and quality validation
- Semantic segmentation for material-aware processing
- Neuroaesthetics optimization
- Quality metrics (LPIPS, FID)

Example Workflows:
- Full luxury estate enhancement pipeline
- Quick iterative enhancement with quality gates
- Material-specific processing with segmentation
- Location-specific atmospheric rendering
- Multi-variant generation with emotional targeting
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Lazy (PEP 562) re-exports. ``custom_nodes`` and ``executor`` pull heavy
# optional runtimes (``torch``, ``numpy``, the ``atmosphere`` stack), so
# importing this package must NOT eagerly drag them in — otherwise the
# pure-Python ``workflow_builder`` / ``workflow_templates`` cannot be imported
# or unit-tested in the torch-free core lane. Each public name resolves to its
# submodule on first access; the node-registry decorators in ``custom_nodes``
# still run the moment any of its symbols is touched, so behavior is unchanged
# for ML-lane consumers. (Mirrors the lazy-import seam discipline in
# docs/testing/COLD_ZONE_COVERAGE_PROGRAM.md and CLAUDE.md.)
_LAZY_EXPORTS = {
    "WorkflowBuilder": "workflow_builder",
    "WorkflowTemplates": "workflow_templates",
    "WorkflowExecutor": "executor",
    "CustomNodeRegistry": "custom_nodes",
    "FluxEnhancementNode": "custom_nodes",
    "SkyGANNode": "custom_nodes",
    "SceneAnalysisNode": "custom_nodes",
    "MaterialSegmentationNode": "custom_nodes",
    "NeuroaestheticsNode": "custom_nodes",
    "QualityValidationNode": "custom_nodes",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public symbol from its submodule on first access (PEP 562)."""
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    submodule = importlib.import_module(f"{__name__}.{module}")
    return getattr(submodule, name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from transformation_portal.comfyui.custom_nodes import (
        CustomNodeRegistry,
        FluxEnhancementNode,
        MaterialSegmentationNode,
        NeuroaestheticsNode,
        QualityValidationNode,
        SceneAnalysisNode,
        SkyGANNode,
    )
    from transformation_portal.comfyui.executor import WorkflowExecutor
    from transformation_portal.comfyui.workflow_builder import WorkflowBuilder
    from transformation_portal.comfyui.workflow_templates import WorkflowTemplates
