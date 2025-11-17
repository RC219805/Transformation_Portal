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

from transformation_portal.comfyui.workflow_builder import WorkflowBuilder
from transformation_portal.comfyui.workflow_templates import WorkflowTemplates
from transformation_portal.comfyui.executor import WorkflowExecutor
from transformation_portal.comfyui.custom_nodes import (
    CustomNodeRegistry,
    FluxEnhancementNode,
    SkyGANNode,
    SceneAnalysisNode,
    MaterialSegmentationNode,
    NeuroaestheticsNode,
    QualityValidationNode
)

__all__ = [
    'WorkflowBuilder',
    'WorkflowTemplates',
    'WorkflowExecutor',
    'CustomNodeRegistry',
    'FluxEnhancementNode',
    'SkyGANNode',
    'SceneAnalysisNode',
    'MaterialSegmentationNode',
    'NeuroaestheticsNode',
    'QualityValidationNode',
]
