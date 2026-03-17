"""Vision-language evaluation backends.

This subpackage provides LLaVA-based quality validation for
APEX Research Ultra workflow (ADR-026 §5).
"""

from transformation_portal.evals.vision_language.llava_backend import (
    LlavaBackendError,
    LlavaGenerationConfig,
    LlavaQualityBackend,
)
from transformation_portal.evals.vision_language.llava_loader import (
    LlavaLoaderError,
    LlavaLoadedArtifacts,
    load_llava_from_manifest_entry,
)
from transformation_portal.evals.vision_language.llava_prompts import (
    LlavaPromptSpec,
    build_architectural_quality_prompt,
    build_depth_quality_prompt,
    build_segmentation_quality_prompt,
)
from transformation_portal.evals.vision_language.llava_schema import (
    VQAIssue,
    VQAParseError,
    VQAResult,
    parse_vqa_result,
)
from transformation_portal.evals.vision_language.llava_scoring import (
    compute_quality_gate_pass,
    recompute_summary_score,
    severity_to_numeric,
)

__all__ = [
    "LlavaBackendError",
    "LlavaGenerationConfig",
    "LlavaQualityBackend",
    "LlavaLoaderError",
    "LlavaLoadedArtifacts",
    "load_llava_from_manifest_entry",
    "LlavaPromptSpec",
    "build_architectural_quality_prompt",
    "build_depth_quality_prompt",
    "build_segmentation_quality_prompt",
    "VQAIssue",
    "VQAParseError",
    "VQAResult",
    "parse_vqa_result",
    "compute_quality_gate_pass",
    "recompute_summary_score",
    "severity_to_numeric",
]
