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
    build_segmentation_quality_prompt,
)
from transformation_portal.evals.vision_language.llava_schema import (
    VQAIssue,
    VQAResult,
    parse_vqa_result,
)
from transformation_portal.evals.vision_language.llava_scoring import recompute_summary_score

__all__ = [
    "LlavaBackendError",
    "LlavaGenerationConfig",
    "LlavaQualityBackend",
    "LlavaLoaderError",
    "LlavaLoadedArtifacts",
    "load_llava_from_manifest_entry",
    "LlavaPromptSpec",
    "build_segmentation_quality_prompt",
    "VQAIssue",
    "VQAResult",
    "parse_vqa_result",
    "recompute_summary_score",
]
