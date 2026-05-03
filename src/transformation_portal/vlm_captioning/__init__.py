"""Optional advisory VLM captioning helpers.

This package is intentionally separate from ``transformation_portal.vlm``.
FastVLM is executed by subprocess only, so importing this package does not
import MLX, CoreML, or model runtime dependencies.
"""

from .fastvlm_runtime import (
    DEFAULT_FASTVLM_PROMPT,
    FASTVLM_MODEL_ROLES,
    FASTVLM_PROMPTS,
    REVIEW_FASTVLM_PROMPT,
    FastVLMRuntimeConfig,
    FastVLMRuntimeResult,
    build_fastvlm_sidecar,
    default_fastvlm_runtime_root,
    infer_fastvlm_model_role,
    prompt_for_fastvlm_model,
    resolve_fastvlm_model_id,
    resolve_fastvlm_model_path,
    resolve_fastvlm_python_executable,
    resolve_fastvlm_runtime_path,
    run_fastvlm_caption,
)
from .image_proxy import VLMImageProxy, build_vlm_image_proxy
from .parser import FastVLMCaptionParse, parse_fastvlm_caption

__all__ = [
    "DEFAULT_FASTVLM_PROMPT",
    "FASTVLM_MODEL_ROLES",
    "FASTVLM_PROMPTS",
    "FastVLMCaptionParse",
    "FastVLMRuntimeConfig",
    "FastVLMRuntimeResult",
    "REVIEW_FASTVLM_PROMPT",
    "VLMImageProxy",
    "build_fastvlm_sidecar",
    "build_vlm_image_proxy",
    "default_fastvlm_runtime_root",
    "infer_fastvlm_model_role",
    "parse_fastvlm_caption",
    "prompt_for_fastvlm_model",
    "resolve_fastvlm_model_id",
    "resolve_fastvlm_model_path",
    "resolve_fastvlm_python_executable",
    "resolve_fastvlm_runtime_path",
    "run_fastvlm_caption",
]
