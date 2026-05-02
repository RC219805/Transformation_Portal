"""Optional advisory VLM captioning helpers.

This package is intentionally separate from ``transformation_portal.vlm``.
FastVLM is executed by subprocess only, so importing this package does not
import MLX, CoreML, or model runtime dependencies.
"""

from .fastvlm_runtime import (
    DEFAULT_FASTVLM_PROMPT,
    FASTVLM_MODEL_ROLES,
    FastVLMRuntimeConfig,
    FastVLMRuntimeResult,
    build_fastvlm_sidecar,
    resolve_fastvlm_model_id,
    resolve_fastvlm_model_path,
    run_fastvlm_caption,
)
from .image_proxy import VLMImageProxy, build_vlm_image_proxy
from .parser import FastVLMCaptionParse, parse_fastvlm_caption

__all__ = [
    "DEFAULT_FASTVLM_PROMPT",
    "FASTVLM_MODEL_ROLES",
    "FastVLMCaptionParse",
    "FastVLMRuntimeConfig",
    "FastVLMRuntimeResult",
    "VLMImageProxy",
    "build_fastvlm_sidecar",
    "build_vlm_image_proxy",
    "parse_fastvlm_caption",
    "resolve_fastvlm_model_id",
    "resolve_fastvlm_model_path",
    "run_fastvlm_caption",
]
