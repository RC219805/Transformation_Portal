"""Evaluation harness package.

This package provides evaluation backends and utilities for assessing
pipeline outputs including:
- Vision-language models (LLaVA) for quality validation
- Traditional image quality metrics (PSNR, SSIM, LPIPS, IoU)
- APEX Research Ultra evaluation harness
- APEX + LLaVA integration for VLM-based quality assessment
- Full benchmark suite
"""

from transformation_portal.evals.apex_harness import (
    ApexEvaluationHarness,
    EvalMetricResult,
    EvalResult,
    brightness_metric,
    contrast_metric,
    sharpness_metric,
)
from transformation_portal.evals.apex_llava_integration import (
    ApexLlavaConfig,
    ApexLlavaIntegrationError,
    build_material_quality_prompt,
    create_apex_harness_with_llava,
    create_apex_harness_without_llava,
    create_ci_smoke_harness,
    create_llava_backend,
    create_production_harness,
    create_quality_max_harness,
)
from transformation_portal.evals.apex_visual import (
    ApexEvalAsset,
    ApexEvalSet,
    DepthBackendRunResult,
    build_apex_eval_report,
    build_depth_backend_benchmark_report,
    load_apex_evalset,
)
from transformation_portal.evals.benchmark_suite import (
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkWeights,
    run_benchmark_batch,
)
from transformation_portal.evals.metrics import (
    dice_coefficient,
    lpips_score,
    lpips_to_score,
    psnr,
    psnr_to_score,
    psnr_torch,
    segmentation_iou,
    ssim,
)

__all__ = [
    # APEX harness
    "ApexEvaluationHarness",
    "EvalMetricResult",
    "EvalResult",
    "brightness_metric",
    "contrast_metric",
    "sharpness_metric",
    # APEX + LLaVA integration
    "ApexLlavaConfig",
    "ApexLlavaIntegrationError",
    "build_material_quality_prompt",
    "create_apex_harness_with_llava",
    "create_apex_harness_without_llava",
    "create_ci_smoke_harness",
    "create_llava_backend",
    "create_production_harness",
    "create_quality_max_harness",
    # APEX visual eval corpus
    "ApexEvalAsset",
    "ApexEvalSet",
    "DepthBackendRunResult",
    "build_apex_eval_report",
    "build_depth_backend_benchmark_report",
    "load_apex_evalset",
    # Benchmark suite
    "BenchmarkResult",
    "BenchmarkSuite",
    "BenchmarkWeights",
    "run_benchmark_batch",
    # Metrics
    "dice_coefficient",
    "lpips_score",
    "lpips_to_score",
    "psnr",
    "psnr_to_score",
    "psnr_torch",
    "segmentation_iou",
    "ssim",
]
