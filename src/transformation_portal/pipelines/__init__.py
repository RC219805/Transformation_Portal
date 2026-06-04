"""Transformation Portal - Processing Pipelines.

Lazy-loading module to prevent import-time dependency explosions.
Heavy pipelines (ML-dependent) are only imported when explicitly accessed.
"""

__all__ = [
    # Unified Luxury Pipeline
    "UnifiedLuxuryPipeline",
    "UnifiedPipelineConfig",
    "ProcessingProfile",
    "SceneType",
    "OutputFormat",
    "PipelineStage",
    "PipelineStatistics",
    "ParallelIOItemResult",
    "ParallelIOPipeline",
    "process_luxury_render",
    "batch_process_luxury_renders",
    # 4K Rendering Pipeline
    "Rendering4KPipeline",
    "PipelineConfig",
    "ProcessingResult",
    "QualityAssessor",
    "QualityFeedbackConfig",
    "QualityLevel",
    "QualityMetrics",
    "ToneMappingMethod",
    # Quality Feedback Bridge
    "QualityFeedbackBridge",
    "QualityTargets",
    "UnifiedQualityMetrics",
    "HeuristicMetrics",
    "PerceptualMetrics",
    "MaterialFidelityMetrics",
    "create_quality_callback_for_pipeline",
    "create_rag_indexing_callback",
    "index_quality_metrics_to_rag",
]


def __getattr__(name: str):
    """Lazy-load pipeline classes and utilities on first access.

    Prevents import-time explosions when heavy ML dependencies
    (rendering_4k_pipeline -> controlnet_aux -> timm) are imported
    but not actually used.
    """
    # Unified Luxury Pipeline exports
    if name in (
        "UnifiedLuxuryPipeline",
        "UnifiedPipelineConfig",
        "ProcessingProfile",
        "SceneType",
        "OutputFormat",
        "PipelineStage",
        "PipelineStatistics",
        "ParallelIOItemResult",
        "ParallelIOPipeline",
        "process_luxury_render",
        "batch_process_luxury_renders",
    ):
        if name in ("ParallelIOItemResult", "ParallelIOPipeline"):
            from .parallel_io import ParallelIOItemResult, ParallelIOPipeline

            return locals()[name]

        from .unified_luxury_pipeline import (
            OutputFormat,
            PipelineStage,
            PipelineStatistics,
            ProcessingProfile,
            SceneType,
            UnifiedLuxuryPipeline,
            UnifiedPipelineConfig,
            batch_process_luxury_renders,
            process_luxury_render,
        )

        return locals()[name]

    # 4K Rendering Pipeline exports (ML-heavy)
    elif name in (
        "Rendering4KPipeline",
        "PipelineConfig",
        "ProcessingResult",
        "QualityAssessor",
        "QualityFeedbackConfig",
        "QualityLevel",
        "QualityMetrics",
        "ToneMappingMethod",
    ):
        from .rendering_4k_pipeline import (
            PipelineConfig,
            ProcessingResult,
            QualityAssessor,
            QualityFeedbackConfig,
            QualityLevel,
            QualityMetrics,
            Rendering4KPipeline,
            ToneMappingMethod,
        )

        return locals()[name]

    # Quality Feedback Bridge exports
    elif name in (
        "QualityFeedbackBridge",
        "QualityTargets",
        "UnifiedQualityMetrics",
        "HeuristicMetrics",
        "PerceptualMetrics",
        "MaterialFidelityMetrics",
        "create_quality_callback_for_pipeline",
        "create_rag_indexing_callback",
        "index_quality_metrics_to_rag",
    ):
        from .quality_feedback_bridge import (
            HeuristicMetrics,
            MaterialFidelityMetrics,
            PerceptualMetrics,
            QualityFeedbackBridge,
            QualityTargets,
            UnifiedQualityMetrics,
            create_quality_callback_for_pipeline,
            create_rag_indexing_callback,
            index_quality_metrics_to_rag,
        )

        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
