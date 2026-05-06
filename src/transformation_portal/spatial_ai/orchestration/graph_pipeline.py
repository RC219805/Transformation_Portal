"""Graph execution bridge for Spatial AI orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from transformation_portal.reporting.contracts import build_stage_report
from transformation_portal.spatial_ai.ingest.linear_decoder import LinearIngestResult
from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationResult

from .error_handler import ErrorRecoveryStrategy, PipelineError
from .graph import ArtifactStore, Executor, build_spatial_ai_graph
from .json_io import write_json_atomic
from .results import PipelineResult

logger = logging.getLogger("transformation_portal.spatial_ai.orchestration.pipeline")


def _build_graph_config(pipeline: Any) -> dict[str, Any]:
    return {
        "strict_ingest": pipeline.config.ingest.get("strict_ingest", False),
        "emit_exr": pipeline.config.ingest.get("emit_exr", False),
        "emit_provenance": pipeline.config.ingest.get("emit_provenance", False),
        "model_size": pipeline.config.segmentation.get("model", {}).get("size", "large"),
        "enable_material_classification": bool(pipeline.config.segmentation.get("material_classification", False)),
        "backend": pipeline.config.materials.get("backend", "heuristic"),
        "device": pipeline.resource_manager.select_device(),
    }


def _compute_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Compute a tight (x, y, w, h) bbox for a boolean/uint8 mask."""
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return (0, 0, 0, 0)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def _build_pipeline_result(
    *,
    pipeline: Any,
    input_path: Path,
    output_dir: Path,
    exec_result: Any,
) -> PipelineResult:
    result = PipelineResult(
        input_path=input_path,
        output_dir=output_dir,
        stages_completed=[sr.stage_id for sr in exec_result.stage_results],
        execution_time=exec_result.total_time_ms / 1000.0,
        peak_memory_mb=pipeline.resource_manager.get_peak_memory_mb(),
        stage_reports=[
            build_stage_report(
                stage=sr.stage_id,
                status=("cached" if getattr(sr, "cache_hit", False) else "completed"),
            )
            for sr in exec_result.stage_results
        ],
    )

    if "ingest.linear_rgb" in exec_result.outputs:
        linear_rgb = exec_result.outputs["ingest.linear_rgb"]
        input_size = exec_result.outputs.get("ingest.input_size", linear_rgb.shape[:2])

        result.linear_image = LinearIngestResult(
            linear_rgb=linear_rgb,
            input_path=input_path,
            input_size=tuple(input_size) if not isinstance(input_size, tuple) else input_size,
            gamma=1.0,
            bit_depth=32,
            dtype="float32",
            input_format=exec_result.outputs.get("ingest.input_format", "TIFF"),
            color_space=exec_result.outputs.get("ingest.color_space", "linear_sRGB"),
        )

    if "segment.masks" in exec_result.outputs:
        masks = exec_result.outputs["segment.masks"]
        scores = exec_result.outputs.get("segment.scores", np.ones(len(masks)))

        areas = exec_result.outputs.get("segment.metadata.area")
        bboxes = exec_result.outputs.get("segment.metadata.bbox")
        stabilities = exec_result.outputs.get("segment.metadata.stability_score")

        metadata: list[MaskMetadata] = []
        num_masks = len(masks)

        for i in range(num_masks):
            mask = masks[i]

            if areas is not None and i < len(areas):
                area = int(areas[i])
            else:
                area = int(np.count_nonzero(mask))

            if bboxes is not None and i < len(bboxes):
                bbox = tuple(int(v) for v in bboxes[i])
            else:
                bbox = _compute_bbox(mask)

            if stabilities is not None and i < len(stabilities):
                stability_score = float(stabilities[i])
            else:
                stability_score = 0.5

            metadata.append(
                MaskMetadata(
                    area=max(1, area),
                    bbox=bbox,
                    stability_score=stability_score,
                )
            )

        result.segmentation = SegmentationResult(
            masks=masks,
            scores=scores,
            metadata=metadata,
        )

    if "materials.pbr_textures" in exec_result.outputs:
        result.materials = exec_result.outputs["materials.pbr_textures"]

    return result


def _build_graph_execution_metadata(exec_result: Any) -> dict[str, Any]:
    return {
        "stages_executed": exec_result.stages_executed,
        "stages_cached": exec_result.stages_cached,
        "total_time_ms": exec_result.total_time_ms,
        "stage_results": [
            {
                "stage_id": sr.stage_id,
                "cache_hit": sr.cache_hit,
                "execution_time_ms": sr.execution_time_ms,
                "cache_key": sr.cache_key,
            }
            for sr in exec_result.stage_results
        ],
    }


def process_with_graph(
    pipeline: Any,
    input_path: Path,
    output_dir: Path,
    save_intermediates: bool,
) -> PipelineResult:
    """Execute pipeline using the ADR-029 execution graph abstraction."""
    logger.info("Using ADR-029 graph-based execution")

    if "reconstruction" in pipeline.config.stages:
        raise PipelineError(
            "graph",
            "Reconstruction is not supported in graph mode (ADR-029). "
            "Either disable graph mode (use_execution_graph=False) or "
            "remove reconstruction from stages.",
        )

    graph_stages = list(pipeline.config.stages)
    graph_config = _build_graph_config(pipeline)
    graph = build_spatial_ai_graph(stages=graph_stages, config=graph_config)

    cache_dir = output_dir / ".cache" / "spatial_ai"
    artifact_store = ArtifactStore(cache_dir=cache_dir)

    executor = Executor(
        artifact_store=artifact_store,
        resource_limits=pipeline.config.resource_limits,
        device=graph_config["device"],
    )

    pipeline.progress_tracker.start_pipeline()
    try:
        exec_result = executor.execute(
            graph=graph,
            inputs={"input_path": str(input_path)},
            output_dir=output_dir,
            config=graph_config,
        )

        result = _build_pipeline_result(
            pipeline=pipeline,
            input_path=input_path,
            output_dir=output_dir,
            exec_result=exec_result,
        )

        if save_intermediates:
            summary_path = output_dir / "pipeline_summary.json"
            result.save_summary(summary_path)
            write_json_atomic(output_dir / "graph_execution.json", _build_graph_execution_metadata(exec_result))

        pipeline.progress_tracker.complete_pipeline(success=True)
        logger.info(
            f"Graph execution completed: {exec_result.stages_executed} executed, "
            f"{exec_result.stages_cached} cached, "
            f"{exec_result.total_time_ms:.1f}ms total"
        )

        return result

    except Exception as e:
        pipeline.progress_tracker.complete_pipeline(success=False)
        logger.error(f"Graph execution failed: {e}")

        result = PipelineResult(
            input_path=input_path,
            output_dir=output_dir,
            stages_completed=[],
            errors=[str(e)],
        )

        if pipeline.config.error_strategy == ErrorRecoveryStrategy.RETURN_PARTIAL:
            return result
        raise PipelineError("graph", f"Graph execution failed: {e}", original_error=e) from e


__all__ = [
    "process_with_graph",
]
