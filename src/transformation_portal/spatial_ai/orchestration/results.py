"""Result models for the Spatial AI orchestration pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformation_portal.reporting.contracts import derive_stage_report_map
from transformation_portal.spatial_ai.ingest.linear_decoder import LinearIngestResult
from transformation_portal.spatial_ai.materials.contracts import PBRTextures
from transformation_portal.spatial_ai.reconstruction.contracts import Scene3D
from transformation_portal.spatial_ai.segmentation.contracts import SegmentationResult

from .json_io import write_json_atomic as _write_json_atomic

logger = logging.getLogger("transformation_portal.spatial_ai.orchestration.pipeline")


@dataclass
class PipelineResult:
    """Result from end-to-end pipeline execution.

    Attributes:
        input_path: Input file path.
        output_dir: Output directory.
        stages_completed: List of completed stages.
        linear_image: Linear ingest result (if ingest stage run).
        segmentation: Segmentation result (if the segmentation stage ran).
        materials: PBR textures per segment (if materials stage run).
        scene_3d: 3D scene reconstruction (if reconstruction stage run).
        execution_time: Total execution time in seconds.
        peak_memory_mb: Peak GPU memory usage in MB.
        errors: List of error messages.
        warnings: List of warning messages.
        metadata: Additional metadata.
    """

    input_path: Path
    output_dir: Path
    stages_completed: List[str]

    linear_image: Optional[LinearIngestResult] = None
    segmentation: Optional[SegmentationResult] = None
    materials: Optional[Dict[str, PBRTextures]] = None
    scene_3d: Optional[Scene3D] = None

    execution_time: float = 0.0
    peak_memory_mb: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_reports: List[Dict[str, Any]] = field(default_factory=list)

    def save_summary(self, path: Path) -> None:
        """Save execution summary as JSON.

        Args:
            path: Output path for summary JSON.
        """
        stage_reports = list(self.stage_reports)
        stage_report_map = derive_stage_report_map(stage_reports)
        summary = {
            "input": str(self.input_path),
            "output_dir": str(self.output_dir),
            "stages_completed": self.stages_completed,
            "execution_time": self.execution_time,
            "peak_memory_mb": self.peak_memory_mb,
            "errors": self.errors,
            "warnings": self.warnings,
            "stage_reports": stage_reports,
            "results": {
                "linear_image": self.linear_image is not None,
                "segmentation": {
                    "completed": (
                        stage_report_map.get("segmentation", {}).get("status") in {"completed", "cached"}
                        if "segmentation" in stage_report_map
                        else self.segmentation is not None
                    ),
                    "num_masks": len(self.segmentation.masks) if self.segmentation else 0,
                },
                "materials": {
                    "completed": (
                        stage_report_map.get("materials", {}).get("status") in {"completed", "cached"}
                        if "materials" in stage_report_map
                        else self.materials is not None
                    ),
                    "num_segments": len(self.materials) if self.materials else 0,
                },
                "scene_3d": {
                    "completed": (
                        stage_report_map.get("reconstruction", {}).get("status") in {"completed", "cached"}
                        if "reconstruction" in stage_report_map
                        else self.scene_3d is not None
                    ),
                    "num_gaussians": self.scene_3d.splats.num_gaussians if self.scene_3d else 0,
                    "rmse": self.scene_3d.rmse if self.scene_3d else None,
                },
            },
            "metadata": self.metadata,
        }

        _write_json_atomic(path, summary)

        logger.info(f"Saved pipeline summary: {path}")


@dataclass
class MultiViewReconstructionResult:
    """Result from multi-view reconstruction pipeline.

    Attributes:
        scene: Reconstructed 3D scene (Scene3D).
        ply_path: Path to exported PLY file.
        sidecar_path: Path to provenance JSON sidecar.
        output_dir: Output directory.
        execution_time: Total execution time in seconds.
        peak_memory_mb: Peak GPU memory usage in MB.
        stages_completed: List of completed stages.
        request_metadata: Original request metadata for traceability.
        errors: List of error messages (if any).
        warnings: List of warning messages (if any).
    """

    scene: Scene3D
    ply_path: Path
    sidecar_path: Path
    output_dir: Path
    execution_time: float = 0.0
    peak_memory_mb: float = 0.0
    stages_completed: List[str] = field(default_factory=list)
    request_metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stage_reports: List[Dict[str, Any]] = field(default_factory=list)

    def save_summary(self, path: Path) -> None:
        """Save reconstruction summary as JSON.

        Args:
            path: Output path for summary JSON.
        """
        summary = {
            "output_dir": str(self.output_dir),
            "ply_path": str(self.ply_path),
            "sidecar_path": str(self.sidecar_path),
            "stages_completed": self.stages_completed,
            "execution_time": self.execution_time,
            "peak_memory_mb": self.peak_memory_mb,
            "errors": self.errors,
            "warnings": self.warnings,
            "stage_reports": self.stage_reports,
            "scene": {
                "num_gaussians": self.scene.splats.num_gaussians,
                "rmse": self.scene.rmse,
                "convergence": self.scene.convergence,
                "quality_score": self.scene.quality_score,
                "iteration": self.scene.iteration,
            },
            "request_metadata": self.request_metadata,
        }

        _write_json_atomic(path, summary)

        logger.info(f"Saved reconstruction summary: {path}")


__all__ = [
    "MultiViewReconstructionResult",
    "PipelineResult",
]
