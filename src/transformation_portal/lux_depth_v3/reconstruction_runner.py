"""Scene-level reconstruction runner for lux_depth_v3."""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Dict

import numpy as np

from transformation_portal.spatial_ai.reconstruction.contracts import Scene3D

from ..ingest.canonical_json import dumps_json
from .io_atomic import atomic_write_bytes
from .scene_context import SceneContext
from .scene_groups import normalize_relative_path
from .security import sanitize_path_component_nonlossy

logger = logging.getLogger(__name__)


def _median_camera_baseline(scene: Scene3D) -> float:
    """Compute median pairwise camera-center distance."""
    centers = [camera.extrinsics[:3, 3].astype(np.float32) for camera in scene.cameras]
    if len(centers) < 2:
        raise ValueError("Scale normalization requires at least 2 cameras")
    baselines = [float(np.linalg.norm(a - b)) for a, b in combinations(centers, 2)]
    if not baselines:
        raise ValueError("Scale normalization requires at least one camera baseline")
    median_baseline = float(np.median(np.asarray(baselines, dtype=np.float32)))
    if not np.isfinite(median_baseline) or median_baseline <= 0.0:
        raise ValueError(f"Invalid median camera baseline for normalization: {median_baseline}")
    return median_baseline


def _normalize_scene_scale(scene: Scene3D) -> Dict[str, float | str]:
    """Normalize scene into canonical scale where median camera baseline is 1.0."""
    baseline_before = _median_camera_baseline(scene)
    scale_factor = float(1.0 / baseline_before)

    scene.splats.positions = (scene.splats.positions * scale_factor).astype(np.float32, copy=False)
    scene.splats.scales = (scene.splats.scales * scale_factor).astype(np.float32, copy=False)
    for camera in scene.cameras:
        camera.extrinsics[:3, 3] = camera.extrinsics[:3, 3] * scale_factor

    baseline_after = _median_camera_baseline(scene)
    if not (0.1 < baseline_after < 10.0):
        raise ValueError(f"Canonical baseline validation failed after normalization: {baseline_after}")

    scale_metadata: Dict[str, float | str] = {
        "method": "median_baseline",
        "scale_factor": scale_factor,
        "baseline_before": baseline_before,
        "baseline_after": baseline_after,
    }
    scene.metadata["scale_normalization"] = scale_metadata
    return scale_metadata


def diagnostics_artifact_path(*, scene_id: str, output_dir: Path) -> Path:
    """Compute deterministic diagnostics artifact path for a scene."""
    safe_scene_id = sanitize_path_component_nonlossy(scene_id)
    return output_dir / f"{safe_scene_id}_diagnostics.json"


def _scene_geometry_stats(scene: Scene3D) -> Dict[str, float | int | list[float]]:
    """Compute compact geometry diagnostics for diffable scene introspection."""
    positions = scene.splats.positions
    bbox_min = positions.min(axis=0).astype(np.float32)
    bbox_max = positions.max(axis=0).astype(np.float32)
    bbox_diag = float(np.linalg.norm((bbox_max - bbox_min).astype(np.float32)))
    return {
        "point_count": int(scene.splats.num_gaussians),
        "bbox_min": [float(v) for v in bbox_min.tolist()],
        "bbox_max": [float(v) for v in bbox_max.tolist()],
        "bbox_diag": bbox_diag,
    }


def _write_scene_diagnostics(
    *,
    scene: Scene3D,
    context: SceneContext,
    output_dir: Path,
    scale_metadata: Dict[str, float | str],
) -> Path:
    """Write deterministic scene diagnostics artifact."""
    baselines = [float(np.linalg.norm(a.extrinsics[:3, 3] - b.extrinsics[:3, 3])) for a, b in combinations(scene.cameras, 2)]
    baseline_array = np.asarray(baselines, dtype=np.float32)
    diagnostics_payload: Dict[str, object] = {
        "schema": "tp.scene_diagnostics.v1",
        "scene_id": context.scene_id,
        "inputs": {
            "image_count": len(context.images),
            "grouping_mode": context.metadata.get("grouping_mode"),
        },
        "cameras": {
            "count": len(scene.cameras),
            "baseline_median": float(np.median(baseline_array)),
            "baseline_min": float(np.min(baseline_array)),
            "baseline_max": float(np.max(baseline_array)),
            "sources": [camera.provenance.source for camera in context.cameras],
            "confidences": [camera.provenance.confidence for camera in context.cameras],
        },
        "geometry": _scene_geometry_stats(scene),
        "scale": scale_metadata,
        "quality": {
            "rmse": float(scene.rmse),
            "iterations": int(scene.iteration),
            "convergence": scene.convergence,
        },
    }
    diagnostics_path = diagnostics_artifact_path(scene_id=context.scene_id, output_dir=output_dir)
    diagnostics_bytes = (
        dumps_json(
            diagnostics_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(diagnostics_path, diagnostics_bytes)
    return diagnostics_path


def run_scene_reconstruction(
    *,
    context: SceneContext,
    output_dir: Path,
    iterations: int = 1000,
    tier: str = "apex_research",
) -> Path:
    """Run reconstruction for a single scene and persist deterministic report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import keeps heavy ML dependencies out of default code path.
    from transformation_portal.spatial_ai.reconstruction.scene_builder import SceneBuilder

    builder = SceneBuilder(tier=tier)
    reconstructed_scene = builder.build_from_images(
        image_paths=list(context.images),
        cameras=[camera.params for camera in context.cameras],
        iterations=iterations,
        gamma=1.0,
    )
    scale_metadata = _normalize_scene_scale(reconstructed_scene)
    diagnostics_path = _write_scene_diagnostics(
        scene=reconstructed_scene,
        context=context,
        output_dir=output_dir,
        scale_metadata=scale_metadata,
    )

    camera_sources = [camera.provenance.source for camera in context.cameras]
    camera_confidences = [camera.provenance.confidence for camera in context.cameras]
    camera_provenance_files = sorted(
        {camera.provenance.file for camera in context.cameras if isinstance(camera.provenance.file, str)}
    )

    payload = {
        "schema": "tp.reconstruction_report.v1",
        "scene_id": context.scene_id,
        "num_views": len(context.images),
        "images": [normalize_relative_path(path, context.dataset_root) for path in context.images],
        "camera_sources": camera_sources,
        "camera_confidences": camera_confidences,
        "rmse": float(reconstructed_scene.rmse),
        "iteration": int(reconstructed_scene.iteration),
        "convergence": reconstructed_scene.convergence,
        "num_gaussians": int(reconstructed_scene.splats.num_gaussians),
        "scene_scale": scale_metadata,
        "diagnostics_path": str(diagnostics_path),
    }
    if camera_provenance_files:
        payload["camera_provenance_files"] = camera_provenance_files

    safe_scene_id = sanitize_path_component_nonlossy(context.scene_id)
    report_path = output_dir / f"{safe_scene_id}_reconstruction_report.json"
    report_bytes = (
        dumps_json(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(report_path, report_bytes)
    logger.info("Reconstruction report written: %s", report_path)
    return report_path
