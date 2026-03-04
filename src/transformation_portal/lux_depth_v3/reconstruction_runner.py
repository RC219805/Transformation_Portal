"""Scene-level reconstruction runner for lux_depth_v3."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams

from .io_atomic import atomic_write_bytes
from .scene_groups import SceneGroup
from .security import sanitize_path_component_nonlossy

logger = logging.getLogger(__name__)


def _normalize_relative_path(path: Path, dataset_root: Path) -> str:
    """Normalize path to lowercase POSIX string relative to dataset root."""
    root_resolved = dataset_root.resolve()
    path_resolved = path.resolve()
    try:
        rel = path_resolved.relative_to(root_resolved)
    except ValueError:
        rel = path_resolved
    return rel.as_posix().lower()


def run_scene_reconstruction(
    *,
    scene: SceneGroup,
    cameras: Tuple[CameraParams, ...],
    dataset_root: Path,
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
        image_paths=list(scene.images),
        cameras=list(cameras),
        iterations=iterations,
        gamma=1.0,
    )

    payload = {
        "schema": "tp.reconstruction_report.v1",
        "scene_id": scene.scene_id,
        "num_views": len(scene.images),
        "images": [_normalize_relative_path(path, dataset_root) for path in scene.images],
        "rmse": float(reconstructed_scene.rmse),
        "iteration": int(reconstructed_scene.iteration),
        "convergence": reconstructed_scene.convergence,
        "num_gaussians": int(reconstructed_scene.splats.num_gaussians),
    }

    safe_scene_id = sanitize_path_component_nonlossy(scene.scene_id)
    report_path = output_dir / f"{safe_scene_id}_reconstruction_report.json"
    report_bytes = (
        json.dumps(
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
