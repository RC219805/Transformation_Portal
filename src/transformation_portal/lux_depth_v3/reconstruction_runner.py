"""Scene-level reconstruction runner for lux_depth_v3."""

from __future__ import annotations

import logging
import shutil
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import cv2 as _cv2
import numpy as np

from transformation_portal.spatial_ai.reconstruction.contracts import Scene3D

from ..ingest.canonical_json import dumps_json
from .io_atomic import atomic_write_bytes
from .reconstruction_manifest import (
    ReconstructionManifest,
    build_reconstruction_manifest,
    load_reconstruction_manifest,
    manifest_image_paths,
    reconstruction_manifest_path,
    write_reconstruction_manifest,
)
from .scene_context import SceneContext
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


def manifest_artifact_path(*, scene_id: str, output_dir: Path) -> Path:
    """Compute deterministic reconstruction manifest artifact path for a scene."""
    return reconstruction_manifest_path(scene_id=scene_id, output_dir=output_dir)


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
    manifest: ReconstructionManifest,
    output_dir: Path,
    scale_metadata: Dict[str, float | str],
) -> Path:
    """Write deterministic scene diagnostics artifact."""
    baselines = [float(np.linalg.norm(a.extrinsics[:3, 3] - b.extrinsics[:3, 3])) for a, b in combinations(scene.cameras, 2)]
    baseline_array = np.asarray(baselines, dtype=np.float32)
    diagnostics_payload: Dict[str, object] = {
        "schema": "tp.scene_diagnostics.v1",
        "scene_id": manifest.scene_id,
        "inputs": {
            "image_count": len(manifest.images),
            "grouping_mode": manifest.reconstruction_parameters.get("grouping_mode"),
        },
        "cameras": {
            "count": len(scene.cameras),
            "baseline_median": float(np.median(baseline_array)),
            "baseline_min": float(np.min(baseline_array)),
            "baseline_max": float(np.max(baseline_array)),
            "sources": [camera.provenance.source for camera in manifest.cameras],
            "confidences": [camera.provenance.confidence for camera in manifest.cameras],
        },
        "geometry": _scene_geometry_stats(scene),
        "scale": scale_metadata,
        "quality": {
            "rmse": float(scene.rmse),
            "iterations": int(scene.iteration),
            "convergence": scene.convergence,
        },
    }
    diagnostics_path = diagnostics_artifact_path(scene_id=manifest.scene_id, output_dir=output_dir)
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


def write_scene_debug_bundle(
    *,
    context: SceneContext,
    segmentation_artifact_paths: Sequence[Path],
    scene_manifest: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    """Write scene debug bundle assets for visual inspection and offline triage.

    Bundle layout:
    - debug/scene_manifest.json
    - debug/cameras.json
    - debug/inputs/<image_name>
    - debug/segmentation_overlay/<image_stem>_overlay.png (best-effort)
    - debug/reprojection_preview.png (best-effort contact sheet)
    """
    debug_dir = output_dir / "debug"
    inputs_dir = debug_dir / "inputs"
    overlays_dir = debug_dir / "segmentation_overlay"
    debug_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    scene_manifest_path = debug_dir / "scene_manifest.json"
    scene_manifest_bytes = (
        dumps_json(
            dict(scene_manifest),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(scene_manifest_path, scene_manifest_bytes)

    cameras_payload = [
        {
            "intrinsics": camera.params.intrinsics.tolist(),
            "extrinsics": camera.params.extrinsics.tolist(),
            "width": int(camera.params.width),
            "height": int(camera.params.height),
            "distortion": camera.params.distortion.tolist() if camera.params.distortion is not None else None,
            "camera_id": camera.params.camera_id,
            "provenance": {
                "source": camera.provenance.source,
                "confidence": camera.provenance.confidence,
                "file": camera.provenance.file,
            },
        }
        for camera in context.cameras
    ]
    cameras_path = debug_dir / "cameras.json"
    cameras_bytes = (
        dumps_json(
            cameras_payload,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(cameras_path, cameras_bytes)

    copied_input_paths: list[Path] = []
    for image_path in context.images:
        if not image_path.exists():
            continue
        destination = inputs_dir / image_path.name
        shutil.copy2(image_path, destination)
        copied_input_paths.append(destination)

    reprojection_preview_path = debug_dir / "reprojection_preview.png"
    for image_path, mask_artifact_path in zip(context.images, segmentation_artifact_paths):
        if not image_path.exists() or not mask_artifact_path.exists():
            continue
        image = _cv2.imread(str(image_path), _cv2.IMREAD_COLOR)
        if image is None:
            continue
        union_mask = _load_union_mask(mask_artifact_path)
        if union_mask is None:
            continue
        if union_mask.shape != image.shape[:2]:
            union_mask = _cv2.resize(
                union_mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=_cv2.INTER_NEAREST,
            ).astype(bool)
        overlay = image.copy()
        overlay[union_mask] = (0, 255, 0)
        blended = _cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        _cv2.imwrite(str(overlays_dir / f"{image_path.stem}_overlay.png"), blended)

    preview_images = []
    for image_path in copied_input_paths[:4]:
        image = _cv2.imread(str(image_path), _cv2.IMREAD_COLOR)
        if image is not None:
            preview_images.append(image)
    if preview_images:
        target_height = min(image.shape[0] for image in preview_images)
        resized = [
            _cv2.resize(
                image,
                (max(1, int(round(image.shape[1] * (target_height / image.shape[0])))), target_height),
                interpolation=_cv2.INTER_AREA,
            )
            for image in preview_images
        ]
        contact_sheet = np.concatenate(resized, axis=1)
        _cv2.imwrite(str(reprojection_preview_path), contact_sheet)

    output_paths = {
        "scene_manifest_path": scene_manifest_path,
        "cameras_path": cameras_path,
    }
    if reprojection_preview_path.exists():
        output_paths["reprojection_preview_path"] = reprojection_preview_path
    return output_paths


def _load_union_mask(mask_artifact_path: Path) -> np.ndarray | None:
    """Load segmentation NPZ and return union mask as boolean array."""
    try:
        with np.load(mask_artifact_path, allow_pickle=False) as payload:
            mask_arrays = [np.asarray(payload[key], dtype=np.float32) for key in sorted(payload.files)]
    except Exception:
        return None
    if not mask_arrays:
        return None
    union_mask = np.zeros(mask_arrays[0].shape, dtype=bool)
    for mask in mask_arrays:
        if mask.shape != union_mask.shape:
            return None
        union_mask |= mask > 0.0
    return union_mask


def run_scene_reconstruction(
    *,
    context: SceneContext,
    output_dir: Path,
    iterations: int = 1000,
    tier: str = "apex_research",
    scene_fingerprint: str | None = None,
    run_card_merkle_root: str | None = None,
) -> Path:
    """Run reconstruction for a single scene and persist deterministic report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import keeps heavy ML dependencies out of default code path.
    from transformation_portal.spatial_ai.reconstruction.scene_builder import SceneBuilder

    manifest = build_reconstruction_manifest(
        context=context,
        iterations=int(iterations),
        tier=str(tier),
    )
    manifest_path = write_reconstruction_manifest(manifest=manifest, output_dir=output_dir)
    loaded_manifest = load_reconstruction_manifest(manifest_path=manifest_path)

    iterations_value = int(loaded_manifest.reconstruction_parameters.get("iterations", iterations))
    tier_value = str(loaded_manifest.reconstruction_parameters.get("tier", tier))
    builder = SceneBuilder(tier=tier_value)
    reconstructed_scene = builder.build_from_images(
        image_paths=list(manifest_image_paths(loaded_manifest)),
        cameras=[camera.params for camera in loaded_manifest.cameras],
        iterations=iterations_value,
        gamma=1.0,
    )
    scale_metadata = _normalize_scene_scale(reconstructed_scene)
    diagnostics_path = _write_scene_diagnostics(
        scene=reconstructed_scene,
        manifest=loaded_manifest,
        output_dir=output_dir,
        scale_metadata=scale_metadata,
    )

    camera_sources = [camera.provenance.source for camera in loaded_manifest.cameras]
    camera_confidences = [camera.provenance.confidence for camera in loaded_manifest.cameras]
    camera_provenance_files = sorted(
        {camera.provenance.file for camera in loaded_manifest.cameras if isinstance(camera.provenance.file, str)}
    )

    payload = {
        "schema": "tp.reconstruction_report.v1",
        "scene_id": loaded_manifest.scene_id,
        "num_views": len(loaded_manifest.images),
        "images": list(loaded_manifest.images),
        "manifest_path": str(manifest_path),
        "camera_sources": camera_sources,
        "camera_confidences": camera_confidences,
        "rmse": float(reconstructed_scene.rmse),
        "iteration": int(reconstructed_scene.iteration),
        "convergence": reconstructed_scene.convergence,
        "num_gaussians": int(reconstructed_scene.splats.num_gaussians),
        "scene_scale": scale_metadata,
        "diagnostics_path": str(diagnostics_path),
    }
    if isinstance(scene_fingerprint, str) and scene_fingerprint:
        payload["scene_fingerprint"] = scene_fingerprint
    if isinstance(run_card_merkle_root, str) and run_card_merkle_root:
        payload["run_card_merkle_root"] = run_card_merkle_root
    if camera_provenance_files:
        payload["camera_provenance_files"] = camera_provenance_files

    safe_scene_id = sanitize_path_component_nonlossy(loaded_manifest.scene_id)
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
