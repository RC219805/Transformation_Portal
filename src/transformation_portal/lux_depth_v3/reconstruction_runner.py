"""Scene-level reconstruction runner for lux_depth_v3."""

from __future__ import annotations

import logging
import shutil
import zipfile
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
from .scene_context import CameraWithProvenance, SceneContext
from .scene_groups import lexical_relative_path
from .security import sanitize_path_component_nonlossy

logger = logging.getLogger(__name__)
RECONSTRUCTION_DIAGNOSTICS_SCHEMA = "tp.reconstruction_diagnostics.v1"


def _camera_center_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """Return camera center C from world-to-camera extrinsics [R|t]."""
    rotation = np.asarray(extrinsics[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsics[:3, 3], dtype=np.float32)
    return (-rotation.T @ translation).astype(np.float32)


def _median_camera_baseline(scene: Scene3D) -> float:
    """Compute median pairwise camera-center distance."""
    centers = [_camera_center_from_extrinsics(camera.extrinsics) for camera in scene.cameras]
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
    return output_dir / f"{safe_scene_id}_reconstruction_diagnostics.json"


def manifest_artifact_path(*, scene_id: str, output_dir: Path) -> Path:
    """Compute deterministic reconstruction manifest artifact path for a scene."""
    return reconstruction_manifest_path(scene_id=scene_id, output_dir=output_dir)


def reprojection_percentiles(errors: np.ndarray) -> Dict[str, float | None]:
    """Compute deterministic reprojection percentile summary."""
    if errors.size == 0:
        return {"p50": None, "p95": None, "p99": None}
    return {
        "p50": float(np.percentile(errors, 50)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
    }


def write_reconstruction_diagnostics(
    *,
    scene: Scene3D,
    manifest: ReconstructionManifest,
    output_dir: Path,
    scene_fingerprint: str | None,
) -> Path:
    """Write deterministic reconstruction diagnostics contract artifact."""
    total_points = int(scene.splats.num_gaussians)

    camera_payloads = []
    points_3d = np.asarray(scene.splats.positions, dtype=np.float32)
    for index, camera in enumerate(scene.cameras):
        rotation = np.asarray(camera.extrinsics[:3, :3], dtype=np.float32)
        translation = np.asarray(camera.extrinsics[:3, 3], dtype=np.float32)
        intrinsics = np.asarray(camera.intrinsics, dtype=np.float32)
        camera_space = (rotation @ points_3d.T).T + translation.reshape(1, 3)
        valid_depth = camera_space[:, 2] > 1e-6

        if np.any(valid_depth):
            projected_x = (intrinsics[0, 0] * camera_space[valid_depth, 0] / camera_space[valid_depth, 2]) + intrinsics[0, 2]
            projected_y = (intrinsics[1, 1] * camera_space[valid_depth, 1] / camera_space[valid_depth, 2]) + intrinsics[1, 2]
            in_bounds = (
                (projected_x >= 0.0)
                & (projected_x < float(camera.width))
                & (projected_y >= 0.0)
                & (projected_y < float(camera.height))
            )
            points_observed = int(np.count_nonzero(in_bounds))
        else:
            points_observed = 0

        camera_rmse = float(scene.rmse)
        reprojection_errors = np.full(points_observed, camera_rmse, dtype=np.float32)
        percentiles = reprojection_percentiles(reprojection_errors)
        camera_payloads.append(
            {
                "camera_id": camera.camera_id or Path(manifest.images[index]).name,
                "points_observed": points_observed,
                "reprojection_rmse": camera_rmse if points_observed > 0 else None,
                "reprojection_max": float(np.max(reprojection_errors)) if points_observed > 0 else None,
                "reprojection_p50": percentiles["p50"],
                "reprojection_p95": percentiles["p95"],
                "reprojection_p99": percentiles["p99"],
            }
        )

    diagnostics_payload: Dict[str, object] = {
        "schema": RECONSTRUCTION_DIAGNOSTICS_SCHEMA,
        "scene_id": manifest.scene_id,
        "scene_fingerprint": scene_fingerprint,
        "camera_count": len(scene.cameras),
        "total_points": total_points,
        "global_rmse": float(scene.rmse),
        "cameras": camera_payloads,
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
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        # File unreadable, invalid format, missing keys, or corrupted ZIP
        return None
    if not mask_arrays:
        return None
    union_mask = np.zeros(mask_arrays[0].shape, dtype=bool)
    for mask in mask_arrays:
        if mask.shape != union_mask.shape:
            return None
        union_mask |= mask > 0.0
    return union_mask


def _build_scene_builder(*, tier: str) -> Any:
    """Lazy-load SceneBuilder to keep heavy ML deps out of module import."""
    from transformation_portal.spatial_ai.reconstruction.scene_builder import SceneBuilder

    return SceneBuilder(tier=tier)


def _camera_views_align(
    execution_cameras: Sequence[CameraWithProvenance],
    manifest_cameras: Sequence[CameraWithProvenance],
) -> bool:
    """Return whether execution and durable camera authority are identical."""

    if len(execution_cameras) != len(manifest_cameras):
        return False
    for execution_camera, manifest_camera in zip(execution_cameras, manifest_cameras):
        execution_params = execution_camera.params
        manifest_params = manifest_camera.params
        execution_distortion = execution_params.distortion
        manifest_distortion = manifest_params.distortion
        if (
            execution_camera.provenance != manifest_camera.provenance
            or execution_params.width != manifest_params.width
            or execution_params.height != manifest_params.height
            or execution_params.camera_id != manifest_params.camera_id
            or not np.array_equal(execution_params.intrinsics, manifest_params.intrinsics)
            or not np.array_equal(execution_params.extrinsics, manifest_params.extrinsics)
            or (execution_distortion is None) != (manifest_distortion is None)
        ):
            return False
        if execution_distortion is not None:
            if manifest_distortion is None or not np.array_equal(execution_distortion, manifest_distortion):
                return False
    return True


def run_scene_reconstruction(
    *,
    context: SceneContext,
    output_dir: Path,
    iterations: int = 1000,
    tier: str = "apex_research",
    scene_fingerprint: str | None = None,
    run_card_merkle_root: str | None = None,
    manifest_context: SceneContext | None = None,
    image_sha256_overrides: Sequence[str] | None = None,
) -> Path:
    """Run reconstruction for a single scene and persist deterministic report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if (manifest_context is None) != (image_sha256_overrides is None):
        raise ValueError("manifest_context and image_sha256_overrides must be provided together for prepared reconstruction")

    durable_context = manifest_context or context
    if durable_context.scene_id != context.scene_id:
        raise ValueError("Reconstruction execution and manifest contexts must have the same scene_id")
    if manifest_context is not None:
        try:
            execution_images = tuple(lexical_relative_path(path, context.dataset_root) for path in context.images)
            durable_images = tuple(
                lexical_relative_path(path, durable_context.dataset_root) for path in durable_context.images
            )
        except ValueError as exc:
            raise ValueError("Reconstruction execution and manifest image identities must align") from exc
        if execution_images != durable_images:
            raise ValueError("Reconstruction execution and manifest image identities must align")
    if not _camera_views_align(context.cameras, durable_context.cameras):
        raise ValueError("Reconstruction execution and manifest cameras must align")

    manifest = build_reconstruction_manifest(
        context=durable_context,
        iterations=int(iterations),
        tier=str(tier),
        image_sha256_overrides=image_sha256_overrides,
        paths_are_canonical=manifest_context is not None,
    )
    manifest_path = write_reconstruction_manifest(manifest=manifest, output_dir=output_dir)

    if manifest_context is None:
        # Preserve the legacy round-trip verification contract. Prepared runs
        # cannot reload the durable originals after snapshot authority is
        # established, so they execute from the already-validated live view.
        execution_manifest = load_reconstruction_manifest(manifest_path=manifest_path)
        execution_image_paths = list(manifest_image_paths(execution_manifest))
        execution_cameras = [camera.params for camera in execution_manifest.cameras]
    else:
        execution_manifest = manifest
        execution_image_paths = list(context.images)
        execution_cameras = [camera.params for camera in context.cameras]

    iterations_value = int(execution_manifest.reconstruction_parameters.get("iterations", iterations))
    tier_value = str(execution_manifest.reconstruction_parameters.get("tier", tier))
    builder = _build_scene_builder(tier=tier_value)
    reconstructed_scene = builder.build_from_images(
        image_paths=execution_image_paths,
        cameras=execution_cameras,
        iterations=iterations_value,
        gamma=1.0,
    )
    scale_metadata = _normalize_scene_scale(reconstructed_scene)
    diagnostics_path = write_reconstruction_diagnostics(
        scene=reconstructed_scene,
        manifest=execution_manifest,
        output_dir=output_dir,
        scene_fingerprint=scene_fingerprint,
    )

    camera_sources = [camera.provenance.source for camera in execution_manifest.cameras]
    camera_confidences = [camera.provenance.confidence for camera in execution_manifest.cameras]
    camera_provenance_files = sorted(
        {camera.provenance.file for camera in execution_manifest.cameras if isinstance(camera.provenance.file, str)}
    )

    payload = {
        "schema": "tp.reconstruction_report.v1",
        "scene_id": execution_manifest.scene_id,
        "num_views": len(execution_manifest.images),
        "images": list(execution_manifest.images),
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

    safe_scene_id = sanitize_path_component_nonlossy(execution_manifest.scene_id)
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
