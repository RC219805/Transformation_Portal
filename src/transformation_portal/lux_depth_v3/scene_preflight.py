"""Deterministic scene preflight validation for reconstruction eligibility."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..ingest.canonical_json import dumps_json
from .io_atomic import atomic_write_bytes
from .scene_context import CameraWithProvenance
from .scene_groups import SceneGroup
from .security import sanitize_path_component_nonlossy

MIN_BASELINE = 1e-3
MAX_BASELINE = 1e4
MAX_RESOLUTION_RATIO = 4.0
MAX_ASPECT_SPAN = 0.35
MIN_FORWARD_COSINE = -0.25


@dataclass(frozen=True)
class ScenePreflightResult:
    """Structured output from scene-level preflight checks."""

    valid: bool
    checks: Dict[str, str]
    metrics: Dict[str, Any]
    reason: Optional[str] = None
    schema: str = "tp.scene_preflight.v1"

    def to_payload(self, *, scene_id: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema": self.schema,
            "scene_id": scene_id,
            "valid": self.valid,
            "checks": dict(self.checks),
            "metrics": dict(self.metrics),
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


def _camera_centers(cameras: Tuple[CameraWithProvenance, ...]) -> list[np.ndarray]:
    centers: list[np.ndarray] = []
    for camera in cameras:
        rotation = np.asarray(camera.params.extrinsics[:3, :3], dtype=np.float32)
        translation = np.asarray(camera.params.extrinsics[:3, 3], dtype=np.float32)
        centers.append((-rotation.T @ translation).astype(np.float32))
    return centers


def _camera_forward_vectors(cameras: Tuple[CameraWithProvenance, ...]) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    for camera in cameras:
        rotation = np.asarray(camera.params.extrinsics[:3, :3], dtype=np.float32)
        forward = rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        norm = float(np.linalg.norm(forward))
        if norm <= 1e-8 or not np.isfinite(norm):
            vectors.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        else:
            vectors.append(forward / norm)
    return vectors


def _baseline_stats(cameras: Tuple[CameraWithProvenance, ...]) -> Tuple[float, float, float]:
    baselines = [float(np.linalg.norm(a - b)) for a, b in combinations(_camera_centers(cameras), 2)]
    baseline_array = np.asarray(baselines, dtype=np.float32)
    return float(np.median(baseline_array)), float(np.min(baseline_array)), float(np.max(baseline_array))


def validate_scene_preflight(
    *,
    scene: SceneGroup,
    cameras: Tuple[CameraWithProvenance, ...],
) -> ScenePreflightResult:
    """Validate whether a scene is geometrically reconstructable."""
    checks: Dict[str, str] = {
        "view_count": "fail",
        "baseline": "fail",
        "fov_overlap": "fail",
        "resolution_consistency": "fail",
        "scale_sanity": "fail",
    }
    metrics: Dict[str, Any] = {
        "image_count": len(scene.images),
        "camera_count": len(cameras),
    }

    if len(scene.images) < 2 or len(cameras) < 2:
        return ScenePreflightResult(valid=False, reason="view_count", checks=checks, metrics=metrics)
    checks["view_count"] = "pass"

    if len(cameras) != len(scene.images):
        return ScenePreflightResult(valid=False, reason="camera_count_mismatch", checks=checks, metrics=metrics)

    median_baseline, min_baseline, max_baseline = _baseline_stats(cameras)
    metrics["baseline_median"] = median_baseline
    metrics["baseline_min"] = min_baseline
    metrics["baseline_max"] = max_baseline

    if (
        not np.isfinite(median_baseline)
        or not np.isfinite(min_baseline)
        or not np.isfinite(max_baseline)
        or median_baseline < MIN_BASELINE
    ):
        return ScenePreflightResult(valid=False, reason="baseline_too_small", checks=checks, metrics=metrics)
    checks["baseline"] = "pass"

    forward_vectors = _camera_forward_vectors(cameras)
    cosine_values = [float(np.dot(a, b)) for a, b in combinations(forward_vectors, 2)]
    max_forward_cosine = max(cosine_values) if cosine_values else -1.0
    metrics["max_forward_cosine"] = max_forward_cosine
    if max_forward_cosine <= MIN_FORWARD_COSINE:
        return ScenePreflightResult(valid=False, reason="no_overlap", checks=checks, metrics=metrics)
    checks["fov_overlap"] = "pass"

    widths = [int(camera.params.width) for camera in cameras]
    heights = [int(camera.params.height) for camera in cameras]
    areas = [float(w * h) for w, h in zip(widths, heights)]
    aspects = [float(w / h) for w, h in zip(widths, heights)]
    resolution_ratio = max(areas) / max(min(areas), 1.0)
    aspect_span = max(aspects) - min(aspects)
    metrics["resolution_ratio"] = resolution_ratio
    metrics["aspect_span"] = aspect_span
    if resolution_ratio > MAX_RESOLUTION_RATIO or aspect_span > MAX_ASPECT_SPAN:
        return ScenePreflightResult(valid=False, reason="resolution_inconsistent", checks=checks, metrics=metrics)
    checks["resolution_consistency"] = "pass"

    if max_baseline > MAX_BASELINE:
        return ScenePreflightResult(valid=False, reason="scale_sanity", checks=checks, metrics=metrics)
    checks["scale_sanity"] = "pass"

    return ScenePreflightResult(valid=True, checks=checks, metrics=metrics)


def preflight_artifact_path(*, scene_id: str, output_dir: Path) -> Path:
    """Compute deterministic path for scene preflight artifact."""
    safe_scene_id = sanitize_path_component_nonlossy(scene_id)
    return output_dir / f"{safe_scene_id}_preflight.json"


def write_scene_preflight_artifact(
    *,
    scene_id: str,
    result: ScenePreflightResult,
    output_dir: Path,
) -> Path:
    """Persist scene preflight artifact as deterministic canonical JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = preflight_artifact_path(scene_id=scene_id, output_dir=output_dir)
    payload = result.to_payload(scene_id=scene_id)
    data = (
        dumps_json(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, data)
    return path
