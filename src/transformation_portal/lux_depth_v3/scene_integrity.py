"""Scene manifest, integrity verification, and deterministic fingerprint helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams

from ..ingest.canonical_json import dumps_json
from .io_atomic import atomic_write_bytes
from .manifest import compute_file_sha256
from .scene_context import CameraWithProvenance, SceneContext
from .scene_groups import lexical_relative_path, normalize_relative_path
from .security import sanitize_path_component_nonlossy

SCENE_MANIFEST_SCHEMA = "tp.scene_manifest.v1"
SCENE_FINGERPRINT_SCHEMA = "tp.scene_fingerprint.v1"
MAX_FOCAL_LENGTH = 10_000.0
MAX_BASELINE = 1_000.0
MIN_BASELINE = 1e-4
DEFAULT_MAX_FORWARD_ANGLE_DEG = 60.0
DEFAULT_MIN_PAIR_FRACTION = 0.3
DEFAULT_WEAK_OVERLAP_THRESHOLD = 0.25


def validate_image_sha256_overrides(
    overrides: Optional[Sequence[str]],
    *,
    expected_count: int,
) -> Optional[tuple[str, ...]]:
    """Validate an aligned set of already-captured canonical SHA-256 digests."""

    if overrides is None:
        return None
    if isinstance(overrides, (str, bytes, bytearray)):
        raise ValueError("image_sha256_overrides must be a sequence of SHA-256 digests")
    try:
        values = tuple(overrides)
    except TypeError as exc:
        raise ValueError("image_sha256_overrides must be a sequence of SHA-256 digests") from exc
    if len(values) != expected_count:
        raise ValueError(
            "image_sha256_overrides must align with context images: " f"expected {expected_count}, got {len(values)}"
        )
    for digest in values:
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("image_sha256_overrides must contain lowercase 64-character hexadecimal digests")
    return values


def _camera_center_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """Return camera center C from world-to-camera extrinsics [R|t]."""
    rotation = np.asarray(extrinsics[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsics[:3, 3], dtype=np.float32)
    return (-rotation.T @ translation).astype(np.float32)


def scene_manifest_artifact_path(*, scene_id: str, output_dir: Path) -> Path:
    """Compute deterministic scene-manifest artifact path."""
    safe_scene_id = sanitize_path_component_nonlossy(scene_id)
    return output_dir / f"{safe_scene_id}_scene_manifest.json"


def build_scene_manifest(
    *,
    context: SceneContext,
    output_root: Path,
    segmentation_artifact_paths: Sequence[Path],
    camera_sidecar_path: Optional[Path] = None,
    camera_sidecar_sha256: Optional[str] = None,
    image_sha256_overrides: Optional[Sequence[str]] = None,
    paths_are_canonical: bool = False,
) -> Dict[str, Any]:
    """Build deterministic, hash-anchored scene manifest for reconstruction gating."""
    output_root_resolved = output_root.resolve()
    validated_image_hashes = validate_image_sha256_overrides(
        image_sha256_overrides,
        expected_count=len(context.images),
    )
    if paths_are_canonical and validated_image_hashes is None:
        raise ValueError("paths_are_canonical requires image_sha256_overrides")

    images_payload = []
    for index, image_path in enumerate(context.images):
        if paths_are_canonical:
            image_relative_path = lexical_relative_path(image_path, context.dataset_root)
            serialized_path = Path(image_path)
        else:
            serialized_path = image_path.resolve()
            image_relative_path = normalize_relative_path(serialized_path, context.dataset_root)
        images_payload.append(
            {
                "path": str(serialized_path),
                "relative_path": image_relative_path if paths_are_canonical else image_relative_path.lower(),
                "sha256": (
                    validated_image_hashes[index]
                    if validated_image_hashes is not None
                    else compute_file_sha256(serialized_path)
                ),
            }
        )

    cameras_payload = [_camera_payload(camera) for camera in context.cameras]

    segmentation_payload = []
    input_hashes: Dict[str, str] = {}
    for artifact_path in sorted({Path(path) for path in segmentation_artifact_paths}, key=lambda value: str(value)):
        resolved = artifact_path.resolve()
        digest = compute_file_sha256(resolved)
        entry: Dict[str, Any] = {
            "path": str(resolved),
            "sha256": digest,
        }
        relative_path = _relative_to_output_root(resolved, output_root_resolved)
        if relative_path:
            entry["relative_path"] = relative_path
            input_hashes[relative_path] = digest
        segmentation_payload.append(entry)

    camera_sidecar_payload: Optional[Dict[str, str]] = None
    if camera_sidecar_sha256 is not None:
        normalized_sidecar_sha256 = camera_sidecar_sha256.strip().lower()
        if camera_sidecar_path is None:
            raise ValueError("camera_sidecar_sha256 requires camera_sidecar_path")
        if len(normalized_sidecar_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_sidecar_sha256
        ):
            raise ValueError("camera_sidecar_sha256 must be a SHA-256 hex digest")
        camera_sidecar_payload = {
            "path": str(camera_sidecar_path.absolute()),
            "sha256": normalized_sidecar_sha256,
        }
    elif camera_sidecar_path is not None and camera_sidecar_path.exists():
        resolved_sidecar = camera_sidecar_path.resolve()
        camera_sidecar_payload = {
            "path": str(resolved_sidecar),
            "sha256": compute_file_sha256(resolved_sidecar),
        }

    manifest: Dict[str, Any] = {
        "schema": SCENE_MANIFEST_SCHEMA,
        "scene_id": context.scene_id,
        "grouping_mode": context.metadata.get("grouping_mode"),
        "images": images_payload,
        "cameras": cameras_payload,
        "segmentation_artifacts": segmentation_payload,
        "inputs": sorted(input_hashes),
        "input_hashes": input_hashes,
    }
    if camera_sidecar_payload:
        manifest["camera_sidecar"] = camera_sidecar_payload
    camera_normalization = context.metadata.get("camera_normalization")
    if isinstance(camera_normalization, Mapping):
        manifest["camera_normalization"] = dict(camera_normalization)
    dataset_health = context.metadata.get("dataset_health")
    if isinstance(dataset_health, Mapping):
        dataset_health_payload = dict(dataset_health)
        manifest["dataset_health"] = dataset_health_payload
        canonical_health = dumps_json(
            dataset_health_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        manifest["dataset_health_hash"] = hashlib.sha256(canonical_health).hexdigest()
    return manifest


def write_scene_manifest(*, scene_manifest: Mapping[str, Any], output_dir: Path) -> Path:
    """Persist deterministic scene-manifest JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_id = str(scene_manifest.get("scene_id", "scene"))
    path = scene_manifest_artifact_path(scene_id=scene_id, output_dir=output_dir)
    data = (
        dumps_json(
            dict(scene_manifest),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, data)
    return path


def verify_scene_integrity(
    scene_manifest: Mapping[str, Any],
    *,
    artifact_index: Optional[Mapping[str, Mapping[str, Any]]] = None,
    base_dir: Optional[Path] = None,
    image_verification_paths: Optional[Sequence[Path]] = None,
) -> None:
    """Verify scene manifest integrity before reconstruction executes."""
    images = scene_manifest.get("images", [])
    if not isinstance(images, list) or not images:
        raise RuntimeError("Scene integrity error: scene manifest must include non-empty images list")

    verification_paths: Optional[tuple[Path, ...]] = None
    if image_verification_paths is not None:
        if isinstance(image_verification_paths, (str, bytes, bytearray)):
            raise RuntimeError("Scene integrity error: image verification paths must be a sequence")
        try:
            verification_paths = tuple(Path(path) for path in image_verification_paths)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Scene integrity error: image verification paths are invalid") from exc
        if len(verification_paths) != len(images):
            raise RuntimeError("Scene integrity error: image verification paths must align with images")

    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise RuntimeError("Scene integrity error: image entries must be objects")
        image_path = (
            verification_paths[index]
            if verification_paths is not None
            else _resolve_manifest_path(image.get("path"), base_dir=base_dir)
        )
        if image_path is None or not image_path.exists():
            raise RuntimeError(f"Scene integrity error: missing image {image.get('path')}")
        expected_image_hash = image.get("sha256")
        if isinstance(expected_image_hash, str):
            actual_image_hash = compute_file_sha256(image_path)
            if actual_image_hash != expected_image_hash:
                raise RuntimeError(f"Scene integrity error: SHA256 mismatch for image {image_path}")

    cameras = scene_manifest.get("cameras", [])
    if isinstance(cameras, list) and len(cameras) != len(images):
        raise RuntimeError("Scene integrity error: camera/image count mismatch")

    segmentation_artifacts = scene_manifest.get("segmentation_artifacts", [])
    if not isinstance(segmentation_artifacts, list):
        raise RuntimeError("Scene integrity error: segmentation_artifacts must be a list")
    for segmentation in segmentation_artifacts:
        if not isinstance(segmentation, dict):
            raise RuntimeError("Scene integrity error: segmentation entries must be objects")
        segmentation_path = _resolve_manifest_path(segmentation.get("path"), base_dir=base_dir)
        if segmentation_path is None or not segmentation_path.exists():
            raise RuntimeError(f"Scene integrity error: missing segmentation artifact {segmentation.get('path')}")
        expected_segmentation_hash = segmentation.get("sha256")
        if isinstance(expected_segmentation_hash, str):
            actual_segmentation_hash = compute_file_sha256(segmentation_path)
            if actual_segmentation_hash != expected_segmentation_hash:
                raise RuntimeError(f"Scene integrity error: SHA256 mismatch for {segmentation_path}")

    if "camera_sidecar" in scene_manifest:
        sidecar = scene_manifest["camera_sidecar"]
        if not isinstance(sidecar, Mapping):
            raise RuntimeError("Scene integrity error: camera_sidecar must be an object")
        sidecar_path = _resolve_manifest_path(sidecar.get("path"), base_dir=base_dir)
        if sidecar_path is None or not sidecar_path.exists():
            raise RuntimeError(f"Scene integrity error: missing camera sidecar {sidecar.get('path')}")
        expected_sidecar_hash = sidecar.get("sha256")
        if (
            not isinstance(expected_sidecar_hash, str)
            or len(expected_sidecar_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_sidecar_hash)
        ):
            raise RuntimeError("Scene integrity error: camera sidecar SHA256 must be lowercase hexadecimal")
        actual_sidecar_hash = compute_file_sha256(sidecar_path)
        if actual_sidecar_hash != expected_sidecar_hash:
            raise RuntimeError(f"Scene integrity error: SHA256 mismatch for camera sidecar {sidecar_path}")

    if artifact_index:
        inputs = scene_manifest.get("inputs", [])
        if not isinstance(inputs, list):
            raise RuntimeError("Scene integrity error: inputs must be a list")
        input_hashes = scene_manifest.get("input_hashes", {})
        if not isinstance(input_hashes, dict):
            raise RuntimeError("Scene integrity error: input_hashes must be an object")
        for relative_path in inputs:
            if not isinstance(relative_path, str) or not relative_path:
                raise RuntimeError("Scene integrity error: invalid manifest input reference")
            index_entry = artifact_index.get(relative_path)
            if not isinstance(index_entry, Mapping):
                raise RuntimeError(f"Scene integrity error: missing artifact index entry for {relative_path}")
            expected_digest = input_hashes.get(relative_path)
            actual_digest = index_entry.get("sha256")
            if isinstance(expected_digest, str) and actual_digest != expected_digest:
                raise RuntimeError(f"Scene integrity error: artifact hash mismatch for {relative_path}")


def _frustum_overlap_score_pair(
    first: CameraWithProvenance,
    second: CameraWithProvenance,
    *,
    max_forward_angle_deg: float,
    require_mutual_facing: bool,
) -> float:
    """Fast geometric proxy that estimates whether two cameras observe the same scene."""
    rotation_a = np.asarray(first.params.extrinsics[:3, :3], dtype=np.float32)
    rotation_b = np.asarray(second.params.extrinsics[:3, :3], dtype=np.float32)

    forward_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    forward_a = rotation_a.T @ forward_axis
    forward_b = rotation_b.T @ forward_axis
    norm_a = float(np.linalg.norm(forward_a))
    norm_b = float(np.linalg.norm(forward_b))
    if norm_a <= 1e-8 or norm_b <= 1e-8:
        return 0.0
    forward_a = forward_a / norm_a
    forward_b = forward_b / norm_b

    center_a = _camera_center_from_extrinsics(first.params.extrinsics)
    center_b = _camera_center_from_extrinsics(second.params.extrinsics)
    baseline = center_b - center_a
    baseline_norm = float(np.linalg.norm(baseline))
    if baseline_norm <= 1e-6:
        return 0.0
    baseline_dir = baseline / baseline_norm

    cosine_threshold = float(np.cos(np.deg2rad(max_forward_angle_deg)))
    alignment_cosine = float(forward_a @ forward_b)
    if alignment_cosine < cosine_threshold:
        return 0.0
    alignment_score = float((alignment_cosine - cosine_threshold) / max(1.0 - cosine_threshold, 1e-6))
    alignment_score = float(np.clip(alignment_score, 0.0, 1.0))

    if not require_mutual_facing:
        return alignment_score

    facing_a = max(0.0, float(forward_a @ baseline_dir))
    facing_b = max(0.0, float(forward_b @ (-baseline_dir)))
    facing_score = float(min(facing_a, facing_b))
    return float(np.clip(alignment_score * facing_score, 0.0, 1.0))


def _camera_graph_connected(*, num_cameras: int, overlap_matrix: Mapping[tuple[int, int], float]) -> bool:
    """Require a connected overlap graph to avoid disjoint multi-scene reconstructions."""
    if num_cameras <= 1:
        return True

    graph: Dict[int, set[int]] = {index: set() for index in range(num_cameras)}
    for (left, right), overlap_score in overlap_matrix.items():
        if overlap_score > 0.0:
            graph[left].add(right)
            graph[right].add(left)

    visited: set[int] = set()
    queue = [0]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        queue.extend(neighbor for neighbor in graph[node] if neighbor not in visited)
    return len(visited) == num_cameras


def summarize_dataset_health(
    *,
    num_cameras: int,
    overlap_matrix: Mapping[tuple[int, int], float],
    weak_threshold: float = DEFAULT_WEAK_OVERLAP_THRESHOLD,
) -> Dict[str, float | int]:
    """Summarize overlap-graph quality into compact deterministic diagnostics."""
    adjacency: Dict[int, set[int]] = {index: set() for index in range(num_cameras)}
    overlaps: list[float] = []
    weak_edges = 0
    for (left, right), score in overlap_matrix.items():
        if score <= 0.0:
            continue
        overlaps.append(float(score))
        adjacency[left].add(right)
        adjacency[right].add(left)
        if score < weak_threshold:
            weak_edges += 1

    visited: set[int] = set()
    largest_component = 0
    components = 0
    for start in range(num_cameras):
        if start in visited:
            continue
        components += 1
        queue = [start]
        component: set[int] = set()
        while queue:
            node = queue.pop(0)
            if node in component:
                continue
            component.add(node)
            queue.extend(neighbor for neighbor in adjacency[node] if neighbor not in component)
        visited |= component
        largest_component = max(largest_component, len(component))

    avg_overlap = float(np.mean(np.asarray(overlaps, dtype=np.float32))) if overlaps else 0.0
    connectivity = float(largest_component / max(1, num_cameras))
    risk_score = float(1.0 - (0.6 * connectivity + 0.4 * avg_overlap))
    risk_score = float(np.clip(risk_score, 0.0, 1.0))
    return {
        "camera_count": int(num_cameras),
        "largest_component": int(largest_component),
        "num_components": int(components),
        "weak_edges": int(weak_edges),
        "avg_overlap": avg_overlap,
        "average_overlap": avg_overlap,
        "risk_score": risk_score,
    }


def build_dataset_triage_report(scene_id: str, health: Mapping[str, Any]) -> str:
    """Build actionable dataset-quality guidance when risk gate rejects reconstruction."""
    issues: list[str] = []

    camera_count = int(health.get("camera_count", 0))
    largest_component = int(health.get("largest_component", camera_count))
    average_overlap = float(health.get("average_overlap", health.get("avg_overlap", 0.0)))
    weak_edges = int(health.get("weak_edges", 0))

    if camera_count > 0 and largest_component < camera_count:
        issues.append(
            f"Camera graph disconnected ({largest_component}/{camera_count} connected). " "Capture additional bridging images."
        )
    if average_overlap < 0.15:
        issues.append(
            f"Low average overlap ({average_overlap:.2f}). " "Increase view overlap to ~60-80% between adjacent shots."
        )
    if camera_count > 0 and weak_edges > camera_count // 2:
        issues.append(
            f"Many weak connections ({weak_edges} weak edges). " "Images may be blurred, poorly exposed, or lack texture."
        )
    if not issues:
        issues.append("Dataset passed basic connectivity checks but still appears unstable.")

    return f"Scene {scene_id} dataset triage:\n - " + "\n - ".join(issues)


def check_camera_geometry_sanity(
    cameras: Sequence[CameraWithProvenance],
    *,
    min_pair_fraction: float = DEFAULT_MIN_PAIR_FRACTION,
    max_forward_angle_deg: float = DEFAULT_MAX_FORWARD_ANGLE_DEG,
    require_mutual_facing: bool = False,
) -> Dict[str, float | int]:
    """Detect degenerate/exploded camera geometry before reconstruction."""
    if len(cameras) < 2:
        raise ValueError("Reconstruction requires >=2 cameras")

    centers: list[np.ndarray] = []
    for camera in cameras:
        intrinsics = np.asarray(camera.params.intrinsics, dtype=np.float32)
        rotation = np.asarray(camera.params.extrinsics[:3, :3], dtype=np.float32)
        translation = np.asarray(camera.params.extrinsics[:3, 3], dtype=np.float32)

        if not np.allclose(rotation @ rotation.T, np.eye(3, dtype=np.float32), atol=1e-2):
            raise ValueError("Invalid rotation matrix")
        if not np.isfinite(translation).all():
            raise ValueError("Camera translation contains NaN/Inf")
        focal_length_x = float(intrinsics[0, 0])
        if focal_length_x <= 0.0 or focal_length_x > MAX_FOCAL_LENGTH:
            raise ValueError("Suspicious focal length")
        centers.append(_camera_center_from_extrinsics(camera.params.extrinsics))

    stacked_centers = np.stack(centers).astype(np.float32)
    spread = float(np.linalg.norm(stacked_centers.max(axis=0) - stacked_centers.min(axis=0)))
    if spread < MIN_BASELINE:
        raise ValueError("All cameras occupy same position")

    baselines = [
        float(np.linalg.norm(stacked_centers[i] - stacked_centers[j]))
        for i in range(len(stacked_centers))
        for j in range(i + 1, len(stacked_centers))
    ]
    baseline_array = np.asarray(baselines, dtype=np.float32)
    if float(np.median(baseline_array)) < MIN_BASELINE:
        raise ValueError("Degenerate camera baseline")
    if float(np.max(baseline_array)) > MAX_BASELINE:
        raise ValueError("Camera translation explosion detected")

    overlap_matrix: Dict[tuple[int, int], float] = {}
    valid_pairs = 0
    overlap_pairs = 0
    for left in range(len(cameras)):
        for right in range(left + 1, len(cameras)):
            overlap_score = _frustum_overlap_score_pair(
                cameras[left],
                cameras[right],
                max_forward_angle_deg=max_forward_angle_deg,
                require_mutual_facing=require_mutual_facing,
            )
            overlap_matrix[(left, right)] = overlap_score
            valid_pairs += 1
            if overlap_score > 0.0:
                overlap_pairs += 1

    if valid_pairs == 0:
        raise ValueError("No valid camera pairs available for overlap check")
    overlap_fraction = float(overlap_pairs / valid_pairs)
    if overlap_fraction < float(min_pair_fraction):
        raise ValueError(
            "Insufficient frustum overlap for reconstruction "
            f"(score={overlap_fraction:.3f}, threshold={float(min_pair_fraction):.3f})"
        )
    if not _camera_graph_connected(num_cameras=len(cameras), overlap_matrix=overlap_matrix):
        raise ValueError("Camera overlap graph disconnected")
    health = summarize_dataset_health(num_cameras=len(cameras), overlap_matrix=overlap_matrix)
    health["pair_overlap_fraction"] = overlap_fraction
    return health


def normalize_camera_poses(
    cameras: Sequence[CameraWithProvenance],
) -> tuple[tuple[CameraWithProvenance, ...], Dict[str, Any]]:
    """Center and scale camera translations so median baseline is approximately 1.0."""
    if len(cameras) < 2:
        raise ValueError("Degenerate camera baseline during normalization")

    centers = np.stack([_camera_center_from_extrinsics(camera.params.extrinsics) for camera in cameras]).astype(np.float32)
    center = centers.mean(axis=0).astype(np.float32)
    centered = centers - center
    baselines = [
        float(np.linalg.norm(centered[i] - centered[j])) for i in range(len(centered)) for j in range(i + 1, len(centered))
    ]
    median_baseline = float(np.median(np.asarray(baselines, dtype=np.float32)))
    if not np.isfinite(median_baseline) or median_baseline <= 1e-6:
        raise ValueError("Degenerate camera baseline during normalization")

    scale = float(1.0 / median_baseline)
    normalized_centers = (centered * scale).astype(np.float32)

    normalized_cameras: list[CameraWithProvenance] = []
    for camera, camera_center in zip(cameras, normalized_centers):
        extrinsics = np.asarray(camera.params.extrinsics, dtype=np.float32).copy()
        rotation = np.asarray(extrinsics[:3, :3], dtype=np.float32)
        extrinsics[:3, 3] = (-rotation @ camera_center).astype(np.float32)
        distortion = (
            np.asarray(camera.params.distortion, dtype=np.float32).copy() if camera.params.distortion is not None else None
        )
        normalized_cameras.append(
            CameraWithProvenance(
                params=CameraParams(
                    intrinsics=np.asarray(camera.params.intrinsics, dtype=np.float32).copy(),
                    extrinsics=extrinsics,
                    width=int(camera.params.width),
                    height=int(camera.params.height),
                    distortion=distortion,
                    camera_id=camera.params.camera_id,
                ),
                provenance=camera.provenance,
            )
        )

    return tuple(normalized_cameras), {
        "center": [float(value) for value in center.tolist()],
        "scale": scale,
        "median_baseline": median_baseline,
        "method": "centered_median_baseline",
    }


def compute_scene_fingerprint(
    *,
    scene_manifest: Mapping[str, Any],
    artifact_index: Mapping[str, Mapping[str, Any]],
    reconstruction_config: Mapping[str, Any],
) -> str:
    """Compute deterministic scene fingerprint from canonical digest fields.

    Fingerprints intentionally avoid hashing the full scene manifest structure so
    unrelated manifest field additions/formatting changes do not perturb cache
    keys. Only identity-bearing, content-addressed inputs are included.
    """
    indexed_artifacts: Dict[str, str] = {}
    for relative_path in sorted(scene_manifest.get("inputs", [])):
        index_entry = artifact_index.get(relative_path)
        if not isinstance(index_entry, Mapping):
            raise RuntimeError(f"Scene fingerprint error: missing artifact index entry for {relative_path}")
        digest = index_entry.get("sha256")
        if not isinstance(digest, str):
            raise RuntimeError(f"Scene fingerprint error: missing sha256 for {relative_path}")
        indexed_artifacts[relative_path] = digest

    image_hashes = [
        {
            "relative_path": image.get("relative_path"),
            "sha256": image.get("sha256"),
        }
        for image in scene_manifest.get("images", [])
        if isinstance(image, Mapping)
    ]
    camera_signatures = [
        camera.get("signature") for camera in scene_manifest.get("cameras", []) if isinstance(camera, Mapping)
    ]
    segmentation_sha256 = {
        str(entry.get("relative_path") or entry.get("path")): str(entry.get("sha256"))
        for entry in scene_manifest.get("segmentation_artifacts", [])
        if isinstance(entry, Mapping) and (entry.get("relative_path") or entry.get("path")) and entry.get("sha256")
    }

    payload = {
        "schema": SCENE_FINGERPRINT_SCHEMA,
        "scene_id": scene_manifest.get("scene_id"),
        "images": image_hashes,
        "segmentation_sha256": segmentation_sha256,
        "camera_sha256": camera_signatures,
        "dataset_health_hash": scene_manifest.get("dataset_health_hash"),
        "artifact_hashes": indexed_artifacts,
        "reconstruction_config": dict(reconstruction_config),
    }
    canonical = dumps_json(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _camera_payload(camera: CameraWithProvenance) -> Dict[str, Any]:
    payload = {
        "intrinsics": camera.params.intrinsics.tolist(),
        "extrinsics": camera.params.extrinsics.tolist(),
        "width": int(camera.params.width),
        "height": int(camera.params.height),
        "distortion": camera.params.distortion.tolist() if camera.params.distortion is not None else None,
        "camera_id": camera.params.camera_id,
        "source": camera.provenance.source,
        "confidence": camera.provenance.confidence,
        "file": camera.provenance.file,
    }
    canonical = dumps_json(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "signature": hashlib.sha1(canonical, usedforsecurity=False).hexdigest()[:12],
        "source": camera.provenance.source,
        "confidence": camera.provenance.confidence,
        "file": camera.provenance.file,
    }


def _relative_to_output_root(path: Path, output_root_resolved: Path) -> Optional[str]:
    try:
        return path.resolve().relative_to(output_root_resolved).as_posix()
    except ValueError:
        return None


def _resolve_manifest_path(path_value: Any, *, base_dir: Optional[Path]) -> Optional[Path]:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path
