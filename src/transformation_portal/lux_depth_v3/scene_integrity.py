"""Scene manifest, integrity verification, and deterministic fingerprint helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ..ingest.canonical_json import dumps_json
from .io_atomic import atomic_write_bytes
from .manifest import compute_file_sha256
from .scene_context import CameraWithProvenance, SceneContext
from .scene_groups import normalize_relative_path
from .security import sanitize_path_component_nonlossy

SCENE_MANIFEST_SCHEMA = "tp.scene_manifest.v1"
SCENE_FINGERPRINT_SCHEMA = "tp.scene_fingerprint.v1"


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
) -> Dict[str, Any]:
    """Build deterministic, hash-anchored scene manifest for reconstruction gating."""
    output_root_resolved = output_root.resolve()

    images_payload = []
    for image_path in context.images:
        resolved = image_path.resolve()
        images_payload.append(
            {
                "path": str(resolved),
                "relative_path": normalize_relative_path(resolved, context.dataset_root),
                "sha256": compute_file_sha256(resolved),
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
    if camera_sidecar_path is not None and camera_sidecar_path.exists():
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
) -> None:
    """Verify scene manifest integrity before reconstruction executes."""
    images = scene_manifest.get("images", [])
    if not isinstance(images, list) or not images:
        raise RuntimeError("Scene integrity error: scene manifest must include non-empty images list")

    for image in images:
        if not isinstance(image, dict):
            raise RuntimeError("Scene integrity error: image entries must be objects")
        image_path = _resolve_manifest_path(image.get("path"), base_dir=base_dir)
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

    sidecar = scene_manifest.get("camera_sidecar")
    if isinstance(sidecar, dict):
        sidecar_path = _resolve_manifest_path(sidecar.get("path"), base_dir=base_dir)
        if sidecar_path is None or not sidecar_path.exists():
            raise RuntimeError(f"Scene integrity error: missing camera sidecar {sidecar.get('path')}")
        expected_sidecar_hash = sidecar.get("sha256")
        if isinstance(expected_sidecar_hash, str):
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


def compute_scene_fingerprint(
    *,
    scene_manifest: Mapping[str, Any],
    artifact_index: Mapping[str, Mapping[str, Any]],
    reconstruction_config: Mapping[str, Any],
) -> str:
    """Compute deterministic scene fingerprint from content-addressed inputs."""
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

    payload = {
        "schema": SCENE_FINGERPRINT_SCHEMA,
        "scene_id": scene_manifest.get("scene_id"),
        "image_hashes": image_hashes,
        "camera_signatures": camera_signatures,
        "artifacts": indexed_artifacts,
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
