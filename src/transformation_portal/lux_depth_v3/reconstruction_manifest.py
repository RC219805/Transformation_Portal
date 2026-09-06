"""Deterministic reconstruction manifest generation and loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, cast

import numpy as np

from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams

from ..ingest.canonical_json import dumps_json
from .io_atomic import atomic_write_bytes
from .manifest import compute_file_sha256
from .scene_context import CameraConfidence, CameraProvenance, CameraSource, CameraWithProvenance, SceneContext
from .scene_groups import lexical_relative_path
from .scene_integrity import validate_image_sha256_overrides
from .security import sanitize_path_component_nonlossy

SCHEMA_RECONSTRUCTION_MANIFEST = "tp.reconstruction_manifest.v1"


@dataclass(frozen=True)
class ReconstructionManifest:
    """Canonical manifest consumed by reconstruction runner."""

    scene_id: str
    dataset_root: str
    images: Tuple[str, ...]
    image_hashes: Tuple[str, ...]
    cameras: Tuple[CameraWithProvenance, ...]
    reconstruction_parameters: Dict[str, Any]
    segmentation_masks: Tuple[str, ...] = field(default_factory=tuple)
    scene_scale: float = 1.0
    schema: str = SCHEMA_RECONSTRUCTION_MANIFEST

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "scene_id": self.scene_id,
            "dataset_root": self.dataset_root,
            "images": list(self.images),
            "image_hashes": list(self.image_hashes),
            "cameras": [_camera_to_payload(camera) for camera in self.cameras],
            "segmentation_masks": list(self.segmentation_masks),
            "reconstruction_parameters": dict(self.reconstruction_parameters),
            "scene_scale": float(self.scene_scale),
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ReconstructionManifest":
        if payload.get("schema") != SCHEMA_RECONSTRUCTION_MANIFEST:
            raise ValueError(
                f"Unsupported reconstruction manifest schema: {payload.get('schema')} (expected {SCHEMA_RECONSTRUCTION_MANIFEST})"
            )
        cameras_payload = payload.get("cameras")
        if not isinstance(cameras_payload, list):
            raise ValueError("Reconstruction manifest cameras must be a list")
        cameras = tuple(_camera_from_payload(entry) for entry in cameras_payload)
        images = tuple(str(value) for value in payload.get("images", []))
        image_hashes = tuple(str(value) for value in payload.get("image_hashes", []))
        if len(images) == 0:
            raise ValueError("Reconstruction manifest must include at least one image path")
        if len(images) != len(image_hashes):
            raise ValueError("Reconstruction manifest image hashes must align with images")
        if len(cameras) != len(images):
            raise ValueError("Reconstruction manifest cameras must align with images")
        return cls(
            scene_id=str(payload["scene_id"]),
            dataset_root=str(payload["dataset_root"]),
            images=images,
            image_hashes=image_hashes,
            cameras=cameras,
            reconstruction_parameters=dict(payload.get("reconstruction_parameters", {})),
            segmentation_masks=tuple(str(value) for value in payload.get("segmentation_masks", [])),
            scene_scale=float(payload.get("scene_scale", 1.0)),
        )


def reconstruction_manifest_path(*, scene_id: str, output_dir: Path) -> Path:
    """Compute deterministic reconstruction manifest path."""
    safe_scene_id = sanitize_path_component_nonlossy(scene_id)
    return output_dir / f"{safe_scene_id}_manifest.json"


def _manifest_image_path(path: Path, dataset_root: Path) -> str:
    """Serialize a loadable path without case-folding filesystem components."""

    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def build_reconstruction_manifest(
    *,
    context: SceneContext,
    iterations: int,
    tier: str,
    image_sha256_overrides: Sequence[str] | None = None,
    paths_are_canonical: bool = False,
) -> ReconstructionManifest:
    """Build deterministic manifest from scene context."""
    validated_image_hashes = validate_image_sha256_overrides(
        image_sha256_overrides,
        expected_count=len(context.images),
    )
    if paths_are_canonical and validated_image_hashes is None:
        raise ValueError("paths_are_canonical requires image_sha256_overrides")
    normalized_images = tuple(
        (
            lexical_relative_path(path, context.dataset_root)
            if paths_are_canonical
            else _manifest_image_path(path, context.dataset_root)
        )
        for path in context.images
    )
    image_hashes = tuple(
        f"sha256:{digest}"
        for digest in (
            validated_image_hashes
            if validated_image_hashes is not None
            else tuple(compute_file_sha256(path) for path in context.images)
        )
    )
    return ReconstructionManifest(
        scene_id=context.scene_id,
        dataset_root=str(context.dataset_root if paths_are_canonical else context.dataset_root.resolve()),
        images=normalized_images,
        image_hashes=image_hashes,
        cameras=tuple(context.cameras),
        reconstruction_parameters={
            "iterations": int(iterations),
            "tier": str(tier),
            "grouping_mode": context.metadata.get("grouping_mode"),
        },
        segmentation_masks=tuple(sorted(context.segmentation_masks.keys())) if context.segmentation_masks else tuple(),
        scene_scale=1.0,
    )


def write_reconstruction_manifest(*, manifest: ReconstructionManifest, output_dir: Path) -> Path:
    """Persist deterministic reconstruction manifest JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = reconstruction_manifest_path(scene_id=manifest.scene_id, output_dir=output_dir)
    data = (
        dumps_json(
            manifest.to_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, data)
    return path


def load_reconstruction_manifest(*, manifest_path: Path) -> ReconstructionManifest:
    """Load and validate reconstruction manifest from disk."""
    payload = json_loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ReconstructionManifest.from_payload(payload)
    _verify_image_hashes(manifest)
    return manifest


def manifest_image_paths(manifest: ReconstructionManifest) -> Tuple[Path, ...]:
    """Resolve manifest-relative image paths to filesystem paths."""
    root = Path(manifest.dataset_root)
    return tuple(root / image_rel for image_rel in manifest.images)


def _verify_image_hashes(manifest: ReconstructionManifest) -> None:
    image_paths = manifest_image_paths(manifest)
    for image_path, expected_hash in zip(image_paths, manifest.image_hashes):
        actual_hash = f"sha256:{compute_file_sha256(image_path)}"
        if actual_hash != expected_hash:
            raise ValueError(
                f"Reconstruction manifest image hash mismatch for {image_path}: expected {expected_hash}, got {actual_hash}"
            )


def _camera_to_payload(camera: CameraWithProvenance) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "intrinsics": camera.params.intrinsics.tolist(),
        "extrinsics": camera.params.extrinsics.tolist(),
        "width": int(camera.params.width),
        "height": int(camera.params.height),
        "provenance": {
            "source": camera.provenance.source,
            "confidence": camera.provenance.confidence,
            "file": camera.provenance.file,
        },
    }
    if camera.params.distortion is not None:
        payload["distortion"] = camera.params.distortion.tolist()
    if camera.params.camera_id is not None:
        payload["camera_id"] = camera.params.camera_id
    return payload


def _camera_from_payload(payload: Dict[str, Any]) -> CameraWithProvenance:
    distortion = payload.get("distortion")
    distortion_array = np.asarray(distortion, dtype=np.float32) if distortion is not None else None
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Reconstruction manifest camera entry missing provenance object")
    camera = CameraWithProvenance(
        params=CameraParams(
            intrinsics=np.asarray(payload["intrinsics"], dtype=np.float32),
            extrinsics=np.asarray(payload["extrinsics"], dtype=np.float32),
            width=int(payload["width"]),
            height=int(payload["height"]),
            distortion=distortion_array,
            camera_id=str(payload["camera_id"]) if payload.get("camera_id") is not None else None,
        ),
        provenance=CameraProvenance(
            source=cast(CameraSource, str(provenance["source"])),
            confidence=cast(CameraConfidence, str(provenance["confidence"])),
            file=str(provenance["file"]) if provenance.get("file") is not None else None,
        ),
    )
    return camera


def json_loads(raw: str) -> Dict[str, Any]:
    """Load JSON payload and validate top-level object type."""
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Reconstruction manifest root must be a JSON object")
    return payload
