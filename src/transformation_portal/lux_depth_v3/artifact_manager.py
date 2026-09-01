"""Artifact management for lux_depth_v3 pipeline.

Extracted from orchestrator.py as part of ADR-043 decomposition.

This module provides:
- Artifact type inference from output paths
- Artifact indexing with integrity hashes (SHA-256)
- Merkle root computation for deterministic artifact manifests
- Output key generation with directory structure preservation

The artifact system ensures:
- Deterministic, reproducible output naming
- Integrity verification through cryptographic hashes
- Audit trail via merkle roots in run cards

Usage:
    from transformation_portal.lux_depth_v3.artifact_manager import (
        ArtifactManager,
        infer_artifact_type,
        build_artifact_index,
        compute_artifact_merkle_root,
        make_output_key,
    )

    # Using ArtifactManager class
    manager = ArtifactManager(output_root=Path("/outputs"))
    index = manager.index_artifacts([path1, path2])
    merkle = manager.compute_merkle_root(index)

    # Using standalone functions
    artifact_type = infer_artifact_type("depth/image_depth.png")
    output_key = make_output_key(input_path, input_root)
"""

from __future__ import annotations

import copy
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._backend_contract import normalize_backend_provenance
from .manifest import CombinedManifest, ConfigFingerprint, compute_file_sha256
from .path_aliasing import normalize_lexical_path, relative_to_path_alias
from .security import sanitize_path_component_nonlossy

logger = logging.getLogger(__name__)

# Phase 3: xxHash support (optional dependency)
try:
    import xxhash

    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    xxhash = None  # type: ignore


def infer_artifact_type(relative_path: str) -> str:
    """Infer canonical artifact type from output-root relative path.

    Categorizes artifacts by directory prefix and file extension:
    - segmentation/: segmentation_mask_npz, segmentation_aux
    - depth/: depth_metadata, depth_u16_png, depth_float_npy, depth_aux
    - v2/: v2_report, v2_output
    - manifests/: batch_manifest, provenance_sidecar, combined_manifest
    - logs/: v2_log
    - pbr/: pbr_normal, pbr_roughness, pbr_ao, pbr_aux
    - reconstruction/: various reconstruction artifact types

    Args:
        relative_path: Path relative to output root (e.g., "depth/image.png")

    Returns:
        Canonical artifact type string
    """
    rel = relative_path.lower()
    name = Path(rel).name

    if rel.startswith("segmentation/"):
        if name.endswith(".npz"):
            return "segmentation_mask_npz"
        return "segmentation_aux"

    if rel.startswith("depth/"):
        if name.endswith("_metadata.json"):
            return "depth_metadata"
        if name.endswith(".png"):
            return "depth_u16_png"
        if name.endswith(".npy"):
            return "depth_float_npy"
        return "depth_aux"

    if rel.startswith("v2/"):
        if name.endswith("_report.json"):
            return "v2_report"
        return "v2_output"

    if rel.startswith("manifests/"):
        if name.startswith("batch_") and name.endswith(".json"):
            return "batch_manifest"
        if name.endswith("_provenance.json"):
            return "provenance_sidecar"
        if name.endswith("_combined.json"):
            return "combined_manifest"
        return "manifest_aux"

    if rel.startswith("logs/"):
        return "v2_log"

    if rel.startswith("pbr/"):
        if "normal" in name:
            return "pbr_normal"
        if "roughness" in name:
            return "pbr_roughness"
        if name.startswith("ao_") or "_ao" in name:
            return "pbr_ao"
        return "pbr_aux"

    if rel.startswith("reconstruction/"):
        if "/debug/" in rel:
            if name == "scene_manifest.json":
                return "reconstruction_debug_scene_manifest_json"
            if name == "cameras.json":
                return "reconstruction_debug_cameras_json"
            if name == "reprojection_preview.png":
                return "reconstruction_debug_preview_png"
            if name.endswith("_overlay.png"):
                return "reconstruction_debug_overlay_png"
            return "reconstruction_debug_aux"
        if name.endswith("_scene_manifest.json"):
            return "reconstruction_scene_manifest"
        if name.endswith("_manifest.json"):
            return "reconstruction_manifest_json"
        if name.endswith("_reconstruction_report.json"):
            return "reconstruction_report"
        if name.endswith("_preflight.json"):
            return "reconstruction_preflight_json"
        if name.endswith("_reconstruction_diagnostics.json"):
            return "reconstruction_diagnostics"
        if name.endswith("_diagnostics.json"):
            return "reconstruction_diagnostics_json"
        return "reconstruction_aux"

    if rel.startswith("captioning/"):
        if name.endswith(".vlm_captioning.sidecar.json"):
            return "vlm_caption_sidecar"
        if name.endswith(".vlm_captioning.raw.txt"):
            return "vlm_caption_raw"
        if name.endswith("_proxy.png") or name.endswith("_proxy.jpg") or name.endswith("_proxy.jpeg"):
            return "vlm_caption_proxy"
        return "vlm_caption_aux"

    return "artifact"


def v2_log_filename(
    output_key_name: str,
    batch_id: Optional[str] = None,
) -> str:
    """Build deterministic, batch-scoped V2 log filename.

    Creates a consistent naming pattern for V2 enhancement logs:
    - Single batch: v2_{output_key}.log
    - With batch ID: v2_{output_key}__{batch_id}.log

    Args:
        output_key_name: The sanitized output key for the image
        batch_id: Optional batch identifier for scoping

    Returns:
        Deterministic log filename
    """
    filename = f"v2_{output_key_name}"
    if batch_id:
        filename += "__" + sanitize_path_component_nonlossy(str(batch_id))
    return f"{filename}.log"


def load_existing_manifest(
    manifest_path: Path,
    *,
    purpose: str,
) -> Optional[CombinedManifest]:
    """Best-effort manifest loader for cached-run preservation paths."""
    if not manifest_path.exists():
        return None
    try:
        return CombinedManifest.load(manifest_path)
    except Exception as exc:
        logger.debug(
            "Failed to load existing manifest for %s: %s",
            purpose,
            exc,
        )
        return None


def coerce_output_paths(raw_paths: Any) -> List[str]:
    """Normalize V2 output path payloads to a list of strings."""
    if isinstance(raw_paths, str):
        return [raw_paths] if raw_paths else []
    if not isinstance(raw_paths, list):
        return []
    return [path_value for path_value in raw_paths if isinstance(path_value, str) and path_value]


def normalize_v2_status(raw_status: Any) -> str:
    """Map runner and manifest status values to the manifest V2 contract."""
    if raw_status is None:
        return "skipped"
    if not isinstance(raw_status, str):
        return str(raw_status)

    normalized = raw_status.strip().lower()
    if not normalized:
        return "skipped"
    if normalized in {"success", "ok"}:
        return "ok"
    if normalized in {"failed", "failure"}:
        return "error"
    return normalized


def restore_materials_v3_from_manifest(
    manifest: Optional[CombinedManifest],
    expected_enhanced_path: Path,
) -> tuple[Optional[dict], float, Optional[Path]]:
    """Restore persisted Materials V3 metadata for cached-depth reruns."""
    if manifest is None or manifest.materials_v3 is None:
        return None, 0.0, None

    materials_v3 = manifest.materials_v3
    materials_v3_metadata: Dict[str, Any] = {
        "version": materials_v3.version,
    }
    if materials_v3.segmentation_metadata is not None:
        materials_v3_metadata["segmentation_metadata"] = copy.deepcopy(
            materials_v3.segmentation_metadata,
        )

    enhanced_path: Optional[Path] = expected_enhanced_path if expected_enhanced_path.exists() else None
    runtime_seconds = materials_v3.runtime_seconds
    restored_runtime = float(runtime_seconds) if runtime_seconds is not None else 0.0
    restored_result = {
        "materials_v3_response_plan": copy.deepcopy(
            materials_v3.response_plan,
        ),
        "materials_v3_pixel_ops": copy.deepcopy(
            materials_v3.pixel_ops,
        ),
        "materials_v3_metadata": materials_v3_metadata,
    }
    return restored_result, restored_runtime, enhanced_path


def preserved_v2_result_from_manifest(
    manifest: Optional[CombinedManifest],
) -> tuple[dict, Optional[Path]]:
    """Rehydrate V2 result fields from the prior manifest when reruns skip V2."""
    if manifest is None or manifest.v2 is None:
        return {"status": "ok"}, None

    previous_v2 = manifest.v2
    preserved_result: Dict[str, Any] = {
        "status": previous_v2.status,
    }
    if previous_v2.report_path:
        preserved_result["report_path"] = previous_v2.report_path

    output_paths = coerce_output_paths(previous_v2.output_paths)
    if output_paths:
        preserved_result["output_paths"] = output_paths
        preserved_result["output"] = output_paths[0]

    if previous_v2.error_message:
        preserved_result["error"] = previous_v2.error_message

    report_path = Path(previous_v2.report_path) if previous_v2.report_path else None
    return preserved_result, report_path


def normalize_backend_provenance_for_reuse(value: Any) -> Optional[str]:
    """Normalize backend provenance identifiers for reuse checks."""
    return normalize_backend_provenance(value)


def has_expanded_stage_a_fingerprint(
    config_fingerprint: Optional[ConfigFingerprint],
) -> bool:
    """Return True when manifest fingerprint carries the expanded Stage A contract."""
    if config_fingerprint is None:
        return False
    return all(
        getattr(config_fingerprint, field_name, None) is not None
        for field_name in (
            "quality_tier",
            "materials_config",
            "pbr_config",
            "apex_depth_gate_config",
            "output_bit_depth",
        )
    )


def segmentation_mask_artifact_path(
    segmentation_dir: Path,
    output_key: Path,
) -> Path:
    """Return canonical segmentation mask artifact path."""
    return segmentation_dir / output_key.parent / f"{output_key.stem}_materials_v3_masks.npz"


def build_artifact_index(
    output_root: Path,
    artifact_paths: List[Path],
) -> List[Dict[str, Any]]:
    """Build deterministic artifact index with size and SHA256.

    Scans artifact paths, computes integrity hashes, and returns
    a sorted index for run card inclusion.

    The index entries contain:
    - artifact_type: Canonical type (e.g., depth_u16_png)
    - path: Relative path from output root
    - relative_path: Same as path (for compatibility)
    - size_bytes: File size in bytes
    - sha256: SHA-256 hash for integrity verification

    Args:
        output_root: Base directory for output artifacts
        artifact_paths: List of paths to include in index

    Returns:
        Sorted list of artifact index entries
    """
    root_resolved = output_root.resolve()
    index_by_relative_path: Dict[str, Dict[str, Any]] = {}

    for candidate in artifact_paths:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.debug(
                "Skipping artifact path due to resolution error (%s): %s",
                candidate,
                exc,
            )
            continue

        if not resolved.is_file():
            continue

        try:
            relative_path = resolved.relative_to(root_resolved).as_posix()
        except ValueError:
            logger.debug(
                "Skipping artifact outside output root: %s",
                resolved,
            )
            continue

        if relative_path in index_by_relative_path:
            continue

        stat = resolved.stat()
        index_by_relative_path[relative_path] = {
            "artifact_type": infer_artifact_type(relative_path),
            "path": relative_path,
            "relative_path": relative_path,
            "size_bytes": stat.st_size,
            "sha256": compute_file_sha256(resolved),
        }

    return [index_by_relative_path[path] for path in sorted(index_by_relative_path)]


def compute_artifact_merkle_root(
    artifact_index: List[Dict[str, Any]],
) -> str:
    """Compute deterministic Merkle root over artifact SHA256 hashes.

    Creates a single hash representing all artifacts for integrity
    verification and audit purposes. The merkle root changes if
    any artifact changes.

    The algorithm:
    1. Sort artifacts by relative_path for determinism
    2. Concatenate SHA-256 bytes in sorted order
    3. Hash the concatenation with SHA-256

    Args:
        artifact_index: List of artifact entries with sha256 field

    Returns:
        64-character hex SHA-256 merkle root

    Raises:
        RuntimeError: If any artifact has invalid sha256 format
    """
    sorted_artifacts = sorted(
        artifact_index,
        key=lambda item: item["relative_path"],
    )
    leaves = []
    for artifact in sorted_artifacts:
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"Invalid artifact sha256 in run card index: {digest!r}")
        try:
            leaves.append(bytes.fromhex(digest))
        except ValueError:
            # Normalize underlying ValueError to the documented RuntimeError contract.
            raise RuntimeError(f"Invalid artifact sha256 in run card index: {digest!r}") from None

    return hashlib.sha256(b"".join(leaves)).hexdigest()


def make_output_key(
    input_path: Path,
    input_root: Path,
    use_xxhash: bool = XXHASH_AVAILABLE,
) -> Path:
    """Compute a stable, sanitized output key for a given input image.

    The final key preserves the relative directory shape under input_root
    (when possible) and emits <stem>_<ext|noext>_<hash8> as the terminal
    component. The 8-character suffix is derived from the POSIX-style
    relative path, using xxh64 when enabled/available or SHA-1 otherwise.

    Args:
        input_path: Path to input image
        input_root: Root directory for relative path calculation
        use_xxhash: Whether to use xxHash (faster) or SHA-1

    Returns:
        Path representing the output key with preserved directory structure
    """
    input_normalized = normalize_lexical_path(input_path)
    root_normalized = normalize_lexical_path(input_root)

    try:
        relpath = relative_to_path_alias(input_normalized, root_normalized)
    except ValueError:
        logger.warning(
            "%s is not relative to %s, using flat naming",
            input_normalized,
            root_normalized,
        )
        relpath = Path(input_normalized.name)

    rel_dir = relpath.parent
    name = relpath.stem
    ext = relpath.suffix

    sanitized_parts = [sanitize_path_component_nonlossy(p) for p in rel_dir.parts]
    ext_label = sanitize_path_component_nonlossy(
        ext.lstrip(".").lower() if ext else "noext",
    )

    hash_input = relpath.as_posix().encode("utf-8")

    if use_xxhash and XXHASH_AVAILABLE:
        hash_suffix = xxhash.xxh64(hash_input).hexdigest()[:8]
    else:
        hash_suffix = hashlib.sha1(
            hash_input,
            usedforsecurity=False,
        ).hexdigest()[:8]

    stem_sanitized = sanitize_path_component_nonlossy(name)
    key_name = f"{stem_sanitized}_{ext_label}_{hash_suffix}"

    if sanitized_parts:
        return Path(*sanitized_parts, key_name)
    return Path(key_name)


class ArtifactManager:
    """Unified artifact management interface for pipeline outputs.

    Provides a single entry point for artifact operations:
    - Type inference from paths
    - Index building with integrity hashes
    - Merkle root computation
    - Output key generation

    This class is the primary interface for artifact management per ADR-043.

    Example:
        manager = ArtifactManager(output_root=Path("/outputs"))

        # Build artifact index
        index = manager.index_artifacts([path1, path2])

        # Compute merkle root
        merkle = manager.compute_merkle_root(index)

        # Generate output key
        key = manager.generate_output_key(input_path, input_root)
    """

    def __init__(self, output_root: Path):
        """Initialize artifact manager.

        Args:
            output_root: Base directory for output artifacts
        """
        self._output_root = output_root.resolve()

    @property
    def output_root(self) -> Path:
        """Return the output root directory."""
        return self._output_root

    def index_artifacts(self, artifact_paths: List[Path]) -> List[Dict[str, Any]]:
        """Build artifact index from paths.

        Args:
            artifact_paths: Paths to artifacts to index

        Returns:
            Sorted list of artifact index entries
        """
        return build_artifact_index(self._output_root, artifact_paths)

    def compute_merkle_root(self, artifact_index: List[Dict[str, Any]]) -> str:
        """Compute merkle root for artifact index.

        Args:
            artifact_index: Artifact index from index_artifacts()

        Returns:
            64-character hex merkle root
        """
        return compute_artifact_merkle_root(artifact_index)

    def generate_output_key(
        self,
        input_path: Path,
        input_root: Optional[Path] = None,
    ) -> Path:
        """Generate output key for an input image.

        Args:
            input_path: Path to input image
            input_root: Root for relative path. Defaults to input_path.parent

        Returns:
            Path representing the output key
        """
        if input_root is None:
            input_root = input_path.parent
        return make_output_key(input_path, input_root)

    def infer_type(self, relative_path: str) -> str:
        """Infer artifact type from relative path.

        Args:
            relative_path: Path relative to output root

        Returns:
            Canonical artifact type string
        """
        return infer_artifact_type(relative_path)
