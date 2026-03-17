"""Manifest-aware Hugging Face model loader utilities.

This module bridges a manifest entry -> strict file verification -> local snapshot
directory that can be consumed by Transformers `from_pretrained(...)`.

Design intent:
- never load from floating Hub refs at inference time
- always resolve and verify required files first
- provide a stable local directory root for processor/model loading
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from transformation_portal.storage.cas_store import ArtifactStore

from transformation_portal.models.hf_lock import (
    HFModelLockError,
    HFModelLockRecord,
    resolve_all_required_files,
)

logger = logging.getLogger(__name__)


class HFManifestLoaderError(RuntimeError):
    """Raised when a manifest-aware local model load cannot be prepared."""


@dataclass(frozen=True)
class HFResolvedLocalModel:
    """Resolved local model with verified files.

    Attributes:
        model_key: Manifest key for this model entry
        repo_id: HuggingFace repository ID
        revision: Pinned commit SHA
        local_root: Local snapshot directory root
        resolved_files: Dictionary mapping relative paths to local absolute paths
    """

    model_key: str
    repo_id: str
    revision: str
    local_root: Path
    resolved_files: dict[str, Path]


def _common_local_root(paths: list[Path]) -> Path:
    """Given verified cached file paths from hf_hub_download, infer the local snapshot root.

    HuggingFace hub typically stores files under:
    .../models--org--repo/snapshots/<commit>/<file>

    Args:
        paths: List of resolved local file paths

    Returns:
        Path to the snapshot root directory

    Raises:
        HFManifestLoaderError: If root cannot be inferred
    """
    if not paths:
        raise HFManifestLoaderError("Cannot infer local root from an empty file list")

    # Typical HF cache path:
    # .../models--org--repo/snapshots/<commit>/<file>
    first = paths[0]
    parts = first.parts
    if "snapshots" not in parts:
        raise HFManifestLoaderError(
            f"Unable to infer Hugging Face snapshot root from path: {first}"
        )
    snapshot_index = parts.index("snapshots")
    root = Path(*parts[: snapshot_index + 2])
    return root


def _infer_local_root_from_hub(
    repo_id: str,
    revision: str,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
) -> Path:
    """Infer local snapshot root by downloading a minimal file.

    When no required_files are specified, we need to resolve the snapshot
    directory by downloading a small file (config.json is typical).

    Args:
        repo_id: HuggingFace repository ID
        revision: Pinned commit SHA
        repo_type: Repository type
        cache_dir: Optional cache directory

    Returns:
        Path to snapshot root

    Raises:
        HFManifestLoaderError: If root cannot be determined
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise HFManifestLoaderError(
            "huggingface_hub is required for HF model resolution"
        ) from exc

    # Try common config files
    config_files = ["config.json", "tokenizer_config.json", "preprocessor_config.json"]

    for config_file in config_files:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=config_file,
                revision=revision,
                repo_type=repo_type,
                cache_dir=cache_dir,
            )
            return _common_local_root([Path(local_path)])
        except Exception:
            continue

    raise HFManifestLoaderError(
        f"Cannot determine local root for '{repo_id}@{revision}': "
        "no common config files found"
    )


def resolve_manifest_model(
    model_key: str,
    payload: dict[str, Any],
    *,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> HFResolvedLocalModel:
    """Resolve a manifest model entry to a local model with verified files.

    Args:
        model_key: Manifest key identifying this model
        payload: Manifest payload dictionary with repo_id, revision, etc.
        cache_dir: Optional HuggingFace cache directory
        force_download: Force re-download even if cached

    Returns:
        HFResolvedLocalModel with local_root and resolved_files

    Raises:
        HFManifestLoaderError: If resolution fails
    """
    try:
        record = HFModelLockRecord.from_mapping(payload)
    except HFModelLockError as exc:
        raise HFManifestLoaderError(f"Invalid manifest payload for '{model_key}': {exc}") from exc

    # Resolve required files if specified
    if record.required_files:
        try:
            resolved_files = resolve_all_required_files(
                record,
                cache_dir=cache_dir,
                force_download=force_download,
            )
        except HFModelLockError as exc:
            raise HFManifestLoaderError(
                f"Failed to resolve required files for '{model_key}': {exc}"
            ) from exc

        local_root = _common_local_root(list(resolved_files.values()))
    else:
        # No explicit required_files - infer root from hub
        local_root = _infer_local_root_from_hub(
            repo_id=record.repo_id,
            revision=record.revision,
            repo_type=record.repo_type,
            cache_dir=cache_dir,
        )
        resolved_files = {}

    logger.info(
        "Resolved manifest model '%s' (%s@%s) to local root: %s",
        model_key,
        record.repo_id,
        record.revision[:8],
        local_root,
    )

    return HFResolvedLocalModel(
        model_key=model_key,
        repo_id=record.repo_id,
        revision=record.revision,
        local_root=local_root,
        resolved_files=resolved_files,
    )


def resolve_into_cas(
    resolved: HFResolvedLocalModel,
    cas: "ArtifactStore",
    target_dir: Path,
) -> Path:
    """Convert HF snapshot into CAS-backed directory.

    This function takes a resolved HF model and:
    1. Adds all files to CAS (deduplication by SHA-256)
    2. Materializes symlinks in the target directory

    The result is a runtime directory where each file is a symlink
    to the deduplicated CAS object.

    Args:
        resolved: Resolved HF model from resolve_manifest_model
        cas: ArtifactStore instance for CAS operations
        target_dir: Target directory for symlinks

    Returns:
        Path to target directory with materialized symlinks

    Example:
        >>> from transformation_portal.storage import ArtifactStore
        >>> cas = ArtifactStore(Path("/cache/cas"))
        >>> resolved = resolve_manifest_model("llava", payload)
        >>> runtime_dir = resolve_into_cas(resolved, cas, Path("runtime/llava"))
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    for relpath, src_path in resolved.resolved_files.items():
        # Add to CAS (deduplication happens here)
        cas_obj = cas.add_file(src_path)

        # Materialize symlink at target location
        dest = target_dir / relpath
        cas.materialize(cas_obj.sha256, dest)

        logger.debug(
            "CAS resolved: %s -> %s",
            relpath,
            cas_obj.sha256[:8],
        )

    logger.info(
        "Resolved %d files into CAS-backed directory: %s",
        len(resolved.resolved_files),
        target_dir,
    )

    return target_dir
