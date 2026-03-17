"""HuggingFace model lock record and file resolution utilities.

This module provides dataclasses and functions for tracking pinned HuggingFace model
entries and resolving required files with strict verification.

Design intent:
- Never load from floating Hub refs at inference time
- Always verify required files before loading
- Provide a stable local directory root for processor/model loading
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class HFModelLockError(RuntimeError):
    """Raised when HuggingFace model lock resolution fails."""


@dataclass(frozen=True)
class HFRequiredFile:
    """Specification for a required model file with optional hash verification."""

    path: str
    sha256: Optional[str] = None
    filesize_bytes: Optional[int] = None


@dataclass(frozen=True)
class HFModelLockRecord:
    """Pinned HuggingFace model entry from manifest.

    Attributes:
        repo_id: HuggingFace repository ID (e.g., "llava-hf/llava-v1.6-mistral-7b-hf")
        revision: Pinned commit SHA (40 hex characters)
        required_files: List of required files with optional verification
        repo_type: Repository type ("model", "dataset", "space")
        provider: Model provider identifier
        license: Model license string
        owner: Internal owner/team identifier
        tier: Model tier classification
    """

    repo_id: str
    revision: str
    required_files: list[HFRequiredFile] = field(default_factory=list)
    repo_type: str = "model"
    provider: str = "huggingface"
    license: Optional[str] = None
    owner: Optional[str] = None
    tier: Optional[str] = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> HFModelLockRecord:
        """Create record from manifest payload dictionary.

        Args:
            payload: Dictionary with repo_id, revision, and optional fields

        Returns:
            HFModelLockRecord instance

        Raises:
            HFModelLockError: If required fields are missing
        """
        repo_id = payload.get("repo_id")
        if not repo_id:
            raise HFModelLockError("Manifest payload missing 'repo_id'")

        revision = payload.get("revision")
        if not revision:
            raise HFModelLockError(f"Manifest payload for '{repo_id}' missing 'revision'")

        # Parse required_files if present
        raw_files = payload.get("required_files", [])
        required_files = []
        for file_entry in raw_files:
            if isinstance(file_entry, str):
                required_files.append(HFRequiredFile(path=file_entry))
            elif isinstance(file_entry, dict):
                required_files.append(
                    HFRequiredFile(
                        path=file_entry.get("path", ""),
                        sha256=file_entry.get("sha256"),
                        filesize_bytes=file_entry.get("filesize_bytes"),
                    )
                )

        return cls(
            repo_id=repo_id,
            revision=revision,
            required_files=required_files,
            repo_type=payload.get("repo_type", "model"),
            provider=payload.get("provider", "huggingface"),
            license=payload.get("license"),
            owner=payload.get("owner"),
            tier=payload.get("tier"),
        )


def _compute_file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash for a file using streaming reads."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _verify_file(file_path: Path, required_file: HFRequiredFile) -> None:
    """Verify a downloaded file against required specifications.

    Args:
        file_path: Path to downloaded file
        required_file: Specification with optional sha256 and filesize

    Raises:
        HFModelLockError: If verification fails
    """
    if not file_path.exists():
        raise HFModelLockError(f"Required file does not exist: {file_path}")

    if required_file.filesize_bytes is not None:
        actual_size = file_path.stat().st_size
        if actual_size != required_file.filesize_bytes:
            raise HFModelLockError(
                f"File size mismatch for {required_file.path}: " f"expected {required_file.filesize_bytes}, got {actual_size}"
            )

    if required_file.sha256 is not None:
        actual_sha256 = _compute_file_sha256(file_path)
        expected_sha256 = required_file.sha256.lower()
        if actual_sha256 != expected_sha256:
            raise HFModelLockError(
                f"SHA-256 mismatch for {required_file.path}: " f"expected {expected_sha256}, got {actual_sha256}"
            )


def resolve_all_required_files(
    record: HFModelLockRecord,
    *,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> dict[str, Path]:
    """Download and verify all required files for a model lock record.

    Args:
        record: HFModelLockRecord with repo_id, revision, and required_files
        cache_dir: Optional HuggingFace cache directory
        force_download: Force re-download even if cached

    Returns:
        Dictionary mapping relative file paths to local absolute paths

    Raises:
        HFModelLockError: If any file fails to download or verify
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise HFModelLockError("huggingface_hub is required for HF model resolution") from exc

    resolved_files: dict[str, Path] = {}

    for required_file in record.required_files:
        if not required_file.path:
            continue

        try:
            local_path = hf_hub_download(
                repo_id=record.repo_id,
                filename=required_file.path,
                revision=record.revision,
                repo_type=record.repo_type,
                cache_dir=cache_dir,
                force_download=force_download,
            )
            local_path = Path(local_path)
        except Exception as exc:
            raise HFModelLockError(f"Failed to download '{required_file.path}' from '{record.repo_id}': {exc}") from exc

        # Verify file if specifications provided
        if required_file.sha256 or required_file.filesize_bytes:
            _verify_file(local_path, required_file)

        resolved_files[required_file.path] = local_path
        logger.debug("Resolved %s -> %s", required_file.path, local_path)

    return resolved_files
