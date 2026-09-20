"""Bounded photographic snapshots using the existing pinned namespace boundary."""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from transformation_portal.lux_depth_v3.execution_evidence import (
    ArtifactEvidenceError,
    _canonicalize_top_level_alias,
    _open_confined_artifact,
    _pin_output_root,
    _secure_atomic_write_bytes,
    _validate_confined_entry_identity,
    _validate_pinned_root_namespace,
)


def directory_path(path: Path, *, allow_missing: bool = False) -> Path:
    """Normalize standard macOS aliases only, then reject linked existing ancestors.

    Validation does not create missing output/cache directories. Execution must
    re-pin the eventual publication namespace before writing.
    """
    candidate = _canonicalize_top_level_alias(Path(os.path.abspath(Path(path).expanduser())))
    existing = candidate
    while not existing.exists() and not existing.is_symlink():
        existing = existing.parent
    if not allow_missing and existing != candidate:
        raise ValueError("Input root must be an existing directory")
    with _pin_output_root(existing):
        pass
    return candidate


@contextmanager
def pinned_directory(path: Path) -> Iterator[None]:
    """Detect namespace replacement across discovery and immutable snapshot creation."""
    with _pin_output_root(path) as root:
        yield
        _validate_pinned_root_namespace(root)


def snapshot(root_path: Path, path: Path, *, maximum_bytes: int, retain_bytes: bool = True) -> tuple[bytes, dict[str, Any]]:
    """Bind one regular, nonlinked file to immutable bytes and a content digest."""
    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes <= 0:
        raise ValueError("Input byte budget must be a positive integer")
    if not isinstance(retain_bytes, bool):
        raise ValueError("retain_bytes must be boolean")
    with _pin_output_root(root_path) as root:
        descriptor, relative = _open_confined_artifact(root, path)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= maximum_bytes:
                raise ValueError("Input must be a bounded regular file without link aliases")
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > maximum_bytes:
                    raise ValueError("Input exceeds byte budget")
                digest.update(chunk)
                if retain_bytes:
                    chunks.append(chunk)
            _validate_confined_entry_identity(root, relative, before, context="photographic input")
            if size != before.st_size:
                raise ValueError("Input changed during snapshot")
        finally:
            os.close(descriptor)
    return b"".join(chunks), {"path": relative, "sha256": digest.hexdigest(), "size_bytes": size}


def write_evidence(root_path: Path, relative: str, data: bytes) -> None:
    """Reuse descriptor-relative, durable completion publication."""
    with _pin_output_root(root_path) as root:
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ArtifactEvidenceError("path_escape", "Evidence destination must be a relative path")
        confined = root.confined_relative_path(Path(relative))
        if confined != relative:
            raise ArtifactEvidenceError("path_escape", "Evidence destination must be canonical")
        _secure_atomic_write_bytes(root, confined, data)
