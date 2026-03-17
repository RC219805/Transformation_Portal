"""Content-addressable storage (CAS) using SHA-256.

This module provides a content-addressable storage layer for model artifacts
and other large files. Files are stored by their SHA-256 hash, enabling:
- Deduplication across models
- Deterministic storage
- Portability beyond HF cache layout

Layout:
    root/
        objects/
            ab/
                abcd1234...  (full sha256)

Example:
    >>> store = ArtifactStore(Path("/cache/cas"))
    >>> obj = store.add_file(Path("model.safetensors"))
    >>> print(f"Stored as: {obj.sha256[:8]}...")
    >>> store.materialize(obj.sha256, Path("runtime/model.safetensors"))
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CASError(RuntimeError):
    """Raised for CAS operation failures."""


@dataclass(frozen=True)
class CASObject:
    """Reference to an object stored in CAS.

    Attributes:
        sha256: SHA-256 hash of the file content (lowercase hex)
        path: Absolute path to the object in CAS
        size_bytes: Size of the object in bytes
    """

    sha256: str
    path: Path
    size_bytes: int


class ArtifactStore:
    """Content-addressable storage (CAS) using SHA-256.

    Objects are stored in a two-level directory structure:
        objects/ab/abcd1234...

    This provides efficient filesystem access while avoiding directory
    bloat from having millions of files in a single directory.

    Example:
        >>> store = ArtifactStore(Path("/cache/cas"))
        >>>
        >>> # Add file to CAS
        >>> obj = store.add_file(Path("model.safetensors"))
        >>> print(f"SHA: {obj.sha256[:8]}..., size: {obj.size_bytes}")
        >>>
        >>> # Materialize at runtime path
        >>> store.materialize(obj.sha256, Path("runtime/model.safetensors"))
        >>>
        >>> # Check if object exists
        >>> if store.has_object(obj.sha256):
        ...     print("Object exists in CAS")
    """

    def __init__(
        self,
        root: Path,
        *,
        create_dirs: bool = True,
    ) -> None:
        """Initialize artifact store.

        Args:
            root: Root directory for CAS storage
            create_dirs: If True, create directories if they don't exist
        """
        self.root = Path(root)
        self.objects_dir = self.root / "objects"

        if create_dirs:
            self.objects_dir.mkdir(parents=True, exist_ok=True)

    def _sha256_file(self, path: Path, chunk_size: int = 1024 * 1024) -> str:
        """Compute SHA-256 hash of a file.

        Args:
            path: Path to file
            chunk_size: Size of chunks to read (default 1MB)

        Returns:
            Lowercase hex SHA-256 hash
        """
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _object_path(self, sha256: str) -> Path:
        """Get path for object in CAS.

        Args:
            sha256: SHA-256 hash

        Returns:
            Path to object location
        """
        sha256 = sha256.lower()
        return self.objects_dir / sha256[:2] / sha256

    def has_object(self, sha256: str) -> bool:
        """Check if object exists in CAS.

        Args:
            sha256: SHA-256 hash to check

        Returns:
            True if object exists
        """
        return self._object_path(sha256).exists()

    def get_object(self, sha256: str) -> Optional[CASObject]:
        """Get CAS object metadata if it exists.

        Args:
            sha256: SHA-256 hash

        Returns:
            CASObject if found, None otherwise
        """
        path = self._object_path(sha256)
        if not path.exists():
            return None

        return CASObject(
            sha256=sha256.lower(),
            path=path,
            size_bytes=path.stat().st_size,
        )

    def add_file(
        self,
        src: Path,
        *,
        verify: bool = True,
    ) -> CASObject:
        """Add file to CAS.

        If an object with the same hash already exists, the existing
        object is returned (deduplication).

        Args:
            src: Source file to add
            verify: If True, verify hash after copy

        Returns:
            CASObject reference

        Raises:
            CASError: If file doesn't exist or copy fails
        """
        if not src.exists():
            raise CASError(f"Source file does not exist: {src}")

        sha = self._sha256_file(src)
        dst = self._object_path(sha)

        # Check for existing object (deduplication)
        if dst.exists():
            logger.debug("CAS hit: %s already exists", sha[:8])
            return CASObject(
                sha256=sha,
                path=dst,
                size_bytes=dst.stat().st_size,
            )

        # Create parent directory
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Copy file to CAS
        try:
            shutil.copy2(src, dst)
        except Exception as exc:
            raise CASError(f"Failed to copy {src} to CAS: {exc}") from exc

        # Verify copy if requested
        if verify:
            actual_sha = self._sha256_file(dst)
            if actual_sha != sha:
                dst.unlink()  # Remove corrupt copy
                raise CASError(f"Hash verification failed after copy: " f"expected {sha}, got {actual_sha}")

        logger.info("CAS add: %s (%d bytes)", sha[:8], dst.stat().st_size)

        return CASObject(
            sha256=sha,
            path=dst,
            size_bytes=dst.stat().st_size,
        )

    def add_bytes(
        self,
        data: bytes,
    ) -> CASObject:
        """Add bytes directly to CAS.

        Args:
            data: Bytes to store

        Returns:
            CASObject reference
        """
        sha = hashlib.sha256(data).hexdigest()
        dst = self._object_path(sha)

        if dst.exists():
            return CASObject(
                sha256=sha,
                path=dst,
                size_bytes=dst.stat().st_size,
            )

        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)

        return CASObject(
            sha256=sha,
            path=dst,
            size_bytes=len(data),
        )

    def materialize(
        self,
        sha256: str,
        dest: Path,
        *,
        use_symlink: bool = True,
        overwrite: bool = True,
    ) -> Path:
        """Materialize CAS object at a destination path.

        Args:
            sha256: SHA-256 hash of object to materialize
            dest: Destination path
            use_symlink: If True, create symlink; if False, copy
            overwrite: If True, overwrite existing destination

        Returns:
            Path to materialized file

        Raises:
            CASError: If object doesn't exist in CAS
        """
        src = self._object_path(sha256)
        if not src.exists():
            raise CASError(f"CAS object missing: {sha256}")

        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() or dest.is_symlink():
            if not overwrite:
                raise CASError(f"Destination exists and overwrite=False: {dest}")
            dest.unlink()

        if use_symlink:
            # Use absolute path for symlink
            dest.symlink_to(src.resolve())
            logger.debug("CAS symlink: %s -> %s", dest, sha256[:8])
        else:
            shutil.copy2(src, dest)
            logger.debug("CAS copy: %s -> %s", dest, sha256[:8])

        return dest

    def gc(
        self,
        referenced_hashes: set[str],
        *,
        dry_run: bool = True,
    ) -> list[str]:
        """Garbage collect unreferenced objects.

        Args:
            referenced_hashes: Set of SHA-256 hashes that are in use
            dry_run: If True, don't delete, just return what would be deleted

        Returns:
            List of SHA-256 hashes that were (or would be) deleted
        """
        referenced = {h.lower() for h in referenced_hashes}
        to_delete = []

        for prefix_dir in self.objects_dir.iterdir():
            if not prefix_dir.is_dir():
                continue

            for obj_path in prefix_dir.iterdir():
                sha = obj_path.name.lower()
                if sha not in referenced:
                    to_delete.append(sha)
                    if not dry_run:
                        obj_path.unlink()
                        logger.info("CAS gc: deleted %s", sha[:8])

        return to_delete
