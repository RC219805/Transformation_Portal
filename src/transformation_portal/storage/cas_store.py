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

Atomicity Contract:
    All write operations use atomic semantics to prevent partial writes:
    1. Write to temporary file (.tmp suffix)
    2. fsync to ensure durability
    3. Atomic rename to final path
    4. Verify hash matches expected value

    This prevents corruption in parallel execution scenarios where:
    - Process A is writing artifact
    - Process B reads partial artifact
    → silent corruption / invalid CAS reuse

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
import tempfile
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

    def _atomic_write_file(
        self,
        src: Path,
        dst: Path,
        expected_sha: str,
    ) -> None:
        """Atomically write a file to CAS with integrity verification.

        Atomicity Contract:
        1. Copy to temporary file in same directory
        2. fsync to ensure durability
        3. Verify hash matches expected value
        4. Atomic rename to final path

        Args:
            src: Source file to copy
            dst: Destination path in CAS
            expected_sha: Expected SHA-256 hash (lowercase hex)

        Raises:
            CASError: If copy fails or hash verification fails
        """
        # Create parent directory
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file in same directory for atomic rename
        # Use non-identifying prefix to avoid leaking hash info in directory listings
        fd, tmp_path_str = tempfile.mkstemp(
            suffix=".tmp",
            prefix=".cas_write_",
            dir=dst.parent,
        )
        tmp_path = Path(tmp_path_str)

        try:
            # Close the fd opened by mkstemp, we'll use shutil.copy2
            os.close(fd)

            # Copy source to temp file
            shutil.copy2(src, tmp_path)

            # fsync the temp file to ensure durability
            with tmp_path.open("rb") as f:
                os.fsync(f.fileno())

            # Verify hash BEFORE atomic rename
            actual_sha = self._sha256_file(tmp_path)
            if actual_sha != expected_sha:
                raise CASError(
                    f"Hash verification failed: expected {expected_sha}, got {actual_sha}"
                )

            # Atomic rename (POSIX guarantee: rename is atomic within same filesystem)
            os.replace(tmp_path, dst)

            # fsync parent directory to ensure rename is durable
            dir_fd = os.open(str(dst.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

            logger.debug("Atomic write complete: %s", expected_sha[:8])

        except Exception as exc:
            # Clean up temp file on failure
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise CASError(f"Atomic write failed for {expected_sha}: {exc}") from exc

    def _atomic_write_bytes(
        self,
        data: bytes,
        dst: Path,
        expected_sha: str,
    ) -> None:
        """Atomically write bytes to CAS with integrity verification.

        Args:
            data: Bytes to write
            dst: Destination path in CAS
            expected_sha: Expected SHA-256 hash (lowercase hex)

        Raises:
            CASError: If write fails or hash verification fails
        """
        # Create parent directory
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file in same directory for atomic rename
        # Use non-identifying prefix to avoid leaking hash info in directory listings
        fd, tmp_path_str = tempfile.mkstemp(
            suffix=".tmp",
            prefix=".cas_write_",
            dir=dst.parent,
        )
        tmp_path = Path(tmp_path_str)

        try:
            # Write bytes and fsync
            os.write(fd, data)
            os.fsync(fd)
            os.close(fd)

            # Verify hash BEFORE atomic rename
            actual_sha = self._sha256_file(tmp_path)
            if actual_sha != expected_sha:
                raise CASError(
                    f"Hash verification failed: expected {expected_sha}, got {actual_sha}"
                )

            # Atomic rename
            os.replace(tmp_path, dst)

            # fsync parent directory
            dir_fd = os.open(str(dst.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

        except Exception as exc:
            # Clean up temp file on failure
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            # Close fd if still open
            try:
                os.close(fd)
            except OSError:
                pass
            raise CASError(f"Atomic write failed for {expected_sha}: {exc}") from exc

    def add_file(
        self,
        src: Path,
        *,
        verify: bool = True,
    ) -> CASObject:
        """Add file to CAS with atomic write guarantees.

        If an object with the same hash already exists, the existing
        object is returned (deduplication).

        Atomicity Contract:
        - Writes use temp file + atomic rename pattern
        - Hash is verified BEFORE making artifact visible
        - Parallel writers cannot corrupt each other

        Args:
            src: Source file to add
            verify: If True, verify hash after copy (always True for atomicity)

        Returns:
            CASObject reference

        Raises:
            CASError: If file doesn't exist, copy fails, or hash verification fails
        """
        if not src.exists():
            raise CASError(f"Source file does not exist: {src}")

        sha = self._sha256_file(src)
        dst = self._object_path(sha)

        # Check for existing object (deduplication)
        if dst.exists():
            # Verify existing object integrity if requested
            if verify:
                actual_sha = self._sha256_file(dst)
                if actual_sha != sha:
                    logger.warning(
                        "Corrupt CAS object detected: %s (expected %s, got %s). Re-adding.",
                        dst,
                        sha[:8],
                        actual_sha[:8],
                    )
                    # Fall through to re-add the file
                else:
                    logger.debug("CAS hit: %s already exists (verified)", sha[:8])
                    return CASObject(
                        sha256=sha,
                        path=dst,
                        size_bytes=dst.stat().st_size,
                    )
            else:
                logger.debug("CAS hit: %s already exists", sha[:8])
                return CASObject(
                    sha256=sha,
                    path=dst,
                    size_bytes=dst.stat().st_size,
                )

        # Atomic write with integrity verification
        self._atomic_write_file(src, dst, sha)

        logger.info("CAS add: %s (%d bytes)", sha[:8], dst.stat().st_size)

        return CASObject(
            sha256=sha,
            path=dst,
            size_bytes=dst.stat().st_size,
        )

    def add_bytes(
        self,
        data: bytes,
        *,
        verify: bool = True,
    ) -> CASObject:
        """Add bytes directly to CAS with atomic write guarantees.

        Atomicity Contract:
        - Writes use temp file + atomic rename pattern
        - Hash is verified BEFORE making artifact visible
        - Parallel writers cannot corrupt each other

        Args:
            data: Bytes to store
            verify: If True, verify existing object integrity

        Returns:
            CASObject reference

        Raises:
            CASError: If write fails or hash verification fails
        """
        sha = hashlib.sha256(data).hexdigest()
        dst = self._object_path(sha)

        if dst.exists():
            # Verify existing object integrity if requested
            if verify:
                actual_sha = self._sha256_file(dst)
                if actual_sha != sha:
                    logger.warning(
                        "Corrupt CAS object detected: %s (expected %s, got %s). Re-adding.",
                        dst,
                        sha[:8],
                        actual_sha[:8],
                    )
                    # Fall through to re-add
                else:
                    return CASObject(
                        sha256=sha,
                        path=dst,
                        size_bytes=dst.stat().st_size,
                    )
            else:
                return CASObject(
                    sha256=sha,
                    path=dst,
                    size_bytes=dst.stat().st_size,
                )

        # Atomic write with integrity verification
        self._atomic_write_bytes(data, dst, sha)

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
