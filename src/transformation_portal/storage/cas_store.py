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
        quarantine/
            {sha256}_{actual_sha16}  (corrupt artifacts for forensics)

Quarantine Policy:
    Corrupt artifacts are moved to quarantine/ for forensic analysis.
    The quarantine is subject to lifecycle policy (default 7 days, 10GB max).
    Use gc_quarantine() to clean up old quarantined artifacts.

Atomicity Contract:
    All write operations use atomic semantics to prevent partial writes:
    1. Write to temporary file (.tmp suffix)
    2. fsync to ensure durability
    3. Verify hash matches expected value
    4. Atomic rename to final path

    This prevents corruption in parallel execution scenarios where:
    - Process A is writing artifact
    - Process B reads partial artifact
    → silent corruption / invalid CAS reuse

Atomicity Contract:
    All write operations use atomic semantics to prevent partial writes:
    1. Write to temporary file (.tmp suffix)
    2. fsync to ensure durability
    3. Verify hash matches expected value
    4. Atomic rename to final path

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
import platform as _platform
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from transformation_portal.core.security.path_safety import safe_cas_path, validate_sha256

logger = logging.getLogger(__name__)

# Quarantine lifecycle policy (prevents unbounded growth)
QUARANTINE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # 7 days
QUARANTINE_MAX_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB

# Cache platform check at module load time for efficiency
_IS_WINDOWS = _platform.system() == "Windows"
_HAS_O_DIRECTORY = hasattr(os, "O_DIRECTORY")


def _fsync_parent_directory(path: Path) -> None:
    """Fsync parent directory for durability (cross-platform).

    On POSIX systems, this ensures the directory entry update from an
    atomic rename is persisted. On Windows, this is a no-op since NTFS
    provides different durability guarantees (rename is visible after
    file fsync completes).

    Args:
        path: Path whose parent directory should be fsynced
    """
    if _IS_WINDOWS:
        # Windows/NTFS: rename durability handled by file fsync
        return

    if not _HAS_O_DIRECTORY:
        # Some platforms may not support directory open
        return

    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        # Directory fsync may fail on some filesystems (e.g., network drives)
        # but the atomic rename itself provides consistency guarantees
        pass


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


class CASFileLock:
    """Per-CAS-ID file lock for preventing parallel write races.

    Uses atomic file creation for cross-process locking with
    exponential backoff and stale lock detection.
    """

    def __init__(self, lock_path: Path, timeout: float = 300.0):
        """Initialize file lock.

        Args:
            lock_path: Path to lock file (will be created)
            timeout: Maximum time to wait for lock (seconds)
        """
        import time

        self._time = time
        self.lock_path = Path(lock_path)
        self.timeout = timeout
        self._acquired = False

    def acquire(self) -> bool:
        """Acquire the lock with exponential backoff."""
        import random

        start_time = self._time.time()
        wait_time = 0.01  # Start with 10ms

        while self._time.time() - start_time < self.timeout:
            try:
                self.lock_path.parent.mkdir(parents=True, exist_ok=True)
                with self.lock_path.open("x") as fd:
                    fd.write(str(self._time.time()))
                self._acquired = True
                return True
            except FileExistsError:
                # Check for stale lock
                try:
                    lock_time = float(self.lock_path.read_text())
                    if self._time.time() - lock_time > self.timeout * 2:
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except (ValueError, OSError):
                    pass
                self._time.sleep(wait_time + random.uniform(0, wait_time * 0.1))
                wait_time = min(wait_time * 2, 1.0)

        return False

    def release(self) -> None:
        """Release the lock."""
        if self._acquired:
            self.lock_path.unlink(missing_ok=True)
            self._acquired = False

    def __enter__(self) -> "CASFileLock":
        if not self.acquire():
            raise TimeoutError(f"Could not acquire CAS lock: {self.lock_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


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
        self.locks_dir = self.root / ".locks"  # Per-CAS-ID lock files

        if create_dirs:
            self.objects_dir.mkdir(parents=True, exist_ok=True)
            self.locks_dir.mkdir(parents=True, exist_ok=True)

    def _get_lock(self, sha256: str) -> CASFileLock:
        """Get a per-CAS-ID lock for coordinating parallel writes.

        Args:
            sha256: SHA-256 hash to lock

        Returns:
            CASFileLock for the given hash
        """
        lock_file = safe_cas_path(self.locks_dir, sha256).with_suffix(".lock")
        return CASFileLock(lock_file)

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
        return safe_cas_path(self.objects_dir, sha256)

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
        sha256 = validate_sha256(sha256)
        path = self._object_path(sha256)
        if not path.exists():
            return None

        return CASObject(
            sha256=sha256,
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
                raise CASError(f"Hash verification failed: expected {expected_sha}, got {actual_sha}")

            # Atomic rename (POSIX guarantee: rename is atomic within same filesystem)
            os.replace(tmp_path, dst)

            # fsync parent directory to ensure rename is durable (cross-platform)
            _fsync_parent_directory(dst)

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
            # Write bytes in a loop to handle partial writes from os.write()
            offset = 0
            while offset < len(data):
                written = os.write(fd, data[offset:])
                if written == 0:
                    raise CASError(f"Write returned 0 bytes at offset {offset}")
                offset += written
            os.fsync(fd)
            os.close(fd)

            # Verify hash BEFORE atomic rename
            actual_sha = self._sha256_file(tmp_path)
            if actual_sha != expected_sha:
                raise CASError(f"Hash verification failed: expected {expected_sha}, got {actual_sha}")

            # Atomic rename
            os.replace(tmp_path, dst)

            # fsync parent directory (cross-platform)
            _fsync_parent_directory(dst)

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
            if isinstance(exc, CASError):
                raise
            raise CASError(f"Atomic write failed for {expected_sha}: {exc}") from exc

    def add_file(
        self,
        src: Path,
        *,
        verify: bool = True,
    ) -> CASObject:
        """Add file to CAS with atomic write guarantees and double-checked locking.

        If an object with the same hash already exists, the existing
        object is returned (deduplication).

        Atomicity Contract:
        - Writes use temp file + atomic rename pattern
        - Hash is verified BEFORE making artifact visible
        - Double-checked locking prevents race conditions
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

        # First check (outside lock) - fast path for existing objects
        if dst.exists():
            if verify:
                actual_sha = self._sha256_file(dst)
                if actual_sha == sha:
                    logger.debug("CAS hit: %s already exists (verified)", sha[:8])
                    return CASObject(
                        sha256=sha,
                        path=dst,
                        size_bytes=dst.stat().st_size,
                    )
                # Corrupt - fall through to re-add with lock
                logger.warning(
                    "Corrupt CAS object detected: %s (expected %s, got %s). Re-adding.",
                    dst,
                    sha[:8],
                    actual_sha[:8],
                )
            else:
                logger.debug("CAS hit: %s already exists", sha[:8])
                return CASObject(
                    sha256=sha,
                    path=dst,
                    size_bytes=dst.stat().st_size,
                )

        # Double-checked locking: acquire per-CAS-ID lock
        with self._get_lock(sha):
            # Second check (inside lock) - another process may have written
            if dst.exists():
                if verify:
                    actual_sha = self._sha256_file(dst)
                    if actual_sha == sha:
                        logger.debug("CAS hit (post-lock): %s", sha[:8])
                        return CASObject(
                            sha256=sha,
                            path=dst,
                            size_bytes=dst.stat().st_size,
                        )
                    # Still corrupt - re-add
                else:
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
        """Add bytes directly to CAS with atomic write guarantees and double-checked locking.

        Atomicity Contract:
        - Writes use temp file + atomic rename pattern
        - Hash is verified BEFORE making artifact visible
        - Double-checked locking prevents race conditions
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

        # First check (outside lock) - fast path for existing objects
        if dst.exists():
            if verify:
                actual_sha = self._sha256_file(dst)
                if actual_sha == sha:
                    return CASObject(
                        sha256=sha,
                        path=dst,
                        size_bytes=dst.stat().st_size,
                    )
                logger.warning("Corrupt CAS object detected: %s. Re-adding.", sha[:8])
            else:
                return CASObject(
                    sha256=sha,
                    path=dst,
                    size_bytes=dst.stat().st_size,
                )

        # Double-checked locking: acquire per-CAS-ID lock
        with self._get_lock(sha):
            # Second check (inside lock)
            if dst.exists():
                if verify:
                    actual_sha = self._sha256_file(dst)
                    if actual_sha == sha:
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

    def verify_object(self, sha256: str) -> bool:
        """Verify CAS object integrity by re-hashing.

        This is critical for detecting silent corruption in the CAS.
        Should be called on read operations where integrity is required.

        Args:
            sha256: Expected SHA-256 hash

        Returns:
            True if hash matches, False if corruption detected

        Raises:
            CASError: If object doesn't exist
        """
        src = self._object_path(sha256)
        if not src.exists():
            raise CASError(f"CAS object missing: {sha256}")

        actual_sha = self._sha256_file(src)
        return actual_sha.lower() == sha256.lower()

    def materialize(
        self,
        sha256: str,
        dest: Path,
        *,
        use_symlink: bool = True,
        overwrite: bool = True,
        verify: bool = True,
    ) -> Path:
        """Materialize CAS object at a destination path.

        Args:
            sha256: SHA-256 hash of object to materialize
            dest: Destination path
            use_symlink: If True, create symlink; if False, copy
            overwrite: If True, overwrite existing destination
            verify: If True (default), verify hash integrity before materializing.
                This prevents silent corruption from being propagated.
                STRONGLY RECOMMENDED to keep as True for all use cases.
                Setting to False emits a warning.

        Returns:
            Path to materialized file

        Raises:
            CASError: If object doesn't exist in CAS or hash verification fails
        """
        src = self._object_path(sha256)
        if not src.exists():
            raise CASError(f"CAS object missing: {sha256}")

        # CRITICAL: Verify hash on read to detect corruption (TOCTOU protection)
        # Default is True - setting to False emits a warning
        if not verify:
            logger.warning(
                "CAS materialize called with verify=False for %s. "
                "This bypasses corruption detection and is NOT RECOMMENDED.",
                sha256[:8],
            )
        else:
            actual_sha = self._sha256_file(src)
            if actual_sha.lower() != sha256.lower():
                # Corruption detected - quarantine instead of delete for forensics
                quarantine_dir = self.root / "quarantine"
                quarantine_dir.mkdir(parents=True, exist_ok=True)
                quarantine_path = quarantine_dir / f"{sha256}_{actual_sha[:16]}"
                logger.error(
                    "CAS corruption detected: expected %s, got %s. Moving to quarantine: %s",
                    sha256[:8],
                    actual_sha[:8],
                    quarantine_path,
                )
                shutil.move(str(src), str(quarantine_path))
                raise CASError(
                    f"CAS hash verification failed: expected {sha256}, got {actual_sha}. "
                    f"Corrupt artifact quarantined at {quarantine_path}. Re-run to regenerate."
                )

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

    def gc_quarantine(
        self,
        *,
        max_age_seconds: int = QUARANTINE_MAX_AGE_SECONDS,
        max_size_bytes: int = QUARANTINE_MAX_SIZE_BYTES,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Garbage collect quarantined artifacts based on lifecycle policy.

        Quarantine cleanup is based on two policies:
        1. Age-based: Delete artifacts older than max_age_seconds
        2. Size-based: If total size exceeds max_size_bytes, delete oldest first

        Args:
            max_age_seconds: Maximum age in seconds (default: 7 days)
            max_size_bytes: Maximum total quarantine size (default: 10GB)
            dry_run: If True, don't delete, just report what would be deleted

        Returns:
            Dictionary with:
            - deleted: List of deleted file names
            - retained: List of retained file names
            - total_size_before: Total size before cleanup
            - total_size_after: Total size after cleanup
        """
        quarantine_dir = self.root / "quarantine"
        if not quarantine_dir.exists():
            return {
                "deleted": [],
                "retained": [],
                "total_size_before": 0,
                "total_size_after": 0,
            }

        now = time.time()
        files_with_info = []

        # Collect all quarantined files with their metadata
        for path in quarantine_dir.iterdir():
            if path.is_file():
                stat = path.stat()
                files_with_info.append(
                    {
                        "path": path,
                        "name": path.name,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                        "age": now - stat.st_mtime,
                    }
                )

        # Sort by age (oldest first) for size-based cleanup
        files_with_info.sort(key=lambda f: f["mtime"])

        total_size_before = sum(f["size"] for f in files_with_info)
        deleted: list[str] = []
        retained: list[dict[str, Any]] = []

        # Phase 1: Age-based cleanup
        for f in files_with_info:
            if f["age"] > max_age_seconds:
                deleted.append(f["name"])
                if not dry_run:
                    f["path"].unlink()
                    logger.info(
                        "CAS quarantine gc: deleted %s (age: %.1f days)",
                        f["name"],
                        f["age"] / 86400,
                    )
            else:
                retained.append(f)

        # Phase 2: Size-based cleanup (if still over limit)
        current_size = sum(f["size"] for f in retained)
        retained_final: list[str] = []

        for f in retained:
            if current_size > max_size_bytes:
                deleted.append(f["name"])
                current_size -= f["size"]
                if not dry_run:
                    f["path"].unlink()
                    logger.info("CAS quarantine gc: deleted %s (size limit)", f["name"])
            else:
                retained_final.append(f["name"])

        total_size_after = current_size if not dry_run else sum(f["size"] for f in files_with_info if f["name"] not in deleted)

        return {
            "deleted": deleted,
            "retained": retained_final,
            "total_size_before": total_size_before,
            "total_size_after": total_size_after,
        }
