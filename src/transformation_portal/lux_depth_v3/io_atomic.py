"""Atomic write operations for lux_depth_v3 artifact writers.

Provides unified atomic write primitives for all lux_depth_v3 artifact types:
- Depth maps (16-bit PNG via cv2)
- PBR maps (8-bit PNG via PIL)
- Future artifact types

All operations use atomic rename (os.replace) and guarantee:
- No partial writes visible to readers
- Deterministic FD cleanup (no leaks)
- No orphaned temp files on failure
- Temp files created in destination directory (same filesystem)
"""

from __future__ import annotations

import errno
import importlib
import logging
import os
import stat
import tempfile
import threading
import time
import unicodedata
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO, Callable, Generator

logger = logging.getLogger(__name__)

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None  # type: ignore


_DEFAULT_NEW_FILE_MODE = 0o644
_IS_WINDOWS = os.name == "nt"
_PUBLICATION_LOCKS_GUARD = threading.Lock()
_PUBLICATION_THREAD_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    error_number
    for error_number in (
        errno.EINVAL,
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if error_number is not None
)


def _filesystem_name_collision_key(name: str) -> str:
    """Return a conservative key for case/Unicode-equivalent path names."""

    normalized = unicodedata.normalize("NFC", name)
    return unicodedata.normalize("NFC", normalized.casefold())


def _optional_platform_module(name: str) -> ModuleType | None:
    """Import a platform lock module without making it a package dependency."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


_FCNTL = _optional_platform_module("fcntl")
_MSVCRT = _optional_platform_module("msvcrt")


def _destination_file_mode(output_path: Path) -> int:
    """Return the existing destination mode or the fixed new-file mode."""
    try:
        return stat.S_IMODE(output_path.stat().st_mode)
    except FileNotFoundError:
        return _DEFAULT_NEW_FILE_MODE


def _apply_file_mode(path: Path, descriptor: int, mode: int) -> None:
    """Apply a deterministic mode to an open temporary file."""
    fchmod = getattr(os, "fchmod", None)
    if fchmod is not None:
        fchmod(descriptor, mode)
    else:  # pragma: no cover - Windows does not expose os.fchmod
        os.chmod(path, mode)


def _write_all(handle: BinaryIO, data: bytes) -> None:
    """Write all bytes, rejecting a zero-length or otherwise short write."""
    remaining = memoryview(data)
    while remaining:
        written = handle.write(remaining)
        if written is None or written <= 0:
            raise OSError("short write while publishing atomic bytes")
        remaining = remaining[written:]


def _fsync_directory(directory: Path) -> None:
    """Persist a directory entry update where the platform supports it.

    Windows does not expose a portable directory-fsync primitive through
    ``os``. On POSIX, only errors that explicitly mean the operation is not
    supported are tolerated; genuine I/O and permission errors propagate.
    """
    if _IS_WINDOWS:
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError as exc:
        if exc.errno in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            logger.debug("Directory fsync is unsupported for %s: %s", directory, exc)
            return
        raise

    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                raise
            logger.debug("Directory fsync is unsupported for %s: %s", directory, exc)
    finally:
        os.close(descriptor)


def _ensure_durable_directory(directory: Path) -> None:
    """Create a directory hierarchy and persist each new parent entry.

    ``mkdir(parents=True)`` makes a hierarchy visible, but an fsync of only the
    leaf directory cannot make the newly created directory entries durable.
    Create missing components from the nearest existing ancestor downward and
    fsync each component's parent immediately after creation. Concurrent
    creators are harmless: an already-created directory is still fenced before
    it is used as the parent of the next component.
    """

    directory = Path(directory)
    missing: list[Path] = []
    current = directory
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent

    for component in reversed(missing):
        component.mkdir(exist_ok=True)
        if not component.is_dir():
            raise NotADirectoryError(errno.ENOTDIR, "publication parent is not a directory", os.fspath(component))
        _fsync_directory(component.parent)


def durable_unlink(path: Path) -> None:
    """Remove a file and durably publish that directory-entry change.

    A missing path is already in the desired state. If directory fsync fails
    after unlinking, the removal is visible but its crash durability is not
    proven and the error propagates.
    """
    path = Path(path)
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def _publication_thread_lock(lock_path: Path) -> threading.Lock:
    """Return the process-local half of a destination-keyed lock."""
    key = os.path.abspath(os.fspath(lock_path))
    with _PUBLICATION_LOCKS_GUARD:
        return _PUBLICATION_THREAD_LOCKS.setdefault(key, threading.Lock())


def _reset_publication_thread_locks_after_fork() -> None:
    """Discard copied thread-lock state in a newly forked child process."""
    global _PUBLICATION_LOCKS_GUARD, _PUBLICATION_THREAD_LOCKS
    _PUBLICATION_LOCKS_GUARD = threading.Lock()
    _PUBLICATION_THREAD_LOCKS = weakref.WeakValueDictionary()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_publication_thread_locks_after_fork)


def _acquire_platform_file_lock(descriptor: int) -> None:
    """Acquire a blocking cross-process lock for one open lock file."""
    if _FCNTL is not None:
        _FCNTL.flock(descriptor, _FCNTL.LOCK_EX)
        return
    if _MSVCRT is not None:  # pragma: no cover - exercised on Windows
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        while True:
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                _MSVCRT.locking(descriptor, _MSVCRT.LK_NBLCK, 1)
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EDEADLK}:
                    raise
                time.sleep(0.05)
    raise RuntimeError("cross-process publication locking is unsupported on this platform")


def _release_platform_file_lock(descriptor: int) -> None:
    """Release a cross-process lock acquired by ``_acquire_platform_file_lock``."""
    if _FCNTL is not None:
        _FCNTL.flock(descriptor, _FCNTL.LOCK_UN)
        return
    if _MSVCRT is not None:  # pragma: no cover - exercised on Windows
        os.lseek(descriptor, 0, os.SEEK_SET)
        _MSVCRT.locking(descriptor, _MSVCRT.LK_UNLCK, 1)
        return
    raise RuntimeError("cross-process publication locking is unsupported on this platform")


def _open_publication_lock(lock_path: Path) -> int:
    """Open one regular, unaliased lock inode without following symlinks."""
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(lock_path, flags, _DEFAULT_NEW_FILE_MODE)
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.lstat(lock_path)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or descriptor_stat.st_nlink != 1
            or not os.path.samestat(descriptor_stat, path_stat)
        ):
            raise OSError(errno.ELOOP, "unsafe publication lock path", os.fspath(lock_path))
        _apply_file_mode(lock_path, descriptor, _DEFAULT_NEW_FILE_MODE)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def publication_lock(destination: Path) -> Generator[None, None, None]:
    """Serialize cooperating publishers for a multi-file evidence contract.

    A stable hidden lock file lives beside the destination. Keeping the file in
    place is intentional: unlinking a lock file can split waiters across two
    inodes. The OS lock is released automatically if a process exits, while the
    process-local lock also serializes threads on platforms whose file locks
    are process-scoped.
    """
    destination = Path(destination)
    _ensure_durable_directory(destination.parent)
    lock_path = destination.with_name(f".{destination.name}.publication.lock")
    thread_lock = _publication_thread_lock(lock_path)

    with thread_lock:
        descriptor = _open_publication_lock(lock_path)
        try:
            _acquire_platform_file_lock(descriptor)
            try:
                yield
            finally:
                _release_platform_file_lock(descriptor)
        finally:
            os.close(descriptor)


def atomic_write_evidence_pair(
    primary_path: Path,
    primary_bytes: bytes,
    sidecar_path: Path,
    sidecar_bytes: bytes,
) -> tuple[Path, Path]:
    """Publish a primary evidence file and its verifying sidecar safely.

    Writers for the same primary destination are serialized across threads and
    cooperating processes. The old sidecar is invalidated durably before the
    new primary is published, so a crash can leave either the old valid pair,
    or a complete primary without a sidecar, but never stale attestation for new
    primary bytes.
    """
    primary_path = Path(primary_path)
    sidecar_path = Path(sidecar_path)
    primary_resolved = primary_path.resolve()
    sidecar_resolved = sidecar_path.resolve()
    if primary_resolved == sidecar_resolved:
        raise ValueError("primary evidence path and sidecar path must differ")
    if primary_resolved.parent != sidecar_resolved.parent:
        raise ValueError("primary evidence and sidecar must share a destination directory")
    if _filesystem_name_collision_key(primary_resolved.name) == _filesystem_name_collision_key(sidecar_resolved.name):
        raise ValueError("primary evidence and sidecar names must differ beyond case and Unicode normalization")
    try:
        if os.path.samefile(primary_path, sidecar_path):
            raise ValueError("primary evidence path and sidecar path must not alias one file")
    except FileNotFoundError:
        pass

    with publication_lock(primary_path):
        sidecar_mode = _destination_file_mode(sidecar_path)
        durable_unlink(sidecar_path)
        atomic_write_bytes(primary_path, primary_bytes)
        atomic_write_bytes(sidecar_path, sidecar_bytes, _preserved_mode=sidecar_mode)
    return primary_path, sidecar_path


@contextmanager
def atomic_temp_file(
    output_path: Path, suffix: str = ".tmp", prefix: str = ".tmp_", create_file: bool = False
) -> Generator[Path, None, None]:
    """Context manager for atomic temp file creation.

    Creates a temporary file path (or file) in the same directory as output_path,
    then atomically renames it on successful exit.

    Args:
        output_path: Final destination path
        suffix: Temp file suffix (default: ".tmp")
        prefix: Temp file prefix (default: ".tmp_")
        create_file: If True, pre-create file with mkstemp (for FD-based writers).
                     If False, generate unique path only (for path-based writers).

    Yields:
        Path to temporary file (may or may not exist yet)

    Ensures:
        - Temp file is in same directory as output_path
        - Atomic rename via os.replace on success
        - Cleanup of temp file on failure
        - No FD leaks

    Example:
        >>> # Path-based writer (cv2.imwrite)
        >>> with atomic_temp_file(Path("output.png"), suffix=".png") as temp_path:
        ...     cv2.imwrite(str(temp_path), image)
        # output.png now exists atomically

        >>> # FD-based writer (requires pre-created file)
        >>> with atomic_temp_file(Path("out.bin"), create_file=True) as temp_path:
        ...     temp_path.write_bytes(data)
        # out.bin now exists atomically
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    destination_mode = _destination_file_mode(output_path)

    if create_file:
        # Pre-create file with mkstemp (for FD-based or direct file access)
        temp_fd, temp_path_str = tempfile.mkstemp(suffix=suffix, dir=output_path.parent, prefix=prefix)
        temp_path = Path(temp_path_str)

        try:
            # Close FD immediately - caller will use path directly
            os.close(temp_fd)

            yield temp_path

            # Preserve an existing mode or use the fixed new-file policy.
            os.chmod(temp_path, destination_mode)

            # Atomic rename on success
            os.replace(temp_path, output_path)

        except BaseException:
            # Cleanup temp file on any failure (including KeyboardInterrupt)
            # Log at debug level since exception will be re-raised and handled by caller
            logger.debug("Cleaning up temp file after failure: %s", temp_path)
            temp_path.unlink(missing_ok=True)
            raise
    else:
        # Generate unique temp path without creating file
        # The caller creates the file; its mode is normalized before publication.
        import uuid

        temp_name = f"{prefix}{uuid.uuid4().hex}{suffix}"
        temp_path = output_path.parent / temp_name

        try:
            yield temp_path

            # Atomic rename on success (temp_path should exist by now)
            if temp_path.exists():
                os.chmod(temp_path, destination_mode)
                os.replace(temp_path, output_path)
            else:
                raise IOError(f"Writer did not create temp file: {temp_path}")

        except BaseException:
            # Cleanup temp file on any failure (including KeyboardInterrupt)
            # Log at debug level since exception will be re-raised and handled by caller
            logger.debug("Cleaning up temp file after failure: %s", temp_path)
            temp_path.unlink(missing_ok=True)
            raise


def atomic_write_bytes(output_path: Path, data: bytes, *, _preserved_mode: int | None = None) -> Path:
    """Atomically and durably write bytes to a file.

    Args:
        output_path: Destination file path
        data: Bytes to write
        _preserved_mode: Internal pair-publication override captured before a
            stale sidecar is invalidated. Ordinary callers must leave this
            unset so the destination mode is discovered immediately before
            publication.

    Returns:
        Path to written file (same as output_path)

    Raises:
        IOError: If writing, replacing, or proving durability fails. An error
            from the directory fsync occurs after replacement, so a complete
            new destination may already be visible even though durability was
            not proven.

    Example:
        >>> path = atomic_write_bytes(Path("output.bin"), b"hello")
        >>> assert path.read_bytes() == b"hello"
    """
    output_path = Path(output_path)
    temp_fd: int | None = None
    temp_path: Path | None = None

    try:
        _ensure_durable_directory(output_path.parent)
        destination_mode = _destination_file_mode(output_path) if _preserved_mode is None else _preserved_mode
        temp_fd, temp_path_str = tempfile.mkstemp(
            suffix=".tmp",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
        )
        temp_path = Path(temp_path_str)

        handle = os.fdopen(temp_fd, "wb")
        temp_fd = None  # The file object now owns the descriptor.
        with handle:
            _write_all(handle, data)
            handle.flush()
            os.fsync(handle.fileno())
            _apply_file_mode(temp_path, handle.fileno(), destination_mode)
            # Persist the mode mutation as well as the payload before rename.
            os.fsync(handle.fileno())

        os.replace(temp_path, output_path)
        _fsync_directory(output_path.parent)
        return output_path
    except BaseException as exc:
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to clean up atomic temp file: %s", temp_path, exc_info=True)

        if isinstance(exc, Exception):
            raise IOError(f"Failed to write {output_path}") from exc
        raise


def atomic_write_pil_png(output_path: Path, pil_image: "Image.Image", optimize: bool = True, **save_kwargs: Any) -> Path:
    """Atomically write PIL Image as PNG.

    Args:
        output_path: Destination file path
        pil_image: PIL Image to save
        optimize: Whether to optimize PNG (default: True)
        **save_kwargs: Additional arguments for PIL Image.save()

    Returns:
        Path to written file (same as output_path)

    Raises:
        ImportError: If PIL not available
        IOError: If write fails

    Example:
        >>> from PIL import Image
        >>> img = Image.new('RGB', (100, 100))
        >>> path = atomic_write_pil_png(Path("output.png"), img)
        >>> assert path.exists()
    """
    if not HAS_PIL:
        raise ImportError("Pillow required for atomic_write_pil_png. Install with: pip install Pillow")

    try:
        # PIL creates the path; atomic_temp_file applies the explicit mode policy.
        with atomic_temp_file(output_path, suffix=".png", create_file=False) as temp_path:
            # Save directly to temp file path
            pil_image.save(temp_path, format="PNG", optimize=optimize, **save_kwargs)
        return Path(output_path)
    except Exception as e:
        raise IOError(f"Failed to write PNG {output_path}") from e


def atomic_write_with_fd(output_path: Path, writer_func: Callable[[int, Path], None], suffix: str = ".tmp") -> Path:
    """Atomically write using a file descriptor-based writer function.

    For writers that need an open file descriptor (e.g., cv2.imwrite with fdopen).

    Args:
        output_path: Destination file path
        writer_func: Callable taking (fd, temp_path) and writing to FD
        suffix: Temp file suffix (default: ".tmp")

    Returns:
        Path to written file (same as output_path)

    Raises:
        IOError: If write fails

    Example:
        >>> def write_cv2_image(fd, temp_path):
        ...     # Close FD first if writer uses path
        ...     os.close(fd)
        ...     cv2.imwrite(str(temp_path), image_data)
        >>> path = atomic_write_with_fd(Path("out.png"), write_cv2_image, ".png")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    destination_mode = _destination_file_mode(output_path)

    temp_fd = None
    temp_path = None

    try:
        # Create temp file with FD
        temp_fd, temp_path_str = tempfile.mkstemp(suffix=suffix, dir=output_path.parent, prefix=".tmp_")
        temp_path = Path(temp_path_str)

        # Call writer function - it's responsible for closing FD if needed
        writer_func(temp_fd, temp_path)

        # Close FD if writer didn't consume it
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass  # Already closed by writer

        # Preserve an existing mode or use the fixed new-file policy.
        os.chmod(temp_path, destination_mode)

        # Atomic rename
        os.replace(temp_path, output_path)

        return Path(output_path)

    except Exception as e:
        # Cleanup
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass

        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)

        raise IOError(f"Failed to write {output_path}") from e
