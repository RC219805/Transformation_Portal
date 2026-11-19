"""
Global Path Safety Module.

This module provides the ONLY allowed way to construct paths from
user or system input. All filesystem access must follow:

    UNTRUSTED INPUT → VALIDATE (whitelist) → CANONICALIZE (join) → USE

NOT:

    UNTRUSTED → join → resolve → hope it's safe  ❌

Key guarantees:
- Pre-sanitization before path construction
- Whitelist-based validation (not blocklist)
- Normalized containment checks before filesystem access where symlinks matter
- CodeQL-compliant patterns

Usage:
    from transformation_portal.core.security.path_safety import (
        PathSafetyError,
        validate_safe_name,
        safe_join_file,
    )

    # Validate BEFORE path construction
    safe_name = validate_safe_name(user_input)
    filepath = safe_join_file(base_dir, safe_name, suffix=".json")
"""

from __future__ import annotations

import errno
import logging
import os
import re
import stat
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class PathSafetyError(ValueError):
    """Raised when path safety validation fails.

    This exception indicates a potential path traversal attack or
    invalid filesystem identifier.
    """

    pass


# Strict filename pattern: alphanumeric, underscores, hyphens only
# No dots, spaces, unicode, or path separators
# Bounded length (1-64) prevents DoS via excessively long names
_SAFE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Strict SHA256 hex pattern for CAS storage validation
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


def validate_safe_name(name: str) -> str:
    """Strict whitelist validation for user-facing file identifiers.

    Guarantees:
    - No path separators (/, \\)
    - No traversal tokens (., ..)
    - No unicode tricks
    - No special characters
    - Bounded length (1-64 chars)

    Args:
        name: The name to validate

    Returns:
        The validated name (unchanged if valid)

    Raises:
        PathSafetyError: If name fails validation

    Example:
        >>> validate_safe_name("valid_name-123")
        'valid_name-123'
        >>> validate_safe_name("../evil")  # Raises PathSafetyError
    """
    if not name:
        logger.warning("Path safety: rejected empty name")
        raise PathSafetyError("Empty name not allowed")

    if not _SAFE_NAME_PATTERN.fullmatch(name):
        logger.warning("Path safety: rejected invalid name %r", name)
        raise PathSafetyError(f"Invalid name: must match [a-zA-Z0-9_-]{{1,64}}, got: {name!r}")

    logger.debug("Path safety: validated name %r", name)
    return name


def validate_sha256(sha: str) -> str:
    """Strict validation for SHA256 hex strings.

    Used for CAS (Content-Addressable Storage) path construction.

    Args:
        sha: SHA256 hex string to validate

    Returns:
        The validated SHA256 string (unchanged if valid)

    Raises:
        PathSafetyError: If not a valid 64-character hex string
    """
    if not sha:
        raise PathSafetyError("Empty SHA256 not allowed")

    # Normalize to lowercase for consistency
    sha_lower = sha.lower()

    if not _SHA256_PATTERN.fullmatch(sha_lower):
        logger.warning("Path safety: rejected invalid SHA256 %r", sha)
        raise PathSafetyError(f"Invalid SHA256: must be 64 hex characters, got: {sha!r}")

    return sha_lower


def safe_join_file(
    base_dir: Path,
    name: str,
    *,
    suffix: str,
) -> Path:
    """Construct a safe file path from a validated name.

    IMPORTANT:
    - `name` is validated automatically via validate_safe_name()
    - `suffix` must be a constant (never user-controlled)
    - No .resolve() or .relative_to() - whitelist validation is sufficient

    Args:
        base_dir: Base directory for the file
        name: User-provided filename stem (will be validated)
        suffix: File extension (must start with '.')

    Returns:
        Safe path: base_dir / f"{validated_name}{suffix}"

    Raises:
        PathSafetyError: If name fails validation or suffix is invalid

    Example:
        >>> safe_join_file(Path("/data"), "pipeline1", suffix=".json")
        PosixPath('/data/pipeline1.json')
    """
    # Validate name BEFORE path construction (required for static analysis)
    safe_name = validate_safe_name(name)

    # Validate suffix format
    if not suffix.startswith("."):
        raise PathSafetyError(f"Suffix must start with '.', got: {suffix!r}")

    # Only alphanumeric suffixes allowed (no .tar.gz etc.)
    suffix_body = suffix[1:]
    if not suffix_body.isalnum():
        raise PathSafetyError(f"Suffix must be alphanumeric after '.', got: {suffix!r}")

    # Construct and return path
    filepath = base_dir / f"{safe_name}{suffix}"
    logger.debug("Path safety: constructed safe path %s", filepath)
    return filepath


def safe_join_subpath(
    base_dir: Path,
    relative_parts: List[str],
) -> Path:
    """Construct a safe path with multiple validated segments.

    For internal (non-user) path construction where multiple directory
    levels are needed. Each segment is validated.

    Args:
        base_dir: Base directory
        relative_parts: List of path segments (each validated)

    Returns:
        Safe path with all segments joined

    Raises:
        PathSafetyError: If any segment fails validation

    Example:
        >>> safe_join_subpath(Path("/data"), ["artifacts", "v1"])
        PosixPath('/data/artifacts/v1')
    """
    if not relative_parts:
        raise PathSafetyError("At least one path segment required")

    path = base_dir
    for part in relative_parts:
        validate_safe_name(part)
        path = path / part

    logger.debug("Path safety: constructed subpath %s", path)
    return path


def safe_cas_path(
    objects_dir: Path,
    sha256: str,
) -> Path:
    """Construct a safe CAS (Content-Addressable Storage) object path.

    Uses standard 2-character prefix sharding for performance.

    Args:
        objects_dir: Base objects directory
        sha256: SHA256 hash of the content

    Returns:
        Path: objects_dir / sha256[:2] / sha256

    Raises:
        PathSafetyError: If SHA256 is invalid

    Example:
        >>> safe_cas_path(Path("/cas"), "abc123...")
        PosixPath('/cas/ab/abc123...')
    """
    safe_sha = validate_sha256(sha256)

    # Normalize before the containment check so both runtime enforcement and
    # static analysis can prove that filesystem access stays under the CAS
    # objects directory, including when a shard is replaced by a symlink.
    normalized_root = os.path.realpath(os.fspath(objects_dir))
    filepath = os.path.realpath(os.path.join(normalized_root, safe_sha[:2], safe_sha))
    root_prefix = normalized_root if normalized_root.endswith(os.sep) else normalized_root + os.sep
    if not filepath.startswith(root_prefix):
        logger.warning("Path safety: rejected CAS path outside objects directory")
        raise PathSafetyError("CAS path escapes objects directory")

    logger.debug("Path safety: constructed CAS path %s", filepath)
    return Path(filepath)


# Windows reports an unresolvable reparse-point cycle with this code.
_WINERROR_CANT_RESOLVE_FILENAME = 1921
_WINDOWS_JUNCTION_TAG = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)


class SymlinkLoopError(OSError, RuntimeError):
    """A path cannot be resolved because one of its components is a symlink loop.

    It is both an ``OSError`` (errno ``ELOOP``, what strict resolution raises
    on Python 3.13+) and a ``RuntimeError`` (what ``Path.resolve`` raised on
    Python 3.11 and 3.12), so handlers written for either keep catching it.
    """


def _symlink_target(path: Path) -> Path | None:
    try:
        metadata = path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return None  # Preserve a missing suffix for paths that will be created later.
    if not stat.S_ISLNK(metadata.st_mode) and getattr(metadata, "st_reparse_tag", None) != _WINDOWS_JUNCTION_TAG:
        return None
    return path.readlink()


def _resolve_path_components(path: Path) -> Path:
    """Walk raw components when the platform resolver cannot check them in order."""
    absolute = path.absolute()
    resolved = Path(os.sep if os.name == "posix" else absolute.anchor)
    pending: list[str | Path] = list(reversed(absolute.parts[1:]))
    links: dict[Path, Path | None] = {}
    while pending:
        component = pending.pop()
        if isinstance(component, Path):
            links[component] = resolved
            continue
        if component == "..":
            resolved = resolved.parent
            continue
        candidate = resolved / component
        if candidate in links:
            cached = links[candidate]
            if cached is None:
                raise OSError(errno.ELOOP, "Symlink loop", candidate)
            resolved = cached
            continue
        target = _symlink_target(candidate)
        if target is None:
            resolved = candidate
            continue
        links[candidate] = None
        pending.append(candidate)  # Complete this link after resolving all its target components.
        if target.anchor:
            resolved /= os.sep if os.name == "posix" else target.anchor
            pending.extend(reversed(target.parts[1:]))
        else:
            pending.extend(reversed(target.parts))
    return resolved


def resolve_rejecting_symlink_loops(path: str | os.PathLike[str]) -> Path:
    """Resolve a path and reject cycles before a later ``..`` can hide them.

    Keep missing components lexically so an output path can be resolved before
    its creation. Check traversal inside symlink targets as well as the input.
    """
    candidate = Path(path)
    try:
        if os.name == "posix":
            try:
                return Path(os.path.realpath(candidate, strict=True))
            except (FileNotFoundError, NotADirectoryError, RecursionError):
                # Strict resolution stops at the first absent component. The
                # walk permits that component but keeps checking later links.
                pass
        resolved = _resolve_path_components(candidate)
        # Windows normalizes '..' before checking links. After checking the
        # original traversal, preserve its canonical drive, casing and spelling.
        return candidate.resolve() if os.name == "nt" else resolved
    except OSError as exc:
        if exc.errno == errno.ELOOP or getattr(exc, "winerror", None) == _WINERROR_CANT_RESOLVE_FILENAME:
            raise SymlinkLoopError(errno.ELOOP, "Symlink loop", os.fspath(path)) from exc
        raise
