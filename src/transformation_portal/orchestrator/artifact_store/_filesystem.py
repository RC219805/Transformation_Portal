"""Descriptor-anchored traversal for private generation source and cleanup paths."""

from __future__ import annotations

import os
from pathlib import Path

from transformation_portal.orchestrator.artifact_store.base import ArtifactStoreError

_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd


def open_directory(path: Path) -> int:
    """Open an absolute directory without following any mutable path symlink.

    Each component is opened relative to its already pinned parent. Callers
    own the returned descriptor and must close it even when later work fails.
    """
    if not path.is_absolute() or ".." in path.parts:
        raise ArtifactStoreError("generation filesystem path must be absolute and canonical")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY") or not _OPEN_SUPPORTS_DIR_FD:
        raise ArtifactStoreError("safe generation filesystem traversal is unavailable on this platform")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_source_file(path: Path) -> int:
    """Pin every ancestor before opening a nonblocking, non-symlink source."""
    parent = open_directory(path.parent)
    try:
        return os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    finally:
        os.close(parent)
