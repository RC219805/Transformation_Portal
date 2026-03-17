"""
Zero-Trust Filesystem Guard.

This module provides a policy-enforced filesystem boundary - a single
choke point for ALL disk IO. Every filesystem operation MUST go through
this class.

Trust Domains:
- "user": User-facing file operations (strict whitelist validation)
- "internal": Internal system paths (validated segments)
- "cas": Content-Addressable Storage (SHA256 validated)

Usage:
    from transformation_portal.core.security.fs_guard import FSGuard, FSContext

    fs = FSGuard(audit_log=Path("/var/log/fs_audit.jsonl"))
    ctx = FSContext(mode="user", base_dir=Path("/data/pipelines"))

    # Safe file access
    filepath = fs.user_file(ctx, "my-pipeline", suffix=".json")
    data = fs.read_text(filepath)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from .path_safety import (
    PathSafetyError,
    safe_cas_path,
    safe_join_file,
    safe_join_subpath,
    validate_safe_name,
    validate_sha256,
)

logger = logging.getLogger(__name__)

Operation = Literal["read", "write", "delete", "symlink", "mkdir", "list"]
TrustMode = Literal["user", "internal", "cas"]


class FSPolicyError(RuntimeError):
    """Raised when a filesystem operation violates security policy.

    This indicates either:
    - Wrong trust context for the operation
    - Attempted escape from base directory
    - Policy-violating operation type
    """

    pass


@dataclass(frozen=True)
class FSContext:
    """Defines trust boundary for filesystem operations.

    Attributes:
        mode: Trust domain - "user" for user input, "internal" for
              system paths, "cas" for content-addressable storage
        base_dir: Root directory that all operations are confined to
    """

    mode: TrustMode
    base_dir: Path

    def __post_init__(self) -> None:
        # Validate base_dir exists or can be created
        if not isinstance(self.base_dir, Path):
            raise FSPolicyError("base_dir must be a Path object")


class FSGuard:
    """Zero-trust filesystem interface.

    All filesystem operations MUST go through this class. It provides:
    - Explicit trust domains (user, internal, cas)
    - Centralized path validation
    - Audit logging of all operations
    - No reliance on .resolve() as security boundary

    Example:
        fs = FSGuard()
        ctx = FSContext(mode="user", base_dir=Path("/data"))

        # User file access (strictly validated)
        path = fs.user_file(ctx, "config", suffix=".json")
        data = fs.read_text(path)

        # Write with audit
        fs.write_text(path, '{"key": "value"}')
    """

    def __init__(self, audit_log: Optional[Path] = None) -> None:
        """Initialize FSGuard.

        Args:
            audit_log: Optional path to JSONL audit log file.
                      If provided, all operations are logged.
        """
        self.audit_log = audit_log
        self._operation_count = 0

    # -----------------------------
    # INTERNAL UTILITIES
    # -----------------------------
    def _log(self, op: Operation, path: Path, extra: Optional[dict] = None) -> None:
        """Log a filesystem operation for audit purposes."""
        self._operation_count += 1

        entry = {
            "ts": time.time(),
            "op": op,
            "path": str(path),
            "seq": self._operation_count,
        }
        if extra:
            entry.update(extra)

        logger.debug("FSGuard %s: %s", op, path)

        if not self.audit_log:
            return

        try:
            self.audit_log.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.warning("FSGuard audit log write failed: %s", e)

    def _ensure_within_base(self, base: Path, path: Path) -> None:
        """Verify path is within base directory.

        NOTE: This is a secondary check. Primary security comes from
        whitelist validation BEFORE path construction.
        """
        # Check if path starts with base_dir (no resolve - rely on construction)
        try:
            # Use parts comparison rather than resolve
            path_parts = path.parts
            base_parts = base.parts

            if len(path_parts) < len(base_parts):
                raise FSPolicyError(f"Path escapes base_dir: {path}")

            if path_parts[: len(base_parts)] != base_parts:
                raise FSPolicyError(f"Path escapes base_dir: {path}")

        except (ValueError, TypeError) as e:
            raise FSPolicyError(f"Invalid path comparison: {e}")

    # -----------------------------
    # USER FILE ACCESS
    # Strictly validated user-facing file operations
    # -----------------------------
    def user_file(self, ctx: FSContext, name: str, *, suffix: str) -> Path:
        """Construct a safe path for user-provided filename.

        Args:
            ctx: FSContext with mode="user"
            name: User-provided name (will be whitelist validated)
            suffix: File extension (must start with '.')

        Returns:
            Safe path within ctx.base_dir

        Raises:
            FSPolicyError: If context mode is not "user"
            PathSafetyError: If name fails validation
        """
        if ctx.mode != "user":
            raise FSPolicyError(f"user_file requires user context, got {ctx.mode!r}")

        path = safe_join_file(ctx.base_dir, name, suffix=suffix)
        self._ensure_within_base(ctx.base_dir, path)
        return path

    # -----------------------------
    # INTERNAL FILE ACCESS
    # For system paths with validated segments
    # -----------------------------
    def internal_path(self, ctx: FSContext, parts: list[str]) -> Path:
        """Construct a safe path for internal system use.

        Args:
            ctx: FSContext with mode="internal"
            parts: List of path segments (each validated)

        Returns:
            Safe path within ctx.base_dir

        Raises:
            FSPolicyError: If context mode is not "internal"
            PathSafetyError: If any segment fails validation
        """
        if ctx.mode != "internal":
            raise FSPolicyError(f"internal_path requires internal context, got {ctx.mode!r}")

        path = safe_join_subpath(ctx.base_dir, parts)
        self._ensure_within_base(ctx.base_dir, path)
        return path

    def internal_file(self, ctx: FSContext, parts: list[str], *, suffix: str) -> Path:
        """Construct a safe file path for internal system use.

        Args:
            ctx: FSContext with mode="internal"
            parts: List of directory segments (each validated)
            suffix: File extension

        Returns:
            Safe file path within ctx.base_dir
        """
        if ctx.mode != "internal":
            raise FSPolicyError(f"internal_file requires internal context, got {ctx.mode!r}")

        if not parts:
            raise PathSafetyError("At least one path segment required")

        # All but last are directory segments
        if len(parts) == 1:
            # Single part is the filename
            return safe_join_file(ctx.base_dir, parts[0], suffix=suffix)

        # Multiple parts: directories + filename
        dir_parts = parts[:-1]
        filename = parts[-1]

        dir_path = safe_join_subpath(ctx.base_dir, dir_parts)
        path = safe_join_file(dir_path, filename, suffix=suffix)
        self._ensure_within_base(ctx.base_dir, path)
        return path

    # -----------------------------
    # CAS ACCESS
    # Content-Addressable Storage with SHA256 validation
    # -----------------------------
    def cas_object(self, ctx: FSContext, sha256: str) -> Path:
        """Get path to a CAS object by SHA256.

        Args:
            ctx: FSContext with mode="cas"
            sha256: SHA256 hash (will be validated)

        Returns:
            Path using 2-char prefix sharding: base/ab/abcd...

        Raises:
            FSPolicyError: If context mode is not "cas"
            PathSafetyError: If SHA256 is invalid
        """
        if ctx.mode != "cas":
            raise FSPolicyError(f"cas_object requires cas context, got {ctx.mode!r}")

        return safe_cas_path(ctx.base_dir, sha256)

    # -----------------------------
    # FILE OPERATIONS
    # All disk IO goes through these methods
    # -----------------------------
    def read_text(self, path: Path, encoding: str = "utf-8") -> str:
        """Read text from a file.

        Args:
            path: Path to read (must be constructed via FSGuard methods)
            encoding: Text encoding (default: utf-8)

        Returns:
            File contents as string
        """
        self._log("read", path)
        return path.read_text(encoding=encoding)

    def read_bytes(self, path: Path) -> bytes:
        """Read bytes from a file.

        Args:
            path: Path to read

        Returns:
            File contents as bytes
        """
        self._log("read", path)
        return path.read_bytes()

    def write_text(
        self,
        path: Path,
        data: str,
        encoding: str = "utf-8",
        atomic: bool = True,
    ) -> None:
        """Write text to a file.

        Args:
            path: Path to write
            data: Text content
            encoding: Text encoding (default: utf-8)
            atomic: If True, write to temp file then rename (default: True)
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._log("write", path, {"size": len(data)})

        if atomic:
            # Atomic write: write to temp, then rename
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(data, encoding=encoding)
            tmp_path.rename(path)
        else:
            path.write_text(data, encoding=encoding)

    def write_bytes(self, path: Path, data: bytes, atomic: bool = True) -> None:
        """Write bytes to a file.

        Args:
            path: Path to write
            data: Binary content
            atomic: If True, write to temp file then rename
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._log("write", path, {"size": len(data)})

        if atomic:
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_bytes(data)
            tmp_path.rename(path)
        else:
            path.write_bytes(data)

    def delete(self, path: Path, missing_ok: bool = True) -> bool:
        """Delete a file.

        Args:
            path: Path to delete
            missing_ok: If True, don't raise if file doesn't exist

        Returns:
            True if file was deleted, False if it didn't exist
        """
        if path.exists():
            self._log("delete", path)
            path.unlink()
            return True
        elif not missing_ok:
            raise FileNotFoundError(f"File not found: {path}")
        return False

    def exists(self, path: Path) -> bool:
        """Check if a path exists."""
        return path.exists()

    def mkdir(self, path: Path, parents: bool = True) -> None:
        """Create a directory.

        Args:
            path: Directory path
            parents: If True, create parent directories
        """
        self._log("mkdir", path)
        path.mkdir(parents=parents, exist_ok=True)

    def list_dir(self, path: Path) -> list[Path]:
        """List directory contents.

        Args:
            path: Directory path

        Returns:
            List of paths in directory
        """
        self._log("list", path)
        return list(path.iterdir())

    def symlink(self, src: Path, dst: Path) -> None:
        """Create a symbolic link.

        Args:
            src: Source path (link target)
            dst: Destination path (link location)
        """
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() or dst.is_symlink():
            dst.unlink()

        self._log("symlink", dst, {"target": str(src)})
        dst.symlink_to(src)

    def copy(self, src: Path, dst: Path) -> None:
        """Copy a file.

        Args:
            src: Source path
            dst: Destination path
        """
        import shutil

        dst.parent.mkdir(parents=True, exist_ok=True)
        self._log("write", dst, {"copy_from": str(src)})
        shutil.copy2(src, dst)

    # -----------------------------
    # STATISTICS
    # -----------------------------
    @property
    def operation_count(self) -> int:
        """Number of operations performed through this guard."""
        return self._operation_count


# Global singleton for convenience (can be overridden in tests)
_default_guard: Optional[FSGuard] = None


def get_fs_guard() -> FSGuard:
    """Get the default FSGuard instance."""
    global _default_guard
    if _default_guard is None:
        _default_guard = FSGuard()
    return _default_guard


def set_fs_guard(guard: FSGuard) -> None:
    """Set the default FSGuard instance (useful for testing)."""
    global _default_guard
    _default_guard = guard
