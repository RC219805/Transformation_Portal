"""
Secure Path Handling.

Prevents path traversal attacks (e.g., ../../../etc/passwd) by enforcing
strict directory confinement.
"""

import logging
import os
from pathlib import Path
from typing import List, Union

from .validation import ValidationError

logger = logging.getLogger(__name__)


class PathValidator:
    """Validates filesystem paths against security policies."""

    def __init__(self, allowed_roots: List[Union[str, Path]]):
        self.allowed_roots = [Path(p).resolve() for p in allowed_roots]

    def is_safe(self, path: Union[str, Path]) -> bool:
        """Check if path is within allowed roots."""
        try:
            target = Path(path).resolve()
            # On Windows, resolve() might handle different drive letters
            # strict=False allows checking non-existent paths (for outputs)

            for root in self.allowed_roots:
                # Check if target is inside root
                # Python 3.9+ has is_relative_to
                if hasattr(target, "is_relative_to"):
                    if target.is_relative_to(root):
                        return True
                else:
                    # Legacy fallback
                    try:
                        target.relative_to(root)
                        return True
                    except ValueError:
                        continue
            return False
        except (OSError, ValueError, RuntimeError):
            return False


def safe_resolve_path(path: Union[str, Path], allowed_root: Union[str, Path] = os.getcwd()) -> Path:
    """
    Resolve a path and ensure it sits within the allowed root.

    Args:
        path: Path to resolve (can be relative or absolute)
        allowed_root: Root directory that path must be within (default: cwd)

    Returns:
        Resolved absolute path

    Raises:
        ValidationError: If path attempts traversal out of root.
    """
    root = Path(allowed_root).resolve()
    target = Path(path).resolve()

    # Check traversal
    try:
        target.relative_to(root)
    except ValueError:
        logger.warning(f"Path traversal blocked: '{path}' resolved to '{target}' " f"which is outside allowed root '{root}'")
        raise ValidationError(f"Path traversal detected: {path} is outside {root}")

    logger.debug(f"Path validated: '{path}' -> '{target}' (within '{root}')")
    return target


def is_safe_path(path: Union[str, Path]) -> bool:
    """Quick check if path is safe (relative to CWD)."""
    try:
        safe_resolve_path(path)
        return True
    except (OSError, ValueError, RuntimeError, ValidationError):
        return False
