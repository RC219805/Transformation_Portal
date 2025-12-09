"""
Path validation and traversal protection.

Prevents directory traversal attacks and ensures paths stay within allowed roots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class PathValidator:
    """
    Path validator with traversal protection.
    
    Ensures all paths stay within allowed root directories.
    """
    
    def __init__(self, allowed_roots: Optional[List[Path]] = None):
        """
        Initialize path validator.
        
        Args:
            allowed_roots: List of allowed root directories (None = allow all)
        """
        self.allowed_roots = [Path(root).resolve() for root in allowed_roots] if allowed_roots else None
    
    def validate(self, path: Path, must_exist: bool = False) -> bool:
        """
        Validate path is safe.
        
        Args:
            path: Path to validate
            must_exist: If True, path must exist
            
        Returns:
            True if path is valid
        """
        path = Path(path)
        
        # Check existence if required
        if must_exist and not path.exists():
            logger.warning(f"Path does not exist: {path}")
            return False
        
        # Resolve path (follows symlinks)
        try:
            resolved = path.resolve()
        except Exception as e:
            logger.warning(f"Failed to resolve path {path}: {e}")
            return False
        
        # Check against allowed roots
        if self.allowed_roots is not None:
            if not any(self._is_relative_to(resolved, root) for root in self.allowed_roots):
                logger.warning(f"Path outside allowed roots: {resolved}")
                return False
        
        return True
    
    def safe_resolve(self, path: Path, root: Optional[Path] = None) -> Path:
        """
        Safely resolve path and ensure it stays within root.
        
        Args:
            path: Path to resolve
            root: Root directory (uses first allowed root if None)
            
        Returns:
            Resolved path
            
        Raises:
            ValueError: If path escapes root
        """
        if root is None:
            if self.allowed_roots:
                root = self.allowed_roots[0]
            else:
                # No root specified, just resolve
                return Path(path).resolve()
        
        root = Path(root).resolve()
        resolved = Path(path).resolve()
        
        if not self._is_relative_to(resolved, root):
            raise ValueError(f"Path escapes allowed root: {resolved} not under {root}")
        
        return resolved
    
    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        """Check if path is relative to root."""
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


def safe_resolve_path(path: Path, root: Path) -> Path:
    """
    Safely resolve path within root directory.
    
    Args:
        path: Path to resolve
        root: Root directory
        
    Returns:
        Resolved path
        
    Raises:
        ValueError: If path escapes root
    """
    validator = PathValidator(allowed_roots=[root])
    return validator.safe_resolve(path, root)


def is_safe_path(path: Path, allowed_roots: Optional[List[Path]] = None) -> bool:
    """
    Check if path is safe (no traversal attacks).
    
    Args:
        path: Path to check
        allowed_roots: List of allowed root directories
        
    Returns:
        True if path is safe
    """
    validator = PathValidator(allowed_roots=allowed_roots)
    return validator.validate(path)
