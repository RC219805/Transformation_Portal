"""
Input sanitization and policy enforcement.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple
import logging

from .validation import InputValidator, ValidationError

logger = logging.getLogger(__name__)


class SanitizationPolicy:
    """
    Input sanitization policy.
    
    Defines rules for sanitizing filenames and inputs.
    """
    
    # Dangerous filename patterns
    DANGEROUS_PATTERNS = [
        r"\.\.",  # Parent directory
        r"^/",  # Absolute path
        r"^~",  # Home directory
        r"[\x00-\x1f]",  # Control characters
    ]
    
    # Safe filename pattern (alphanumeric, dash, underscore, dot)
    SAFE_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+$')
    
    @classmethod
    def is_safe_filename(cls, filename: str) -> bool:
        """
        Check if filename is safe.
        
        Args:
            filename: Filename to check
            
        Returns:
            True if filename is safe
        """
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, filename):
                return False
        
        # Check length
        if len(filename) > 255:
            return False
        
        # Check against safe pattern
        return cls.SAFE_PATTERN.match(filename) is not None
    
    @classmethod
    def sanitize_filename(cls, filename: str, replacement: str = "_") -> str:
        """
        Sanitize filename by replacing unsafe characters.
        
        Args:
            filename: Filename to sanitize
            replacement: Replacement character for unsafe chars
            
        Returns:
            Sanitized filename
        """
        # Remove parent directory references
        filename = filename.replace("..", "")
        
        # Remove leading slashes and tildes
        filename = filename.lstrip("/~")
        
        # Remove control characters
        filename = re.sub(r'[\x00-\x1f]', '', filename)
        
        # Replace unsafe characters
        filename = re.sub(r'[^a-zA-Z0-9._-]', replacement, filename)
        
        # Limit length
        if len(filename) > 255:
            # Preserve extension
            stem = filename[:245]
            suffix = filename[-10:] if "." in filename[-10:] else ""
            filename = stem + suffix
        
        return filename


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename (convenience function).
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    return SanitizationPolicy.sanitize_filename(filename)


def validate_input_file(
    path: Path,
    allowed_extensions: Tuple[str, ...] = (".tif", ".tiff", ".jpg", ".jpeg", ".png"),
    max_size_mb: float = 500.0,
    strict: bool = True
) -> bool:
    """
    Validate input file (convenience function).
    
    Args:
        path: Path to file
        allowed_extensions: Allowed file extensions
        max_size_mb: Maximum file size in MB
        strict: If True, raise exception on failure
        
    Returns:
        True if valid (or raises ValidationError if strict=True)
        
    Raises:
        ValidationError: If strict=True and validation fails
    """
    validator = InputValidator(
        allowed_extensions=allowed_extensions,
        max_size_mb=max_size_mb,
        enable_magic_bytes=True
    )
    
    result = validator.validate_file(path, strict=strict)
    
    if not result.valid:
        if strict:
            raise ValidationError(
                f"File validation failed: {', '.join(result.errors)}",
                path=path
            )
        return False
    
    return True
