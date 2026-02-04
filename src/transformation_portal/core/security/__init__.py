"""
Core Security Module

Consolidated security validation and sanitization from:
- lux_depth_v2/hardening/
- src/transformation_portal/hardening/

Provides unified input validation, path traversal protection,
and secure file handling.
"""

from .path import (
    PathValidator,
    is_safe_path,
    safe_resolve_path,
)
from .sanitization import (
    SanitizationPolicy,
    sanitize_filename,
    validate_input_file,
)
from .validation import (
    InputValidator,
    ValidationError,
    ValidationResult,
)

__all__ = [
    "InputValidator",
    "ValidationResult",
    "ValidationError",
    "PathValidator",
    "safe_resolve_path",
    "is_safe_path",
    "SanitizationPolicy",
    "sanitize_filename",
    "validate_input_file",
]
