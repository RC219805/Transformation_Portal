"""
Core Security Module

Consolidated security validation and sanitization from:
- lux_depth_v2/hardening/
- src/transformation_portal/hardening/

Provides unified input validation, path traversal protection,
and secure file handling.
"""

from .validation import (
    InputValidator,
    ValidationResult,
    ValidationError,
)
from .path import (
    PathValidator,
    safe_resolve_path,
    is_safe_path,
)
from .sanitization import (
    SanitizationPolicy,
    sanitize_filename,
    validate_input_file,
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
