"""
Core Security Module

Consolidated security validation and sanitization from:
- lux_depth_v2/hardening/
- src/transformation_portal/hardening/

Provides unified input validation, path traversal protection,
and secure file handling.
"""

from .model_lock import (
    ModelLockError,
    is_model_lock_strict_enabled,
    is_pinned_revision,
    load_model_lock_manifest,
    manifest_revision_for_repo,
    model_lock_manifest_path,
    resolve_model_lock_revision,
)
from .path import PathValidator, is_safe_path, safe_resolve_path
from .sanitization import SanitizationPolicy, sanitize_filename, validate_input_file
from .serialization import RestrictedUnpickler, safe_pickle_load
from .validation import InputValidator, ValidationError, ValidationResult

__all__ = [
    "InputValidator",
    "ValidationResult",
    "ValidationError",
    "ModelLockError",
    "load_model_lock_manifest",
    "manifest_revision_for_repo",
    "resolve_model_lock_revision",
    "is_pinned_revision",
    "is_model_lock_strict_enabled",
    "model_lock_manifest_path",
    "PathValidator",
    "safe_resolve_path",
    "is_safe_path",
    "SanitizationPolicy",
    "sanitize_filename",
    "validate_input_file",
    "RestrictedUnpickler",
    "safe_pickle_load",
]
